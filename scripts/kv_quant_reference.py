import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchlib import (
    MemorySampler,
    bytes_to_mib,
    current_rss_bytes,
    now_iso,
    read_jsonl,
    system_info,
    write_json,
)


def make_random_orthogonal(dim, dtype, seed=1234):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    matrix = torch.randn(dim, dim, generator=generator, dtype=torch.float32)
    q, _ = torch.linalg.qr(matrix)
    return q.to(dtype=dtype)


def random_3d_rotations(n_blocks, dtype, seed=5678):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    axis = torch.randn(n_blocks, 3, generator=generator, dtype=torch.float32)
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    angle = torch.rand(n_blocks, generator=generator, dtype=torch.float32) * (2 * math.pi)
    kx, ky, kz = axis.unbind(-1)
    c = torch.cos(angle)
    s = torch.sin(angle)
    one_c = 1 - c
    r = torch.zeros(n_blocks, 3, 3, dtype=torch.float32)
    r[:, 0, 0] = c + kx * kx * one_c
    r[:, 0, 1] = kx * ky * one_c - kz * s
    r[:, 0, 2] = kx * kz * one_c + ky * s
    r[:, 1, 0] = ky * kx * one_c + kz * s
    r[:, 1, 1] = c + ky * ky * one_c
    r[:, 1, 2] = ky * kz * one_c - kx * s
    r[:, 2, 0] = kz * kx * one_c - ky * s
    r[:, 2, 1] = kz * ky * one_c + kx * s
    r[:, 2, 2] = c + kz * kz * one_c
    return r.to(dtype=dtype)


def apply_block_rotations(x, rotations, inverse=False):
    orig_shape = x.shape
    dim = orig_shape[-1]
    pad = (-dim) % 3
    if pad:
        x = F.pad(x, (0, pad))
    blocks = x.reshape(*x.shape[:-1], -1, 3)
    r = rotations.transpose(-1, -2) if inverse else rotations
    y = torch.einsum("...bi,bij->...bj", blocks, r)
    y = y.reshape(*x.shape[:-2], -1)
    if pad:
        y = y[..., :dim]
    return y.reshape(orig_shape)


def quantize_symmetric(flat, bits):
    qmax = (2 ** (bits - 1)) - 1
    scale = flat.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / qmax
    q = torch.round(flat / scale).clamp(-qmax, qmax).to(torch.int8)
    return q, scale


def quantize_turbo(x, bits, rotation):
    shape = x.shape
    dim = shape[-1]
    flat = x.reshape(-1, dim).to(rotation.dtype)
    rotated = flat @ rotation
    q, scale = quantize_symmetric(rotated, bits)
    deq = (q.to(rotation.dtype) * scale) @ rotation.T
    return q, scale, deq.reshape(shape).to(x.dtype)


def quantize_rotor(x, bits, rotations):
    shape = x.shape
    dim = shape[-1]
    rotated = apply_block_rotations(x, rotations, inverse=False)
    flat = rotated.reshape(-1, dim)
    q, scale = quantize_symmetric(flat, bits)
    deq_rotated = (q.to(rotated.dtype) * scale).reshape(shape)
    deq = apply_block_rotations(deq_rotated, rotations, inverse=True)
    return q, scale, deq.to(x.dtype)


def kv_cache_bytes(past_key_values):
    total = 0
    for key, value in past_key_values:
        total += key.numel() * key.element_size()
        total += value.numel() * value.element_size()
    return total


def cosine_mse(original, reconstructed):
    a = original.reshape(-1, original.shape[-1]).float()
    b = reconstructed.reshape(-1, reconstructed.shape[-1]).float()
    cos = F.cosine_similarity(a, b, dim=-1)
    return {
        "cosine_mean": float(cos.mean().item()),
        "cosine_min": float(cos.min().item()),
        "mse": float(F.mse_loss(b, a).item()),
    }


def packed_bits_for(key, value, bits, include_qjl):
    head_dim = key.shape[-1]
    n_values = key.numel() + value.numel()
    n_vectors = key.reshape(-1, head_dim).shape[0] + value.reshape(-1, head_dim).shape[0]
    packed = n_values * bits
    packed += n_vectors * 16
    if include_qjl:
        packed += n_values
    return packed


def main():
    parser = argparse.ArgumentParser(description="Reference CPU KV-cache compression benchmark for TurboQuant/RotorQuant.")
    parser.add_argument("--model-path", default="facebook/opt-125m")
    parser.add_argument("--method", choices=["turboquant", "rotorquant"], required=True)
    parser.add_argument("--dataset", default="datasets/processed/long_wikitext2.jsonl")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=1536)
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--include-qjl-residual-bits", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(args.dataset, limit=args.max_samples)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    model.to("cpu")
    model.eval()

    records = []
    with MemorySampler() as sampler:
        for idx, row in enumerate(rows):
            encoded = tokenizer(row["text"], return_tensors="pt", truncation=True, max_length=args.max_input_tokens)
            with torch.no_grad():
                start = time.perf_counter()
                outputs = model(**encoded, use_cache=True)
                prefill_seconds = time.perf_counter() - start
            past = outputs.past_key_values
            head_dim = past[0][0].shape[-1]
            transform = (
                make_random_orthogonal(head_dim, past[0][0].dtype)
                if args.method == "turboquant"
                else random_3d_rotations(math.ceil(head_dim / 3), past[0][0].dtype)
            )

            layer_reports = []
            packed_bits = 0
            quant_seconds = 0.0
            for layer_idx, (key, value) in enumerate(past):
                start = time.perf_counter()
                if args.method == "turboquant":
                    _, _, key_deq = quantize_turbo(key, args.bits, transform)
                    _, _, value_deq = quantize_turbo(value, args.bits, transform)
                else:
                    _, _, key_deq = quantize_rotor(key, args.bits, transform)
                    _, _, value_deq = quantize_rotor(value, args.bits, transform)
                quant_seconds += time.perf_counter() - start
                packed_bits += packed_bits_for(key, value, args.bits, args.include_qjl_residual_bits)
                key_report = cosine_mse(key, key_deq)
                value_report = cosine_mse(value, value_deq)
                layer_reports.append({"layer": layer_idx, "key": key_report, "value": value_report})

            fp_bytes = kv_cache_bytes(past)
            packed_bytes = math.ceil(packed_bits / 8)
            records.append(
                {
                    "sample_index": idx,
                    "sample_id": row.get("id"),
                    "dataset": row.get("dataset"),
                    "method": args.method,
                    "bits": args.bits,
                    "input_tokens": int(encoded["input_ids"].shape[-1]),
                    "prefill_seconds": prefill_seconds,
                    "quantize_dequantize_seconds": quant_seconds,
                    "fp_cache_mib": bytes_to_mib(fp_bytes),
                    "packed_cache_mib": bytes_to_mib(packed_bytes),
                    "compression_ratio": fp_bytes / packed_bytes if packed_bytes else None,
                    "rss_after_sample_mib": bytes_to_mib(current_rss_bytes()),
                    "layer_reports": layer_reports,
                }
            )

    means = {}
    for key in ("prefill_seconds", "quantize_dequantize_seconds", "fp_cache_mib", "packed_cache_mib", "compression_ratio"):
        values = [r[key] for r in records if r.get(key) is not None]
        if values:
            means[f"{key}_mean"] = sum(values) / len(values)

    summary = {
        "created_at": now_iso(),
        "method": args.method,
        "model_path": args.model_path,
        "dataset": args.dataset,
        "args": vars(args),
        "system": system_info(),
        "num_samples": len(records),
        "peak_rss_mib": bytes_to_mib(sampler.peak_rss),
        "records": records,
        **means,
    }
    output_path = out_dir / f"{args.method}_kv_reference_{Path(args.dataset).stem}_{int(time.time())}.json"
    write_json(output_path, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
