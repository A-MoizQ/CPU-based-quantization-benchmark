import argparse
import json
import math
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchlib import (
    MemorySampler,
    append_jsonl,
    bytes_to_gib,
    bytes_to_mib,
    current_rss_bytes,
    directory_size_bytes,
    now_iso,
    read_jsonl,
    system_info,
    write_json,
)


def kv_cache_bytes(past_key_values):
    if past_key_values is None:
        return 0
    total = 0
    for layer in past_key_values:
        if isinstance(layer, (tuple, list)):
            tensors = layer
        else:
            tensors = tuple(getattr(layer, name) for name in ("key_cache", "value_cache") if hasattr(layer, name))
        for tensor in tensors[:2]:
            if torch.is_tensor(tensor):
                total += tensor.numel() * tensor.element_size()
    return total


def load_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, trust_remote_code=args.trust_remote_code)
    load_kwargs = {
        "device_map": None,
        "low_cpu_mem_usage": True,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.dtype == "float32":
        load_kwargs["torch_dtype"] = torch.float32
    elif args.dtype == "bfloat16":
        load_kwargs["torch_dtype"] = torch.bfloat16
    elif args.dtype == "float16":
        load_kwargs["torch_dtype"] = torch.float16

    if args.loader == "gptq":
        from transformers import GPTQConfig

        load_kwargs["quantization_config"] = GPTQConfig(bits=args.bits, use_exllama=False)
    elif args.loader == "awq":
        from awq import AutoAWQForCausalLM

        awq_device_map = {"": "cpu"} if args.awq_device_map == "cpu" else args.awq_device_map
        awq_kwargs = {
            "device_map": awq_device_map,
            "fuse_layers": False,
            "safetensors": True,
            "trust_remote_code": args.trust_remote_code,
        }
        if args.awq_device_map != "cpu" and args.offload_folder:
            awq_kwargs["offload_folder"] = args.offload_folder
        model = AutoAWQForCausalLM.from_quantized(
            args.model_path,
            **awq_kwargs,
        )
        model.eval()
        return tokenizer, model

    model = AutoModelForCausalLM.from_pretrained(args.model_path, **load_kwargs)
    model.to("cpu")
    model.eval()
    return tokenizer, model


def model_forward(model, **kwargs):
    try:
        return model(**kwargs)
    except TypeError:
        inner = getattr(model, "model", None)
        if inner is None:
            raise
        return inner(**kwargs)


def run_one_sample(model, tokenizer, row, args):
    encoded = tokenizer(
        row["text"],
        return_tensors="pt",
        truncation=True,
        max_length=args.max_input_tokens,
    )
    input_ids = encoded["input_ids"].to("cpu")
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to("cpu")

    labels = input_ids.clone() if args.compute_loss and input_ids.shape[-1] > 1 else None
    forward_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask, "use_cache": True}
    if labels is not None:
        forward_kwargs["labels"] = labels

    with torch.no_grad():
        start = time.perf_counter()
        outputs = model_forward(model, **forward_kwargs)
        prefill_seconds = time.perf_counter() - start

    past = getattr(outputs, "past_key_values", None)
    kv_prefill_bytes = kv_cache_bytes(past)
    loss = getattr(outputs, "loss", None)
    loss_value = float(loss.detach().cpu().item()) if loss is not None else None
    ppl = math.exp(loss_value) if loss_value is not None and loss_value < 20 else None

    decode_tokens = 0
    decode_seconds = 0.0
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    with torch.no_grad():
        start = time.perf_counter()
        for _ in range(args.max_new_tokens):
            outputs = model_forward(model, input_ids=next_token, past_key_values=past, use_cache=True)
            past = getattr(outputs, "past_key_values", None)
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            decode_tokens += 1
        decode_seconds = time.perf_counter() - start

    return {
        "sample_id": row.get("id"),
        "dataset": row.get("dataset"),
        "input_tokens": int(input_ids.shape[-1]),
        "generated_tokens": decode_tokens,
        "prefill_seconds": prefill_seconds,
        "prefill_tokens_per_second": float(input_ids.shape[-1] / prefill_seconds) if prefill_seconds else 0.0,
        "decode_seconds": decode_seconds,
        "decode_tokens_per_second": float(decode_tokens / decode_seconds) if decode_seconds else 0.0,
        "kv_cache_prefill_mib": bytes_to_mib(kv_prefill_bytes),
        "kv_cache_after_decode_mib": bytes_to_mib(kv_cache_bytes(past)),
        "loss": loss_value,
        "perplexity": ppl,
        "rss_after_sample_mib": bytes_to_mib(current_rss_bytes()),
    }


def summarize(records):
    numeric_keys = [
        "input_tokens",
        "generated_tokens",
        "prefill_seconds",
        "prefill_tokens_per_second",
        "decode_seconds",
        "decode_tokens_per_second",
        "kv_cache_prefill_mib",
        "kv_cache_after_decode_mib",
        "loss",
        "perplexity",
        "rss_after_sample_mib",
    ]
    summary = {"num_samples": len(records)}
    for key in numeric_keys:
        values = [r[key] for r in records if r.get(key) is not None]
        if values:
            summary[f"{key}_mean"] = sum(values) / len(values)
            summary[f"{key}_min"] = min(values)
            summary[f"{key}_max"] = max(values)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Benchmark Hugging Face causal LM artifacts on CPU.")
    parser.add_argument("--model-path", required=True, help="HF model id or local quantized model folder.")
    parser.add_argument("--method", required=True, help="Label for the method, e.g. fp32, gptq, awq, smoothquant.")
    parser.add_argument("--dataset", default="datasets/processed/wikitext2.jsonl")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--max-input-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--loader", choices=["auto", "gptq", "awq"], default="auto")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--compute-loss", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--offload-folder", default="results/offload")
    parser.add_argument(
        "--awq-device-map",
        default="cpu",
        help="Device map passed to AutoAWQ. CPU benchmarking should use the default 'cpu'.",
    )
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(max(1, min(args.threads, 2)))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.method}_{Path(args.dataset).stem}_{int(time.time())}"
    sample_path = out_dir / f"{stem}.jsonl"
    summary_path = out_dir / f"{stem}_summary.json"

    rows = read_jsonl(args.dataset, limit=args.max_samples)
    meta = {
        "created_at": now_iso(),
        "method": args.method,
        "loader": args.loader,
        "model_path": args.model_path,
        "dataset": args.dataset,
        "system": system_info(),
        "args": vars(args),
        "artifact_size_gib": bytes_to_gib(directory_size_bytes(args.model_path)),
    }

    with MemorySampler() as sampler:
        load_start = time.perf_counter()
        tokenizer, model = load_model(args)
        load_seconds = time.perf_counter() - load_start
        load_rss_mib = bytes_to_mib(current_rss_bytes())

        records = []
        for idx, row in enumerate(rows):
            record = run_one_sample(model, tokenizer, row, args)
            record.update({"method": args.method, "sample_index": idx})
            append_jsonl(sample_path, record)
            records.append(record)

    summary = summarize(records)
    summary.update(meta)
    summary.update(
        {
            "load_seconds": load_seconds,
            "rss_after_load_mib": load_rss_mib,
            "peak_rss_mib": bytes_to_mib(sampler.peak_rss),
            "peak_rss_delta_mib": bytes_to_mib(sampler.peak_delta_bytes),
            "sample_metrics_jsonl": str(sample_path),
        }
    )
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
