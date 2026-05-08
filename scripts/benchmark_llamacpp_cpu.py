import argparse
import json
import shlex
import subprocess
import time
from pathlib import Path

from benchlib import MemorySampler, bytes_to_gib, bytes_to_mib, directory_size_bytes, now_iso, read_jsonl, system_info, write_json


def build_command(args, prompt):
    cmd = [
        args.binary,
        "-m",
        args.model,
        "-p",
        prompt,
        "-n",
        str(args.max_new_tokens),
        "-c",
        str(args.ctx_size),
        "-t",
        str(args.threads),
    ]
    if args.extra_args:
        cmd.extend(shlex.split(args.extra_args))
    return cmd


def run_sample(args, row):
    prompt = row["text"]
    if len(prompt) > args.max_prompt_chars:
        prompt = prompt[: args.max_prompt_chars]
    cmd = build_command(args, prompt)
    start = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    with MemorySampler(pid=proc.pid) as sampler:
        stdout, stderr = proc.communicate()
    elapsed = time.perf_counter() - start
    return {
        "sample_id": row.get("id"),
        "dataset": row.get("dataset"),
        "returncode": proc.returncode,
        "elapsed_seconds": elapsed,
        "tokens_per_second": args.max_new_tokens / elapsed if elapsed else None,
        "peak_rss_mib": bytes_to_mib(sampler.peak_rss),
        "peak_rss_delta_mib": bytes_to_mib(sampler.peak_delta_bytes),
        "stdout_tail": stdout[-2000:],
        "stderr_tail": stderr[-4000:],
        "command": cmd,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark llama.cpp-compatible CPU inference.")
    parser.add_argument("--binary", required=True, help="Path to llama-cli/main binary.")
    parser.add_argument("--model", required=True, help="Path to GGUF model.")
    parser.add_argument("--method", required=True, help="Label, e.g. gguf_q4, turboquant_llamacpp, rotorquant_llamacpp.")
    parser.add_argument("--dataset", default="datasets/processed/long_wikitext2.jsonl")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--ctx-size", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-prompt-chars", type=int, default=7000)
    parser.add_argument(
        "--extra-args",
        default="",
        help="Extra llama.cpp args, e.g. cache quantization flags from a TurboQuant/RotorQuant fork.",
    )
    args = parser.parse_args()

    rows = read_jsonl(args.dataset, limit=args.max_samples)
    records = [run_sample(args, row) for row in rows]
    ok = [r for r in records if r["returncode"] == 0]
    elapsed = [r["elapsed_seconds"] for r in ok]
    tok_s = [r["tokens_per_second"] for r in ok if r.get("tokens_per_second") is not None]
    peak = [r["peak_rss_mib"] for r in records]

    summary = {
        "created_at": now_iso(),
        "method": args.method,
        "binary": args.binary,
        "model": args.model,
        "dataset": args.dataset,
        "args": vars(args),
        "system": system_info(),
        "model_size_gib": bytes_to_gib(directory_size_bytes(args.model)),
        "num_samples": len(records),
        "num_success": len(ok),
        "elapsed_seconds_mean": sum(elapsed) / len(elapsed) if elapsed else None,
        "tokens_per_second_mean": sum(tok_s) / len(tok_s) if tok_s else None,
        "peak_rss_mib_max": max(peak) if peak else None,
        "records": records,
    }

    out = Path(args.output_dir) / f"{args.method}_llamacpp_{Path(args.dataset).stem}_{int(time.time())}.json"
    write_json(out, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
