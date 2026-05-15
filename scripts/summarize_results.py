import argparse
import json
from pathlib import Path


SUMMARY_KEYS = [
    "method",
    "model_path",
    "model",
    "dataset",
    "num_samples",
    "artifact_size_gib",
    "model_size_gib",
    "load_seconds",
    "rss_after_load_mib",
    "peak_rss_mib",
    "peak_rss_mib_max",
    "decode_tokens_per_second_mean",
    "tokens_per_second_mean",
    "prefill_tokens_per_second_mean",
    "kv_cache_prefill_mib_mean",
    "kv_cache_after_decode_mib_mean",
    "fp_cache_mib_mean",
    "packed_cache_mib_mean",
    "compression_ratio_mean",
    "perplexity_mean",
]


def load_json_files(root):
    for path in sorted(Path(root).glob("*.json")):
        if path.name.endswith("_comparison.json"):
            continue
        try:
            yield path, json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue


def main():
    parser = argparse.ArgumentParser(description="Summarize benchmark JSON files into a Markdown table.")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    rows = []
    for path, payload in load_json_files(args.results_dir):
        row = {"file": path.name}
        for key in SUMMARY_KEYS:
            value = payload.get(key)
            if isinstance(value, float):
                value = round(value, 4)
            row[key] = value
        rows.append(row)

    columns = ["file"] + SUMMARY_KEYS
    print("| " + " | ".join(columns) + " |")
    print("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        print("| " + " | ".join("" if row.get(col) is None else str(row.get(col)) for col in columns) + " |")


if __name__ == "__main__":
    main()
