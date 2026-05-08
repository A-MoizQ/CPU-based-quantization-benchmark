import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset

from benchlib import write_json


def normalize_text(text):
    text = re.sub(r"\s+", " ", text or "").strip()
    return text


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def build_wikitext(max_samples):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    rows = []
    for item in ds:
        text = normalize_text(item.get("text", ""))
        if len(text) < 200 or text.startswith("="):
            continue
        rows.append({"id": f"wikitext2-{len(rows)}", "dataset": "wikitext2", "text": text})
        if len(rows) >= max_samples:
            break
    return rows


def build_lambada(max_samples):
    ds = load_dataset("lambada", split="validation")
    rows = []
    for item in ds:
        text = normalize_text(item.get("text", ""))
        if len(text) < 120:
            continue
        rows.append({"id": f"lambada-{len(rows)}", "dataset": "lambada", "text": text})
        if len(rows) >= max_samples:
            break
    return rows


def build_long_context(short_rows, max_samples, target_words):
    words = []
    for row in short_rows:
        words.extend(row["text"].split())
    rows = []
    cursor = 0
    stride = max(128, target_words // 2)
    while cursor + target_words <= len(words) and len(rows) < max_samples:
        text = " ".join(words[cursor : cursor + target_words])
        rows.append(
            {
                "id": f"long-wikitext2-{len(rows)}",
                "dataset": "long_wikitext2",
                "text": text,
                "source": "concatenated wikitext-2 test split",
                "target_words": target_words,
            }
        )
        cursor += stride
    return rows


def main():
    parser = argparse.ArgumentParser(description="Prepare small research datasets for CPU quantization benchmarks.")
    parser.add_argument("--output-dir", default="datasets/processed")
    parser.add_argument("--wikitext-samples", type=int, default=128)
    parser.add_argument("--lambada-samples", type=int, default=128)
    parser.add_argument("--long-samples", type=int, default=24)
    parser.add_argument(
        "--long-target-words",
        type=int,
        default=1400,
        help="Approximate words per long-context sample. 1400 words is usually around 1800-2200 tokens.",
    )
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    wikitext_rows = build_wikitext(args.wikitext_samples)
    lambada_rows = build_lambada(args.lambada_samples)
    long_rows = build_long_context(
        wikitext_rows,
        max_samples=args.long_samples,
        target_words=args.long_target_words,
    )

    files = {
        "wikitext2": out / "wikitext2.jsonl",
        "lambada": out / "lambada.jsonl",
        "long_wikitext2": out / "long_wikitext2.jsonl",
    }
    write_jsonl(files["wikitext2"], wikitext_rows)
    write_jsonl(files["lambada"], lambada_rows)
    write_jsonl(files["long_wikitext2"], long_rows)

    manifest = {
        "files": {name: str(path) for name, path in files.items()},
        "counts": {
            "wikitext2": len(wikitext_rows),
            "lambada": len(lambada_rows),
            "long_wikitext2": len(long_rows),
        },
        "long_target_words": args.long_target_words,
        "notes": [
            "Use wikitext2/lambada for general speed and lightweight quality checks.",
            "Use long_wikitext2 for KV-cache footprint and decode-speed stress tests.",
        ],
    }
    write_json(out / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
