import argparse
import json
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def ratio(new, old):
    if new is None or old in (None, 0):
        return None
    return new / old


def reduction(old, new):
    if new is None or old in (None, 0):
        return None
    return 1.0 - (new / old)


def pick(payload, *keys):
    for key in keys:
        if payload.get(key) is not None:
            return payload[key]
    return None


def main():
    parser = argparse.ArgumentParser(description="Compare one benchmark result against a baseline.")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    args = parser.parse_args()

    base = load(args.baseline)
    cand = load(args.candidate)

    base_decode = pick(base, "decode_tokens_per_second_mean", "tokens_per_second_mean")
    cand_decode = pick(cand, "decode_tokens_per_second_mean", "tokens_per_second_mean")
    base_memory = pick(base, "peak_rss_mib", "peak_rss_mib_max", "rss_after_load_mib")
    cand_memory = pick(cand, "peak_rss_mib", "peak_rss_mib_max", "rss_after_load_mib")
    base_artifact = pick(base, "artifact_size_gib", "model_size_gib")
    cand_artifact = pick(cand, "artifact_size_gib", "model_size_gib")
    base_kv = pick(base, "kv_cache_after_decode_mib_mean", "fp_cache_mib_mean")
    cand_kv = pick(cand, "kv_cache_after_decode_mib_mean", "packed_cache_mib_mean")

    comparison = {
        "baseline": args.baseline,
        "candidate": args.candidate,
        "baseline_method": base.get("method"),
        "candidate_method": cand.get("method"),
        "decode_speedup_x": ratio(cand_decode, base_decode),
        "peak_memory_reduction_fraction": reduction(base_memory, cand_memory),
        "peak_memory_reduction_percent": None,
        "artifact_size_reduction_fraction": reduction(base_artifact, cand_artifact),
        "artifact_size_reduction_percent": None,
        "kv_cache_reduction_fraction": reduction(base_kv, cand_kv),
        "kv_cache_reduction_percent": None,
        "baseline_decode_metric": base_decode,
        "candidate_decode_metric": cand_decode,
        "baseline_peak_memory_mib": base_memory,
        "candidate_peak_memory_mib": cand_memory,
        "baseline_artifact_size_gib": base_artifact,
        "candidate_artifact_size_gib": cand_artifact,
        "baseline_kv_cache_mib": base_kv,
        "candidate_kv_cache_mib": cand_kv,
    }
    for key in ("peak_memory_reduction", "artifact_size_reduction", "kv_cache_reduction"):
        frac = comparison[f"{key}_fraction"]
        comparison[f"{key}_percent"] = None if frac is None else frac * 100
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
