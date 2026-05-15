# Datasets

The current experiment uses only the processed WikiText-2 JSONL files:

- `datasets/processed/wikitext2.jsonl` for the AWQ INT4 CPU baseline.
- `datasets/processed/long_wikitext2.jsonl` for TurboQuant-style and RotorQuant-style KV-cache tests.

Regenerate them with:

```bash
python scripts/prepare_datasets.py
```

The processed files are ignored because they can be regenerated. The benchmark result files that describe the runs are stored in `results/`.

Not used in the current comparison: `lambada.jsonl` and broader evaluation datasets. Those should be added later for paper-grade accuracy reporting.
