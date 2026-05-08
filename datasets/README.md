# Datasets

This folder is for benchmark datasets and derived prompt files.

Run:

```bash
python scripts/prepare_datasets.py
```

The script writes small JSONL files under `datasets/processed/`:

- `wikitext2.jsonl`: short/medium language-modeling prompts.
- `lambada.jsonl`: prediction-oriented prompts.
- `long_wikitext2.jsonl`: concatenated long-context prompts for KV-cache tests.
- `manifest.json`: dataset sizes and generation settings.

The downloaded Hugging Face cache is not stored in this repo.
