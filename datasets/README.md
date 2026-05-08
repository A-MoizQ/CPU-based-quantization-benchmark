# Datasets

This folder is for benchmark datasets and derived prompt files.

The documented experiment model for this project is `TinyLlama/TinyLlama-1.1B-Chat-v1.0`. The dataset files are model-independent JSONL prompt files, but the benchmark commands in the root README assume TinyLlama for both Colab quantization and CPU tests.

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
