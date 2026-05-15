# LLM Quantization Comparison

This repo is a Colab-ready and CPU-benchmark scaffold for comparing Google TurboQuant against GPTQ, AWQ, SmoothQuant, RotorQuant, and a standard INT8 baseline.

The important distinction:

- GPTQ, AWQ, SmoothQuant, and LLM.int8 mostly quantize model weights or weight/activation compute and can produce saved model artifacts.
- TurboQuant and RotorQuant are KV-cache compression methods. They do not produce a normal quantized weight checkpoint; their artifact is a compressed runtime cache/backend plus memory and reconstruction metrics.

## Model Targets

| Purpose | Model/artifact |
| --- | --- |
| SmoothQuant (OPT-only library) | `facebook/opt-125m` |
| All other methods — HF/Colab | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| Practical CPU test | TinyLlama 1.1B GGUF through `llama.cpp` |

**SmoothQuant exception:** the official `mit-han-lab/smoothquant` library (`smooth_lm`, `quantize_opt`, `Int8OPTForCausalLM`) is hardcoded to the OPT architecture. TinyLlama/LLaMA cannot be used with it. All other notebooks use TinyLlama for a consistent comparison. Document this difference when presenting results.

On 4 GB RAM, `llama.cpp`/GGUF is the more realistic CPU path than full PyTorch for 1B+ models.

## Status

| Notebook | Method | Model | Artifact | Status |
| --- | --- | --- | --- | --- |
| `notebooks/00_fp16_baseline.ipynb` | FP16 baseline | TinyLlama | HF model folder | ⬜ pending |
| `notebooks/01_llm_int8_bitsandbytes.ipynb` | LLM.int8 | TinyLlama | HF/bitsandbytes model folder | ⬜ pending |
| `notebooks/02_gptq_gptqmodel.ipynb` | GPTQ INT4 | TinyLlama | GPTQ model folder | ⬜ pending |
| `notebooks/03_awq_autoawq.ipynb` | AWQ INT4 | TinyLlama | AWQ model folder | ⬜ pending |
| `notebooks/04_smoothquant_official.ipynb` | SmoothQuant W8A8 | OPT-125m | scales `.pt` (fake W8A8) | ✅ done |
| `notebooks/05_turboquant_kv_cache.ipynb` | TurboQuant KV cache | TinyLlama | `compressed_kv_cache.pt` + metrics | ✅ done |
| `notebooks/06_rotorquant_kv_cache.ipynb` | RotorQuant KV cache | TinyLlama | `compressed_kv_cache.pt` + metrics | ✅ done |

CPU benchmark runs (via scripts) are pending for all methods.

## Current Results Layout

```text
results/
├── smoothquant/
│   ├── metrics_path2.json          ← fake W8A8 metrics (opt-125m)
│   └── smoothquant_scales_and_config.pt
├── turboquant_kv/
│   ├── metrics.json                ← KV compression + cosine/MSE metrics
│   └── compressed_kv_cache.pt
└── rotorquant_kv/
    ├── metrics.json                ← KV compression + cosine/MSE metrics
    └── compressed_kv_cache.pt
```

Pending notebooks will add:

```text
results/
├── fp16_baseline/model/
├── llm_int8/model/
├── gptq_int4/model/
└── awq_int4/model/
```

CPU benchmark script outputs write to `results/` as timestamped `.jsonl` and `_summary.json` files.

## Quick CPU Runbook

Use this path for the actual 4 GB RAM / i5 6th gen experiments.

1. Create a Python environment and install the CPU tooling:

```bash
python -m pip install -r requirements-cpu.txt
```

2. Prepare the small benchmark datasets (already done — `datasets/processed/` is populated):

```bash
python scripts/prepare_datasets.py
```

3. Run a small TinyLlama CPU check before trying every artifact:

```bash
python scripts/benchmark_transformers_cpu.py \
  --model-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --method tinyllama_fp32_check \
  --dataset datasets/processed/wikitext2.jsonl \
  --max-samples 2 \
  --max-input-tokens 256 \
  --max-new-tokens 8 \
  --threads 4
```

4. Benchmark the saved AWQ artifact:

```bash
python scripts/benchmark_transformers_cpu.py \
  --model-path weights/awq_int4/model \
  --method awq \
  --loader awq \
  --dataset datasets/processed/wikitext2.jsonl \
  --max-samples 8 \
  --max-input-tokens 512 \
  --max-new-tokens 16 \
  --threads 4 \
  --compute-loss
```

5. Benchmark TurboQuant/RotorQuant-style KV-cache footprint on long prompts:

```bash
python scripts/kv_quant_reference.py \
  --model-path weights/awq_int4/model \
  --method turboquant \
  --loader awq \
  --dataset datasets/processed/long_wikitext2.jsonl \
  --max-samples 4 \
  --max-input-tokens 1536 \
  --bits 3 \
  --threads 4

python scripts/kv_quant_reference.py \
  --model-path weights/awq_int4/model \
  --method rotorquant \
  --loader awq \
  --dataset datasets/processed/long_wikitext2.jsonl \
  --max-samples 4 \
  --max-input-tokens 1536 \
  --bits 3 \
  --threads 4
```

6. Summarize and compare results:

```bash
python scripts/summarize_results.py --results-dir results

python scripts/compare_results.py \
  --baseline results/fp32_<dataset>_<timestamp>_summary.json \
  --candidate results/gptq_<dataset>_<timestamp>_summary.json
```

## Notebook Index

Run the FP16 baseline first, then each quantization notebook on the same Colab GPU, model, prompt, and token count.

| Notebook | Method | Artifact | Primary use |
| --- | --- | --- | --- |
| `notebooks/00_fp16_baseline.ipynb` | FP16 baseline | HF model folder | Baseline memory and generation speed |
| `notebooks/01_llm_int8_bitsandbytes.ipynb` | LLM.int8 | HF/bitsandbytes model folder | Standard INT8 baseline |
| `notebooks/02_gptq_gptqmodel.ipynb` | GPTQ INT4 | GPTQ model folder | Weight-only 4-bit checkpoint |
| `notebooks/03_awq_autoawq.ipynb` | AWQ INT4 | AWQ model folder | Weight-only 4-bit checkpoint |
| `notebooks/04_smoothquant_official.ipynb` | SmoothQuant W8A8 | smoothing scales `.pt` | W8A8 path — OPT-125m only |
| `notebooks/05_turboquant_kv_cache.ipynb` | TurboQuant-style KV cache | `compressed_kv_cache.pt` + metrics | KV-cache memory study |
| `notebooks/06_rotorquant_kv_cache.ipynb` | RotorQuant-style KV cache | `compressed_kv_cache.pt` + metrics | KV-cache memory study |

## Research Sources

| Method | Paper | Code used or referenced | Short quote |
| --- | --- | --- | --- |
| TurboQuant | [arXiv:2504.19874](https://arxiv.org/abs/2504.19874), [HF paper page](https://huggingface.co/papers/2504.19874) | [tonbistudio/turboquant-pytorch](https://github.com/tonbistudio/turboquant-pytorch), [OnlyTerp/turboquant](https://github.com/OnlyTerp/turboquant) | "absolute quality neutrality with 3.5 bits per channel" |
| RotorQuant | [Scrya technical report](https://scrya.com/rotorquant) | [scrya-com/rotorquant](https://github.com/scrya-com/rotorquant) | "44x fewer parameters" |
| GPTQ | [arXiv:2210.17323](https://arxiv.org/abs/2210.17323), [HF paper page](https://huggingface.co/papers/2210.17323) | [IST-DASLab/gptq](https://github.com/IST-DASLab/gptq), [HF GPTQ docs](https://huggingface.co/docs/transformers/quantization/gptq) | "accurate post-training quantization" |
| AWQ | [arXiv:2306.00978](https://arxiv.org/abs/2306.00978), [MLSys page](https://proceedings.mlsys.org/paper_files/paper/2024/hash/42a452cbafa9dd64e9ba4aa95cc1ef21-Abstract-Conference.html) | [mit-han-lab/llm-awq](https://github.com/mit-han-lab/llm-awq), [HF AWQ docs](https://huggingface.co/docs/transformers/quantization/awq) | "1% of salient weights" |
| SmoothQuant | [arXiv:2211.10438](https://arxiv.org/abs/2211.10438), [HF paper page](https://huggingface.co/papers/2211.10438) | [mit-han-lab/smoothquant](https://github.com/mit-han-lab/smoothquant), [Intel Neural Compressor docs](https://intel.github.io/neural-compressor/latest/docs/source/smooth_quant.html) | "training-free, accuracy-preserving" |
| LLM.int8 | [arXiv:2208.07339](https://arxiv.org/abs/2208.07339), [HF paper page](https://huggingface.co/papers/2208.07339) | [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes), [HF bitsandbytes docs](https://huggingface.co/docs/transformers/quantization/bitsandbytes) | "only half the required memory" |

Source status checked on 2026-05-09.

## CPU Inference Caveats

Saved model folders are not equally portable to CPU:

- GPTQ: most CPU-friendly among the 4-bit notebook paths. Reload with `GPTQConfig(bits=4, use_exllama=False)`.
- SmoothQuant: the fake W8A8 artifact (scales `.pt`) does not load as a standard HF checkpoint for CPU inference. For CPU, the recommended path is an ONNX Runtime or Intel Neural Compressor export. Treat this as an experimental limitation in the write-up.
- AWQ: optimized primarily for CUDA/TinyChat-style kernels; CPU loading depends on the runtime stack.
- LLM.int8: bitsandbytes is primarily a CUDA inference path; CPU support varies by installed bitsandbytes backend.
- TurboQuant/RotorQuant: KV-cache methods — CPU testing via `kv_quant_reference.py` measures compression and reconstruction quality, not inference throughput. Use `benchmark_llamacpp_cpu.py` for throughput.

## Recommended Benchmark Protocol

Use the same values across all notebooks:

- `MODEL_ID` (TinyLlama for all except SmoothQuant)
- prompt and input length
- `MAX_NEW_TOKENS = 64`
- Colab runtime/GPU type
- calibration text count and sequence length where relevant

Suggested accuracy checks after artifact creation:

- WikiText-2 perplexity with `lm-evaluation-harness`
- LAMBADA last-word accuracy
- HellaSwag zero-shot accuracy

Suggested performance checks:

- saved artifact size on disk
- peak GPU memory during load/generation
- tokens/sec at batch size 1
- CPU load and CPU tokens/sec for checkpoint-based methods

## Reproducibility Notes

The TurboQuant and RotorQuant notebooks include self-contained reference implementations for cache compression so you can generate artifacts immediately in Colab. They are deliberately marked as reference paths, not production kernels. For final numbers, run the linked implementation repositories or a serving-engine integration when available.

The SmoothQuant notebook produces fake W8A8 (Path 2) because `torch_int` — the CUDA extension required for real INT8 kernels — does not compile against current Colab PyTorch/CUDA versions. This is an upstream compatibility issue with `mit-han-lab/torch-int`, not a code error. The fake W8A8 path is valid for accuracy and perplexity comparison.

## CPU Benchmark Scripts

Install the lightweight CPU requirements:

```bash
python -m pip install -r requirements-cpu.txt
```

### Hugging Face / PyTorch CPU Artifacts

```bash
python scripts/benchmark_transformers_cpu.py \
  --model-path weights/awq_int4/model \
  --method awq \
  --loader awq \
  --dataset datasets/processed/wikitext2.jsonl \
  --max-samples 8 \
  --max-input-tokens 512 \
  --max-new-tokens 16 \
  --threads 4 \
  --compute-loss
```

### TurboQuant / RotorQuant Reference Metrics

```bash
python scripts/kv_quant_reference.py \
  --model-path weights/awq_int4/model \
  --method turboquant \
  --loader awq \
  --dataset datasets/processed/long_wikitext2.jsonl \
  --max-input-tokens 1536 \
  --bits 3 \
  --threads 4

python scripts/kv_quant_reference.py \
  --model-path weights/awq_int4/model \
  --method rotorquant \
  --loader awq \
  --dataset datasets/processed/long_wikitext2.jsonl \
  --max-input-tokens 1536 \
  --bits 3 \
  --threads 4
```

### Summaries

```bash
python scripts/summarize_results.py --results-dir results

python scripts/compare_results.py \
  --baseline results/fp32_wikitext2_<timestamp>_summary.json \
  --candidate results/gptq_wikitext2_<timestamp>_summary.json
```