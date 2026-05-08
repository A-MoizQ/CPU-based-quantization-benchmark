# LLM Quantization Comparison

This repo is a Colab-ready and CPU-benchmark scaffold for comparing Google TurboQuant against GPTQ, AWQ, SmoothQuant, RotorQuant, and a standard INT8 baseline.

The important distinction:

- GPTQ, AWQ, SmoothQuant, and LLM.int8 mostly quantize model weights or weight/activation compute and can produce saved model artifacts.
- TurboQuant and RotorQuant are KV-cache compression methods. They do not produce a normal quantized weight checkpoint; their artifact is a compressed runtime cache/backend plus memory and reconstruction metrics.

## Recommended CPU Target

For the final i5 6th gen / 4 GB RAM experiments, use TinyLlama as the single documented model target. The Hugging Face model is useful for Colab quantization and PyTorch-based checks, while GGUF TinyLlama through `llama.cpp` is the practical CPU inference path.

Recommended model target:

| Purpose | Model/artifact |
| --- | --- |
| HF/Colab quantization | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| Practical CPU test | TinyLlama 1.1B GGUF through `llama.cpp` |

On 4 GB RAM, `llama.cpp`/GGUF is usually the more realistic CPU path than full PyTorch for 1B+ models.

## Quick CPU Runbook

Use this path for the actual 4 GB RAM / i5 6th gen experiments.

1. Create a Python environment and install the CPU tooling:

```bash
python -m pip install -r requirements-cpu.txt
```

2. Prepare the small benchmark datasets:

```bash
python scripts/prepare_datasets.py
```

This downloads small slices of WikiText-2 and LAMBADA, then creates a longer concatenated WikiText-2 set for KV-cache experiments. The processed files live in `datasets/processed/`.

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

4. Benchmark a saved GPTQ/AWQ/SmoothQuant artifact:

```bash
python scripts/benchmark_transformers_cpu.py \
  --model-path /path/to/quantized_model \
  --method gptq \
  --loader gptq \
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
  --model-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --method turboquant \
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
  --baseline /path/to/baseline_summary.json \
  --candidate /path/to/candidate_summary.json
```

## Notebook Index

Run the FP16 baseline first, then run each quantization notebook on the same Colab GPU, model, prompt, and token count.

| Notebook | Method | Artifact | Primary use |
| --- | --- | --- | --- |
| `notebooks/00_fp16_baseline.ipynb` | FP16 baseline | HF model folder | Baseline memory and generation speed |
| `notebooks/01_llm_int8_bitsandbytes.ipynb` | LLM.int8 | HF/bitsandbytes model folder | Standard INT8 baseline |
| `notebooks/02_gptq_gptqmodel.ipynb` | GPTQ INT4 | GPTQ model folder | Weight-only 4-bit checkpoint |
| `notebooks/03_awq_autoawq.ipynb` | AWQ INT4 | AWQ model folder | Weight-only 4-bit checkpoint |
| `notebooks/04_smoothquant_official.ipynb` | SmoothQuant W8A8 | SmoothQuant export or smoothing scales | W8A8 path |
| `notebooks/05_turboquant_kv_cache.ipynb` | TurboQuant-style KV cache | `compressed_kv_cache.pt` plus metrics | KV-cache memory study |
| `notebooks/06_rotorquant_kv_cache.ipynb` | RotorQuant-style KV cache | `compressed_kv_cache.pt` plus metrics | KV-cache memory study |

All notebooks should use `TinyLlama/TinyLlama-1.1B-Chat-v1.0` as the documented experiment model. Keep `MODEL_ID` consistent across notebooks for the real comparison.

## Output Layout

Each notebook writes to:

```text
/content/quant_outputs/<method>/
```

Typical files:

- `model/`: saved quantized model when the method supports a checkpoint.
- `compressed_kv_cache.pt`: saved KV-cache artifact for TurboQuant/RotorQuant notebooks.
- `metrics.json`: latency, peak GPU memory where available, disk footprint, compression ratio, and reconstruction metrics.

Download that `/content/quant_outputs` folder from Colab after each run if you want to preserve artifacts.

## Research Sources

| Method | Paper | Code used or referenced | Short quote |
| --- | --- | --- | --- |
| TurboQuant | [arXiv:2504.19874](https://arxiv.org/abs/2504.19874), [HF paper page](https://huggingface.co/papers/2504.19874) | [tonbistudio/turboquant-pytorch](https://github.com/tonbistudio/turboquant-pytorch), [OnlyTerp/turboquant](https://github.com/OnlyTerp/turboquant) | "absolute quality neutrality with 3.5 bits per channel" |
| RotorQuant | [Scrya technical report](https://scrya.com/rotorquant) | [scrya-com/rotorquant](https://github.com/scrya-com/rotorquant) | "44x fewer parameters" |
| GPTQ | [arXiv:2210.17323](https://arxiv.org/abs/2210.17323), [HF paper page](https://huggingface.co/papers/2210.17323) | [IST-DASLab/gptq](https://github.com/IST-DASLab/gptq), [HF GPTQ docs](https://huggingface.co/docs/transformers/quantization/gptq) | "accurate post-training quantization" |
| AWQ | [arXiv:2306.00978](https://arxiv.org/abs/2306.00978), [MLSys page](https://proceedings.mlsys.org/paper_files/paper/2024/hash/42a452cbafa9dd64e9ba4aa95cc1ef21-Abstract-Conference.html) | [mit-han-lab/llm-awq](https://github.com/mit-han-lab/llm-awq), [HF AWQ docs](https://huggingface.co/docs/transformers/quantization/awq) | "1% of salient weights" |
| SmoothQuant | [arXiv:2211.10438](https://arxiv.org/abs/2211.10438), [HF paper page](https://huggingface.co/papers/2211.10438) | [mit-han-lab/smoothquant](https://github.com/mit-han-lab/smoothquant), [Intel Neural Compressor docs](https://intel.github.io/neural-compressor/latest/docs/source/smooth_quant.html) | "training-free, accuracy-preserving" |
| LLM.int8 | [arXiv:2208.07339](https://arxiv.org/abs/2208.07339), [HF paper page](https://huggingface.co/papers/2208.07339) | [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes), [HF bitsandbytes docs](https://huggingface.co/docs/transformers/quantization/bitsandbytes) | "only half the required memory" |

Source status checked on 2026-05-08.

## CPU Inference Caveats

Saved model folders are not equally portable to CPU:

- GPTQ: most CPU-friendly among the 4-bit notebook paths. Reload with `GPTQConfig(bits=4, use_exllama=False)`.
- SmoothQuant: for CPU, prefer an ONNX Runtime or Intel Neural Compressor export path. Treat any model-family-specific export limitation as part of the experimental limitations.
- AWQ: optimized primarily for CUDA/TinyChat-style kernels; CPU loading depends on the runtime stack.
- LLM.int8: bitsandbytes is primarily an inference/runtime quantization path; CPU support depends on installed bitsandbytes backends.
- TurboQuant/RotorQuant: these target KV-cache storage during generation, so CPU testing requires an inference engine integration, not only a saved HF checkpoint.

## Recommended Benchmark Protocol

Use the same values across all notebooks:

- `MODEL_ID`
- prompt and input length
- `MAX_NEW_TOKENS`
- Colab runtime/GPU type
- calibration text count and sequence length where relevant

Suggested accuracy checks after artifact creation:

- WikiText-2 perplexity with `lm-evaluation-harness`
- LAMBADA last-word accuracy
- HellaSwag zero-shot accuracy for LLMs
- SST-2 or MNLI for a BERT-style side experiment, if you expand beyond causal LMs

Suggested performance checks:

- saved artifact size on disk
- peak GPU memory during load/generation
- tokens/sec at batch size 1
- tokens/sec at larger batch sizes if the method supports batching
- CPU load and CPU tokens/sec for checkpoint-based methods

## Reproducibility Notes

The TurboQuant and RotorQuant notebooks include self-contained reference implementations for cache compression so you can generate artifacts immediately in Colab. They are deliberately marked as reference paths, not production kernels. For final numbers, run the linked implementation repositories or a serving-engine integration when available.

The checked-in notebooks are ready to open directly. `scripts/create_notebooks.py` now acts as a small sanity check that lists them.

## CPU Benchmark Scripts

Install the lightweight CPU requirements:

```bash
python -m pip install -r requirements-cpu.txt
```

Prepare datasets:

```bash
python scripts/prepare_datasets.py
```

This creates:

- `datasets/processed/wikitext2.jsonl`
- `datasets/processed/lambada.jsonl`
- `datasets/processed/long_wikitext2.jsonl`
- `datasets/processed/manifest.json`

Use `wikitext2` and `lambada` for normal speed/quality checks. Use `long_wikitext2` for TurboQuant/RotorQuant-style KV-cache stress because long prompts create a larger cache.

### Hugging Face / PyTorch CPU Artifacts

Benchmark a local HF model or quantized checkpoint:

```bash
python scripts/benchmark_transformers_cpu.py \
  --model-path /path/to/model_or_quantized_artifact \
  --method fp32_or_gptq_or_awq \
  --loader auto \
  --dataset datasets/processed/wikitext2.jsonl \
  --max-samples 8 \
  --max-input-tokens 512 \
  --max-new-tokens 16 \
  --threads 4 \
  --compute-loss
```

For GPTQ CPU loading, use:

```bash
python scripts/benchmark_transformers_cpu.py \
  --model-path /path/to/gptq_model \
  --method gptq \
  --loader gptq \
  --dataset datasets/processed/wikitext2.jsonl \
  --threads 4
```

Logged metrics include:

- artifact size
- load time
- RSS after load
- peak RSS
- prompt/prefill tokens per second
- decode tokens per second
- KV-cache memory before/after decode
- optional loss/perplexity

### TurboQuant / RotorQuant Reference Metrics

These methods are KV-cache methods, so the included script measures CPU cache compression footprint and quantize/dequantize cost on real model KV caches:

```bash
python scripts/kv_quant_reference.py \
  --model-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --method turboquant \
  --dataset datasets/processed/long_wikitext2.jsonl \
  --max-input-tokens 1536 \
  --bits 3 \
  --threads 4

python scripts/kv_quant_reference.py \
  --model-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --method rotorquant \
  --dataset datasets/processed/long_wikitext2.jsonl \
  --max-input-tokens 1536 \
  --bits 3 \
  --threads 4
```

This logs:

- FP KV-cache MiB
- theoretical packed KV-cache MiB
- compression ratio
- CPU quantize/dequantize time
- cosine similarity and MSE for reconstructed keys/values
- peak RSS

For true CPU inference speed with TurboQuant/RotorQuant, use an inference engine integration. The generic `llama.cpp` runner is:

```bash
python scripts/benchmark_llamacpp_cpu.py \
  --binary /path/to/llama-cli \
  --model /path/to/model.gguf \
  --method gguf_q4_or_turboquant_or_rotorquant \
  --dataset datasets/processed/long_wikitext2.jsonl \
  --ctx-size 2048 \
  --max-new-tokens 32 \
  --threads 4 \
  --extra-args "PUT_ENGINE_SPECIFIC_CACHE_QUANT_FLAGS_HERE"
```

### Summaries

Summarize result JSON files:

```bash
python scripts/summarize_results.py --results-dir results
```

Compare a candidate against a baseline:

```bash
python scripts/compare_results.py \
  --baseline results/fp32_wikitext2_summary.json \
  --candidate results/gptq_wikitext2_summary.json
```

This reports decode speedup, peak-memory reduction, artifact-size reduction, and KV-cache reduction when those fields are present.
