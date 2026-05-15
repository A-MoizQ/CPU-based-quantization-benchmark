# AWQ INT4 KV-Cache Quantization Benchmark

This repository is currently focused on one narrow experiment: use a saved AWQ INT4 TinyLlama artifact as the CPU model path, then compare TurboQuant-style and RotorQuant-style KV-cache compression on long WikiText-2 prompts.

The current research question is:

> After weight-only AWQ INT4 compression, does a TurboQuant-style or RotorQuant-style 3-bit KV-cache reference path give the better CPU-side memory/quality tradeoff for long prompts?

This is not yet a full paper benchmark suite. It is a small reference experiment that records CPU feasibility, cache footprint, and reconstruction quality.

## What To Report

For the current write-up, report these items:

| Area | Report this |
| --- | --- |
| Base model artifact | `weights/awq_int4/model`, AWQ INT4 TinyLlama artifact |
| Hardware | CPU, core count, RAM, OS, Python version from each result JSON `system` block |
| AWQ CPU baseline | artifact size, load time, peak RSS, prefill tokens/s, decode tokens/s, perplexity |
| KV-cache methods | TurboQuant-style reference and RotorQuant-style reference |
| KV settings | `bits=3`, `max_input_tokens=1536`, `max_samples=4`, long WikiText-2 |
| KV memory | float KV-cache MiB, packed KV-cache MiB, compression ratio, percent reduction |
| KV quality | key/value cosine mean, key/value cosine minimum, key/value MSE |
| Limitations | small sample count, reference implementations, no end-to-end generated-text quality for KV methods yet |

The headline metric from the current long-context KV files is cache footprint:

- float KV cache: `66.0 MiB`
- packed 3-bit cache: `6.7031 MiB`
- compression ratio: `9.846x`
- cache reduction: `89.84%`

## Result Files

Lightweight result files are tracked under `results/`. Large `.pt` cache dumps, `.dat` offload files, and model weights are intentionally ignored.

| File | Meaning |
| --- | --- |
| `results/awq_wikitext2_1778763649_summary.json` | main AWQ INT4 CPU benchmark summary, 8 short WikiText-2 samples |
| `results/awq_wikitext2_1778763649.jsonl` | per-sample AWQ INT4 CPU benchmark records |
| `results/awq_wikitext2_1778763388_summary.json` | earlier 1-sample AWQ smoke run |
| `results/turboquant_kv_reference_long_wikitext2_1778750659.json` | TurboQuant-style 3-bit KV-cache reference run |
| `results/rotorquant_kv_reference_long_wikitext2_1778753695.json` | RotorQuant-style 3-bit KV-cache reference run |
| `results/summary.md` | generated table from `scripts/summarize_results.py` |
| `results/turboquant_vs_rotorquant_comparison.json` | generated comparison from `scripts/compare_results.py` |

Important observation: the saved KV reference JSONs currently record `model_path` as `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, while the AWQ CPU benchmark records `weights/awq_int4/model`. If the paper claim must be strictly "TurboQuant vs RotorQuant on AWQ INT4", rerun the two KV commands below and use the new JSON files whose `model_path` and `loader` fields show the AWQ artifact.

## Observed Results

### AWQ INT4 CPU Baseline

From `results/awq_wikitext2_1778763649_summary.json`:

| Metric | Value |
| --- | ---: |
| samples | 8 |
| artifact size | 0.7165 GiB |
| load time | 10.553 s |
| RSS after load | 1042.25 MiB |
| peak RSS | 1620.07 MiB |
| prefill speed | 8.007 tokens/s |
| decode speed | 0.1517 tokens/s |
| perplexity | 11.9013 |

This shows the AWQ INT4 artifact can run on the target small CPU/RAM setup, but generation is extremely slow. Treat throughput as a feasibility result, not a deployment-ready result.

### TurboQuant vs RotorQuant KV Reference

From the current long WikiText-2 result files:

| Metric | TurboQuant-style | RotorQuant-style |
| --- | ---: | ---: |
| samples | 4 | 4 |
| max input tokens | 1536 | 1536 |
| float KV cache | 66.0 MiB | 66.0 MiB |
| packed KV cache | 6.7031 MiB | 6.7031 MiB |
| compression ratio | 9.846x | 9.846x |
| peak RSS | 3185.67 MiB | 3142.64 MiB |
| prefill mean | 92.219 s | 134.639 s |
| quantize/dequantize mean | 0.339 s | 1.004 s |
| key cosine mean | 0.9700 | 0.9392 |
| value cosine mean | 0.9704 | 0.9629 |
| key cosine minimum | 0.8929 | 0.7975 |
| value cosine minimum | 0.8575 | 0.8236 |
| key MSE mean | 0.4031 | 0.8487 |
| value MSE mean | 0.00418 | 0.00494 |

What was observed:

- Both methods reached the same packed KV-cache size because both were configured as 3-bit reference compressors with the same packing estimate.
- TurboQuant-style reconstruction was better in this run: higher key/value cosine and lower key/value MSE.
- RotorQuant-style peak RSS was only about `1.35%` lower in the generic comparison output, but its reconstruction quality was worse and its quantize/dequantize path was slower in this reference implementation.
- The current evidence favors TurboQuant-style KV compression for this specific small CPU reference setup.

Layer-averaged reconstruction quality (88 layer reports = 4 samples * 22 layers):

- TurboQuant-style: key cosine mean `0.96996`, key MSE mean `0.40308`, value cosine mean `0.97044`, value MSE mean `0.00418`.
- RotorQuant-style: key cosine mean `0.93922`, key MSE mean `0.84870`, value cosine mean `0.96293`, value MSE mean `0.00494`.

Methodology note: these averages are computed across every per-layer report from the four long-context samples (22 layers each), without weighting by layer size.

## Commands Used

Benchmark the saved AWQ artifact:

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

Benchmark TurboQuant/RotorQuant-style KV-cache footprint on long prompts:

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

Regenerate the tracked summary and comparison files:

```bash
python scripts/summarize_results.py --results-dir results > results/summary.md

python scripts/compare_results.py \
  --baseline results/turboquant_kv_reference_long_wikitext2_1778750659.json \
  --candidate results/rotorquant_kv_reference_long_wikitext2_1778753695.json \
  > results/turboquant_vs_rotorquant_comparison.json
```

## Research Direction

The useful direction is not "AWQ vs GPTQ vs SmoothQuant vs everything" yet. The current direction is narrower:

1. Establish that AWQ INT4 makes the model small enough to run on constrained CPU memory.
2. Add KV-cache compression because long prompts still grow runtime memory.
3. Compare TurboQuant-style and RotorQuant-style 3-bit KV-cache reconstruction under the same prompt length and sample count.
4. Expand from reference metrics to paper-grade results: more samples, repeated runs, generated text quality, perplexity after compressed-cache decode, and an implementation closer to each method's production design.

## Not Used Yet

These parts remain in the repository as scaffolding, but they were not used for the current AWQ INT4 TurboQuant/RotorQuant comparison:

| Method | Current status |
| --- | --- |
| FP16 baseline notebook | not run for the current result set |
| LLM.int8 notebook | not run for the current result set |
| GPTQ INT4 notebook | not run for the current result set |
| SmoothQuant | older OPT-only/fake-W8A8 artifact exists, not part of this comparison |
| llama.cpp CPU script | available, not part of the current reported results |
| Colab notebook matrix | useful later, not needed for this narrow write-up |

Do not mix these unused methods into the current paper table unless they are rerun under a consistent protocol.
