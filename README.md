# dsc180-quantization

Experiments on **low-bit KV-cache quantization** for large language models, done for UCSD DSC-180A.

The project tests the main claim behind the quantization method [KIVI](https://arxiv.org/abs/2402.02750), using both synthetic data and a real model:

> Kivi differs from the norm because the **key** cache should be quantized **per channel**. The **value** cache should be quantized **per token**.

We take real query/key/value tensors from **Qwen3-1.7B** running on a **LongBench** prompt, quantize them to 2 bits each way, and measure how much error each method adds to the attention computation.

---

## Background

During autoregressive decoding, a transformer stores the keys (K) and values (V) of all previous tokens so it doesn't have to compute them again. For long contexts this **KV cache** often uses more memory than the model weights. Quantizing it to 2 bits cuts that memory by about 8× compared with fp16. The issue with quantization is how to do that without hurting attention much.

KIVI's observation:

- **Keys** have a few **outlier channels**: certain dimensions are much larger than the rest across *every* token. If you quantize per token (one scale per row), those outliers inflate the scale and wipe out the precision of the other channels. Giving each channel its own scale by quantizing per key avoids this.
- **Values** have no consistent channel outliers. Their error matters through the attention output `A @ V`, which mixes tokens together, so quantizing **per token** keeps each token's error contained.
- The most recent **R tokens** (the *residual*) stay in full precision, because they are still being filled into groups and they usually get a lot of attention.

`figures/kv_magnitudes.png` shows this structure in the real Qwen3 cache. `figures/error_comparison.png` compares the error of the two methods.

## Repository layout

```
src/
  inference/
    pull_cache.py      # Run Qwen3-1.7B on a LongBench sample; save Q/K/V per layer
  quantization/
    quantize.py        # Asymmetric min/max quantization + per-channel / per-token grouped quantizers
    metrics.py         # Relative-error metrics on K, V, attention weights and attention output
    synthesizer.py     # Synthetic K/V/Q generator with injected outliers (for controlled experiments)
    visualize.py       # KIVI-style 3D magnitude plots and error bar charts
    main.py            # Full experiment: quantize every layer/head, compute errors, save CSV + figures
tests/                 # pytest unit tests for quantization, metrics and plotting
figures/               # Generated plots
kv_snapshots/          # (gitignored) saved Q/K/V tensors from pull_cache.py
results/               # Per-head error CSV written by main.py
```

## Method

### Quantization (`quantize.py`)

All quantizers use **asymmetric uniform (min/max) quantization**, with a zero point `z = min` and a scale `s = (max − min) / (2^bits − 1)`, taken along an axis:

```
q   = clip(round((X − z) / s), 0, 2^bits − 1)
X̂   = q · s + z
```

The code uses *fake quantization* (quantize, then dequantize right away), so error can be measured in floating point.

| Function | Used for | Grouping | Full-precision residual |
|---|---|---|---|
| `per_channel_quantization` | Keys | Groups of `G` tokens; each channel gets its own scale inside a group | Trailing `l mod R` tokens |
| `per_token_quantization` | Values | Each token gets its own scale over groups of `G` channels | Last `R` tokens |

Defaults: `bits=2`, `G=32`, `R=128`.

### Error metrics (`metrics.py`)

Every metric is a **norm-ratio relative error**, `‖X − X̂‖ / ‖X‖`:

| Metric | What it measures |
|---|---|
| `key_reconstruction_error` | `K` vs `K̂` |
| `attention_score_error` | `softmax(QKᵀ/√d)` vs `softmax(QK̂ᵀ/√d)` |
| `value_reconstruction_error` | `V` vs `V̂` |
| `attention_output_error` | `A·V` vs `A·V̂` |

Key metrics depend only on how K is quantized, and value metrics only on how V is quantized. This matches the split in KIVI Table 2.

### Real-model pipeline

1. **`pull_cache.py`** loads `Qwen/Qwen3-1.7B` with a custom attention hook that records the query tensor in each layer. It then runs one forward pass on a LongBench `multifieldqa_en` sample and saves Q, K and V for every layer to `kv_snapshots/layer.pt`.
2. **`main.py`** loads that snapshot. For every layer and every KV head, it quantizes K and V with both methods. Qwen3 uses grouped-query attention, so it then scores each quantized head against each query head that shares it, using the last token's query (the next decoding step). It prints the mean, median and worst-case error per method, plus a per-layer breakdown. Per-head results go to `results/head_errors.csv` and figures to `figures/`.

We also checked to see if this relationship held up with synthetic experiments: `Synthesizer` makes random K with a few channels scaled by 10× (channel outliers) and V with randomly scattered 10× entries. To use it, swap it in at the top of `main.py`.

## Getting started

The project uses [uv](https://docs.astral.sh/uv/) and Python 3.13.

```bash
uv sync                                         # create .venv and install dependencies

# 1. Pull a real KV cache (downloads Qwen3-1.7B and LongBench from Hugging Face)
uv run python src/inference/pull_cache.py

# 2. Quantize, compute errors, write results/head_errors.csv and figures/
uv run python src/quantization/main.py

# Run the tests
uv run pytest
```

The first step needs enough memory to run a 1.7B model on a long prompt. A GPU or Apple Silicon (MPS) is recommended, but it will run on CPU.

### Configuration

- `longbench.py`: `LONGBENCH_TASK`, `SAMPLE_INDEX`
- `pull_cache.py`: model name
- `main.py`: `BITS`, `GROUP_SIZE`, `RESIDUAL_LENGTH`

## One-prompt generation with a packed KIVI cache

The offline tensor experiment above remains available. To compare actual answers,
`src/inference/compare_longbench.py` runs the same complete LongBench QA prompt
with an ordinary cache and the packed cache in `src/inference/packed_kivi_cache.py`.
Both use Qwen3-1.7B, native bfloat16 model computation, greedy decoding, and thinking
mode disabled. No model weights are quantized.

```bash
PYTHONPATH=src uv run python -m inference.compare_longbench
# Optional: choose a particular example and write to a new output directory.
PYTHONPATH=src uv run python -m inference.compare_longbench \
  --task multifieldqa_en --sample-index 0 \
  --output-dir kv_snapshots/my_kivi_run
```

By default, the runner selects the first complete example whose formatted
prompt contains 129–4096 tokens. It does not truncate the document or choose
examples based on their answers. Each new output directory contains the exact
prompt and settings, both answers, reference answers, English QA F1 scores,
cache sizes, and a decode trace. Existing output directories are rejected so
previous results cannot be overwritten. Supported tasks are `multifieldqa_en`
and `qasper`; both use LongBench's normalized token-overlap QA F1 definition.

The cache keeps older keys in per-channel groups and older values in per-token
groups, packing four 2-bit codes into each byte. Scales and minimum values are
stored alongside the codes. Recent keys and values stay in full precision.
During prefill, ordinary attention processes the prompt and the cache prepares
its compressed history for later steps. During decode, each new token is added;
completed key blocks and old value rows are then packed for the next step.
Qwen applies RoPE to Q/K before the cache receives K. Values are not rotated.

This implementation supports one unpadded prompt followed by single-token
greedy decoding. It temporarily unpacks each layer's cache to floating point
for standard attention. Stored-cache byte counts include scales and residuals,
but exclude model weights, temporary tensors, and allocator overhead. These
counts are not peak device-memory or speed benchmarks. Beam search, batched
requests, chunked prefill, and the paper's fused GPU kernels are outside this
implementation.

Tests check known packed bytes, quantization against the original NumPy
quantizers, residual boundary updates, token order, released residual storage,
and a tiny Qwen model with shared KV heads. Disabling quantization reproduces
ordinary cached and uncached attention across prefill and decode.

## References

- Liu et al., *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache*, ICML 2024. [arXiv:2402.02750](https://arxiv.org/abs/2402.02750)
- Bai et al., *LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding*, 2023. [arXiv:2308.14508](https://arxiv.org/abs/2308.14508)
- [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)
