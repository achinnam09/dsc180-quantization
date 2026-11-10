import csv
from pathlib import Path

import numpy as np

from metrics import compute_error_metrics
from quantize import per_channel_quantization, per_token_quantization
from synthesizer import Synthesizer
from visualize import (
    plot_error_comparison,
    plot_errors_by_layer,
    plot_errors_by_position,
    plot_magnitudes,
)
import torch

RESIDUAL_LENGTH = 128
GROUP_SIZE = 32
BITS = 2
# Average the error metrics over the last NUM_QUERY_TOKENS query positions of the
# prompt instead of using only the final token.
NUM_QUERY_TOKENS = 100

ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"

data = torch.load(ROOT / "kv_snapshots" / "layer.pt")
#synth = Synthesizer()
#Q, K, V = synth.generate()
Q = data["query"]
K = data["keys"]
V = data["values"]

print("Saved shapes (layers x (batch, heads, tokens, channels)):")
for name, layers in data.items():
    print(f"  {name:<7} {len(layers)} x {tuple(layers[0].shape)}")

# Each method is (how to quantize K, how to quantize V). 'channel' and 'token'
# use one scheme for both caches; 'kivi' is the paper's combination.
quantization_method_map = {
    'channel': (per_channel_quantization, per_channel_quantization),
    'token': (per_token_quantization, per_token_quantization),
    'kivi': (per_channel_quantization, per_token_quantization),
}

num_tokens = K[0].shape[2]
positions = range(num_tokens - NUM_QUERY_TOKENS, num_tokens)
print(f"Averaging over query positions {positions.start}..{positions.stop - 1} "
      f"({NUM_QUERY_TOKENS} tokens of {num_tokens})")

# When the query at position t is processed, the cache holds only tokens 0..t.
# So for each position we quantize that prefix, which puts the residual
# (full-precision) boundary where it would really be at that step, and score the
# query against it. Causal attention means later tokens are never attended to.
records = []
for method, (key_quantizer, value_quantizer) in quantization_method_map.items():
    for lay in range(len(K)):
        group = Q[lay].shape[1] // K[lay].shape[1]
        for h in range(K[lay].shape[1]):
            K_head = K[lay][0, h].float().numpy()
            V_head = V[lay][0, h].float().numpy()
            Q_heads = Q[lay][0, h * group:(h + 1) * group].float().numpy()
            for t in positions:
                K_prefix, V_prefix = K_head[:t + 1], V_head[:t + 1]
                K_hat = key_quantizer(K_prefix, bits=BITS, G=GROUP_SIZE, R=RESIDUAL_LENGTH)
                V_hat = value_quantizer(V_prefix, bits=BITS, G=GROUP_SIZE, R=RESIDUAL_LENGTH)
                for g in range(group):
                    errors = compute_error_metrics(Q_heads[g, t:t + 1], K_prefix, V_prefix, K_hat, V_hat)
                    records.append({"method": method, "layer": lay, "kv_head": h,
                                    "query_head": h * group + g, "position": t, **errors})
        print(f"  {method}: layer {lay + 1}/{len(K)} done", end="\r")
    print()

metric_names = list(errors)
avg_errors, std_errors = {}, {}
for method in quantization_method_map:
    rows = [r for r in records if r["method"] == method]
    avg_errors[method] = {name: float(np.mean([r[name] for r in rows])) for name in metric_names}
    std_errors[method] = {name: float(np.std([r[name] for r in rows])) for name in metric_names}

    print(f"{method} ({len(rows)} rows = layers x query heads x {NUM_QUERY_TOKENS} positions):")
    for name in metric_names:
        values = np.array([r[name] for r in rows])
        worst = rows[int(values.argmax())]
        print(f"  {name:<28} mean {values.mean():.3f}  std {values.std():.3f}  median {np.median(values):.3f}  "
              f"max {values.max():.3f} (layer {worst['layer']}, head {worst['query_head']}, pos {worst['position']})")

    print("  by layer (mean):        V-only output   K+V output")
    for lay in range(len(K)):
        layer_rows = [r for r in rows if r["layer"] == lay]
        print(f"    layer {lay:>2}: {np.mean([r['attention_output_error'] for r in layer_rows]):>19.3f}"
              f"{np.mean([r['combined_output_error'] for r in layer_rows]):>13.3f}")

RESULTS_DIR.mkdir(exist_ok=True)
with open(RESULTS_DIR / "head_errors.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=records[0].keys())
    writer.writeheader()
    writer.writerows(records)
with open(RESULTS_DIR / "error_summary.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["method", "metric", "mean", "std"])
    for method in avg_errors:
        for name in metric_names:
            writer.writerow([method, name, avg_errors[method][name], std_errors[method][name]])
print(f"Saved per-(layer, head, position) errors and summary to {RESULTS_DIR}")

FIGURES_DIR.mkdir(exist_ok=True)
plot_magnitudes(K[0][0, 0].float().numpy(), V[0][0, 0].float().numpy(), FIGURES_DIR / "kv_magnitudes.png")
plot_error_comparison(avg_errors, FIGURES_DIR / "error_comparison.png", std_by_method=std_errors,
                      title=f"Relative error, averaged over the last {NUM_QUERY_TOKENS} tokens")
plot_errors_by_layer(records, FIGURES_DIR / "errors_by_layer.png")
plot_errors_by_position(records, FIGURES_DIR / "errors_by_position.png", residual_length=RESIDUAL_LENGTH)
print(f"Saved figures to {FIGURES_DIR}")
