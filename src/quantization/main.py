import csv
from pathlib import Path

import numpy as np

from metrics import compute_error_metrics
from quantize import per_channel_quantization, per_token_quantization
from synthesizer import Synthesizer
from visualize import plot_error_comparison, plot_magnitudes
import torch

RESIDUAL_LENGTH = 128
GROUP_SIZE = 32
BITS = 2
FIGURES_DIR = Path(__file__).resolve().parents[2] / "figures"

data = torch.load(Path(__file__).resolve().parents[2] / "kv_snapshots" / "layer.pt")
#synth = Synthesizer()
#Q, K, V = synth.generate()
Q = data["query"]
K = data["keys"]
V = data["values"]

print("Saved shapes (layers x (batch, heads, tokens, channels)):")
for name, layers in data.items():
    print(f"  {name:<7} {len(layers)} x {tuple(layers[0].shape)}")

quantization_method_map = {
    'channel': per_channel_quantization,
    'token': per_token_quantization
}

records = []
for method, quantization_method in quantization_method_map.items():
    for lay in range(len(K)):
        group = Q[lay].shape[1] // K[lay].shape[1]
        for h in range(K[lay].shape[1]):
            K_head = K[lay][0, h].float().numpy()
            V_head = V[lay][0, h].float().numpy()
            K_hat = quantization_method(K_head, bits=BITS, G=GROUP_SIZE, R=RESIDUAL_LENGTH)
            V_hat = quantization_method(V_head, bits=BITS, G=GROUP_SIZE, R=RESIDUAL_LENGTH)
            for q in range(h * group, (h + 1) * group):
                Q_head = Q[lay][0, q, -1:].float().numpy()
                errors = compute_error_metrics(Q_head, K_head, V_head, K_hat, V_hat)
                records.append({"method": method, "layer": lay, "kv_head": h, "query_head": q, **errors})

metric_names = list(errors)
avg_errors = {}
for method in quantization_method_map:
    rows = [r for r in records if r["method"] == method]
    avg_errors[method] = {name: float(np.mean([r[name] for r in rows])) for name in metric_names}

    print(f"{method} ({len(rows)} query heads):")
    for name in metric_names:
        values = np.array([r[name] for r in rows])
        worst = rows[int(values.argmax())]
        print(f"  {name:<28} mean {values.mean():.3f}  median {np.median(values):.3f}  "
              f"max {values.max():.3f} (layer {worst['layer']}, head {worst['query_head']})")

    print("  attention_output_error by layer (mean):")
    for lay in range(len(K)):
        layer_values = [r["attention_output_error"] for r in rows if r["layer"] == lay]
        print(f"    layer {lay:>2}: {np.mean(layer_values):.3f}")

RESULTS_PATH = Path(__file__).resolve().parents[2] / "results" / "head_errors.csv"
RESULTS_PATH.parent.mkdir(exist_ok=True)
with open(RESULTS_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=records[0].keys())
    writer.writeheader()
    writer.writerows(records)
print(f"Saved per-head errors to {RESULTS_PATH}")

FIGURES_DIR.mkdir(exist_ok=True)
plot_magnitudes(K[0][0, 0].float().numpy(), V[0][0, 0].float().numpy(), FIGURES_DIR / "kv_magnitudes.png")
plot_error_comparison(avg_errors, FIGURES_DIR / "error_comparison.png")
print(f"Saved figures to {FIGURES_DIR}")
