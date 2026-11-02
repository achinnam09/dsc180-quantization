from pathlib import Path

from metrics import compute_error_metrics
from quantize import per_channel_quantization, per_token_quantization
from synthesizer import Synthesizer
from visualize import plot_error_comparison, plot_magnitudes
import torch

RESIDUAL_LENGTH = 128
GROUP_SIZE = 32
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

avg_errors = {}
for method, quantization_method in quantization_method_map.items():
    count = 0
    totals = {}
    for lay in range(len(K)):
        group = Q[lay].shape[1] // K[lay].shape[1]                
        for h in range(K[lay].shape[1]):
            K_head = K[lay][0, h].float().numpy()
            V_head = V[lay][0, h].float().numpy()
            K_hat = quantization_method(K_head, bits=2, G=GROUP_SIZE, R=RESIDUAL_LENGTH)
            V_hat = quantization_method(V_head, bits=2, G=GROUP_SIZE, R=RESIDUAL_LENGTH)
            for q in range(h * group, (h + 1) * group):
                Q_head = Q[lay][0, q, -1:].float().numpy()
                errors = compute_error_metrics(Q_head, K_head, V_head, K_hat, V_hat)
                for name, value in errors.items():
                    totals[name] = totals.get(name, 0) + value
                count += 1
    avg_errors[method] = {name: total / count for name, total in totals.items()}

print(f"Average error over {count} query heads:")
for method, errors in avg_errors.items():
    print(f"  {method}: " + ", ".join(f"{name} {value:.3f}" for name, value in errors.items()))

FIGURES_DIR.mkdir(exist_ok=True)
plot_magnitudes(K[0][0, 0].float().numpy(), V[0][0, 0].float().numpy(), FIGURES_DIR / "kv_magnitudes.png")
plot_error_comparison(avg_errors, FIGURES_DIR / "error_comparison.png")
print(f"Saved figures to {FIGURES_DIR}")
