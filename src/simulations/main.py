from pathlib import Path

from metrics import compute_error_metrics
from quantize import per_channel_quantization, per_token_quantization
from synthesizer import Synthesizer
from visualize import plot_error_comparison, plot_magnitudes

RESIDUAL_LENGTH = 128
GROUP_SIZE = 32
FIGURES_DIR = Path(__file__).resolve().parents[2] / "figures"


synth = Synthesizer()
Q, K, V = synth.generate()

quantization_method_map = {
    'channel': per_channel_quantization,
    'token': per_token_quantization
}

errors_by_method = {}
for method, quantization_method in quantization_method_map.items():
    K_hat = quantization_method(K, bits=2, G=GROUP_SIZE, R=RESIDUAL_LENGTH)
    V_hat = quantization_method(V, bits=2, G=GROUP_SIZE, R=RESIDUAL_LENGTH)
    errors_by_method[method] = compute_error_metrics(Q, K, V, K_hat, V_hat)

FIGURES_DIR.mkdir(exist_ok=True)
plot_magnitudes(K, V, FIGURES_DIR / "kv_magnitudes.png")
plot_error_comparison(errors_by_method, FIGURES_DIR / "error_comparison.png")
print(f"Saved figures to {FIGURES_DIR}")
