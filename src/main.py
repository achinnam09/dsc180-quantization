from pathlib import Path

from metrics import compute_error_metrics
from quantize import per_channel_quantization, per_token_quantization
from synthesizer import Synthesizer
from visualize import plot_error_comparison, plot_magnitudes

RESIDUAL_LENGTH = 128
GROUP_SIZE = 32
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"


synth = Synthesizer()
Q, K, V = synth.generate()

quantization_method_map = {
    'channel': per_channel_quantization,
    'token': per_token_quantization
}

for key_quantization_method, value_quantization_method in [
    ('channel', 'token'),
    ('token', 'channel'),
    ('channel', 'channel'),
    ('token', 'token')
]:
    print(f"Simulating per {key_quantization_method} quantization for K and per {value_quantization_method} for V")

    K_hat = quantization_method_map[key_quantization_method](K, bits=2, G=GROUP_SIZE, R=RESIDUAL_LENGTH)
    V_hat = quantization_method_map[value_quantization_method](V, bits=2, G=GROUP_SIZE, R=RESIDUAL_LENGTH)

    errors = compute_error_metrics(Q, K, V, K_hat, V_hat)
    border = "+------------------------------+------------------+"
    print(border)
    print(f"| {'Metric':<28} | {'Error':>16} |")
    print(border)
    for name, error in errors.items():
        label = name.replace('_', ' ').capitalize()
        print(f"| {label:<28} | {error:>16.6f} |")
    print(border)
    print()

# Quantize K and V with the same method so each run fills one column of KIVI
# Table 2: key metrics depend only on K_hat, value metrics only on V_hat.
errors_by_method = {}
for method, quantization_method in quantization_method_map.items():
    K_hat = quantization_method(K, bits=2, G=GROUP_SIZE, R=RESIDUAL_LENGTH)
    V_hat = quantization_method(V, bits=2, G=GROUP_SIZE, R=RESIDUAL_LENGTH)
    errors_by_method[method] = compute_error_metrics(Q, K, V, K_hat, V_hat)

FIGURES_DIR.mkdir(exist_ok=True)
plot_magnitudes(K, V, FIGURES_DIR / "kv_magnitudes.png")
plot_error_comparison(errors_by_method, FIGURES_DIR / "error_comparison.png")
print(f"Saved figures to {FIGURES_DIR}")
