from metrics import compute_error_metrics
from quantize import per_channel_quantization, per_token_quantization
from synthesizer import Synthesizer

RESIDUAL_LENGTH = 128
GROUP_SIZE = 32


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
