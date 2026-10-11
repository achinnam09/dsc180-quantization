from src.kivi import per_channel_quantization, per_token_quantization
from src.synthesizer import Synthesizer


synth = Synthesizer()
Q, K, V = synth.generate()

quantization_method_map = {
    'channel': per_channel_quantization,
    'per token': per_token_quantization
}

for key_quantization_method, value_quantization_method in [
    ('channel', 'token'),
    ('token', 'channel'),
    ('channel', 'channel'),
    ('token', 'token')
]:
    print(f"Simulating per {key_quantization_method} quantization for K and per {value_quantization_method} for V")

    K_hat = quantization_method_map[key_quantization_method](K, bits=2, G=32)
    V_hat = quantization_method_map[value_quantization_method](K, bits=2, G=32)

    # calculate + print error metrics
