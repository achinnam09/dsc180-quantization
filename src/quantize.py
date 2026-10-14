import numpy as np
import matplotlib.pyplot as plt

def quantize(X, bits, axis):
    z = X.min(axis=axis, keepdims=True)
    s = (X.max(axis=axis, keepdims=True) - z) / ((2**bits) - 1)
    s = np.maximum(s, 1e-8)

    q = np.round((X - z) / s)
    q = np.clip(q, 0, 2**bits - 1)

    return q, z, s

def dequantize(q, z, s):
    return (q * s) + z

def fake_quant(X, bits, axis):
    q, z, s = quantize(X, bits, axis)
    return dequantize(q, z, s)


def per_channel_quantization(K, bits=2, G=32, R=128):
    l, d = K.shape
    Kq = K.copy()
    quantized_rows = l - (l % R)
    for i in range(0, quantized_rows, G):
        end = min(i + G, quantized_rows)
        Kq[i:end] = fake_quant(K[i:end], bits, axis=0)
    return Kq

def per_token_quantization(V, bits=2, G=32, R=128):
    l, d = V.shape
    Vq = V.copy()
    quantized_rows = max(0, l - R)
    if quantized_rows == 0:
        return Vq
    for c in range(0, d, G):
        Vq[:quantized_rows, c:c+G] = fake_quant(
            V[:quantized_rows, c:c+G], bits, axis=1
        )
    return Vq
