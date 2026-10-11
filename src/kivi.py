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


def per_channel_quantization(K, bits, G):
    l, d = K.shape
    Kq = np.empty_like(K)
    for i in range(0, l, G):
        Kq[i:i+G] = fake_quant(K[i:i+G], bits, axis=0)
    return Kq

def per_token_quantization(V, bits, G):
    l, d = V.shape
    Vq = np.empty_like(V)
    for c in range(0, d, G):
        Vq[:, c:c+G] = fake_quant(V[:, c:c+G], bits, axis=1)
    return Vq
