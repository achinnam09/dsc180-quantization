import numpy as np

from quantize import per_channel_quantization, per_token_quantization


def test_per_channel_quantization():
    # Each column has its own scale + last row stays unquantized.
    X = np.array([
        [0.0, 100.0],
        [0.4, 104.0],
        [1.4, 114.0],
        [3.0, 130.0],
        [0.7, 107.0],
    ])

    result = per_channel_quantization(X, bits=2, G=4, R=4)

    np.testing.assert_allclose(result, [
        [0, 100],
        [0, 100],
        [1, 110],
        [3, 130],
        [0.7, 107],
    ])


def test_per_token_quantization():
    # Quantize first row, but keep last row in full precision
    X = np.array([
        [0.0, 0.4, 1.4, 3.0],
        [0.7, 1.2, 2.3, 3.4],
    ])

    result = per_token_quantization(X, bits=2, G=4, R=1)

    np.testing.assert_allclose(result, [
        [0.0, 0.0, 1.0, 3.0],
        [0.7, 1.2, 2.3, 3.4],
    ])
