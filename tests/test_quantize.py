import numpy as np

from quantization.quantize import dequantize, fake_quant, quantize


def test_quantize():
    # With 2 bits, values become one of four codes: 0, 1, 2, 3.
    X = np.array([[-2.0, -1.0, 0.0, 1.0]])

    q, z, s = quantize(X, bits=2, axis=1)

    np.testing.assert_allclose(q, [[0, 1, 2, 3]])
    np.testing.assert_allclose(z, [[-2]])
    np.testing.assert_allclose(s, [[1]])


def test_dequantize():
    q = np.array([[0, 1, 2, 3]])

    result = dequantize(q, z=-2, s=1)

    np.testing.assert_allclose(result, [[-2, -1, 0, 1]])


def test_fake_quant():
    # Values between quantization levels round to the nearest level.
    X = np.array([[0.0, 0.4, 1.4, 3.0]])

    result = fake_quant(X, bits=2, axis=1)

    np.testing.assert_allclose(result, [[0, 0, 1, 3]])


def test_constant_values():
    X = np.array([[5.0, 5.0, 5.0]])

    result = fake_quant(X, bits=2, axis=1)

    np.testing.assert_allclose(result, X)
