import numpy as np

from quantization.metrics import attention_weights, compute_error_metrics, relative_error


def test_relative_error():
    # Error norm is sqrt(6**2 + 16**2) and original norm is sqrt(2**2 + 4**2).
    original = np.array([[-2.0, 4.0]])
    reconstructed = np.array([[-8.0, 20.0]])
    error = relative_error(original, reconstructed)

    np.testing.assert_allclose(error, np.sqrt(292 / 20))


def test_relative_error_handles_zeros():
    original = np.array([[0.0, 0.0]])
    reconstructed = np.array([[0.0, 1.0]])
    error = relative_error(original, reconstructed, eps=0.1)

    np.testing.assert_allclose(error, 10)


def test_attention_weights():
    # Four channels give a scale of sqrt(4) = 2.
    Q = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    K = np.array([[2000.0, 0.0, 0.0, 0.0], [2002.0, 0.0, 0.0, 0.0]])

    A = attention_weights(Q, K)

    # A one-point score difference gives about 27% / 73%; tied scores split equally.
    np.testing.assert_allclose(A, [[0.26894142, 0.73105858], [0.5, 0.5]])


def test_unchanged_matrices_have_zero_error():
    Q = np.array([[1.0]])
    K = np.array([[1.0], [2.0]])
    V = np.array([[2.0], [4.0]])

    errors = compute_error_metrics(Q, K, V, K, V)

    assert all(error == 0 for error in errors.values())


def test_key_only_quantization_isolates_key_metrics():
    rng = np.random.default_rng(0)
    Q, K, V = rng.normal(size=(1, 4)), rng.normal(size=(6, 4)), rng.normal(size=(6, 4))
    K_hat = K + 0.5 * rng.normal(size=K.shape)

    errors = compute_error_metrics(Q, K, V, K_hat, V)

    # V is exact, so the value-only metrics are zero, and quantizing "both"
    # caches is the same as quantizing only K.
    assert errors["value_reconstruction_error"] == 0
    assert errors["attention_output_error"] == 0
    assert errors["key_output_error"] > 0
    np.testing.assert_allclose(errors["combined_output_error"], errors["key_output_error"])


def test_value_only_quantization_isolates_value_metrics():
    rng = np.random.default_rng(1)
    Q, K, V = rng.normal(size=(1, 4)), rng.normal(size=(6, 4)), rng.normal(size=(6, 4))
    V_hat = V + 0.5 * rng.normal(size=V.shape)

    errors = compute_error_metrics(Q, K, V, K, V_hat)

    assert errors["attention_score_error"] == 0
    assert errors["key_output_error"] == 0
    assert errors["attention_output_error"] > 0
    np.testing.assert_allclose(errors["combined_output_error"], errors["attention_output_error"])
