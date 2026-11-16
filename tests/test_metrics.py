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


def test_relative_error_uses_frobenius_ratio_for_a_matrix():
    # Hand-computed squared sums: error = 1 + 4 + 4 + 1, original = 9 + 16.
    # Zero entries also distinguish this from elementwise relative division.
    original = np.array([[3.0, 0.0], [0.0, 4.0]])
    reconstructed = np.array([[2.0, 2.0], [2.0, 3.0]])
    np.testing.assert_allclose(relative_error(original, reconstructed), np.sqrt(10) / 5)


def test_nonzero_metrics_against_hand_calculated_attention():
    Q = np.array([[1.0]])
    K = np.array([[0.0], [np.log(3.0)]])
    K_hat = np.zeros_like(K)
    V = np.array([[2.0], [6.0]])
    V_hat = np.array([[4.0], [6.0]])
    errors = compute_error_metrics(Q, K, V, K_hat, V_hat)

    # A = [1/4, 3/4], A_hat = [1/2, 1/2]. The value-only
    # output compares 5 with 5.5, using A rather than A_hat.
    expected = {
        "key_reconstruction_error": 1.0,
        "attention_score_error": np.sqrt(0.125 / 0.625),
        "value_reconstruction_error": 2 / np.sqrt(40),
        "attention_output_error": 0.1,
    }
    for name, value in expected.items():
        np.testing.assert_allclose(errors[name], value)


def test_unchanged_matrices_have_zero_error():
    Q = np.array([[1.0]])
    K = np.array([[1.0], [2.0]])
    V = np.array([[2.0], [4.0]])

    errors = compute_error_metrics(Q, K, V, K, V)

    assert all(error == 0 for error in errors.values())
