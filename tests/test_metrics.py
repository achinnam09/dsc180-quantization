import numpy as np

from metrics import attention_weights, compute_error_metrics, relative_error


def test_relative_error():
    # Relative errors are 0.5 and 1, so their norm is sqrt(0.5**2 + 1**2).
    original = np.array([[-2.0, 4.0]])
    reconstructed = np.array([[-1.0, 8.0]])

    error = relative_error(original, reconstructed)

    np.testing.assert_allclose(error, np.sqrt(1.25))


def test_relative_error_handles_zeros():
    original = np.array([[0.0, 2.0]])
    reconstructed = np.array([[0.0, 1.0]])

    error = relative_error(original, reconstructed)

    np.testing.assert_allclose(error, 0.5)
    # A changed zero uses the denominator floor instead of dividing by zero.
    np.testing.assert_allclose(relative_error([[0.0]], [[1.0]], eps=0.1), 10)


def test_attention_weights():
    Q = np.array([[1.0]])
    K = np.array([[0.0], [np.log(2)]])

    A = attention_weights(Q, K)

    np.testing.assert_allclose(A, [[1 / 3, 2 / 3]])
    # Large equal scores should still give equal weights without overflow.
    np.testing.assert_allclose(attention_weights(Q, [[1000.0], [1000.0]]), [[0.5, 0.5]])


def test_unchanged_matrices_have_zero_error():
    Q = np.array([[1.0]])
    K = np.array([[1.0], [2.0]])
    V = np.array([[2.0], [4.0]])

    errors = compute_error_metrics(Q, K, V, K, V)

    assert errors == {
        "key_reconstruction_error": 0.0,
        "attention_score_error": 0.0,
        "value_reconstruction_error": 0.0,
        "attention_output_error": 0.0,
    }


def test_value_errors_use_full_precision_attention():
    Q = np.array([[0.0, 0.0]])  # Equal weights for the two tokens.
    K = np.array([[1.0, 1.0], [2.0, 2.0]])
    V = np.array([[2.0, 4.0], [6.0, 8.0]])
    V_hat = np.array([[4.0, 8.0], [6.0, 8.0]])

    errors = compute_error_metrics(Q, K, V, K, V_hat)

    assert errors["key_reconstruction_error"] == 0
    assert errors["attention_score_error"] == 0
    np.testing.assert_allclose(errors["value_reconstruction_error"], np.sqrt(2))
    # Original output is [4, 6], reconstructed output is [5, 8].
    np.testing.assert_allclose(errors["attention_output_error"], 5 / 12)


def test_key_errors_do_not_change_value_output_metric():
    Q = np.array([[1.0]])
    K = np.array([[1.0], [2.0]])
    K_hat = np.array([[1.0], [3.0]])
    V = np.array([[2.0], [4.0]])

    errors = compute_error_metrics(Q, K, V, K_hat, V)

    np.testing.assert_allclose(errors["key_reconstruction_error"], 0.5)
    expected_A = [[1 / (1 + np.exp(1)), np.exp(1) / (1 + np.exp(1))]]
    expected_A_hat = [[1 / (1 + np.exp(2)), np.exp(2) / (1 + np.exp(2))]]
    expected_error = relative_error(expected_A, expected_A_hat)
    np.testing.assert_allclose(errors["attention_score_error"], expected_error)
    assert errors["value_reconstruction_error"] == 0
    assert errors["attention_output_error"] == 0
