import numpy as np


def relative_error(original, reconstructed, eps=1e-8):
    original = np.asarray(original, dtype=np.float64)
    reconstructed = np.asarray(reconstructed, dtype=np.float64)
    denominator = np.maximum(np.abs(original), eps)
    return float(np.linalg.norm((original - reconstructed) / denominator))


def attention_weights(Q, K):
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    scores = Q @ K.T
    # Subtract the row maximum so exponentiation does not overflow.
    weights = np.exp(scores - scores.max(axis=1, keepdims=True))
    return weights / weights.sum(axis=1, keepdims=True)


def compute_error_metrics(Q, K, V, K_hat, V_hat):
    A = attention_weights(Q, K)
    A_hat = attention_weights(Q, K_hat)

    return {
        "key_reconstruction_error": relative_error(K, K_hat),
        "attention_score_error": relative_error(A, A_hat),
        "value_reconstruction_error": relative_error(V, V_hat),
        "attention_output_error": relative_error(A @ V, A @ V_hat),
    }
