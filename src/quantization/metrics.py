import numpy as np


def relative_error(original, reconstructed, eps=1e-8):
    denominator = max(np.linalg.norm(original), eps)
    return float(np.linalg.norm((original - reconstructed)) / denominator)


def attention_weights(Q, K):
    scores = (Q @ K.T) / np.sqrt(Q.shape[1])
    # Subtract the row maximum so exponentiation does not overflow.
    weights = np.exp(scores - scores.max(axis=1, keepdims=True))
    return weights / weights.sum(axis=1, keepdims=True)


def compute_error_metrics(Q, K, V, K_hat, V_hat):
    A = attention_weights(Q, K)
    A_hat = attention_weights(Q, K_hat)

    O = A @ V
    return {
        # Key cache only: V is left exact.
        "key_reconstruction_error": relative_error(K, K_hat),
        "attention_score_error": relative_error(A, A_hat),
        "key_output_error": relative_error(O, A_hat @ V),
        # Value cache only: the attention weights are left exact.
        "value_reconstruction_error": relative_error(V, V_hat),
        "attention_output_error": relative_error(O, A @ V_hat),
        # Both caches quantized together, which is what happens in practice.
        "combined_output_error": relative_error(O, A_hat @ V_hat),
    }
