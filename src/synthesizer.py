import numpy as np


class Synthesizer:
    def __init__(self, num_tokens=1000, num_channels=128, num_queries=1):
        self.num_tokens = num_tokens
        self.num_channels = num_channels
        self.num_queries = num_queries
        self.key_matrices = []
        self.value_matrices = []
        self.query_matrices = []

    def generate_key_matrix(self):
        K = np.random.randn(self.num_tokens, self.num_channels).astype(np.float32)

        # bump a group of channels to simulate outliers for key matrices
        channels = np.random.choice(
            self.num_channels, max(1, self.num_channels // 20), replace=False
        )
        K[:, channels] *= 10
        return K

    def generate_value_matrix(self):
        V = np.random.randn(self.num_tokens, self.num_channels).astype(np.float32)

        # Amplify scattered entries without a fixed channel or token pattern.
        positions = np.random.choice(V.size, max(1, V.size // 100), replace=False)
        V.flat[positions] *= 10
        return V

    def generate_query_matrix(self):
        return np.random.randn(self.num_queries, self.num_channels).astype(np.float32)

    def generate(self):
        """Generate, save, and return one Q/K/V set"""
        K = self.generate_key_matrix()
        V = self.generate_value_matrix()
        Q = self.generate_query_matrix()
        self.key_matrices.append(K)
        self.value_matrices.append(V)
        self.query_matrices.append(Q)
        return Q, K, V
