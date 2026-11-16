import torch
from transformers.cache_utils import Cache, DynamicLayer


def fake_quant(x, bits, dim):
    x_float = x.float()
    minimum = x_float.amin(dim=dim, keepdim=True)
    maximum = x_float.amax(dim=dim, keepdim=True)
    step = ((maximum - minimum) / (2**bits - 1)).clamp(min=1e-8)
    level = torch.round((x_float - minimum) / step)  # an integer from 0 to 2^bits - 1
    return (level * step + minimum).to(x.dtype)


def quantize_keys(keys, bits, group_size):
    batch, heads, tokens, channels = keys.shape
    groups = keys.reshape(batch, heads, tokens // group_size, group_size, channels)
    return fake_quant(groups, bits, dim=3).reshape(batch, heads, tokens, channels)


def quantize_values(values, bits, group_size):
    batch, heads, tokens, channels = values.shape
    groups = values.reshape(batch, heads, tokens, channels // group_size, group_size)
    return fake_quant(groups, bits, dim=4).reshape(batch, heads, tokens, channels)


class KIVILayer(DynamicLayer):
    """The cache for one layer of the model.

    `self.keys` / `self.values` hold every token so far. The oldest tokens are quantized,
    the newest are kept in full precision:
    - Keys are quantized in blocks of `residual_length` tokens, once a full block has built up.
    - Values keep only the newest `residual_length` tokens in full precision.
    """

    def __init__(self, bits=2, group_size=32, residual_length=128):
        super().__init__()
        self.bits = bits
        self.group_size = group_size
        self.residual_length = residual_length
        self.num_quantized_keys = 0
        self.num_quantized_values = 0

    def update(self, new_keys, new_values, *args, **kwargs):
        """Called by every attention layer on every forward pass with the new tokens' keys and values.
        Returns the keys and values that attention should use."""
        is_prefill = not self.is_initialized
        if is_prefill:
            self.lazy_initialization(new_keys, new_values)  # creates empty self.keys / self.values

        # 1. Add the new tokens to the end of the cache.
        self.keys = torch.cat([self.keys, new_keys], dim=2)
        self.values = torch.cat([self.values, new_values], dim=2)
        num_tokens = self.keys.shape[2]

        # 2. Keys: quantize up to the largest multiple of residual_length.
        start = self.num_quantized_keys
        end = num_tokens - num_tokens % self.residual_length
        if end > start:
            self.keys[:, :, start:end] = quantize_keys(self.keys[:, :, start:end], self.bits, self.group_size)
            self.num_quantized_keys = end

        # 3. Values: quantize everything except the newest residual_length tokens.
        start = self.num_quantized_values
        end = num_tokens - self.residual_length
        if end > start:
            self.values[:, :, start:end] = quantize_values(self.values[:, :, start:end], self.bits, self.group_size)
            self.num_quantized_values = end

        # In KIVI, attention over the prompt itself uses the original values; only what's stored is quantized.
        if is_prefill:
            return new_keys, new_values
        return self.keys, self.values


class KIVICache(Cache):
    def __init__(self, num_layers, bits=2, group_size=32, residual_length=128):
        super().__init__(layers=[KIVILayer(bits, group_size, residual_length) for _ in range(num_layers)])
