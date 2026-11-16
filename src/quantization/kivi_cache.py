"""KIVI KV cache for Hugging Face generation (fake quantization).

Implements KIVI (Liu et al., 2024; Algorithm 1 and the official code
github.com/jy-yuan/KIVI) as a transformers `Cache`, so every layer's attention
runs on the quantized cache and quantization error propagates through the
layers and across generated tokens.

Quantization rules, applied to post-RoPE keys and values (the cache receives
keys after RoPE, exactly as in KIVI):

* Keys, 'channel' mode: tokens collect in a full-precision residual. Whenever
  the cache length is a multiple of R, the residual is quantized per channel in
  groups of G tokens. After prefill of length l, the first l - (l % R) tokens
  are quantized.
* Values, 'token' mode: the newest R tokens stay in full precision. Older
  tokens are quantized per token, in groups of G channels.

Either rule can be used for either tensor (key_mode / value_mode), so the
per-token-keys and per-channel-values baselines run through the same code.

This is *fake* quantization: values are quantized and immediately dequantized,
then stored in the model dtype. The numbers attention sees are the same as
with real low-bit storage, but no memory is saved.
"""

import torch
from transformers import Cache
from transformers.cache_utils import DynamicLayer

MODES = ("channel", "token", "none")


def fake_quant(x, bits, dim):
    """Asymmetric min/max quantize-then-dequantize along `dim`, computed in float32."""
    x32 = x.float()
    mn = x32.amin(dim=dim, keepdim=True)
    mx = x32.amax(dim=dim, keepdim=True)
    levels = 2 ** bits - 1
    scale = ((mx - mn) / levels).clamp_min(1e-8)
    q = ((x32 - mn) / scale).round().clamp(0, levels)
    return (q * scale + mn).to(x.dtype)


def quantize_per_channel(x, bits, group_size):
    """Quantize (..., tokens, dim) per channel, in groups of `group_size` tokens.

    The token count must be a multiple of group_size (KIVI only ever quantizes
    full groups of keys).
    """
    *lead, n, d = x.shape
    assert n % group_size == 0, f"{n} tokens is not a multiple of group size {group_size}"
    groups = x.reshape(*lead, n // group_size, group_size, d)
    return fake_quant(groups, bits, dim=-2).reshape(x.shape)


def quantize_per_token(x, bits, group_size):
    """Quantize (..., tokens, dim) per token, in groups of `group_size` channels."""
    d = x.shape[-1]
    if d % group_size == 0:
        groups = x.reshape(*x.shape[:-1], d // group_size, group_size)
        return fake_quant(groups, bits, dim=-1).reshape(x.shape)
    # Last group is smaller, matching the numpy version in quantize.py.
    return torch.cat([fake_quant(x[..., c:c + group_size], bits, dim=-1)
                      for c in range(0, d, group_size)], dim=-1)


def quantized_length(mode, length, residual_length):
    """How many of the first `length` tokens are quantized under `mode`."""
    if mode == "channel":
        return length - length % residual_length
    if mode == "token":
        return max(0, length - residual_length)
    return 0


class KIVILayer(DynamicLayer):
    """One layer's cache. Stores [quantized (dequantized) prefix | full-precision residual]."""

    # Cropping would desync the quantized-token counts; nothing we run needs it.
    is_croppable = False

    def __init__(self, bits=2, group_size=32, residual_length=128, key_mode="channel",
                 value_mode="token", quantize_prefill_attention=True,
                 quantize_before_attention=False):
        super().__init__()
        assert key_mode in MODES and value_mode in MODES
        assert residual_length % group_size == 0, "KIVI requires R to be a multiple of G"
        self.bits = bits
        self.group_size = group_size
        self.residual_length = residual_length
        self.key_mode = key_mode
        self.value_mode = value_mode
        self.quantize_prefill_attention = quantize_prefill_attention
        self.quantize_before_attention = quantize_before_attention
        # Number of leading tokens of self.keys / self.values that are quantized.
        self.num_quantized_keys = 0
        self.num_quantized_values = 0

    def reset(self):
        super().reset()
        self.num_quantized_keys = 0
        self.num_quantized_values = 0

    def _quantize_new(self, x, mode, num_quantized):
        """Quantize tokens that have just become due; returns (tensor, new count).

        Builds a new tensor rather than writing in place, so a tensor already
        returned to attention is never changed afterwards.
        """
        target = quantized_length(mode, x.shape[-2], self.residual_length)
        if target <= num_quantized or self.bits >= 16:
            return x, num_quantized
        segment = x[..., num_quantized:target, :]
        if mode == "channel":
            segment = quantize_per_channel(segment, self.bits, self.group_size)
        else:
            segment = quantize_per_token(segment, self.bits, self.group_size)
        x = torch.cat([x[..., :num_quantized, :], segment, x[..., target:, :]], dim=-2)
        return x, target

    def _flush(self):
        self.keys, self.num_quantized_keys = self._quantize_new(
            self.keys, self.key_mode, self.num_quantized_keys)
        self.values, self.num_quantized_values = self._quantize_new(
            self.values, self.value_mode, self.num_quantized_values)

    def update(self, key_states, value_states, *args, **kwargs):
        is_prefill = not self.is_initialized
        if is_prefill:
            self.lazy_initialization(key_states, value_states)
        self.keys = torch.cat([self.keys, key_states], dim=-2)
        self.values = torch.cat([self.values, value_states], dim=-2)

        if is_prefill:
            # Prefill: attention sees either the quantized cache (mentor's
            # version) or the exact K/V (KIVI paper); the stored cache is
            # quantized either way.
            exact_keys, exact_values = self.keys, self.values
            self._flush()
            if self.quantize_prefill_attention:
                return self.keys, self.values
            return exact_keys, exact_values

        if self.quantize_before_attention:
            # Algorithm 1's pseudocode: quantize due tokens, then attend.
            self._flush()
            return self.keys, self.values
        # Official KIVI code: attend with the newly filled residual still in
        # full precision, then quantize it for the next step.
        keys, values = self.keys, self.values
        self._flush()
        return keys, values


class KIVICache(Cache):
    """KIVI cache for every layer of a model.

    Args:
        config: the model config (used for the number of layers).
        bits: quantization bits (2 or 4 in the paper; >= 16 disables quantization).
        group_size: G, tokens per key group / channels per value group.
        residual_length: R, full-precision residual length.
        key_mode, value_mode: 'channel', 'token' or 'none'. KIVI is
            key_mode='channel', value_mode='token'.
        quantize_prefill_attention: if True (default), prefill attention uses the
            quantized cache so errors propagate from the first forward pass. If
            False, prefill attention uses exact K/V, as in the KIVI paper.
        quantize_before_attention: if True, newly due tokens are quantized
            before that step's attention (Algorithm 1 pseudocode). Default False
            follows the official code, which quantizes after.
    """

    def __init__(self, config, bits=2, group_size=32, residual_length=128, key_mode="channel",
                 value_mode="token", quantize_prefill_attention=True,
                 quantize_before_attention=False):
        num_layers = config.get_text_config(decoder=True).num_hidden_layers
        layers = [KIVILayer(bits, group_size, residual_length, key_mode, value_mode,
                            quantize_prefill_attention, quantize_before_attention)
                  for _ in range(num_layers)]
        super().__init__(layers=layers)
