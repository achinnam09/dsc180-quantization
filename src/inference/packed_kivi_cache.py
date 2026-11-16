"""Packed 2-bit KIVI cache for a single unpadded prompt and greedy decoding.

Qwen calls Cache.update after RoPE. Prefill attention uses the original tensors;
subsequent steps use reconstructed history plus full-precision recent tokens.
Cache conversion prepares the NEXT step, matching the KIVI reference ordering.
Storage is packed, but attention temporarily unpacks to use ordinary SDPA. This
is a correctness implementation, not the paper's fused CUDA attention kernels.
"""
from dataclasses import dataclass

import torch
from transformers.cache_utils import Cache, CacheLayerMixin


@dataclass
class PackedTensor:
    codes: torch.Tensor
    minimum: torch.Tensor
    scale: torch.Tensor
    shape: tuple
    axis: int
    dtype: torch.dtype

    @classmethod
    def pack(cls, tensor, group_size, axis):
        moved = tensor.movedim(axis, -1)
        if group_size % 4 or moved.shape[-1] % group_size:
            raise ValueError("The quantized axis must contain complete groups of a multiple of four")
        groups = moved.float().reshape(*moved.shape[:-1], -1, group_size)
        minimum = groups.amin(dim=-1, keepdim=True)
        scale = ((groups.amax(dim=-1, keepdim=True) - minimum) / 3).clamp_min(1e-8)
        codes = ((groups - minimum) / scale).round().clamp(0, 3).to(torch.uint8)
        # Four independent 2-bit codes occupy one byte, lowest bits first.
        packed = codes[..., 0::4] | (codes[..., 1::4] << 2) | (codes[..., 2::4] << 4) | (codes[..., 3::4] << 6)
        return cls(packed, minimum, scale, tuple(moved.shape), axis, tensor.dtype)

    def unpack(self):
        codes = torch.stack([(self.codes >> shift) & 3 for shift in (0, 2, 4, 6)], dim=-1)
        groups = codes.flatten(-2).float() * self.scale + self.minimum
        return groups.reshape(self.shape).movedim(-1, self.axis).to(self.dtype)

    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in (self.codes, self.minimum, self.scale))


class KIVILayer(CacheLayerMixin):
    is_sliding = False
    supports_early_init = False

    def __init__(self, group_size=32, residual_length=128, enabled=True):
        super().__init__()
        if group_size <= 0 or group_size % 4 or residual_length <= 0 or residual_length % group_size:
            raise ValueError("Require group_size divisible by 4 and residual_length divisible by group_size")
        self.group_size = group_size
        self.residual_length = residual_length
        self.enabled = enabled
        self.key_blocks = []
        self.value_blocks = []
        self.length = 0
        self.key_flushes = 0
        self.value_flushes = 0
        self.prefill_calls = 0
        self.decode_calls = 0

    def lazy_initialization(self, key_states, value_states):
        if key_states.ndim != 4 or key_states.shape != value_states.shape or key_states.shape[0] != 1:
            raise ValueError("Expected matching [1, kv_heads, tokens, head_dim] tensors")
        if key_states.shape[-1] % self.group_size:
            raise ValueError("head_dim must be divisible by group_size")
        self.dtype, self.device = key_states.dtype, key_states.device
        self.keys = key_states[..., :0, :].clone()
        self.values = value_states[..., :0, :].clone()
        self.is_initialized = True

    def update(self, key_states, value_states, *args, **kwargs):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        elif key_states.shape[-2] != 1:
            raise ValueError("After prefill this cache accepts one decode token at a time")
        if key_states.shape != value_states.shape or key_states.shape[:2] != self.keys.shape[:2]:
            raise ValueError("Key/value shapes or batch/head counts changed")
        if key_states.shape[-1] != self.keys.shape[-1] or key_states.shape[-2] == 0:
            raise ValueError("Invalid head dimension or empty update")

        if self.length == 0:
            self.prefill_calls += 1
        else:
            self.decode_calls += 1
        self.keys = torch.cat((self.keys, key_states), dim=-2)
        self.values = torch.cat((self.values, value_states), dim=-2)
        self.length += key_states.shape[-2]

        # Materialize BEFORE flushing: newly eligible residual data remains full
        # precision for this attention operation, then is packed for future steps.
        attention_keys = torch.cat([b.unpack() for b in self.key_blocks] + [self.keys], dim=-2)
        attention_values = torch.cat([b.unpack() for b in self.value_blocks] + [self.values], dim=-2)
        if self.enabled:
            key_rows = self.keys.shape[-2] // self.residual_length * self.residual_length
            if key_rows:
                self.key_blocks.append(PackedTensor.pack(self.keys[..., :key_rows, :], self.group_size, -2))
                self.keys = self.keys[..., key_rows:, :].clone()
                self.key_flushes += 1
            value_rows = max(0, self.values.shape[-2] - self.residual_length)
            if value_rows:
                self.value_blocks.append(PackedTensor.pack(self.values[..., :value_rows, :], self.group_size, -1))
                self.values = self.values[..., value_rows:, :].clone()
                self.value_flushes += 1
        return attention_keys, attention_values

    def get_seq_length(self):
        return self.length

    def get_mask_sizes(self, query_length):
        return self.length + query_length, 0

    def get_max_length(self):
        return -1

    def reset(self):
        self.__init__(self.group_size, self.residual_length, self.enabled)

    def reorder_cache(self, beam_idx):
        raise NotImplementedError("Beam search is not supported; use greedy decoding")

    def nbytes(self):
        residuals = sum(t.numel() * t.element_size() for t in (self.keys, self.values) if t is not None)
        return residuals + sum(b.nbytes() for b in self.key_blocks + self.value_blocks)


class KIVICache(Cache):
    def __init__(self, num_layers, group_size=32, residual_length=128, enabled=True):
        super().__init__(layers=[KIVILayer(group_size, residual_length, enabled) for _ in range(num_layers)])

    def stats(self):
        return {
            "stored_tensor_bytes": sum(layer.nbytes() for layer in self.layers),
            "layers": [
                {
                    "tokens": layer.length,
                    "key_residual_tokens": 0 if layer.keys is None else layer.keys.shape[-2],
                    "value_residual_tokens": 0 if layer.values is None else layer.values.shape[-2],
                    "key_flushes": layer.key_flushes, "value_flushes": layer.value_flushes,
                    "prefill_calls": layer.prefill_calls, "decode_calls": layer.decode_calls,
                }
                for layer in self.layers
            ],
        }
