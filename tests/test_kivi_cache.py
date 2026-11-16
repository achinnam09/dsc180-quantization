import numpy as np
import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.cache_utils import DynamicCache

from inference.packed_kivi_cache import KIVICache, KIVILayer, PackedTensor
from quantization.quantize import per_channel_quantization, per_token_quantization


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_known_packed_byte_and_constant_group(dtype):
    x = torch.tensor([[[[0., 1., 2., 3., 7., 7., 7., 7.]]]], dtype=dtype)
    packed = PackedTensor.pack(x, 4, -1)
    assert packed.codes.dtype == torch.uint8
    assert packed.codes.flatten().tolist() == [228, 0]
    torch.testing.assert_close(packed.unpack(), x)
    assert packed.codes.numel() == x.numel() // 4


@pytest.mark.parametrize('prefill_length', [3, 7, 8, 9, 16, 17])
def test_prefill_and_decode_against_independent_matrix_quantizers(prefill_length):
    generator = torch.Generator().manual_seed(42)
    k = torch.randn(1, 2, 26, 8, generator=generator)
    v = torch.randn(1, 2, 26, 8, generator=generator)
    cache = KIVILayer(group_size=4, residual_length=8)
    ends = [prefill_length] + list(range(prefill_length + 1, 27))
    start = 0
    previous_k = previous_v = None
    for end in ends:
        attention_k, attention_v = cache.update(k[..., start:end, :], v[..., start:end, :])
        expected_k = k[..., :end, :] if start == 0 else torch.cat((previous_k, k[..., start:end, :]), -2)
        expected_v = v[..., :end, :] if start == 0 else torch.cat((previous_v, v[..., start:end, :]), -2)
        torch.testing.assert_close(attention_k, expected_k)
        torch.testing.assert_close(attention_v, expected_v)
        stored_k = torch.cat([b.unpack() for b in cache.key_blocks] + [cache.keys], -2)
        stored_v = torch.cat([b.unpack() for b in cache.value_blocks] + [cache.values], -2)
        for head in range(2):
            np.testing.assert_allclose(stored_k[0, head], per_channel_quantization(k[0, head, :end].numpy(), G=4, R=8), atol=1e-6)
            np.testing.assert_allclose(stored_v[0, head], per_token_quantization(v[0, head, :end].numpy(), G=4, R=8), atol=1e-6)
        assert cache.get_seq_length() == end
        assert cache.get_mask_sizes(1) == (end + 1, 0)
        assert cache.keys.shape[-2] == end % 8
        assert cache.values.shape[-2] == min(end, 8)
        # Cloning the tail must release the larger original tensor allocation.
        assert cache.keys.untyped_storage().nbytes() == cache.keys.numel() * cache.keys.element_size()
        previous_k, previous_v = stored_k, stored_v
        start = end
    cache.reset()
    assert cache.get_seq_length() == 0 and cache.nbytes() == 0


def test_tiny_qwen_prefill_decode_positions_and_grouped_query_heads():
    torch.manual_seed(10)
    config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                        head_dim=8, attention_dropout=0.0)
    model = Qwen3ForCausalLM(config).eval()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]])
    normal = DynamicCache(config=config)
    disabled = KIVICache(2, 4, 8, enabled=False)
    quantized = KIVICache(2, 4, 8)
    differences = []
    with torch.inference_mode():
        start = 0
        for end in [7, 8, 9, 10, 11, 12, 13]:
            token = ids[:, start:end]
            a = model(token, past_key_values=normal, use_cache=True).logits[:, -1]
            b = model(token, past_key_values=disabled, use_cache=True).logits[:, -1]
            c = model(token, past_key_values=quantized, use_cache=True).logits[:, -1]
            whole = model(ids[:, :end], use_cache=False).logits[:, -1]
            torch.testing.assert_close(a, b)
            torch.testing.assert_close(a, whole, rtol=1e-4, atol=1e-6)
            if end <= 8:
                torch.testing.assert_close(a, c)
            differences.append((a - c).abs().max().item())
            assert torch.isfinite(c).all()
            start = end
    assert max(differences) > 1e-6  # quantized history actually reaches attention
    for layer in quantized.layers:
        assert layer.length == 13 and layer.keys.shape[1] == 2
        assert layer.prefill_calls == 1 and layer.decode_calls == 6
        assert layer.key_flushes > 0 and layer.value_flushes > 0


def test_invalid_configuration_and_multitoken_decode_are_rejected():
    with pytest.raises(ValueError):
        KIVILayer(group_size=6, residual_length=8)
    cache = KIVILayer(4, 8)
    x = torch.ones(1, 1, 2, 8)
    cache.update(x, x)
    with pytest.raises(ValueError):
        cache.update(x, x)


def test_qa_f1_normalization_overlap_and_multiple_references():
    from inference.compare_longbench import qa_f1
    assert qa_f1('The RED, fox!', ['red fox']) == 1.0
    assert qa_f1('red fox', ['red dog']) == 0.5
    assert qa_f1('fox', ['red fox']) == pytest.approx(2 / 3)
    assert qa_f1('fox', ['dog', 'fox']) == 1.0
    assert qa_f1('', ['']) == 0.0
