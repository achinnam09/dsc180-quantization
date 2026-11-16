import numpy as np
import pytest
import torch
from transformers import DynamicCache, Qwen3Config, Qwen3ForCausalLM

from quantization.kivi_cache import KIVICache, KIVILayer
from quantization.quantize import per_channel_quantization, per_token_quantization

# Small G and R so short sequences cross many group and residual boundaries.
G, R, BITS = 4, 16, 2
HEADS, DIM = 2, 16


def random_kv(length, batch=1, seed=0):
    g = torch.Generator().manual_seed(seed)
    K = torch.randn(batch, HEADS, length, DIM, generator=g)
    K[..., 3] *= 10  # an outlier channel, like real keys
    V = torch.randn(batch, HEADS, length, DIM, generator=g)
    return K, V


def offline(K, V):
    """Quantize each (batch, head) with the numpy quantizers from Task 1a."""
    Kq, Vq = torch.empty_like(K), torch.empty_like(V)
    for b in range(K.shape[0]):
        for h in range(K.shape[1]):
            Kq[b, h] = torch.from_numpy(per_channel_quantization(K[b, h].numpy(), bits=BITS, G=G, R=R))
            Vq[b, h] = torch.from_numpy(per_token_quantization(V[b, h].numpy(), bits=BITS, G=G, R=R))
    return Kq, Vq


def make_layer(**kwargs):
    return KIVILayer(bits=BITS, group_size=G, residual_length=R, **kwargs)


# ---------------------------------------------------------------- cache layer

@pytest.mark.parametrize("prompt_len", [1, 5, 15, 16, 17, 40, 48])
def test_streaming_cache_matches_offline_prefix_quantization(prompt_len):
    """After prefill and after every decode step, the stored cache equals the
    offline quantizers applied to the whole prefix. So the streaming cache
    quantizes every token exactly once, at the right time, and never changes
    it again."""
    total = 70
    K, V = random_kv(total, batch=2)
    layer = make_layer()

    layer.update(K[..., :prompt_len, :], V[..., :prompt_len, :])
    for L in range(prompt_len, total + 1):
        if L > prompt_len:
            layer.update(K[..., L - 1:L, :], V[..., L - 1:L, :])
        Kq, Vq = offline(K[..., :L, :], V[..., :L, :])
        torch.testing.assert_close(layer.keys, Kq, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(layer.values, Vq, rtol=1e-5, atol=1e-5)
        # Residual bounds: fewer than R full-precision keys, exactly min(L, R) values.
        assert L - layer.num_quantized_keys < R
        assert L - layer.num_quantized_values == min(L, R)


def test_prefill_flag_controls_what_attention_sees():
    K, V = random_kv(40)

    quantized = make_layer(quantize_prefill_attention=True)
    k_out, v_out = quantized.update(K, V)
    torch.testing.assert_close(k_out, quantized.keys)  # attention uses the quantized cache
    assert not torch.equal(k_out, K)

    exact = make_layer(quantize_prefill_attention=False)
    k_out, v_out = exact.update(K, V)
    assert torch.equal(k_out, K) and torch.equal(v_out, V)  # attention uses exact K/V
    torch.testing.assert_close(exact.keys, quantized.keys)  # but the stored cache is quantized


def test_decode_ordering_official_vs_pseudocode():
    """The step that fills the key residual to R: the official code attends with
    those R keys in full precision; Algorithm 1's pseudocode quantizes them first."""
    K, V = random_kv(2 * R)
    official = make_layer()
    pseudo = make_layer(quantize_before_attention=True)
    for layer in (official, pseudo):
        layer.update(K[..., :R + 3, :], V[..., :R + 3, :])

    # Decode up to length 2R, where the second key group of R becomes due.
    for L in range(R + 4, 2 * R + 1):
        k_off, _ = official.update(K[..., L - 1:L, :], V[..., L - 1:L, :])
        k_ps, _ = pseudo.update(K[..., L - 1:L, :], V[..., L - 1:L, :])

    assert torch.equal(k_off[..., R:, :], K[..., R:2 * R, :])  # still full precision
    assert not torch.equal(k_ps[..., R:, :], K[..., R:2 * R, :])  # already quantized
    # Either way, the stored state afterwards is the same.
    torch.testing.assert_close(official.keys, pseudo.keys)


def test_returned_tensor_not_modified_after_return():
    """Quantizing after attention must not change the tensor attention received."""
    K, V = random_kv(R)
    layer = make_layer()
    layer.update(K[..., :R - 1, :], V[..., :R - 1, :])
    k_out, _ = layer.update(K[..., R - 1:, :], V[..., R - 1:, :])
    assert torch.equal(k_out, K)  # returned before the flush...
    assert not torch.equal(layer.keys, K)  # ...and the flush made a new tensor


def test_modes_none_and_16_bits_do_not_quantize():
    K, V = random_kv(50)
    for layer in (make_layer(key_mode="none", value_mode="none"),
                  KIVILayer(bits=16, group_size=G, residual_length=R)):
        layer.update(K, V)
        assert torch.equal(layer.keys, K) and torch.equal(layer.values, V)


def test_per_token_keys_baseline_matches_offline():
    K, V = random_kv(45)
    layer = make_layer(key_mode="token", value_mode="channel")
    layer.update(K, V)
    expected_k = torch.from_numpy(per_token_quantization(K[0, 0].numpy(), bits=BITS, G=G, R=R))
    expected_v = torch.from_numpy(per_channel_quantization(V[0, 0].numpy(), bits=BITS, G=G, R=R))
    torch.testing.assert_close(layer.keys[0, 0], expected_k, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(layer.values[0, 0], expected_v, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------- full model

@pytest.fixture(scope="module")
def tiny_model():
    # Same architecture as Qwen3-1.7B (GQA, q/k norm, RoPE), just tiny and random.
    torch.manual_seed(0)
    config = Qwen3Config(vocab_size=128, hidden_size=64, intermediate_size=128,
                         num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                         head_dim=16, max_position_embeddings=512)
    return Qwen3ForCausalLM(config).eval()


def prompt(batch=1, length=40, seed=1):
    return torch.randint(0, 128, (batch, length), generator=torch.Generator().manual_seed(seed))


def generate(model, ids, cache, n=25):
    with torch.no_grad():
        return model.generate(ids, attention_mask=torch.ones_like(ids), past_key_values=cache,
                              max_new_tokens=n, do_sample=False, pad_token_id=0)


def kivi(model, **kwargs):
    return KIVICache(model.config, bits=BITS, group_size=G, residual_length=R, **kwargs)


def test_cache_with_quantization_off_matches_default_cache(tiny_model):
    ids = prompt()
    expected = generate(tiny_model, ids, DynamicCache(config=tiny_model.config))
    got = generate(tiny_model, ids, kivi(tiny_model, key_mode="none", value_mode="none"))
    assert torch.equal(got, expected)


def test_quantization_reaches_every_layer(tiny_model):
    ids = prompt()
    cache = kivi(tiny_model)
    out = generate(tiny_model, ids, cache, n=25)
    total = out.shape[1] - 1  # the last generated token is never fed back in
    assert len(cache.layers) == tiny_model.config.num_hidden_layers
    for layer in cache.layers:
        assert layer.keys.shape[-2] == total
        assert layer.num_quantized_keys == total - total % R
        assert layer.num_quantized_values == total - R


def test_prefill_flag_on_real_forward_pass(tiny_model):
    ids = prompt()
    with torch.no_grad():
        exact = tiny_model(ids, past_key_values=DynamicCache(config=tiny_model.config)).logits
        paper = tiny_model(ids, past_key_values=kivi(tiny_model, quantize_prefill_attention=False)).logits
        mentor = tiny_model(ids, past_key_values=kivi(tiny_model, quantize_prefill_attention=True)).logits
    torch.testing.assert_close(paper, exact)  # KIVI paper: prefill attention is exact
    assert not torch.allclose(mentor, exact)  # mentor's version: quantized from the start


def test_errors_propagate_through_layers(tiny_model):
    """With a quantized cache, layer 0 sees the same input, but every later
    layer's input differs from the fp run, so quantization error carries forward."""
    ids = prompt()
    with torch.no_grad():
        fp = tiny_model(ids, past_key_values=DynamicCache(config=tiny_model.config),
                        output_hidden_states=True).hidden_states
        q = tiny_model(ids, past_key_values=kivi(tiny_model), output_hidden_states=True).hidden_states
    assert torch.equal(fp[0], q[0])  # embeddings: before any attention
    for layer_idx in range(1, len(fp)):
        assert not torch.allclose(fp[layer_idx], q[layer_idx])


def test_batched_generation_matches_single(tiny_model):
    ids = prompt(batch=2)
    batched = generate(tiny_model, ids, kivi(tiny_model))
    for b in range(2):
        single = generate(tiny_model, ids[b:b + 1], kivi(tiny_model))
        assert torch.equal(batched[b:b + 1], single)
