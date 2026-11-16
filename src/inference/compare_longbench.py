"""Run one complete LongBench QA prompt with normal and packed KIVI caches."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import string
import time
import zipfile

from huggingface_hub import hf_hub_download
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from .packed_kivi_cache import KIVICache

MODEL_ID = "Qwen/Qwen3-1.7B"
DATASET_ID = "zai-org/LongBench"


def qa_f1(prediction, references):
    """LongBench English QA F1: normalize, count token overlap, best reference.

    Definition: https://github.com/THUDM/LongBench/blob/main/LongBench/metrics.py
    """
    def tokens(text):
        text = text.lower().translate(str.maketrans('', '', string.punctuation))
        return re.sub(r'\b(a|an|the)\b', ' ', text).split()

    predicted = tokens(prediction)
    scores = []
    for reference in references:
        expected = tokens(reference)
        overlap = sum((Counter(predicted) & Counter(expected)).values())
        scores.append(2 * overlap / (len(predicted) + len(expected)) if overlap else 0.0)
    return max(scores, default=0.0)


def select_prompt(tokenizer, task, maximum, sample_index=None):
    archive = Path(hf_hub_download(DATASET_ID, "data.zip", repo_type="dataset"))
    with zipfile.ZipFile(archive) as z:
        member = next(name for name in z.namelist() if name.endswith(f"/{task}.jsonl") or name == f"{task}.jsonl")
        samples = [json.loads(line) for line in z.read(member).splitlines()]
    for index, sample in enumerate(samples):
        if sample_index is not None and index != sample_index:
            continue
        prompt = (
            "Read the document and answer the question using the document. "
            "Give only a brief answer, without explanation.\n\n"
            f"Document:\n{sample['context']}\n\nQuestion: {sample['input']}\nAnswer:"
        )
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False,
            return_tensors="pt", return_dict=True,
        )
        length = encoded['input_ids'].shape[1]
        if 128 < length <= maximum:
            return sample, index, prompt, encoded, archive.parent.name
        if sample_index is not None:
            raise ValueError(f"Selected prompt has {length} tokens; require 128 < length <= {maximum}; no truncation")
    raise ValueError("No complete prompt fits the requested token budget")


def cache_stats(cache):
    if isinstance(cache, KIVICache):
        return cache.stats()
    return {
        "stored_tensor_bytes": sum(t.numel() * t.element_size() for layer in cache.layers for t in (layer.keys, layer.values)),
        "layers": [{"tokens": layer.get_seq_length()} for layer in cache.layers],
    }


def generate(model, tokenizer, inputs, cache, max_new_tokens):
    # Same greedy decode loop for both methods. Cache type is the only change.
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    generated, trace = [], []
    start = time.perf_counter()
    with torch.inference_mode():
        output = model(**inputs, past_key_values=cache, use_cache=True, logits_to_keep=1)
        first_logits = output.logits[0, -1].float().cpu()
        if not torch.isfinite(first_logits).all():
            raise ValueError("Non-finite prefill logits")
        prefill = cache_stats(cache)
        prompt_length = inputs['input_ids'].shape[1]
        for step in range(max_new_tokens):
            token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
            token_id = token.item()
            generated.append(token_id)
            if token_id in eos:
                break
            if step + 1 == max_new_tokens:
                break
            # Position indices are derived from the cache's total length by Qwen.
            output = model(input_ids=token, past_key_values=cache, use_cache=True, logits_to_keep=1)
            if not torch.isfinite(output.logits).all():
                raise ValueError("Non-finite decode logits")
            expected_length = prompt_length + step + 1
            if any(layer.get_seq_length() != expected_length for layer in cache.layers):
                raise AssertionError("Cache token count is inconsistent across layers")
            state = cache_stats(cache)
            trace.append({"decode_step": step + 1, "stored_tensor_bytes": state['stored_tensor_bytes'], **state['layers'][0]})
    return {
        "answer": tokenizer.decode(generated, skip_special_tokens=True).strip(),
        "generated_token_ids": generated, "generated_tokens": len(generated),
        "stopped_on_eos": generated[-1] in eos,
        "elapsed_seconds": time.perf_counter() - start,
        "prefill_cache": prefill, "final_cache": cache_stats(cache),
        "decode_trace_layer_0": trace,
    }, first_logits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=['multifieldqa_en', 'qasper'], default='multifieldqa_en')
    parser.add_argument('--sample-index', type=int)
    parser.add_argument('--max-prompt-tokens', type=int, default=4096)
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'])
    parser.add_argument('--dtype', choices=['bfloat16', 'float32'], default='bfloat16')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error('--max-new-tokens must be positive')
    output_dir = args.output_dir or Path(__file__).resolve().parents[2] / 'kv_snapshots' / ('kivi_generation_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    output_dir.mkdir(parents=True, exist_ok=False)
    revision = Path(hf_hub_download(MODEL_ID, 'config.json')).parent.name
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=revision)
    sample, index, prompt, inputs, dataset_revision = select_prompt(tokenizer, args.task, args.max_prompt_tokens, args.sample_index)
    device = args.device or ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    metadata = {
        'model': MODEL_ID, 'model_revision': revision, 'dataset': DATASET_ID,
        'dataset_revision': dataset_revision, 'task': args.task, 'sample_index': index,
        'sample_id': sample.get('_id'), 'question': sample['input'], 'reference_answers': sample['answers'],
        'prompt': prompt, 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
        'input_ids': inputs['input_ids'][0].tolist(), 'prompt_tokens': inputs['input_ids'].shape[1],
        'truncated': False, 'selection': 'first complete example in dataset order within token budget' if args.sample_index is None else 'explicit index',
        'device': device, 'dtype': args.dtype, 'torch': str(torch.__version__), 'transformers': transformers.__version__,
        'bits': 2, 'group_size': 32, 'residual_length': 128, 'enable_thinking': False,
        'decoding': 'greedy', 'max_new_tokens': args.max_new_tokens,
        'storage_note': 'Packed 2-bit persistent history; temporary full-precision unpacking for SDPA. Bytes include scales and residuals, exclude model and temporaries.',
    }
    (output_dir / 'prompt.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f"Selected {args.task}[{index}], {metadata['prompt_tokens']} complete prompt tokens; device={device}", flush=True)
    # CPU loading followed by transfer avoids the direct-to-MPS loading crash.
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=revision, dtype=getattr(torch, args.dtype), attn_implementation='sdpa').to(device).eval()
    inputs = inputs.to(device)
    runs, first_logits = {}, {}
    for name in ('normal', 'kivi'):
        print(f"Running {name} prefill and greedy decode", flush=True)
        cache = DynamicCache(config=model.config) if name == 'normal' else KIVICache(model.config.num_hidden_layers)
        runs[name], first_logits[name] = generate(model, tokenizer, inputs, cache, args.max_new_tokens)
        runs[name]['qa_f1'] = qa_f1(runs[name]['answer'], sample['answers'])
        (output_dir / f'{name}.json').write_text(json.dumps(runs[name], indent=2) + '\n')
        print(f"{name}: {runs[name]['answer']!r}; QA F1={runs[name]['qa_f1']:.4f}", flush=True)
        del cache
        if device == 'mps':
            torch.mps.empty_cache()
    torch.testing.assert_close(first_logits['normal'], first_logits['kivi'], rtol=1e-4, atol=1e-5)
    report = {
        'metadata': metadata, 'runs': runs,
        'validation': {'prefill_logits_max_abs_difference': (first_logits['normal'] - first_logits['kivi']).abs().max().item()},
        'same_answer': runs['normal']['answer'] == runs['kivi']['answer'],
        'prefill_cache_storage_ratio_normal_over_kivi': runs['normal']['prefill_cache']['stored_tensor_bytes'] / runs['kivi']['prefill_cache']['stored_tensor_bytes'],
    }
    (output_dir / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f"Saved comparison to {output_dir}", flush=True)


if __name__ == '__main__':
    main()
