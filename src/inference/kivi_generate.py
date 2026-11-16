"""Generate an answer to one LongBench prompt with an fp16 or KIVI-quantized KV cache.

Usage:
    uv run python src/inference/kivi_generate.py --configs fp16 kivi2 kivi4 token2

Every config runs the same prompt with greedy decoding, so outputs differ only
because of the KV cache.
"""

import argparse
import json
import sys
import time
import zipfile
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from quantization.kivi_cache import KIVICache  # noqa: E402

MODEL_NAME = "Qwen/Qwen3-1.7B"

# name -> KIVICache keyword arguments (None = ordinary full-precision cache).
CONFIGS = {
    "fp16": None,
    "kivi2": dict(bits=2, key_mode="channel", value_mode="token"),
    "kivi4": dict(bits=4, key_mode="channel", value_mode="token"),
    "token2": dict(bits=2, key_mode="token", value_mode="token"),
    "channel2": dict(bits=2, key_mode="channel", value_mode="channel"),
}


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(device=None):
    # Load straight onto one device. device_map="auto" offloaded part of the
    # model to disk on an 8 GB Mac, which makes generation very slow.
    device = device or pick_device()
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return model, tokenizer


def load_longbench(task):
    zip_path = hf_hub_download(repo_id="zai-org/LongBench", filename="data.zip", repo_type="dataset")
    with zipfile.ZipFile(zip_path) as z:
        member = next(n for n in z.namelist() if n.endswith(f"{task}.jsonl"))
        with z.open(member) as f:
            return [json.loads(line) for line in f]


def build_inputs(tokenizer, sample, max_context_tokens, device):
    # Middle truncation of the context, as LongBench's own evaluation does.
    ids = tokenizer(sample["context"], add_special_tokens=False)["input_ids"]
    if len(ids) > max_context_tokens:
        half = max_context_tokens // 2
        ids = ids[:half] + ids[-half:]
    context = tokenizer.decode(ids)
    chat = [{"role": "user", "content": f"{context}\n\nQuestion: {sample['input']}\nAnswer briefly."}]
    return tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True,
                                         enable_thinking=False, return_tensors="pt",
                                         return_dict=True).to(device)


def make_cache(model, config_name, quantize_prefill_attention=True):
    kwargs = CONFIGS[config_name]
    if kwargs is None:
        return DynamicCache(config=model.config)
    return KIVICache(model.config, group_size=32, residual_length=128,
                     quantize_prefill_attention=quantize_prefill_attention, **kwargs)


def generate(model, tokenizer, inputs, cache, max_new_tokens):
    with torch.no_grad():
        out = model.generate(**inputs, past_key_values=cache, max_new_tokens=max_new_tokens,
                             do_sample=False, pad_token_id=tokenizer.eos_token_id)
    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default="multifieldqa_en")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    parser.add_argument("--max-context-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--exact-prefill", action="store_true",
                        help="KIVI paper setting: prefill attention uses exact K/V")
    args = parser.parse_args()

    model, tokenizer = load_model()
    sample = load_longbench(args.task)[args.index]
    inputs = build_inputs(tokenizer, sample, args.max_context_tokens, model.device)
    print(f"device: {model.device}  prompt tokens: {inputs['input_ids'].shape[1]}")
    print(f"question: {sample['input']}")
    print(f"reference answer(s): {sample['answers']}\n")

    for name in args.configs:
        cache = make_cache(model, name, quantize_prefill_attention=not args.exact_prefill)
        start = time.time()
        answer = generate(model, tokenizer, inputs, cache, args.max_new_tokens)
        print(f"[{name:>8}] ({time.time() - start:5.1f}s) {answer}")


if __name__ == "__main__":
    main()
