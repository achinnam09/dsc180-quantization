import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from kivi_cache import KIVICache
from longbench import build_inputs

MODEL_NAME = "Qwen/Qwen3-1.7B"
BITS = 2
GROUP_SIZE = 32
RESIDUAL_LENGTH = 128

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
prompt = build_inputs(tokenizer).to(model.device)
print("prompt tokens:", prompt["input_ids"].shape[1])


@torch.no_grad()
def generate(cache):
    """Run the prompt through the model (prefill), then generate one token at a time (decode),
    always picking the most likely next token, until end-of-sequence."""
    output = model(**prompt, past_key_values=cache)  # prefill
    tokens = []
    while True:
        token = output.logits[0, -1].argmax()  # most likely next token
        tokens.append(token)
        if token == tokenizer.eos_token_id:
            return torch.stack(tokens)
        output = model(input_ids=token.view(1, 1), past_key_values=cache)  # decode one token


ref_tokens = generate(DynamicCache(config=model.config))
kivi_cache = KIVICache(model.config.num_hidden_layers, BITS, GROUP_SIZE, RESIDUAL_LENGTH)
kivi_tokens = generate(kivi_cache)

print("\n--- normal cache answer ---")
print(tokenizer.decode(ref_tokens))
print("\n--- KIVI cache answer ---")
print(tokenizer.decode(kivi_tokens))
