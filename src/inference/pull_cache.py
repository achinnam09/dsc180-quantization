from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer, BatchEncoding
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from pathlib import Path
import torch

from longbench import build_inputs


QKVs = {"query": [], "keys": [], "values": []}


def capture_attention(module, query, key, value, attention_mask, **kwargs):
    if module.layer_idx == len(QKVs["query"]):
        QKVs["query"].append(query.detach().cpu())
    return sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)


AttentionInterface.register("capture", capture_attention)

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype="auto", device_map="auto", attn_implementation="capture")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

assert tokenizer

inputs = build_inputs(tokenizer)
assert isinstance(inputs, BatchEncoding)
inputs.to(model.device)
print("prompt tokens:", inputs["input_ids"].shape[1])

with torch.no_grad():
    outputs = model(**inputs, use_cache=True)
    

for layer in outputs.past_key_values.layers:

    QKVs["keys"].append(layer.keys.cpu())
    QKVs["values"].append(layer.values.cpu())

output_dir = Path(__file__).resolve().parents[2] / "kv_snapshots"
output_dir.mkdir(exist_ok=True)

torch.save(
    QKVs,
    output_dir / "layer.pt",
)
print(f"Saved Q/K/V to {output_dir / 'layer.pt'}")