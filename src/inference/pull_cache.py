from transformers import AutoModelForImageTextToText, AutoTokenizer, BatchEncoding
from pathlib import Path
import torch

model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3.5-0.8B", dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")

assert tokenizer


chat = [
  {"role": "user", "content": "Hello, how are you?"},
  {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
  {"role": "user", "content": "Tell me about the data science major at UCSD"},
]

inputs = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True)
assert isinstance(inputs, BatchEncoding)
inputs.to(model.device)

with torch.no_grad():
    outputs = model(**inputs, use_cache=True)

layer = outputs.past_key_values.layers[3]

output_dir = Path(__file__).resolve().parents[2] / "kv_snapshots"
output_dir.mkdir(exist_ok=True)

torch.save(
    {"keys": layer.keys.cpu(), "values": layer.values.cpu()},
    output_dir / "layer_3.pt",
)
print(f"Saved K/V to {output_dir / 'layer_3.pt'}")
