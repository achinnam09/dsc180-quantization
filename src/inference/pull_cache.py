from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer, BatchEncoding
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from pathlib import Path
import torch
import json
import zipfile
from huggingface_hub import hf_hub_download


QKVs = {"query": [], "keys": [], "values": []}


def capture_attention(module, query, key, value, attention_mask, **kwargs):
    if module.layer_idx == len(QKVs["query"]):
        QKVs["query"].append(query.detach().cpu())
    return sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)


AttentionInterface.register("capture", capture_attention)

LONGBENCH_TASK = "multifieldqa_en"   # any task name from the dataset card
SAMPLE_INDEX = 0
MAX_CONTEXT_TOKENS = 4096            # keeps the saved Q/K/V tensors a manageable size


def load_longbench(task):
    zip_path = hf_hub_download(
        repo_id="zai-org/LongBench", filename="data.zip", repo_type="dataset"
    )
    with zipfile.ZipFile(zip_path) as z:
        member = next(n for n in z.namelist() if n.endswith(f"{task}.jsonl"))
        with z.open(member) as f:
            return [json.loads(line) for line in f]

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype="auto", device_map="auto", attn_implementation="capture")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

assert tokenizer


# Long enough (well over the residual length of 128 tokens) that the
# quantizers have full groups of tokens to work on.
# Replacing long_text for the longbench model
long_text = """
For most of history, people who wanted to know where they were relied on the sky. Sailors crossing open water
watched the height of the sun at noon and the angle of familiar stars at night, and from those two observations
they could work out how far north or south they had traveled. Latitude, in other words, was always within reach
of anyone with a clear horizon and a simple instrument. Longitude was a different story. Because the Earth turns
fifteen degrees every hour, knowing how far east or west you had gone required knowing the time at your home port
at the exact moment you measured local noon. Without a reliable clock, a navigator could only estimate, and
estimates on long voyages drifted by hundreds of miles.

The consequences of that drift were severe. Ships ran aground on coastlines that their crews believed were still
days away. Supplies ran short because voyages took longer than planned, and sailors fell ill from scurvy while
searching for islands they had already passed. Governments and merchants understood that whoever solved the
problem would gain an enormous advantage in trade and in war, and several nations offered large prizes for a
practical method of finding longitude at sea.

Two broad approaches competed for those prizes. Astronomers argued that the sky itself could serve as a clock.
The moon moves steadily against the background stars, so if its position could be predicted precisely, a
navigator who measured the angle between the moon and a known star could look up the corresponding time at a
reference location. This lunar distance method demanded careful observation, accurate tables, and long
calculations, but it required no special equipment beyond a good sextant and a printed almanac.

Clockmakers took the opposite view. If a mechanical clock could keep accurate time through months of rolling
decks, changing temperatures, and salty air, the navigator would simply compare it with local noon. The
difficulty was that ordinary pendulum clocks failed completely on a moving ship, and even the best spring-driven
watches gained or lost several minutes a day as temperatures changed. Building a marine timekeeper meant
rethinking almost every part of a clock: compensating for metal that expanded in the heat, reducing friction so
that oil would not thicken and slow the mechanism, and isolating the regulating parts from the motion of the sea.

Over several decades, a series of increasingly refined sea clocks showed that the mechanical approach could work.
Trial voyages demonstrated errors of only a few seconds per day, which translated into positions accurate to
within a handful of miles. Even so, the instruments were expensive and difficult to manufacture, so for many
years ships carried both chronometers and almanacs, using each method to check the other.

Eventually, improvements in manufacturing made chronometers affordable, and they became standard equipment on
ocean-going vessels. The lunar distance method faded from everyday use, although navigators continued to learn
it as a backup. In the twentieth century, radio time signals let ships correct their clocks while underway, and
later, satellite navigation systems reduced the entire problem to reading a number from a screen.

The story of longitude is often told as a contest between two rival methods, but it is also a lesson in how
measurement shapes what people are able to do. Once position could be known reliably, maps improved, shipping
routes became safer and shorter, and scientific expeditions could record exactly where their observations were
made. A problem that had cost countless lives was solved not by a single discovery but by steady refinement of
instruments, tables, and techniques, each one reducing error a little further than the last.
"""
'''
chat = [
  {"role": "user", "content": f"Summarize the following passage in a few sentences:\n\n{long_text}"},
]
'''

sample = load_longbench(LONGBENCH_TASK)[SAMPLE_INDEX]

# Truncate the context from the middle (keep both ends), the same convention
# LongBench's own evaluation uses, so the question at the end is never cut off.
ids = tokenizer(sample["context"], add_special_tokens=False)["input_ids"]
if len(ids) > MAX_CONTEXT_TOKENS:
    half = MAX_CONTEXT_TOKENS // 2
    ids = ids[:half] + ids[-half:]
context = tokenizer.decode(ids)

chat = [
    {"role": "user", "content": f"{context}\n\nQuestion: {sample['input']}\nAnswer briefly."},
]

inputs = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True)
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