import json
import zipfile

from huggingface_hub import hf_hub_download

LONGBENCH_TASK = "multifieldqa_en"   # any task name from the dataset card
SAMPLE_INDEX = 0


def load_longbench(task):
    zip_path = hf_hub_download(
        repo_id="zai-org/LongBench", filename="data.zip", repo_type="dataset"
    )
    with zipfile.ZipFile(zip_path) as z:
        member = next(n for n in z.namelist() if n.endswith(f"{task}.jsonl"))
        with z.open(member) as f:
            return [json.loads(line) for line in f]


def build_inputs(tokenizer, task=LONGBENCH_TASK, index=SAMPLE_INDEX):
    """Tokenized chat prompt for one LongBench sample."""
    sample = load_longbench(task)[index]
    chat = [
        {"role": "user", "content": f"{sample['context']}\n\nQuestion: {sample['input']}\nAnswer briefly."},
    ]
    return tokenizer.apply_chat_template(
        chat, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )
