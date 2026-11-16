"""Run LongBench tasks with an fp16 or KIVI-quantized KV cache, then score them.

Follows the official LongBench v1 protocol (pred.py / eval.py): per-task prompt
templates and max generation lengths, middle truncation of the prompt to
--max-length tokens, the chat template only for non-few-shot tasks, greedy
decoding, and the official metrics.

Defaults match the KIVI paper: its 8 LongBench tasks and a 4096-token limit.

Examples:
    # Task 3: one Qasper prompt, every config
    uv run python src/evaluation/run_longbench.py --tasks qasper --n 1 --show

    # Subset: 30 prompts from each of KIVI's 8 tasks
    uv run python src/evaluation/run_longbench.py --n 30 --run-name subset30

    # Re-score finished runs without generating
    uv run python src/evaluation/run_longbench.py --run-name subset30 --score-only

Predictions go to results/longbench/<run-name>/<config>/<task>.jsonl, one line
per prompt. Re-running the same command skips prompts already done, so an
interrupted run (e.g. a DSMLP pod timing out) resumes where it stopped.
"""

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from evaluation.longbench_metrics import score_dataset, score_sample  # noqa: E402
from inference.kivi_generate import CONFIGS, load_longbench, load_model, make_cache  # noqa: E402

CONFIG_DIR = Path(__file__).resolve().parent / "config"
DATASET2PROMPT = json.loads((CONFIG_DIR / "dataset2prompt.json").read_text())
DATASET2MAXLEN = json.loads((CONFIG_DIR / "dataset2maxlen.json").read_text())

# The LongBench tasks reported in the KIVI paper (Table 4).
KIVI_TASKS = ["qasper", "qmsum", "multi_news", "trec", "triviaqa", "samsum", "lcc", "repobench-p"]
# LongBench feeds these raw, without the chat template ("chat models are better
# off without build prompts on these tasks").
NO_CHAT_TASKS = ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]

log = logging.getLogger("longbench")


def select_samples(task, n, seed):
    """A fixed random subset of n prompts per task: the same seed gives the same prompts."""
    data = load_longbench(task)
    indices = sorted(random.Random(f"{seed}-{task}").sample(range(len(data)), min(n, len(data))))
    return [(i, data[i]) for i in indices]


def build_inputs(tokenizer, task, sample, max_length, device):
    """LongBench pred.py: fill the template, middle-truncate, then (maybe) the chat template."""
    prompt = DATASET2PROMPT[task].format(**sample)
    ids = tokenizer(prompt, truncation=False)["input_ids"]
    truncated = len(ids) > max_length
    if truncated:
        # Keep both ends: the instructions and the question are at the start and end.
        half = max_length // 2
        prompt = (tokenizer.decode(ids[:half], skip_special_tokens=True)
                  + tokenizer.decode(ids[-half:], skip_special_tokens=True))
    if task not in NO_CHAT_TASKS:
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                               add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
    return inputs, len(ids), truncated


def generate(model, tokenizer, task, inputs, cache):
    context_length = inputs["input_ids"].shape[-1]
    kwargs = dict(max_new_tokens=DATASET2MAXLEN[task], num_beams=1, do_sample=False,
                  temperature=None, top_p=None, top_k=None,
                  pad_token_id=tokenizer.eos_token_id, past_key_values=cache)
    if task == "samsum":
        # Same special case as LongBench: stop at a newline so the model doesn't
        # keep inventing more "Dialogue:" examples.
        kwargs["min_length"] = context_length + 1
        kwargs["eos_token_id"] = [tokenizer.eos_token_id,
                                  tokenizer.encode("\n", add_special_tokens=False)[-1]]
    with torch.no_grad():
        output = model.generate(**inputs, **kwargs)[0]
    return tokenizer.decode(output[context_length:], skip_special_tokens=True)


def read_jsonl(path):
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def run(args, out_dir):
    model, tokenizer = load_model(args.device)
    log.info("model on %s; configs %s; tasks %s; n=%d; max_length=%d; prefill attention %s",
             model.device, args.configs, args.tasks, args.n, args.max_length,
             "exact (KIVI paper)" if args.exact_prefill else "quantized (mentor)")

    for task in args.tasks:
        samples = select_samples(task, args.n, args.seed)
        for i, sample in samples:
            inputs = None
            for config in args.configs:
                path = out_dir / config / f"{task}.jsonl"
                if any(r["index"] == i for r in read_jsonl(path)):
                    continue  # already done (resuming)
                if inputs is None:
                    inputs, prompt_tokens, truncated = build_inputs(
                        tokenizer, task, sample, args.max_length, model.device)
                cache = make_cache(model, config, quantize_prefill_attention=not args.exact_prefill)
                start = time.time()
                pred = generate(model, tokenizer, task, inputs, cache)
                seconds = time.time() - start
                score = score_sample(task, pred, sample["answers"], sample["all_classes"])
                record = {"index": i, "pred": pred, "answers": sample["answers"],
                          "all_classes": sample["all_classes"], "length": sample["length"],
                          "prompt_tokens": prompt_tokens, "input_tokens": inputs["input_ids"].shape[-1],
                          "truncated": truncated, "score": score, "seconds": round(seconds, 2)}
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "a") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                log.info("%-12s #%-4d %-8s score %.3f  %5.1fs  %d tokens%s", task, i, config, score,
                         seconds, inputs["input_ids"].shape[-1], " (truncated)" if truncated else "")
                if args.show:
                    log.info("    pred:    %r", pred.strip()[:300])
                    log.info("    answers: %r", [a[:300] for a in sample["answers"]])


def score(out_dir, configs, tasks):
    """LongBench scores (0-100) per config and task, plus the average over tasks."""
    table = {}
    for config in configs:
        row = {}
        for task in tasks:
            records = read_jsonl(out_dir / config / f"{task}.jsonl")
            if records:
                row[task] = score_dataset(task, [r["pred"] for r in records],
                                          [r["answers"] for r in records], records[0]["all_classes"])
                row[f"{task}_n"] = len(records)
        scored = [row[t] for t in tasks if t in row]
        if scored:
            row["average"] = round(sum(scored) / len(scored), 2)
            table[config] = row
    (out_dir / "scores.json").write_text(json.dumps(table, indent=2))

    header = f"{'config':<10}" + "".join(f"{t[:11]:>12}" for t in tasks) + f"{'average':>10}"
    lines = [header]
    for config, row in table.items():
        lines.append(f"{config:<10}" + "".join(
            f"{row[t]:>12.2f}" if t in row else f"{'-':>12}" for t in tasks) + f"{row['average']:>10.2f}")
    log.info("scores (LongBench, 0-100):\n%s", "\n".join(lines))
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", nargs="+", default=KIVI_TASKS, choices=list(DATASET2PROMPT))
    parser.add_argument("--n", type=int, default=1, help="prompts per task")
    parser.add_argument("--seed", type=int, default=0, help="which random subset of prompts")
    parser.add_argument("--configs", nargs="+", default=["fp16", "kivi4", "kivi2", "token2"],
                        choices=list(CONFIGS))
    parser.add_argument("--max-length", type=int, default=4096, help="prompt token limit (KIVI: 4096)")
    parser.add_argument("--exact-prefill", action="store_true",
                        help="KIVI paper setting: prefill attention uses exact K/V")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--device", default=None, help="cuda / mps / cpu (default: best available)")
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--show", action="store_true", help="log each prediction and its answers")
    args = parser.parse_args()

    run_name = args.run_name or f"n{args.n}_{'exactprefill' if args.exact_prefill else 'quantprefill'}"
    out_dir = ROOT / "results" / "longbench" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    # Log to a file as well as the terminal: DSMLP background pods don't keep stdout.
    # Handlers go on our own logger only, so other libraries' INFO chatter
    # (e.g. every Hugging Face HTTP request) isn't printed.
    formatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
    for handler in (logging.StreamHandler(), logging.FileHandler(out_dir / "run.log")):
        handler.setFormatter(formatter)
        log.addHandler(handler)
    log.setLevel(logging.INFO)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    if not args.score_only:
        run(args, out_dir)
    score(out_dir, args.configs, args.tasks)


if __name__ == "__main__":
    main()
