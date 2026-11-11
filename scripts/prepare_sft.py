"""Download Alpaca-cleaned and Dolly 15k, normalize them to prompt/response pairs,
and hold out Dolly prompts for the instruction-following evaluation.

    python scripts/prepare_sft.py

Writes data/sft/{train.jsonl, val.jsonl, eval_prompts.jsonl, meta.json}.
"""

import argparse
import json
import os
import random


def join(instruction: str, extra: str) -> str:
    instruction, extra = instruction.strip(), (extra or "").strip()
    return f"{instruction}\n\n{extra}" if extra else instruction


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/sft")
    ap.add_argument("--n-eval", type=int, default=200, help="held-out Dolly prompts for evaluation")
    ap.add_argument("--max-eval-prompt-chars", type=int, default=1000, help="keep eval prompts short enough to leave room to answer")
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from datasets import load_dataset

    alpaca = [
        {"prompt": join(r["instruction"], r["input"]), "response": r["output"].strip(), "source": "alpaca"}
        for r in load_dataset("yahma/alpaca-cleaned", split="train")
    ]
    dolly = [
        {"prompt": join(r["instruction"], r["context"]), "response": r["response"].strip(), "source": "dolly", "category": r["category"]}
        for r in load_dataset("databricks/databricks-dolly-15k", split="train")
    ]
    alpaca = [r for r in alpaca if r["prompt"] and r["response"]]
    dolly = [r for r in dolly if r["prompt"] and r["response"]]

    rng = random.Random(args.seed)
    rng.shuffle(dolly)
    eval_rows, dolly_rest = [], []
    for r in dolly:
        if len(eval_rows) < args.n_eval and len(r["prompt"]) <= args.max_eval_prompt_chars:
            eval_rows.append({"prompt": r["prompt"], "reference": r["response"], "category": r["category"]})
        else:
            dolly_rest.append(r)

    pool = alpaca + dolly_rest
    rng.shuffle(pool)
    n_val = int(len(pool) * args.val_frac)
    val_rows, train_rows = pool[:n_val], pool[n_val:]

    os.makedirs(args.out_dir, exist_ok=True)
    write_jsonl(os.path.join(args.out_dir, "train.jsonl"), train_rows)
    write_jsonl(os.path.join(args.out_dir, "val.jsonl"), val_rows)
    write_jsonl(os.path.join(args.out_dir, "eval_prompts.jsonl"), eval_rows)
    meta = {"train": len(train_rows), "val": len(val_rows), "eval_prompts": len(eval_rows), "alpaca": len(alpaca), "dolly": len(dolly)}
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
