"""Answer the held-out Dolly prompts with one or more checkpoints.

    python scripts/eval_sft.py base=out/final/ckpt.pt full_masked=out/sft/full_masked/ckpt.pt

Writes out/eval/<name>.jsonl (prompt, reference, response, stopped) and prints
the on-format rate: the share of answers that close with <|end|> before the
token limit. These files are the input to the pairwise LLM-judge comparison.
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from slm.chat import prompt_ids  # noqa: E402
from slm.tokenizer import END_ID, load_tokenizer  # noqa: E402
from slm.utils import load_checkpoint, setup_device  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", help="name=path/to/ckpt.pt")
    ap.add_argument("--prompts", default="data/sft/eval_prompts.jsonl")
    ap.add_argument("--out-dir", default="out/eval")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(args.prompts, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    device, ctx, _ = setup_device("cuda")
    os.makedirs(args.out_dir, exist_ok=True)

    rates = {}
    for spec in args.models:
        name, path = spec.split("=", 1)
        model, ckpt = load_checkpoint(path, device)
        model.eval()
        tok = load_tokenizer(ckpt["tokenizer"])
        torch.manual_seed(args.seed)
        results, stopped = [], 0
        for r in rows:
            ids = prompt_ids(tok, r["prompt"])
            room = model.cfg.block_size - len(ids)
            if room <= 0:
                continue
            x = torch.tensor([ids], device=device)
            with ctx:
                out = model.generate(x, min(args.max_new_tokens, room), args.temperature, args.top_k, stop_ids=(END_ID,))
            new = out[0, len(ids) :].tolist()
            ended = bool(new) and new[-1] == END_ID
            stopped += ended
            results.append({**r, "response": tok.decode([t for t in new if t != END_ID]), "stopped": ended})
        with open(os.path.join(args.out_dir, f"{name}.jsonl"), "w", encoding="utf-8") as f:
            for res in results:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
        rates[name] = {"prompts": len(results), "on_format": stopped / len(results)}
        print(f"{name}: {len(results)} prompts, on-format (closed with <|end|>) {stopped / len(results):.1%}")
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(rates, f, indent=2)


if __name__ == "__main__":
    main()
