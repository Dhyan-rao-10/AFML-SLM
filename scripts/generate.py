"""Sample from a checkpoint.

    python scripts/generate.py --ckpt out/ablations/modern_s0/ckpt.pt --prompt "Once upon a time"
    python scripts/generate.py --ckpt out/sft/full_masked/ckpt.pt --chat --prompt "Give three tips for sleeping well."
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from slm.chat import prompt_ids  # noqa: E402
from slm.tokenizer import END_ID, EOT_ID, load_tokenizer  # noqa: E402
from slm.utils import load_checkpoint, setup_device  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompt", default="")
    ap.add_argument("--chat", action="store_true", help="wrap the prompt in the chat template")
    ap.add_argument("--num-samples", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device, ctx, _ = setup_device("cuda")
    model, ckpt = load_checkpoint(args.ckpt, device)
    model.eval()
    tok = load_tokenizer(ckpt["tokenizer"])

    if args.chat:
        ids, stop = prompt_ids(tok, args.prompt), (END_ID,)
    else:
        ids, stop = [EOT_ID] + tok.encode(args.prompt).ids, (EOT_ID,)
    for _ in range(args.num_samples):
        x = torch.tensor([ids], device=device)
        with ctx:
            out = model.generate(x, args.max_new_tokens, args.temperature, args.top_k, stop_ids=stop)
        new = [t for t in out[0, len(ids) :].tolist() if t not in stop]
        print(args.prompt + ("\n" if args.chat else "") + tok.decode(new))
        print("-" * 60)


if __name__ == "__main__":
    main()
