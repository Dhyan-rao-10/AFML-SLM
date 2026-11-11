"""Run the architecture ablations or the scaling sweep, one training run at a time.

    python scripts/sweep.py ablations --seeds 0 1
    python scripts/sweep.py scaling --worker dhyan
    python scripts/sweep.py ablations --dry-run

Finished runs (those with a summary.json) are skipped, and so are runs another
worker is training right now (see scripts/claims.py), so several Colab sessions
can share one results folder. An interrupted run resumes from its checkpoint.
Summarize with scripts/summarize.py out/<sweep>.
"""

import argparse
import os
import random
import subprocess
import sys

from claims import status, try_claim

MODERN = ["model.pos_emb=rope", "model.norm=rmsnorm", "model.mlp=swiglu", "model.n_kv_head=2"]

SWEEPS = {
    # GPT-2-style baseline, then one LLaMA-style component swapped in at a time,
    # then all four together. Same data, token budget and eval batches for all.
    "ablations": {
        "baseline": ("configs/ablation_base.json", []),
        "rope": ("configs/ablation_base.json", ["model.pos_emb=rope"]),
        "rmsnorm": ("configs/ablation_base.json", ["model.norm=rmsnorm"]),
        "swiglu": ("configs/ablation_base.json", ["model.mlp=swiglu"]),
        "gqa": ("configs/ablation_base.json", ["model.n_kv_head=2"]),
        "modern": ("configs/ablation_base.json", MODERN),
    },
    # Three sizes trained on 20 tokens per parameter each, largest (slowest) first.
    "scaling": {
        "l": ("configs/scaling_l.json", []),
        "m": ("configs/scaling_m.json", []),
        "s": ("configs/scaling_s.json", []),
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep", choices=SWEEPS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--only", nargs="*", help="run only these variants")
    ap.add_argument("--worker", default="local", help="your name, when sharing a results folder")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("extra", nargs="*", help="key=value overrides applied to every run")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    jobs = [
        (name, seed, config, overrides)
        for name, (config, overrides) in SWEEPS[args.sweep].items()
        if not args.only or name in args.only
        for seed in args.seeds
    ]
    if args.sweep == "ablations":
        # Each worker takes the runs in its own order, so sessions started at the
        # same time rarely reach for the same run.
        random.Random(args.worker).shuffle(jobs)

    busy = []
    for name, seed, config, overrides in jobs:
        out_dir = f"out/{args.sweep}/{name}_s{seed}"
        st = status(os.path.join(root, out_dir), args.worker)
        if st != "free":
            print(f"skip {out_dir}: {st}")
            if st.startswith("running"):
                busy.append(out_dir)
            continue
        cmd = [sys.executable, "scripts/train.py", "--config", config, f"out_dir={out_dir}", f"seed={seed}", *overrides, *args.extra]
        print(" ".join(cmd), flush=True)
        if not args.dry_run:
            if not try_claim(os.path.join(root, out_dir), args.worker):
                print(f"skip {out_dir}: claimed by another worker just now")
                continue
            subprocess.run(cmd, cwd=root, check=True)
    if busy:
        print(f"still running in other sessions: {', '.join(busy)}")


if __name__ == "__main__":
    main()
