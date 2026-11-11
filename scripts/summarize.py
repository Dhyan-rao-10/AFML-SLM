"""Tabulate and plot a sweep's results.

    python scripts/summarize.py out/ablations

Groups runs by variant (the folder name before _s<seed>), averages over seeds,
and writes <sweep_dir>/summary.md and val_curves.png. If the variants differ in
size by more than 2x, also plots final loss against parameters (scaling.png).
"""

import argparse
import glob
import json
import os
import statistics
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_runs(sweep_dir):
    runs = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(sweep_dir, "*", "summary.json"))):
        run_dir = os.path.dirname(path)
        variant = os.path.basename(run_dir).rsplit("_s", 1)[0]
        with open(path) as f:
            summary = json.load(f)
        with open(os.path.join(run_dir, "log.jsonl")) as f:
            # A resumed run re-evaluates at its restart iteration; keep the last row per iteration.
            by_iter = {r["iter"]: r for r in map(json.loads, f) if "val_loss" in r}
        curve = [by_iter[i] for i in sorted(by_iter)]
        runs[variant].append({**summary, "curve": curve})
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_dir")
    args = ap.parse_args()
    runs = load_runs(args.sweep_dir)
    if not runs:
        raise SystemExit(f"no finished runs under {args.sweep_dir}")

    rows = []
    for variant, rs in runs.items():
        losses = [r["final_val_loss"] for r in rs]
        rows.append({
            "variant": variant,
            "seeds": len(rs),
            "params_m": rs[0]["params"] / 1e6,
            "tokens_m": rs[0]["tokens"] / 1e6,
            "val_loss": statistics.mean(losses),
            "spread": (max(losses) - min(losses)) / 2,
            "tok_s": statistics.mean(r["tok_per_s"] for r in rs),
            "minutes": statistics.mean(r["wall_time_s"] for r in rs) / 60,
        })
    rows.sort(key=lambda r: r["val_loss"])
    base = next((r["val_loss"] for r in rows if r["variant"] == "baseline"), None)

    lines = [
        "| Variant | Seeds | Params (M) | Tokens (M) | Val loss | ± half-range | Δ vs baseline | Tok/s | Minutes |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        delta = f"{r['val_loss'] - base:+.4f}" if base is not None else "–"
        lines.append(
            f"| {r['variant']} | {r['seeds']} | {r['params_m']:.2f} | {r['tokens_m']:.0f} | {r['val_loss']:.4f} | "
            f"{r['spread']:.4f} | {delta} | {r['tok_s']:,.0f} | {r['minutes']:.1f} |"
        )
    table = "\n".join(lines)
    with open(os.path.join(args.sweep_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(table + "\n")
    print(table)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for variant, rs in sorted(runs.items()):
        curve = rs[0]["curve"]
        xs = [p["tokens"] / 1e6 for p in curve]
        ys = [statistics.mean(r["curve"][i]["val_loss"] for r in rs if i < len(r["curve"])) for i in range(len(curve))]
        ax.plot(xs[1:], ys[1:], label=variant)  # skip the step-0 loss, which flattens the plot
    ax.set_xlabel("Training tokens (millions)")
    ax.set_ylabel("Validation loss (nats/token)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.sweep_dir, "val_curves.png"), dpi=150)

    sizes = [r["params_m"] for r in rows]
    if max(sizes) > 2 * min(sizes):
        fig, ax = plt.subplots(figsize=(6, 4.5))
        pts = sorted((r["params_m"], r["val_loss"], r["variant"]) for r in rows)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o")
        for p, loss, name in pts:
            ax.annotate(name, (p, loss), textcoords="offset points", xytext=(6, 6))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Parameters (millions, log scale)")
        ax.set_ylabel("Final validation loss (log scale)")
        ax.grid(alpha=0.3, which="both")
        fig.tight_layout()
        fig.savefig(os.path.join(args.sweep_dir, "scaling.png"), dpi=150)


if __name__ == "__main__":
    main()
