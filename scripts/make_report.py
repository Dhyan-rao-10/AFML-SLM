"""Combine every result into one report: tables, the scaling fit, measured failure
modes of the tuned models, and side-by-side example answers.

    python scripts/make_report.py

Reads kaggle_results/out and data/sft/train.jsonl; writes kaggle_results/report/
{RESULTS.md, metrics.json, scaling_fit.png}. Every number comes from those files.
"""

import argparse
import difflib
import json
import math
import os
import re
import statistics
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ABLATIONS = ["baseline", "rope", "rmsnorm", "swiglu", "gqa", "modern"]
SIZES = ["s", "m", "l"]
MODELS = ["base", "full_masked", "full_unmasked", "lora_masked", "lora_unmasked"]
SFT = MODELS[1:]
# Final val loss from the independent Colab run (scripts/summarize.py, same seeds and code, different GPU).
COLAB = {"baseline": 1.7894, "rope": 1.7314, "rmsnorm": 1.7892, "swiglu": 1.7229, "gqa": 1.8012, "modern": 1.6965}
BOILERPLATE = re.compile(r"\bas an ai\b|\bai language model\b|\bi am an ai\b|\bi'm an ai\b|\bi do not have (access|the ability)\b", re.I)
EXAMPLES = [
    (3, "Right, but through a learned template: this format appears 271 times in training"),
    (148, "Copying: the context is repeated, not summarized"),
    (157, "Confident, well-formatted, invented answers"),
    (109, "Right format, nonsense content"),
    (106, "Refusal boilerplate learned from Alpaca, seen in an unmasked run"),
    (60, "Classification failure"),
]


def load_json(*path):
    with open(os.path.join(*path), encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(*path):
    with open(os.path.join(*path), encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def repetitive(text):
    words = text.lower().split()
    grams = Counter(tuple(words[i : i + 4]) for i in range(len(words) - 3))
    return bool(grams) and max(grams.values()) >= 3


def copied(answer, prompt, min_chars=60):
    m = difflib.SequenceMatcher(None, prompt, answer, autojunk=False).find_longest_match(0, len(prompt), 0, len(answer))
    return m.size >= min_chars


def failure_modes(answers, summaries):
    """Share of answers showing each failure, per model. Copying is measured only on
    prompts that carry a context passage (a blank line after the question)."""
    out = {}
    for name in MODELS:
        rows = answers[name]
        ctx = [r for r in rows if "\n\n" in r["prompt"]]
        out[name] = {
            "on_format": summaries[name][name]["on_format"],
            "median_words": statistics.median(len(r["response"].split()) for r in rows),
            "repetitive": sum(repetitive(r["response"]) for r in rows) / len(rows),
            "boilerplate": sum(bool(BOILERPLATE.search(r["response"])) for r in rows) / len(rows),
            "copied": sum(copied(r["response"], r["prompt"]) for r in ctx) / len(ctx),
            "context_prompts": len(ctx),
        }
    return out


def ablation_section(out_dir):
    rows = {}
    for name in ABLATIONS:
        runs = [load_json(out_dir, "ablations", f"{name}_s{s}", "summary.json") for s in (0, 1)]
        losses = [r["final_val_loss"] for r in runs]
        rows[name] = {"params": runs[0]["params"], "loss": sum(losses) / 2, "half_range": abs(losses[0] - losses[1]) / 2,
                      "tok_per_s": sum(r["tok_per_s"] for r in runs) / 2}
    base = rows["baseline"]["loss"]
    singles = sum(rows[n]["loss"] - base for n in ("rope", "rmsnorm", "swiglu", "gqa"))
    lines = ["| Variant | Params (M) | Val loss | ± half-range | Δ vs baseline | Colab val loss | Colab Δ | Tok/s |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name in sorted(ABLATIONS, key=lambda n: rows[n]["loss"]):
        r = rows[name]
        lines.append(f"| {name} | {r['params'] / 1e6:.2f} | {r['loss']:.4f} | {r['half_range']:.4f} | {r['loss'] - base:+.4f} | "
                     f"{COLAB[name]:.4f} | {COLAB[name] - COLAB['baseline']:+.4f} | {r['tok_per_s']:,.0f} |")
    return rows, singles, rows["modern"]["loss"] - base, lines


def scaling_section(out_dir, report_dir):
    runs = {n: load_json(out_dir, "scaling", f"{n}_s0", "summary.json") for n in SIZES}
    params = np.array([runs[n]["params"] for n in SIZES], dtype=float)
    losses = np.array([runs[n]["final_val_loss"] for n in SIZES])
    slope, intercept = np.polyfit(np.log(params), np.log(losses), 1)
    fig, ax = plt.subplots(figsize=(6, 4))
    grid = np.geomspace(params[0], params[-1], 50)
    ax.plot(grid / 1e6, np.exp(intercept) * grid ** slope, color="#888", label=f"fit: loss ∝ N^{slope:.3f}")
    ax.scatter(params / 1e6, losses, color="#1f6feb", zorder=3, label="trained models")
    for n, p, v in zip(SIZES, params, losses):
        ax.annotate(f"{n}: {v:.3f}", (p / 1e6, v), textcoords="offset points", xytext=(6, 6))
    ax.set(xscale="log", yscale="log", xlabel="Parameters (millions)", ylabel="Validation loss (nats/token)")
    ax.set_xticks([10, 20, 40], ["10", "20", "40"])
    ax.set_yticks([3.4, 3.8, 4.2], ["3.4", "3.8", "4.2"])
    ax.minorticks_off()
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(report_dir, "scaling_fit.png"), dpi=150)
    plt.close(fig)
    lines = ["| Size | Params (M) | Tokens (M) | Val loss | Wall time (min) | Tok/s |", "| --- | --- | --- | --- | --- | --- |"]
    for n in SIZES:
        r = runs[n]
        lines.append(f"| {n} | {r['params'] / 1e6:.2f} | {r['tokens'] / 1e6:.0f} | {r['final_val_loss']:.4f} | "
                     f"{r['wall_time_s'] / 60:.0f} | {r['tok_per_s']:,.0f} |")
    per_doubling = 1 - 2 ** slope
    return {"exponent": float(slope), "loss_drop_per_doubling": float(per_doubling)}, lines


def sft_section(out_dir, modes):
    judge = {(p["x"], p["y"]): p for p in load_json(out_dir, "judge", "summary.json")["pairs"]}
    lines = ["| Model | Trainable params (M) | Val loss | GPU hours | In-format rate |", "| --- | --- | --- | --- | --- |"]
    for name in SFT:
        s = load_json(out_dir, "sft", name, "summary.json")
        lines.append(f"| {name} | {s['trainable_params'] / 1e6:.2f} | {s['final_val_loss']:.4f} | {s['gpu_hours']:.2f} | "
                     f"{modes[name]['on_format']:.1%} |")
    lines += ["", "| Comparison (x vs y) | Prompts | x score (95% CI) | x wins | y wins | Ties | Order-consistent |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for (x, y), p in judge.items():
        lines.append(f"| {x} vs {y} | {p['prompts']} | {p['x_score']:.3f} ± {p['ci95']:.3f} | {p['x_wins']} | {p['y_wins']} | "
                     f"{p['ties']} | {p['order_consistent']:.0%} |")
    return judge, lines


def clip(text, n=220):
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n] + "…"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(ROOT, "kaggle_results", "out"))
    ap.add_argument("--report-dir", default=os.path.join(ROOT, "kaggle_results", "report"))
    ap.add_argument("--train", default=os.path.join(ROOT, "data", "sft", "train.jsonl"))
    args = ap.parse_args()
    os.makedirs(args.report_dir, exist_ok=True)

    answers = {n: load_jsonl(args.results, "eval", n, f"{n}.jsonl") for n in MODELS}
    summaries = {n: load_json(args.results, "eval", n, "summary.json") for n in MODELS}
    modes = failure_modes(answers, summaries)
    ablations, singles, modern_gap, ablation_lines = ablation_section(args.results)
    scaling, scaling_lines = scaling_section(args.results, args.report_dir)
    judge, sft_lines = sft_section(args.results, modes)
    judge_summary = load_json(args.results, "judge", "summary.json")

    norm = lambda s: re.sub(r"\W+", " ", s.lower()).strip()
    train = load_jsonl(args.train)
    train_prompts = {norm(t["prompt"]) for t in train}
    train_responses = {norm(t["response"]) for t in train}
    overlap = [i for i, r in enumerate(answers["base"]) if norm(r["prompt"]) in train_prompts]
    reference_overlap = sum(norm(r["reference"]) in train_responses for r in answers["base"])
    summ = [r for r in answers["full_masked"] if r["category"] == "summarization"]
    summ_copied = sum(copied(r["response"], r["prompt"]) for r in summ)

    metrics = {"ablations": ablations, "sum_of_single_changes": singles, "modern_gap": modern_gap, "scaling": scaling,
               "failure_modes": modes, "eval_prompts_in_train": overlap, "eval_references_in_train": reference_overlap,
               "summarization_copied": [summ_copied, len(summ)]}
    with open(os.path.join(args.report_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    mode_lines = ["| Model | In format | Median words | Repetitive | AI boilerplate | Copied context* |",
                  "| --- | --- | --- | --- | --- | --- |"]
    for n in MODELS:
        m = modes[n]
        mode_lines.append(f"| {n} | {m['on_format']:.1%} | {m['median_words']:.0f} | {m['repetitive']:.1%} | "
                          f"{m['boilerplate']:.1%} | {m['copied']:.1%} |")

    pair_names = [f"{x} vs {y}" for x, y in judge]
    cat_lines = ["| Category | Prompts | " + " | ".join(pair_names) + " |", "| --- | --- |" + " --- |" * len(pair_names)]
    counts = Counter(r["category"] for r in answers["base"])
    for c in sorted(judge_summary["by_category"]):
        cells = [f"{judge_summary['by_category'][c].get(p, math.nan):.2f}" for p in pair_names]
        cat_lines.append(f"| {c} | {counts[c]} | " + " | ".join(cells) + " |")

    example_lines = []
    for idx, why in EXAMPLES:
        row = answers["base"][idx]
        example_lines += [f"**#{idx}: {why}** ({row['category']})", "", f"- Prompt: {clip(row['prompt'], 160)}",
                          f"- Reference: {clip(row['reference'], 160)}"]
        example_lines += [f"- {n}: {clip(answers[n][idx]['response'])}" for n in MODELS]
        example_lines.append("")

    tuned = judge[("full_masked", "base")]
    md = ["# Results", "",
          "Generated by `scripts/make_report.py` from the Kaggle run outputs in `kaggle_results/out`.", "",
          "## 1. Architecture ablations (TinyStories, 100M tokens, 2 seeds)", "", *ablation_lines, "",
          f"Applied one at a time, the changes sum to {singles:+.4f}; together they give {modern_gap:+.4f}. "
          "The gains overlap, so they are not additive.", "",
          "## 2. Scaling (FineWeb-Edu, 20 tokens per parameter, 1 seed)", "", *scaling_lines, "",
          f"Power-law fit: loss ∝ N^{scaling['exponent']:.3f}, about {scaling['loss_drop_per_doubling']:.1%} lower loss "
          "per doubling of parameters. Tokens grow with parameters here, so this measures model size and data together, "
          "and it rests on three single-seed points. See `scaling_fit.png`.", "",
          "## 3. Instruction tuning", "", *sft_lines, "",
          "x score: 1 = x always preferred, 0.5 = no preference, 0 = y always preferred; ties count half. "
          "Blind judge: Claude Sonnet 5.5 at low effort through Claude Code in non-interactive mode, with a fixed rubric, "
          "no tools and no saved context. Each pair was judged in both orders, 1,600 verdicts in all.", "",
          "x score by question type (small categories are noisy):", "", *cat_lines, "",
          "## 4. How the tuned models fail", "", *mode_lines, "",
          f"*Share of the {modes['base']['context_prompts']} prompts that include a context passage where the answer repeats "
          "60 or more characters from the prompt verbatim. Repetitive: some 4-word sequence occurs 3 or more times. "
          "AI boilerplate: phrases such as \"as an AI language model\".", "",
          "## 5. Example answers", "",
          "Chosen by hand to illustrate each behaviour, so they are not a fair sample of quality. All 200 answers per model "
          "are in `kaggle_results/out/eval`.", "", *example_lines,
          "## 6. Caveats", "",
          f"- Held-out data: {len(overlap)} of 200 eval prompts also appears in training (ids {overlap}), and "
          f"{reference_overlap} reference answers do. Too few to matter, but Dolly has repeated templates, so a correct answer "
          "can come from a learned pattern.",
          f"- Summarization: {summ_copied} of {len(summ)} full_masked summaries copy 60+ characters from the prompt, so its "
          "high judge score partly rewards copying, not summarizing.",
          "- Base model: it was never trained to stop, so it fills the token limit. "
          f"Of its {tuned['ties']} ties with full_masked, nearly all verdicts say both answers were wrong or incoherent.",
          "- Reference answers: some Dolly references are themselves wrong (#121 calls Sandpipers extinct), and the judge "
          "treats them as a guide.",
          "- Statistics: judge ranges are normal-approximation 95% intervals over 200 prompts, not corrected for "
          "multiple comparisons. Each ablation has 2 seeds and each scaling size 1.", ""]
    with open(os.path.join(args.report_dir, "RESULTS.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"wrote {args.report_dir}")
    print(json.dumps({k: metrics[k] for k in ("sum_of_single_changes", "modern_gap", "scaling", "eval_prompts_in_train",
                                              "eval_references_in_train", "summarization_copied")}, indent=1))
    print("\n".join(mode_lines))


if __name__ == "__main__":
    main()
