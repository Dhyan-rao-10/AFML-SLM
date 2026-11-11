"""Blind pairwise LLM judge over the held-out Dolly answers from scripts/eval_sft.py.

    python scripts/judge.py --limit 5        # pilot: 5 prompts per pair
    python scripts/judge.py                  # everything (resumes where it stopped)
    python scripts/judge.py --summary-only   # rebuild the summary from saved verdicts

Each verdict is one call to the Claude Code CLI (`claude -p`), so it runs on a Claude
subscription rather than a paid API key. Every call is isolated: no tools, safe mode
(no CLAUDE.md, memory, skills, hooks or MCP servers), a fixed rubric as the whole system
prompt, an empty working directory, and no saved session. The judge sees the instruction,
the human reference answer and two unlabeled answers; each pair is judged in both orders
(A/B and B/A) because judges favour one position.

Writes <out-dir>/judgments.jsonl (one line per verdict), summary.json and summary.md.
"""

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# (x, y): x is the model we expect to be better, so "x score" > 0.5 means it is.
PAIRS = [
    ("full_masked", "base"),           # does instruction tuning help at all?
    ("full_masked", "lora_masked"),    # full fine-tuning vs LoRA
    ("full_masked", "full_unmasked"),  # loss masking, full fine-tuning
    ("lora_masked", "lora_unmasked"),  # loss masking, LoRA
]

RUBRIC = """You judge answers written by small, weak language models to a user's instruction.

You get the instruction, a reference answer written by a person, and two candidate answers, A and B.
Decide which candidate responds better to the instruction. In order of importance:
1. Correct: no false facts or wrong reasoning. The reference shows one good answer; a candidate may
   differ from it in wording, length or approach and still be correct.
2. Relevant and on task: it does what was asked, in the form asked.
3. Coherent: readable, not repetitive, not rambling into unrelated text.

Judge content, not length or style: a short correct answer beats a long wrong one. An answer may be
cut off at a length limit; judge what is there. The order in which A and B are shown means nothing.
If neither is better (both equally good, or both wrong or incoherent), the winner is "tie".

Give a one-sentence reason, then the winner."""

SCHEMA = {
    "type": "object",
    "properties": {"reason": {"type": "string"}, "winner": {"type": "string", "enum": ["A", "B", "tie"]}},
    "required": ["reason", "winner"],
    "additionalProperties": False,
}


def find_claude():
    exe = os.path.join(os.environ.get("APPDATA", ""), "npm", "node_modules", "@anthropic-ai", "claude-code", "bin", "claude.exe")
    return exe if os.path.exists(exe) else shutil.which("claude")


def load_answers(eval_dir, name):
    with open(os.path.join(eval_dir, name, f"{name}.jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def message(row, a, b):
    return (f"<instruction>\n{row['prompt']}\n</instruction>\n\n"
            f"<reference_answer>\n{row['reference']}\n</reference_answer>\n\n"
            f"<answer_A>\n{a}\n</answer_A>\n\n<answer_B>\n{b}\n</answer_B>")


def judge(claude, rubric_file, workdir, args, text):
    cmd = [claude, "-p", "--safe-mode", "--tools", "", "--strict-mcp-config", "--no-session-persistence",
           "--model", args.model, "--effort", args.effort, "--output-format", "json",
           "--system-prompt-file", rubric_file, "--json-schema", json.dumps(SCHEMA)]
    r = subprocess.run(cmd, input=text, capture_output=True, text=True, encoding="utf-8", cwd=workdir, timeout=600)
    try:
        out = json.loads(r.stdout)
    except ValueError:
        return None, f"exit {r.returncode}: {(r.stdout + r.stderr).strip()[:300]}"
    verdict = out.get("structured_output")
    if out.get("is_error") or not isinstance(verdict, dict) or verdict.get("winner") not in ("A", "B", "tie"):
        return None, str(out.get("result") or out.get("subtype"))[:300]
    return {**verdict, "cost_usd": out.get("total_cost_usd", 0.0), "usage": out.get("usage", {})}, None


def run(args):
    claude = find_claude()
    if not claude:
        sys.exit("claude CLI not found; install it with: npm install -g @anthropic-ai/claude-code")
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "judgments.jsonl")
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            done = {(j["x"], j["y"], j["idx"], j["order"]) for j in map(json.loads, f)}

    answers = {n: load_answers(args.eval_dir, n) for n in {m for p in PAIRS for m in p}}
    todo = []
    for x, y in PAIRS:
        n = len(answers[x]) if args.limit is None else min(args.limit, len(answers[x]))
        for idx in range(n):
            assert answers[x][idx]["prompt"] == answers[y][idx]["prompt"], f"{x}/{y} prompt {idx} differs"
            for order in (0, 1):  # order 0: A = x, B = y; order 1: swapped
                if (x, y, idx, order) not in done:
                    todo.append((x, y, idx, order))
    random.Random(0).shuffle(todo)  # interleave pairs, so a stop part-way leaves every pair partly done
    print(f"{len(done)} verdicts already saved; {len(todo)} to go with {args.model} (effort {args.effort})", flush=True)
    if not todo:
        return

    workdir = tempfile.mkdtemp(prefix="judge-work-")  # stays empty: the judge sees no files
    rubric_file = os.path.join(tempfile.mkdtemp(prefix="judge-rubric-"), "rubric.txt")
    with open(rubric_file, "w", encoding="utf-8") as f:
        f.write(RUBRIC)

    lock, stop = threading.Lock(), threading.Event()
    stats = {"ok": 0, "errors": 0, "streak": 0, "cost": 0.0}
    t0 = time.time()

    def one(item):
        if stop.is_set():
            return
        x, y, idx, order = item
        row = answers[x][idx]
        a, b = (answers[x][idx]["response"], answers[y][idx]["response"])[:: 1 if order == 0 else -1]
        verdict, err = judge(claude, rubric_file, workdir, args, message(row, a, b))
        with lock:
            if err:
                stats["errors"] += 1
                stats["streak"] += 1
                print(f"error on {x} vs {y} #{idx}: {err}", flush=True)
                # A usage limit or several failures in a row: stop; a later run resumes.
                if "limit" in err.lower() or stats["streak"] >= 3:
                    stop.set()
                return
            stats["streak"] = 0
            winner = {"A": (x, y)[order], "B": (y, x)[order], "tie": "tie"}[verdict["winner"]]
            rec = {"x": x, "y": y, "idx": idx, "order": order, "category": row.get("category"),
                   "verdict": verdict["winner"], "winner": winner, "reason": verdict["reason"],
                   "cost_usd": verdict["cost_usd"],
                   "input_tokens": sum(verdict["usage"].get(k, 0) for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")),
                   "output_tokens": verdict["usage"].get("output_tokens", 0)}
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            stats["ok"] += 1
            stats["cost"] += verdict["cost_usd"] or 0.0
            if stats["ok"] % 20 == 0 or stats["ok"] == len(todo):
                rate = stats["ok"] / (time.time() - t0) * 60
                print(f"{stats['ok']}/{len(todo)} verdicts ({rate:.1f}/min), "
                      f"API-equivalent cost so far ${stats['cost']:.2f}", flush=True)

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(one, todo))
    print(f"saved {stats['ok']} verdicts, {stats['errors']} errors, API-equivalent cost ${stats['cost']:.2f}"
          + ("; stopped early, run again later to continue" if stop.is_set() else ""), flush=True)


def summarize(out_dir):
    with open(os.path.join(out_dir, "judgments.jsonl"), encoding="utf-8") as f:
        js = [json.loads(line) for line in f]
    by_prompt = defaultdict(dict)
    for j in js:
        by_prompt[(j["x"], j["y"], j["idx"])][j["order"]] = j

    rows, by_cat = [], defaultdict(lambda: defaultdict(list))
    for x, y in PAIRS:
        prompts = [v for (px, py, _), v in by_prompt.items() if (px, py) == (x, y) and len(v) == 2]
        if not prompts:
            continue
        # Per prompt: x scores 1 per order it wins and 0.5 per tie, averaged over the two orders.
        scores, outcomes = [], Counter()
        for v in prompts:
            s = sum(1.0 if j["winner"] == x else 0.5 if j["winner"] == "tie" else 0.0 for j in v.values()) / 2
            scores.append(s)
            by_cat[v[0]["category"]][f"{x} vs {y}"].append(s)
            w = {j["winner"] for j in v.values()}
            outcomes["x" if s > 0.5 else "y" if s < 0.5 else "tie"] += 1
            outcomes["inconsistent"] += {x, y} <= w  # the two orders picked different models
        n = len(scores)
        mean = sum(scores) / n
        sd = math.sqrt(sum((s - mean) ** 2 for s in scores) / max(n - 1, 1))
        rows.append({"x": x, "y": y, "prompts": n, "x_score": mean, "ci95": 1.96 * sd / math.sqrt(n),
                     "x_wins": outcomes["x"], "y_wins": outcomes["y"], "ties": outcomes["tie"],
                     "order_consistent": 1 - outcomes["inconsistent"] / n})

    summary = {"pairs": rows, "judgments": len(js), "api_equivalent_cost_usd": sum(j.get("cost_usd") or 0 for j in js),
               "by_category": {c: {p: sum(v) / len(v) for p, v in d.items()} for c, d in by_cat.items()}}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    lines = ["| x | y | Prompts | x score (95% CI) | x wins | y wins | Ties | Order-consistent |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        lines.append(f"| {r['x']} | {r['y']} | {r['prompts']} | {r['x_score']:.3f} ± {r['ci95']:.3f} | {r['x_wins']} | "
                     f"{r['y_wins']} | {r['ties']} | {r['order_consistent']:.0%} |")
    lines += ["", "x score: 1 = x always preferred, 0.5 = no preference, 0 = y always preferred (ties count half).",
              "Order-consistent: share of prompts where the two orderings did not pick different models."]
    pair_names = [f"{r['x']} vs {r['y']}" for r in rows]
    lines += ["", "x score by Dolly category:", "", "| Category | " + " | ".join(pair_names) + " |",
              "| --- |" + " --- |" * len(pair_names)]
    for c in sorted(summary["by_category"]):
        cells = [f"{summary['by_category'][c][p]:.2f}" if p in summary["by_category"][c] else "–" for p in pair_names]
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default=os.path.join(ROOT, "kaggle_results", "out", "eval"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "kaggle_results", "out", "judge"))
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--limit", type=int, default=None, help="judge only the first N prompts of each pair")
    ap.add_argument("--workers", type=int, default=3, help="judge calls running at once")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # the summary table prints "±"
    if not args.summary_only:
        run(args)
    summarize(args.out_dir)


if __name__ == "__main__":
    main()
