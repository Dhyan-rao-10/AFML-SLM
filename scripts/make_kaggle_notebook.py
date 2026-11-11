"""Build kaggle/AFML_SLM_kaggle.ipynb: a self-contained notebook for a Kaggle
background run ("Save & Run All"). It writes the project's source files, then runs
scripts/pipeline.py with one data-prep worker and one training worker per GPU.

    python scripts/make_kaggle_notebook.py

Re-run after editing any source file so the notebook carries the current code.
"""

import glob
import json
import os

from make_colab_notebook import ROOT, SOURCES, code, md

SKIP = {"scripts/make_colab_notebook.py", "scripts/make_kaggle_notebook.py"}

INTRO = """# What Matters in a Small Language Model — Kaggle runner

Runs the whole AFML default project in the background: architecture ablations, scaling sweep,
instruction tuning and evaluation. Both of Kaggle's T4 GPUs train at once while the CPU prepares data.

**Settings (right-hand panel → Session options)**
- **Accelerator: GPU T4 x2**
- **Internet: On** (needs a phone-verified Kaggle account)

**Run it:** *Save Version* (top right) → **Save & Run All (Commit)** → *Save*. You can close the tab and turn
your computer off; it runs on Kaggle's servers for up to ~11 hours and saves everything in its output.

**When you come back:** open the notebook → the version's **Logs** and **Output**. The last cell prints either
`ALL DONE` or `NOT FINISHED YET`. If not finished: *Add Input* → *Your Work* → *Notebooks* → this notebook's
latest version → *Add*, then **Save & Run All** again. It restores everything and continues where it stopped."""

CONFIG = """# Kaggle stops a run after 12 hours. Training pauses itself (resumable) before this budget runs out,
# leaving time to save outputs.
TIME_BUDGET_HOURS = 11.25"""

SETUP = """import glob, json, os, shutil, subprocess, sys, time, urllib.request

START = time.time()
WORK, INPUT = '/kaggle/working', '/kaggle/input'
os.chdir(WORK)
for d in ('slm', 'scripts', 'configs', 'tests', 'out', 'data'):
    os.makedirs(d, exist_ok=True)

# Continue from a previous run of this notebook if its output was added as input.
states = glob.glob(f'{INPUT}/**/pipeline_state.json', recursive=True)
if states:
    prev = os.path.dirname(max(states, key=lambda p: json.load(open(p))['time']))
    print(f'restoring the previous run from {prev} (a few minutes)', flush=True)
    for sub in ('out', 'data'):
        if os.path.isdir(f'{prev}/{sub}'):
            shutil.copytree(f'{prev}/{sub}', sub, dirs_exist_ok=True)
    # Nobody is running anything yet, and failed jobs deserve another try.
    for marker in ('CLAIM', 'FAILED', '.failed', '.prep_exited'):
        for p in glob.glob(f'**/{marker}', recursive=True):
            os.remove(p)
    for p in glob.glob('**/.claiming', recursive=True):
        os.rmdir(p)
else:
    print('starting fresh (no previous run attached as input)')

try:
    urllib.request.urlopen('https://huggingface.co', timeout=10)
except Exception:
    raise SystemExit('No internet. In the right-hand panel set Session options → Internet → On, then run again.')

!nvidia-smi --query-gpu=name,memory.total --format=csv
!pip install -q tokenizers datasets pytest"""

TESTS = """!python -m pytest tests -q"""

RUN = """import torch

deadline = START + TIME_BUDGET_HOURS * 3600
n_gpu = torch.cuda.device_count()
assert n_gpu >= 1, 'No GPU. In the right-hand panel set Session options → Accelerator → GPU T4 x2.'
print(f'{n_gpu} GPU(s); stopping new work by {time.ctime(deadline)}', flush=True)
if os.path.exists('data/.prep_exited'):
    os.remove('data/.prep_exited')

procs, start_sizes = {}, {}
def launch(name, gpu):
    log = open(f'out/pipeline_{name}.log', 'a')
    start_sizes[name] = os.path.getsize(f'out/pipeline_{name}.log')
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': gpu}
    procs[name] = subprocess.Popen([sys.executable, 'scripts/pipeline.py', name, '--deadline', str(deadline)],
                                   stdout=log, stderr=subprocess.STDOUT, env=env)

launch('prep', '')
for i in range(n_gpu):
    launch(f'gpu{i}', str(i))

seen = {name: start_sizes[name] for name in procs}  # show only this run's lines
while True:
    for name in procs:
        with open(f'out/pipeline_{name}.log', encoding='utf-8', errors='replace') as f:
            f.seek(seen[name])
            new = f.read()
            seen[name] = f.tell()
        if new:
            print(new, end='', flush=True)
    if all(p.poll() is not None for p in procs.values()):
        break
    if time.time() > deadline + 20 * 60:
        print('hard stop: terminating workers so the outputs can be saved', flush=True)
        for p in procs.values():
            p.terminate()
        break
    time.sleep(30)"""

RESULTS = """sys.path.insert(0, 'scripts')
from pipeline import all_states

for sweep in ('ablations', 'scaling'):
    if glob.glob(f'out/{sweep}/*/summary.json'):
        subprocess.run([sys.executable, 'scripts/summarize.py', f'out/{sweep}'])
for path in sorted(glob.glob('out/sft/*/summary.json') + glob.glob('out/eval/*/summary.json')):
    print(path, open(path).read())

states = {job['out_dir']: st for job, st in all_states('results', 0)}
left = {d: st for d, st in states.items() if st != 'done'}
json.dump({'time': time.time(), 'complete': not left, 'left': left}, open('pipeline_state.json', 'w'), indent=2)
print()
if not left:
    print('ALL DONE: every job finished. Results are in this version\\'s Output (out/).')
else:
    print(f'NOT FINISHED YET: {len(states) - len(left)} of {len(states)} jobs done; left: {left}')
    print('To continue: Add Input → Your Work → Notebooks → this notebook (latest version) → Add, then Save & Run All again.')"""

RESULTS_IMAGES = """from IPython.display import Image, display
for path in ('out/ablations/val_curves.png', 'out/scaling/val_curves.png', 'out/scaling/scaling.png'):
    if os.path.exists(path):
        print(path)
        display(Image(path))"""


def main():
    cells = [md(INTRO), md("## Time budget"), code(CONFIG), md("## Setup"), code(SETUP)]
    cells.append(md("## Project code\nGenerated from the repository by `scripts/make_kaggle_notebook.py`; edit the "
                    "repository, not these cells."))
    for pattern in SOURCES:
        for path in sorted(glob.glob(os.path.join(ROOT, pattern))):
            rel = os.path.relpath(path, ROOT).replace("\\", "/")
            if rel in SKIP:
                continue
            with open(path, encoding="utf-8") as f:
                body = f.read()
            # %%writefile refuses an empty cell body (e.g. slm/__init__.py).
            body = body or "# (empty)\n"
            cells.append(code(f"%%writefile {rel}\n{body}", hidden=True))
    cells += [
        md("## Check the code\nUnit tests: causality, RoPE, LoRA, loss masking, overfitting a batch."), code(TESTS),
        md("## Run everything\nOne data-prep worker and one training worker per GPU. GPU workers take the most important "
           "job they can start: evaluation, then fine-tuning, then the scaling sweep (largest model first), then the "
           "ablations. Progress lines appear here every 30 seconds; each job's full output is in its `out/` folder."),
        code(RUN),
        md("## Results"), code(RESULTS), code(RESULTS_IMAGES),
    ]
    nb = {
        "nbformat": 4,
        "nbformat_minor": 4,
        "metadata": {
            "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
            "language_info": {"name": "python"},
            "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True},
        },
        "cells": cells,
    }
    out = os.path.join(ROOT, "kaggle", "AFML_SLM_kaggle.ipynb")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"wrote {out} ({len(cells)} cells, {os.path.getsize(out) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
