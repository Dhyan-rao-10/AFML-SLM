"""Build colab/AFML_SLM.ipynb: one self-contained notebook that writes the project's
source files, then runs the pipeline on a Colab GPU. Several teammates can run it at
once against one shared Drive folder; jobs are claimed so none runs twice.

    python scripts/make_colab_notebook.py

Re-run after editing any source file so the notebook carries the current code.
"""

import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCES = ["slm/*.py", "scripts/*.py", "configs/*.json", "tests/*.py"]
SKIP = {"scripts/make_colab_notebook.py", "scripts/make_kaggle_notebook.py"}

INTRO = """# What Matters in a Small Language Model — Colab runner

Runs the AFML default project: architecture ablations, scaling sweep, instruction tuning and evaluation.
**The whole team can run this notebook at the same time.** Everyone shares one Drive folder; each session
claims the next job nobody else is running, so work splits itself.

**One-time setup per teammate**
1. Dhyan shares the Drive folder `AFML-SLM` with you as **Editor**.
2. In Google Drive, open *Shared with me*, right-click `AFML-SLM` → *Organize → Add shortcut* → **My Drive**.
   The shortcut must sit directly in My Drive and be named exactly `AFML-SLM`.

**Every session**
1. *Runtime → Change runtime type →* **T4 GPU**.
2. Type your name in the first code cell below.
3. *Runtime → Run all*, and allow Drive access **with the same Google account you added the shortcut in**.
4. Keep the tab open. If Colab disconnects or runs out of time, just **Run all** again later. Finished jobs are
   skipped; a job whose session died is picked up again after 15 minutes (by you or anyone) and resumes from its
   last checkpoint.

Results, logs and checkpoints go to `AFML-SLM/out`; prepared data is cached in `AFML-SLM/data_cache`."""

WHO = """WORKER = ''  # <- your first name, e.g. 'dhruv'. Every teammate uses a different name.

# Order to work through the steps. Step 2's largest model is the longest job (~7 h of GPU time across
# sessions), so at least one person should start with Step 2: STEPS = [2, 1, 3, 4]
STEPS = [1, 2, 3, 4]

WORKER = WORKER.strip().lower().replace(' ', '_')
assert WORKER, 'Type your name in WORKER above, then Run all again.'"""

SETUP = """import os, shutil, time
from google.colab import drive

drive.mount('/content/drive')
DRIVE = '/content/drive/MyDrive/AFML-SLM'  # shared results and cached data
WORK = '/content/slm'                     # code and training data (fast local disk)
if not os.path.isdir(DRIVE):
    raise SystemExit('MyDrive/AFML-SLM not found. Add the shared AFML-SLM folder to My Drive as a shortcut '
                     '(see the instructions at the top), and mount Drive with that same Google account.')
for d in (f'{DRIVE}/out', f'{DRIVE}/data_cache'):
    os.makedirs(d, exist_ok=True)
for d in ('slm', 'scripts', 'configs', 'tests', 'data'):
    os.makedirs(f'{WORK}/{d}', exist_ok=True)
if not os.path.islink(f'{WORK}/out'):
    os.symlink(f'{DRIVE}/out', f'{WORK}/out')
os.chdir(WORK)

!nvidia-smi --query-gpu=name,memory.total --format=csv
!pip install -q tokenizers datasets"""

HELPERS = """from IPython.display import Image, display

def sh(cmd):
    \"\"\"Run a shell command with live output; stop the notebook if it fails.\"\"\"
    get_ipython().system(cmd)
    code = get_ipython().user_ns.get('_exit_code', 0)
    if code:
        raise RuntimeError(f'exit code {code}: {cmd}')

def age_min(path):
    return (time.time() - os.path.getmtime(path)) / 60

def ensure_data(name, cmd):
    \"\"\"Get data/<name> onto this machine: copy it from the shared Drive cache, wait while a teammate
    is building it, or build it here and cache it for everyone.\"\"\"
    local, cache = f'data/{name}', f'{DRIVE}/data_cache/{name}'
    done = f'{cache}/.complete'   # written last, so a half-finished copy is never trusted
    building = f'{cache}/.building'  # heartbeat of whoever is building it
    if os.path.exists(f'{local}/meta.json') and os.path.exists(done):
        return
    while not os.path.exists(done) and os.path.exists(building) and age_min(building) < 20 \\
            and (open(building).read().split() or [''])[0] != WORKER:
        print(f'a teammate is preparing {name}; checking again in a minute', flush=True)
        time.sleep(60)
    if os.path.exists(done):
        if not os.path.exists(f'{local}/meta.json'):
            print(f'copying {name} data from Drive', flush=True)
            shutil.copytree(cache, local, dirs_exist_ok=True)
        return
    os.makedirs(cache, exist_ok=True)
    beat = lambda: open(building, 'w').write(f'{WORKER} {time.ctime()}')
    beat()
    if not os.path.exists(f'{local}/meta.json'):
        sh(f'{cmd} --heartbeat {building}' if 'prepare_pretrain' in cmd else cmd)
    beat()
    print(f'caching {name} data to Drive for the team', flush=True)
    shutil.copytree(local, cache, dirs_exist_ok=True)
    open(done, 'w').close()
    os.remove(building)

def claimed(out_dir, cmd):
    \"\"\"Run cmd unless out_dir is finished or a teammate is running it right now.\"\"\"
    sh(f'python scripts/claims.py --worker {WORKER} --out-dir {out_dir} -- {cmd}')

def done(out_dir):
    return os.path.exists(f'{out_dir}/summary.json')

def show(path):
    if os.path.exists(path):
        display(Image(path))"""

TESTS = """sh('python -m pytest tests -q')"""

STEP1 = """def step1():
    ensure_data('tinystories', 'python scripts/prepare_pretrain.py --dataset tinystories')
    sh(f'python scripts/sweep.py ablations --seeds 0 1 --worker {WORKER}')
    if os.path.isdir('out/ablations') and any(done(f'out/ablations/{d}') for d in os.listdir('out/ablations')):
        sh('python scripts/summarize.py out/ablations')
        show('out/ablations/val_curves.png')"""

STEP2 = """def step2():
    ensure_data('fineweb-edu', 'python scripts/prepare_pretrain.py --dataset fineweb-edu --max-tokens 1000000000')
    sh(f'python scripts/sweep.py scaling --worker {WORKER}')
    if os.path.isdir('out/scaling') and any(done(f'out/scaling/{d}') for d in os.listdir('out/scaling')):
        sh('python scripts/summarize.py out/scaling')
        show('out/scaling/val_curves.png')
        show('out/scaling/scaling.png')"""

STEP3 = """SFT_RUNS = ['full_masked', 'full_unmasked', 'lora_masked', 'lora_unmasked']

def step3():
    if not done('out/scaling/l_s0'):
        print('Step 3 needs the finished 46M model (out/scaling/l_s0); skipping it for now.')
        return
    ensure_data('sft', 'python scripts/prepare_sft.py')
    for name in SFT_RUNS:
        mode, mask = name.split('_')
        claimed(f'out/sft/{name}', f'python scripts/sft.py --config configs/sft_{mode}.json '
                                   f'out_dir=out/sft/{name} mask_prompt={str(mask == \"masked\").lower()}')"""

STEP4 = """def step4():
    if not all(done(f'out/sft/{n}') for n in SFT_RUNS):
        print('Step 4 needs all four fine-tuned models from Step 3; skipping it for now.')
        return
    models = ['base=out/scaling/l_s0/ckpt.pt'] + [f'{n}=out/sft/{n}/ckpt.pt' for n in SFT_RUNS]
    claimed('out/eval', 'python scripts/eval_sft.py ' + ' '.join(models))"""

RUN = """for s in STEPS:
    print(f'\\n=== Step {s} ===', flush=True)
    {1: step1, 2: step2, 3: step3, 4: step4}[s]()
print('\\nNothing left for this session. Jobs listed as running belong to teammates; Run all again later to pick up whatever is left.')"""


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text, hidden=False):
    meta = {"cellView": "form", "jupyter": {"source_hidden": True}} if hidden else {}
    return {"cell_type": "code", "metadata": meta, "execution_count": None, "outputs": [], "source": text.splitlines(keepends=True)}


def main():
    cells = [md(INTRO), md("## Who is running this session?"), code(WHO), md("## Setup"), code(SETUP)]
    cells.append(md("## Project code\nGenerated from the repository by `scripts/make_colab_notebook.py`; edit the "
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
        md("## Helpers"), code(HELPERS),
        md("## Check the code\nUnit tests: causality, RoPE, LoRA, loss masking, overfitting a batch. About 30 seconds."), code(TESTS),
        md("## Step 1 — Architecture ablations (extension 1)\nGPT-2-style baseline vs. RoPE, RMSNorm, SwiGLU and "
           "grouped-query attention, swapped in one at a time, plus all four together. ~14M parameters, 100M TinyStories "
           "tokens per run, 2 seeds each: 12 runs of ~19 minutes on a T4."), code(STEP1),
        md("## Step 2 — Scaling sweep (extension 2)\nThree sizes (~9M, ~20M, ~46M parameters) on FineWeb-Edu, 20 tokens per "
           "parameter each. The 46M model is the longest job in the project (~7 h on a T4, spread over sessions). The first "
           "person to get here prepares 1B FineWeb-Edu tokens (~1 h); everyone else copies them from Drive."), code(STEP2),
        md("## Step 3 — Instruction tuning (extension 3)\nFine-tunes the 46M model four ways: full vs. LoRA, loss on the "
           "response only vs. on every token. Starts once Step 2's 46M model is finished."), code(STEP3),
        md("## Step 4 — Answer the held-out prompts\nThe base model and the four tuned models answer the 200 held-out Dolly "
           "prompts; answers are saved to `out/eval/` for the pairwise comparison."), code(STEP4),
        md("## Run\nWorks through `STEPS` in order, skipping jobs that are finished or that a teammate is running."), code(RUN),
    ]

    nb = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "cells": cells,
    }
    out = os.path.join(ROOT, "colab", "AFML_SLM.ipynb")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"wrote {out} ({len(cells)} cells, {os.path.getsize(out) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
