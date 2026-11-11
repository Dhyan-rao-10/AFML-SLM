"""Run the whole project as a queue of jobs shared by several workers on one machine:
one CPU worker that prepares data, and one worker per GPU that trains.

    python scripts/pipeline.py prep --deadline <unix time>
    CUDA_VISIBLE_DEVICES=0 python scripts/pipeline.py gpu0 --deadline <unix time>
    CUDA_VISIBLE_DEVICES=1 python scripts/pipeline.py gpu1 --deadline <unix time>

GPU workers always take the most important job they can start: evaluation, then
fine-tuning, then the scaling sweep (largest model first, since it is the longest
job), then the ablations. Jobs are claimed through scripts/claims.py so no two
workers run the same one. Before the deadline, training pauses itself (resumable)
and no job is started that could not finish in time; a later session resumes.

Set SLM_SMOKE_TEST=1 to shrink every job to a few steps (for testing the plumbing).
"""

import argparse
import json
import os
import subprocess
import sys
import time
import traceback

from claims import status, try_claim
from sweep import SWEEPS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
PAUSED = 75
SFT_RUNS = ["full_masked", "full_unmasked", "lora_masked", "lora_unmasked"]
SMOKE = os.environ.get("SLM_SMOKE_TEST") == "1"

# name: (command, minutes needed). Data prep cannot pause, so it only starts with enough time left.
DATA = {
    "tinystories": ([PY, "scripts/prepare_pretrain.py", "--dataset", "tinystories"], 30),
    "sft": ([PY, "scripts/prepare_sft.py"], 10),
    "fineweb-edu": ([PY, "scripts/prepare_pretrain.py", "--dataset", "fineweb-edu", "--max-tokens", "1000000000"], 120),
}
SMOKE_DATA = {
    "tinystories": ["--max-tokens", "3000000", "--tokenizer-docs", "3000"],
    "fineweb-edu": ["--max-tokens", "3000000", "--val-docs", "300", "--tokenizer-docs", "3000"],
}
SMOKE_TIME = 30  # smoke-test jobs take seconds, so time requirements shrink by this factor
PREP_EXITED = os.path.join(ROOT, "data", ".prep_exited")


def need(minutes):
    """Minutes of session left that a step requires (shrunk in smoke tests)."""
    return minutes / (SMOKE_TIME if SMOKE else 1)
SMOKE_TRAIN = ["batch_size=4", "grad_accum=1", "max_iters=30", "eval_interval=10", "eval_iters=2", "log_interval=10"]
SMOKE_SFT = ["max_train_examples=200", "max_val_examples=32", "epochs=1", "batch_size=4", "grad_accum=1", "eval_interval=10"]


def log(worker, msg):
    print(f"[{worker} {time.strftime('%H:%M')}] {msg}", flush=True)


def data_state(name):
    d = os.path.join(ROOT, "data", name)
    if os.path.exists(os.path.join(d, ".complete")):
        return "ready"
    if os.path.exists(os.path.join(d, ".failed")):
        return "failed"
    # The prep worker has stopped (e.g. not enough time left), so this data won't arrive this session.
    return "unavailable" if os.path.exists(PREP_EXITED) else "pending"


def jobs(deadline):
    """Every job as a dict, most important first."""
    out = []
    ev = ["--max-new-tokens", "8"] if SMOKE else []
    for name in ["base"] + SFT_RUNS:
        ckpt = "out/scaling/l_s0/ckpt.pt" if name == "base" else f"out/sft/{name}/ckpt.pt"
        dep = "out/scaling/l_s0" if name == "base" else f"out/sft/{name}"
        out.append({"out_dir": f"out/eval/{name}", "deps": [dep], "data": ["sft"], "minutes": 45,
                    "cmd": [PY, "scripts/eval_sft.py", f"{name}={ckpt}", "--out-dir", f"out/eval/{name}", *ev]})
    for name in SFT_RUNS:
        mode, mask = name.split("_")
        out.append({"out_dir": f"out/sft/{name}", "deps": ["out/scaling/l_s0"], "data": ["sft"], "minutes": 40,
                    "cmd": [PY, "scripts/sft.py", "--config", f"configs/sft_{mode}.json", f"out_dir=out/sft/{name}",
                            f"mask_prompt={str(mask == 'masked').lower()}", *(SMOKE_SFT if SMOKE else [])]})
    for sweep, data, seeds in (("scaling", "fineweb-edu", [0]), ("ablations", "tinystories", [0, 1])):
        for name, (config, overrides) in SWEEPS[sweep].items():
            for seed in seeds:
                out_dir = f"out/{sweep}/{name}_s{seed}"
                # Training pauses itself at the deadline, so it may start with little time left.
                out.append({"out_dir": out_dir, "deps": [], "data": [data], "minutes": 5,
                            "cmd": [PY, "scripts/train.py", "--config", config, f"out_dir={out_dir}", f"seed={seed}",
                                    *overrides, f"deadline={deadline}", *(SMOKE_TRAIN if SMOKE else [])]})
    return out


def job_state(job, worker, known=None):
    """'done', 'failed', 'blocked', 'waiting', 'later' (not this session) or 'runnable'.
    known maps out_dir -> state for jobs already assessed, so 'later' and 'blocked'
    pass on to the jobs that depend on them."""
    known = known or {}
    st = status(os.path.join(ROOT, job["out_dir"]), worker)
    if st in ("done", "failed"):
        return st
    if st.startswith("running"):
        return "waiting"
    for dep in job["deps"]:
        dst = known.get(dep) or status(os.path.join(ROOT, dep), worker)
        if dst in ("failed", "blocked"):
            return "blocked"
        if dst == "later":
            return "later"
        if dst != "done":
            return "waiting"
    for name in job["data"]:
        state = data_state(name)
        if state == "failed":
            return "blocked"
        if state == "unavailable":
            return "later"
        if state != "ready":
            return "waiting"
    return "runnable"


def all_states(worker, deadline):
    """[(job, state)] in priority order. Assessed dependencies-first (the list runs
    from eval down to the ablations), so each job sees its dependencies' states."""
    all_jobs = jobs(deadline)
    known = {}
    for job in reversed(all_jobs):
        known[job["out_dir"]] = job_state(job, worker, known)
    return [(job, known[job["out_dir"]]) for job in all_jobs]


def run_job(job, worker):
    out_dir = os.path.join(ROOT, job["out_dir"])
    if not try_claim(out_dir, worker):
        return None  # another worker got it first
    log(worker, f"start {job['out_dir']}")
    t0 = time.time()
    with open(os.path.join(out_dir, "stdout.log"), "a", encoding="utf-8") as f:
        rc = subprocess.run(job["cmd"], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
    minutes = (time.time() - t0) / 60
    if rc == 0:
        with open(os.path.join(out_dir, "summary.json")) as f:
            s = json.load(f)
        extra = f", val loss {s['final_val_loss']:.4f}" if "final_val_loss" in s else ""
        log(worker, f"done  {job['out_dir']} in {minutes:.0f} min{extra}")
    elif rc == PAUSED:
        log(worker, f"pause {job['out_dir']} at the deadline (resumable)")
    else:
        with open(os.path.join(out_dir, "stdout.log"), encoding="utf-8", errors="replace") as f:
            tail = f.readlines()[-15:]
        with open(os.path.join(out_dir, "FAILED"), "w") as f:
            f.write(f"exit code {rc}\n" + "".join(tail))
        log(worker, f"FAIL  {job['out_dir']} (exit {rc}); last lines:\n" + "".join(tail))
    return rc


def gpu_worker(worker, deadline):
    while True:
        left = (deadline - time.time()) / 60
        states = all_states(worker, deadline)
        runnable = [job for job, st in states if st == "runnable"]
        startable = [job for job in runnable if need(job["minutes"]) <= left]
        if startable:
            if run_job(startable[0], worker) == PAUSED:
                return "paused"
            continue
        if runnable:
            log(worker, f"{len(runnable)} job(s) left need more time than remains; stopping until the next session")
            return "paused"
        if any(st == "waiting" for _, st in states) and left > need(10):
            time.sleep(10 if SMOKE else 60)
            continue
        if all(st == "done" for _, st in states):
            log(worker, "all jobs done")
            return "complete"
        if any(st in ("later", "waiting") for _, st in states):
            log(worker, "the remaining jobs need data or time this session doesn't have; stopping until the next session")
            return "paused"
        stuck = [job["out_dir"] for job, st in states if st in ("failed", "blocked")]
        log(worker, f"nothing more I can do; failed or blocked: {', '.join(stuck) or 'none'}")
        return "stopped"


def prep_worker(deadline):
    for name, (cmd, minutes) in DATA.items():
        if data_state(name) == "ready":
            continue
        if (deadline - time.time()) / 60 < need(minutes):
            log("prep", f"not enough time left to prepare {name}; next session will")
            return "paused"
        d = os.path.join(ROOT, "data", name)
        os.makedirs(d, exist_ok=True)
        if os.path.exists(os.path.join(d, ".failed")):
            os.remove(os.path.join(d, ".failed"))
        full = cmd + (SMOKE_DATA.get(name, []) if SMOKE else [])
        log("prep", f"preparing {name}")
        t0 = time.time()
        with open(os.path.join(ROOT, "data", f"prepare_{name}.log"), "a", encoding="utf-8") as f:
            rc = subprocess.run(full, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
        if rc == 0:
            open(os.path.join(d, ".complete"), "w").close()
            log("prep", f"{name} ready in {(time.time() - t0) / 60:.0f} min")
        else:
            open(os.path.join(d, ".failed"), "w").write(f"exit code {rc}\n")
            log("prep", f"FAIL preparing {name} (exit {rc}); see data/prepare_{name}.log")
    return "complete"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("worker", help="'prep' for data preparation, or a GPU worker name such as gpu0")
    ap.add_argument("--deadline", type=float, required=True, help="unix time by which to stop")
    args = ap.parse_args()
    os.chdir(ROOT)
    if args.worker == "prep":
        try:
            result = prep_worker(args.deadline)
        finally:
            open(PREP_EXITED, "w").close()
    else:
        result = None
        for attempt in range(5):
            try:
                result = gpu_worker(args.worker, args.deadline)
                break
            except Exception:
                # Losing a GPU worker for the rest of a long run costs hours; log and carry on.
                log(args.worker, "unexpected error (retrying in a minute):\n" + traceback.format_exc())
                time.sleep(5 if SMOKE else 60)
        else:
            result = "crashed"
    print(f"[{args.worker}] finished: {result}", flush=True)


if __name__ == "__main__":
    main()
