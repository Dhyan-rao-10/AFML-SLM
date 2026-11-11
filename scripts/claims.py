"""Let several people's Colab sessions share one Drive folder without running the
same job twice.

A job is a run directory such as out/ablations/rope_s0. A worker claims it by
writing CLAIM there. The job counts as taken while any file in the directory
changed within STALE_MIN minutes: training appends to log.jsonl every few
seconds, so a live run keeps its claim fresh, while a run whose session died
goes stale and can be taken over (train.py then resumes from ckpt_last.pt).

    python scripts/claims.py --worker dhyan --out-dir out/sft/full_masked -- python scripts/sft.py ...

runs the command only if the job is neither finished nor running elsewhere.
"""

import argparse
import json
import os
import subprocess
import sys
import time

STALE_MIN = 15


def last_activity(out_dir: str) -> float:
    # The .claiming lock is not activity: it exists exactly while someone checks the job.
    times = [os.path.getmtime(os.path.join(out_dir, f)) for f in os.listdir(out_dir) if f != ".claiming"]
    return max(times, default=0.0)


def owner(out_dir: str) -> str | None:
    try:
        with open(os.path.join(out_dir, "CLAIM")) as f:
            return json.load(f)["worker"]
    except (OSError, ValueError, KeyError):
        return None


def status(out_dir: str, worker: str) -> str:
    """'done', 'failed', 'free', or 'running (<name>)'."""
    if os.path.exists(os.path.join(out_dir, "summary.json")):
        return "done"
    if os.path.exists(os.path.join(out_dir, "FAILED")):
        return "failed"
    if not os.path.isdir(out_dir):
        return "free"
    who = owner(out_dir)
    if who != worker and time.time() - last_activity(out_dir) < STALE_MIN * 60:
        return f"running ({who or 'someone'})"
    return "free"  # never started, abandoned, or our own run from a session that died


def claim(out_dir: str, worker: str):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "CLAIM"), "w") as f:
        json.dump({"worker": worker, "time": time.strftime("%Y-%m-%d %H:%M:%S")}, f)


def try_claim(out_dir: str, worker: str) -> bool:
    """Claim out_dir if it is free, atomically: two workers checking the same job at
    the same instant cannot both get it. mkdir either creates the lock or fails."""
    os.makedirs(out_dir, exist_ok=True)
    lock = os.path.join(out_dir, ".claiming")
    try:
        os.mkdir(lock)
    except FileExistsError:
        try:
            if time.time() - os.path.getmtime(lock) < 60:
                return False  # another worker is claiming this job right now
            os.rmdir(lock)  # left behind by a worker that crashed mid-claim
        except FileNotFoundError:
            return False  # the other worker just finished claiming; let the caller look again
        return try_claim(out_dir, worker)
    try:
        if status(out_dir, worker) != "free":
            return False
        claim(out_dir, worker)
        return True
    finally:
        os.rmdir(lock)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="-- command to run")
    args = ap.parse_args()
    cmd = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd

    if not try_claim(args.out_dir, args.worker):
        print(f"skip {args.out_dir}: {status(args.out_dir, args.worker)}")
        return
    sys.exit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
