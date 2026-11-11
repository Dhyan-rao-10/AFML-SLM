"""Pretrain a model from scratch on token bins.

    python scripts/train.py --config configs/ablation_base.json out_dir=out/test model.pos_emb=rope

Any key=value after the flags overrides the config (dotted keys reach model.*).
Writes <out_dir>/{log.jsonl, summary.json, ckpt.pt}.
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from slm.data import TokenBins  # noqa: E402
from slm.model import GPT, GPTConfig  # noqa: E402
from slm.utils import JsonlLogger, cosine_lr, load_config, save_checkpoint, setup_device  # noqa: E402

DEFAULTS = {
    "data_dir": "data/tinystories",
    "out_dir": "out/run",
    "model": {},  # GPTConfig fields; vocab_size comes from the data's meta.json
    "batch_size": 32,
    "grad_accum": 4,
    "max_tokens": 100_000_000,  # training budget; ignored if max_iters or tokens_per_param is set
    "tokens_per_param": None,  # e.g. 20 -> budget = 20 x total parameters
    "max_iters": None,
    "lr": 1e-3,
    "min_lr": 1e-4,
    "warmup_iters": 100,
    "weight_decay": 0.1,
    "beta1": 0.9,
    "beta2": 0.95,
    "grad_clip": 1.0,
    "eval_interval": 250,
    "eval_iters": 40,
    "log_interval": 20,
    "seed": 0,
    "device": "cuda",
    "compile": False,  # torch.compile needs Triton, which is not bundled on Windows
    "save_checkpoint": True,
    "resume": True,  # continue from <out_dir>/ckpt_last.pt if a previous session was cut off
    "deadline": None,  # unix time; save resume state and exit with PAUSED once it passes
}

PAUSED = 75  # exit code: stopped at the deadline, resumable


def dataset_id(meta: dict) -> str:
    """Fingerprint of the token data: the tokenizer file plus the token counts."""
    with open(meta["tokenizer"], "rb") as f:
        digest = hashlib.sha1(f.read()).hexdigest()
    return f"{digest}:{meta['train_tokens']}:{meta['val_tokens']}"


def save_resume_state(path, model, optimizer, scaler, it, best_val, elapsed, data_id):
    """Everything needed to continue mid-run. Written to a temp file first, so a
    disconnect during the write cannot corrupt the previous state."""
    state = {
        "model": {k.removeprefix("_orig_mod."): v for k, v in model.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "iter": it,
        "best_val": best_val,
        "elapsed": elapsed,
        "rng": torch.get_rng_state(),
        "data_id": data_id,
    }
    torch.save(state, path + ".tmp")
    os.replace(path + ".tmp", path)


@torch.no_grad()
def evaluate(model, batches, device, ctx):
    model.eval()
    losses = []
    for x, y in batches:
        with ctx:
            _, loss = model(x.to(device), y.to(device))
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("overrides", nargs="*", help="key=value pairs")
    args = ap.parse_args()
    cfg = load_config(DEFAULTS, args.config, args.overrides)

    torch.manual_seed(cfg["seed"])
    device, ctx, scaler = setup_device(cfg["device"])

    with open(os.path.join(cfg["data_dir"], "meta.json")) as f:
        meta = json.load(f)
    model_cfg = GPTConfig(**{**cfg["model"], "vocab_size": meta["vocab_size"]})
    model = GPT(model_cfg).to(device)
    n_params = model.num_params()
    n_params_nonemb = model.num_params(non_embedding=True)

    T = model_cfg.block_size
    tokens_per_iter = cfg["batch_size"] * cfg["grad_accum"] * T
    if cfg["max_iters"] is not None:
        max_iters = cfg["max_iters"]
    else:
        budget = cfg["tokens_per_param"] * n_params if cfg["tokens_per_param"] else cfg["max_tokens"]
        max_iters = math.ceil(budget / tokens_per_iter)

    os.makedirs(cfg["out_dir"], exist_ok=True)
    with open(os.path.join(cfg["out_dir"], "config.json"), "w") as f:
        json.dump({**cfg, "model": vars(model_cfg)}, f, indent=2)
    print(f"model: {n_params / 1e6:.2f}M params ({n_params_nonemb / 1e6:.2f}M non-embedding)")
    print(f"training: {max_iters} iters x {tokens_per_iter:,} tokens = {max_iters * tokens_per_iter / 1e6:.0f}M tokens on {device}")

    data = TokenBins(cfg["data_dir"], T, device)
    val_batches = data.fixed_batches("val", cfg["eval_iters"], cfg["batch_size"])
    optimizer = model.configure_optimizer(cfg["lr"], cfg["weight_decay"], (cfg["beta1"], cfg["beta2"]), device.split(":")[0])
    train_model = torch.compile(model) if cfg["compile"] else model
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    last_path = os.path.join(cfg["out_dir"], "ckpt_last.pt")
    log_path = os.path.join(cfg["out_dir"], "log.jsonl")
    data_id = dataset_id(meta)
    start_it, best_val, elapsed_before = 0, float("inf"), 0.0
    if cfg["resume"] and os.path.exists(last_path):
        state = torch.load(last_path, map_location=device, weights_only=False)
        if state.get("data_id") == data_id:
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scaler.load_state_dict(state["scaler"])
            torch.set_rng_state(state["rng"].cpu())  # the CPU generator needs a CPU tensor
            start_it, best_val, elapsed_before = state["iter"], state["best_val"], state["elapsed"]
            print(f"resumed from iter {start_it}")
        else:
            # The token data was rebuilt since this state was saved (e.g. in a new Colab
            # session), so its token ids may not match the model's. Start over.
            print("resume state was saved with different data; starting this run fresh")
    if start_it == 0 and os.path.exists(log_path):
        os.remove(log_path)
    logger = JsonlLogger(log_path)

    x, y = data.get_batch("train", cfg["batch_size"])
    t_start = t_last = time.time()
    for it in range(start_it, max_iters + 1):
        lr = cosine_lr(it, max_iters, cfg["lr"], cfg["min_lr"], cfg["warmup_iters"])
        for group in optimizer.param_groups:
            group["lr"] = lr

        if it % cfg["eval_interval"] == 0 or it == max_iters:
            val_loss = evaluate(model, val_batches, device, ctx)
            elapsed = elapsed_before + time.time() - t_start
            logger.log(iter=it, tokens=it * tokens_per_iter, val_loss=val_loss, elapsed=elapsed)
            print(f"iter {it}: val loss {val_loss:.4f}")
            if val_loss < best_val:
                best_val = val_loss
                if cfg["save_checkpoint"] and it > 0:
                    save_checkpoint(os.path.join(cfg["out_dir"], "ckpt.pt"), model, iter=it, val_loss=val_loss, tokenizer=meta["tokenizer"], train_config=cfg)
            if it > start_it and it < max_iters:
                save_resume_state(last_path, model, optimizer, scaler, it, best_val, elapsed, data_id)
        if it == max_iters:
            break

        for _ in range(cfg["grad_accum"]):
            with ctx:
                _, loss = train_model(x, y)
            x, y = data.get_batch("train", cfg["batch_size"])  # fetch the next batch while the GPU works
            scaler.scale(loss / cfg["grad_accum"]).backward()
        scaler.unscale_(optimizer)  # so clipping sees true gradient sizes under fp16
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"]).item()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if cfg["deadline"] and time.time() > cfg["deadline"]:
            # Out of time (e.g. a 12-hour Kaggle run): keep everything for the next session.
            elapsed = elapsed_before + time.time() - t_start
            save_resume_state(last_path, model, optimizer, scaler, it + 1, best_val, elapsed, data_id)
            print(f"deadline reached after iter {it}; paused, resumable from {last_path}")
            sys.exit(PAUSED)

        if it % cfg["log_interval"] == 0:
            now = time.time()
            tok_s = tokens_per_iter * (cfg["log_interval"] if it > start_it else 1) / (now - t_last)
            t_last = now
            logger.log(iter=it, tokens=(it + 1) * tokens_per_iter, train_loss=loss.item(), lr=lr, grad_norm=grad_norm, tok_per_s=tok_s)
            print(f"iter {it}: loss {loss.item():.4f}, lr {lr:.2e}, grad norm {grad_norm:.2f}, {tok_s:,.0f} tok/s")

    elapsed = elapsed_before + time.time() - t_start
    if os.path.exists(last_path):
        os.remove(last_path)  # the run is complete; ckpt.pt holds the best weights
    summary = {
        "out_dir": cfg["out_dir"],
        "model": vars(model_cfg),
        "seed": cfg["seed"],
        "params": n_params,
        "params_non_embedding": n_params_nonemb,
        "tokens": max_iters * tokens_per_iter,
        "final_val_loss": val_loss,
        "best_val_loss": best_val,
        "wall_time_s": elapsed,
        "tok_per_s": max_iters * tokens_per_iter / elapsed,
        "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if device.startswith("cuda") else None,
    }
    with open(os.path.join(cfg["out_dir"], "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"done in {elapsed / 60:.1f} min, final val loss {val_loss:.4f}")


if __name__ == "__main__":
    main()
