"""Instruction-tune a pretrained checkpoint: full fine-tuning or LoRA, with the
loss on the response only (mask_prompt=true) or on every token.

    python scripts/sft.py --config configs/sft_full.json base_ckpt=out/final/ckpt.pt
    python scripts/sft.py --config configs/sft_lora.json mask_prompt=false out_dir=out/sft/lora_nomask

Validation loss is always measured on response tokens only, so masked and
unmasked runs are compared on the same quantity. LoRA weights are merged into
the saved checkpoint, so every SFT checkpoint loads like a pretrained one.
"""

import argparse
import json
import math
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from slm.chat import collate, encode_example  # noqa: E402
from slm.lora import apply_lora, lora_state_dict, merge_lora  # noqa: E402
from slm.tokenizer import load_tokenizer  # noqa: E402
from slm.utils import JsonlLogger, cosine_lr, load_checkpoint, load_config, save_checkpoint, setup_device  # noqa: E402

DEFAULTS = {
    "base_ckpt": "out/final/ckpt.pt",
    "sft_dir": "data/sft",
    "out_dir": "out/sft/run",
    "mode": "full",  # "full" | "lora"
    "mask_prompt": True,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate", "up", "down"],
    "epochs": 2,
    "batch_size": 16,
    "grad_accum": 2,
    "lr": 1e-4,
    "min_lr_ratio": 0.1,
    "warmup_ratio": 0.03,
    "weight_decay": 0.0,
    "grad_clip": 1.0,
    "eval_interval": 200,
    "max_val_examples": 1000,
    "max_train_examples": None,
    "seed": 0,
    "device": "cuda",
}


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def encode_all(tok, rows, block_size, mask_prompt):
    out = [encode_example(tok, r["prompt"], r["response"], block_size, mask_prompt) for r in rows]
    kept = [e for e in out if e is not None]
    return kept, len(out) - len(kept)


@torch.no_grad()
def evaluate(model, examples, batch_size, device, ctx):
    model.eval()
    total, count = 0.0, 0
    for i in range(0, len(examples), batch_size):
        x, y = collate(examples[i : i + batch_size], device)
        with ctx:
            _, loss = model(x, y)
        n = (y != -100).sum().item()
        total += loss.item() * n
        count += n
    model.train()
    return total / count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("overrides", nargs="*", help="key=value pairs")
    args = ap.parse_args()
    cfg = load_config(DEFAULTS, args.config, args.overrides)

    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    device, ctx, scaler = setup_device(cfg["device"])
    model, base = load_checkpoint(cfg["base_ckpt"], device)
    tok = load_tokenizer(base["tokenizer"])
    T = model.cfg.block_size

    train_rows = read_jsonl(os.path.join(cfg["sft_dir"], "train.jsonl"))
    if cfg["max_train_examples"]:
        train_rows = train_rows[: cfg["max_train_examples"]]
    val_rows = read_jsonl(os.path.join(cfg["sft_dir"], "val.jsonl"))[: cfg["max_val_examples"]]
    train, dropped = encode_all(tok, train_rows, T, cfg["mask_prompt"])
    val, _ = encode_all(tok, val_rows, T, mask_prompt=True)
    print(f"{len(train):,} train examples ({dropped:,} longer than {T} tokens dropped), {len(val):,} val")

    if cfg["mode"] == "lora":
        apply_lora(model, cfg["lora_r"], cfg["lora_alpha"], cfg["lora_dropout"], tuple(cfg["lora_targets"]))
        model.to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"{cfg['mode']}: {trainable / 1e6:.2f}M of {total / 1e6:.2f}M parameters trainable")

    optimizer = model.configure_optimizer(cfg["lr"], cfg["weight_decay"], (0.9, 0.95), device.split(":")[0])
    examples_per_step = cfg["batch_size"] * cfg["grad_accum"]
    steps_per_epoch = math.ceil(len(train) / examples_per_step)
    max_steps = cfg["epochs"] * steps_per_epoch
    warmup = max(1, int(cfg["warmup_ratio"] * max_steps))
    os.makedirs(cfg["out_dir"], exist_ok=True)
    with open(os.path.join(cfg["out_dir"], "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    logger = JsonlLogger(os.path.join(cfg["out_dir"], "log.jsonl"))
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    val_loss = evaluate(model, val, cfg["batch_size"], device, ctx)
    logger.log(step=0, val_loss=val_loss)
    print(f"step 0: val loss {val_loss:.4f}")
    t_start = time.time()
    step = 0
    model.train()
    for epoch in range(cfg["epochs"]):
        order = list(range(len(train)))
        random.shuffle(order)
        for s in range(steps_per_epoch):
            idx = order[s * examples_per_step : (s + 1) * examples_per_step]
            micro = [idx[i : i + cfg["batch_size"]] for i in range(0, len(idx), cfg["batch_size"])]
            lr = cosine_lr(step, max_steps, cfg["lr"], cfg["lr"] * cfg["min_lr_ratio"], warmup)
            for group in optimizer.param_groups:
                group["lr"] = lr
            for m in micro:
                x, y = collate([train[i] for i in m], device)
                with ctx:
                    _, loss = model(x, y)
                scaler.scale(loss / len(micro)).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % 20 == 0:
                logger.log(step=step, epoch=epoch, train_loss=loss.item(), lr=lr)
                print(f"step {step}/{max_steps}: loss {loss.item():.4f}, lr {lr:.2e}")
            if step % cfg["eval_interval"] == 0 or step == max_steps:
                val_loss = evaluate(model, val, cfg["batch_size"], device, ctx)
                logger.log(step=step, val_loss=val_loss, elapsed=time.time() - t_start)
                print(f"step {step}: val loss {val_loss:.4f}")

    elapsed = time.time() - t_start
    peak_mem = torch.cuda.max_memory_allocated() / 1e9 if device.startswith("cuda") else None
    if cfg["mode"] == "lora":
        torch.save(lora_state_dict(model), os.path.join(cfg["out_dir"], "adapter.pt"))
        merge_lora(model)
    save_checkpoint(os.path.join(cfg["out_dir"], "ckpt.pt"), model, tokenizer=base["tokenizer"], sft_config=cfg, base_ckpt=cfg["base_ckpt"])
    summary = {
        "out_dir": cfg["out_dir"],
        "mode": cfg["mode"],
        "mask_prompt": cfg["mask_prompt"],
        "trainable_params": trainable,
        "total_params": total,
        "train_examples": len(train),
        "steps": max_steps,
        "final_val_loss": val_loss,
        "wall_time_s": elapsed,
        "gpu_hours": elapsed / 3600,
        "peak_mem_gb": peak_mem,
    }
    with open(os.path.join(cfg["out_dir"], "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"done in {elapsed / 60:.1f} min, val loss {val_loss:.4f}")


if __name__ == "__main__":
    main()
