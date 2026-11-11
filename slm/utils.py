"""Config loading, LR schedule, device setup and checkpoint helpers."""

import copy
import json
import math
import os
from contextlib import nullcontext

import torch

from .model import GPT, GPTConfig


def load_config(defaults: dict, path: str | None, overrides: list[str]) -> dict:
    """defaults <- JSON file <- key=value overrides. Dotted keys reach nested dicts,
    e.g. model.pos_emb=rope. Values parse as JSON when they can, else stay strings."""
    cfg = copy.deepcopy(defaults)
    if path:
        with open(path) as f:
            _merge(cfg, json.load(f))
    for item in overrides:
        key, _, raw = item.partition("=")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        node = cfg
        *parents, leaf = key.split(".")
        for k in parents:
            node = node.setdefault(k, {})
        node[leaf] = value
    return cfg


def _merge(base: dict, new: dict):
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v


def cosine_lr(it: int, max_iters: int, lr: float, min_lr: float, warmup_iters: int) -> float:
    """Linear warmup, then cosine decay from lr to min_lr."""
    if it < warmup_iters:
        return lr * (it + 1) / warmup_iters
    progress = min(1.0, (it - warmup_iters) / max(1, max_iters - warmup_iters))
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * progress)) * (lr - min_lr)


def setup_device(requested: str = "cuda"):
    """Return (device, autocast context, grad scaler).

    Ampere and newer (RTX 30xx/40xx, A100, L4) train in bf16, which needs no loss
    scaling. Older GPUs such as Colab's T4 lack bf16, so they train in fp16 with a
    GradScaler that keeps small gradients from underflowing. On CPU, and in bf16,
    the scaler is disabled and its methods pass straight through."""
    if requested.startswith("cuda") and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if torch.cuda.get_device_capability() >= (8, 0):
            ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            return requested, ctx, torch.amp.GradScaler("cuda", enabled=False)
        ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
        return requested, ctx, torch.amp.GradScaler("cuda", enabled=True)
    return "cpu", nullcontext(), torch.amp.GradScaler("cpu", enabled=False)


def save_checkpoint(path: str, model: GPT, **extra):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    state = {k.removeprefix("_orig_mod."): v for k, v in model.state_dict().items()}
    torch.save({"model": state, "model_config": vars(model.cfg).copy(), **extra}, path)


def load_checkpoint(path: str, device: str = "cpu"):
    """Return (model, checkpoint dict)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = GPT(GPTConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["model"])
    return model.to(device), ckpt


class JsonlLogger:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.f = open(path, "a", encoding="utf-8")

    def log(self, **row):
        self.f.write(json.dumps(row) + "\n")
        self.f.flush()
