"""Low-rank adaptation (Hu et al., 2021) written out by hand.

A frozen weight W is adapted as W + (alpha / r) * B @ A, where A is (r x in) and
B is (out x r). B starts at zero, so the adapted model is exactly the base model
at step 0 and training only has to learn the low-rank update.
"""

import math

import torch
import torch.nn as nn

DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        update = (self.dropout(x) @ self.lora_A.T) @ self.lora_B.T
        return self.base(x) + update * self.scaling

    def merged(self) -> nn.Linear:
        """Fold the update into a plain Linear: no extra cost at inference time."""
        out = nn.Linear(self.base.in_features, self.base.out_features, bias=False)
        delta = (self.lora_B @ self.lora_A) * self.scaling
        out.weight.data.copy_(self.base.weight.data + delta.to(self.base.weight.dtype))
        return out.to(self.base.weight.device)


def apply_lora(model: nn.Module, r: int, alpha: float, dropout: float = 0.0, targets=DEFAULT_TARGETS):
    """Freeze every base weight, then wrap each Linear whose attribute name is in targets."""
    for p in model.parameters():
        p.requires_grad_(False)
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if name in targets and isinstance(child, nn.Linear):
                setattr(module, name, LoRALinear(child, r, alpha, dropout))
    return model


def merge_lora(model: nn.Module):
    """Replace every LoRALinear with its merged Linear, in place."""
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                setattr(module, name, child.merged())
    for p in model.parameters():
        p.requires_grad_(True)
    return model


def lora_state_dict(model: nn.Module) -> dict:
    return {k: v for k, v in model.state_dict().items() if "lora_" in k}
