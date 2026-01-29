"""Simple LoRA utilities for PyTorch linear layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812


@dataclass(frozen=True)
class LoraConfig:
    enabled: bool = False
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.05
    target_modules: Iterable[str] | None = None


class LoRALinear(nn.Linear):
    """Linear layer with LoRA adapters while preserving original weight names."""

    def __init__(self, in_features: int, out_features: int, bias: bool, *, r: int, alpha: float, dropout: float):
        super().__init__(in_features, out_features, bias=bias)
        self.r = int(r)
        self.lora_alpha = float(alpha)
        self.scaling = self.lora_alpha / self.r if self.r > 0 else 1.0

        if self.r > 0:
            self.lora_A = nn.Linear(in_features, self.r, bias=False)
            self.lora_B = nn.Linear(self.r, out_features, bias=False)
            self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

            nn.init.normal_(self.lora_A.weight, std=0.01)
            nn.init.zeros_(self.lora_B.weight)
        else:
            self.lora_A = None
            self.lora_B = None
            self.lora_dropout = nn.Identity()

    @classmethod
    def from_linear(cls, linear: nn.Linear, *, r: int, alpha: float, dropout: float) -> "LoRALinear":
        lora = cls(linear.in_features, linear.out_features, linear.bias is not None, r=r, alpha=alpha, dropout=dropout)
        with torch.no_grad():
            lora.weight.copy_(linear.weight)
            if linear.bias is not None and lora.bias is not None:
                lora.bias.copy_(linear.bias)
        return lora

    def forward(self, x):
        result = F.linear(x, self.weight, self.bias)
        if self.r > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
            result = result + lora_out
        return result


def _matches_target(name: str, target_modules: Iterable[str] | None) -> bool:
    if target_modules is None:
        return True
    return any(key in name for key in target_modules)


def apply_lora(
    model: nn.Module,
    *,
    rank: int,
    alpha: float,
    dropout: float,
    target_modules: Iterable[str] | None = None,
) -> int:
    """Replace Linear layers with LoRA-augmented versions. Returns count of replaced modules."""
    replaced = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and not isinstance(module, LoRALinear) and _matches_target(name, target_modules):
            parent = model
            *parents, child_name = name.split(".")
            for part in parents:
                parent = getattr(parent, part)
            setattr(parent, child_name, LoRALinear.from_linear(module, r=rank, alpha=alpha, dropout=dropout))
            replaced += 1
    return replaced


def mark_only_lora_as_trainable(model: nn.Module) -> None:
    """Freeze all parameters except LoRA adapters."""
    for _, param in model.named_parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad = True


def get_trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]