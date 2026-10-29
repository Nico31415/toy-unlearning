"""ResNet-18 with the CIFAR stem, BatchNorm handling, and the S-rescaling infrastructure."""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm


def resnet18_cifar(num_classes: int = 10) -> nn.Module:
    """torchvision ResNet-18 with the CIFAR stem: 3x3 stride-1 first conv, no maxpool.

    The ImageNet stem (7x7 stride-2 conv plus maxpool, as in the Lippl & Lindsey
    code) shrinks a 32x32 image to 8x8 before the first residual block.
    """
    from torchvision.models import resnet18

    model = resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")
    model.maxpool = nn.Identity()
    return model


class TinyNet(nn.Module):
    """Small conv net with BatchNorm and an `fc` head, for fast tests only."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.features(x))


def build_model(arch: str, num_classes: int) -> nn.Module:
    if arch == "resnet18":
        return resnet18_cifar(num_classes)
    if arch == "tiny":
        return TinyNet(num_classes)
    raise ValueError(f"unknown arch {arch!r}")


def set_train_mode(model: nn.Module, freeze_bn: bool) -> None:
    """Training mode; with `freeze_bn`, BatchNorm layers use and keep their running
    statistics while their affine parameters still receive gradients."""
    model.train()
    if freeze_bn:
        for module in model.modules():
            if isinstance(module, _BatchNorm):
                module.eval()


# --------------------------------------------------------------------------- rescaling

RESCALE_MODES = ("none", "all", "weights", "body_head")


def _head(model: nn.Module) -> nn.Linear:
    head = getattr(model, "fc", None)
    if not isinstance(head, nn.Linear):
        raise ValueError("rescaling expects the final layer to be a Linear module named `fc`")
    return head


@torch.no_grad()
def rescale_(model: nn.Module, mode: str = "none", S: float = 1.0,
             alpha_body: float = 1.0, alpha_head: float = 1.0) -> None:
    """Rescale a trained model in place, before unlearning.

    none       no change
    all        every parameter times S, including biases and BatchNorm affine
               parameters (the `finetune_scaling` convention of Lippl & Lindsey)
    weights    every conv and linear weight times S; biases and BatchNorm untouched
    body_head  every conv weight times alpha_body and the final linear layer
               (weight and bias) times alpha_head; the analogue of the toy's
               (alpha_in, alpha_out) layer rescaling

    BatchNorm running statistics are buffers and are never touched here. With frozen
    BatchNorm they are fixed, so rescaling conv weights changes the network's
    function; use `recalibrate_bn` to recompute them instead.
    """
    if mode not in RESCALE_MODES:
        raise ValueError(f"unknown rescale mode {mode!r}; expected one of {RESCALE_MODES}")
    if mode == "all":
        for p in model.parameters():
            p.mul_(S)
    elif mode == "weights":
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                module.weight.mul_(S)
    elif mode == "body_head":
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                module.weight.mul_(alpha_body)
        head = _head(model)
        head.weight.mul_(alpha_head)
        if head.bias is not None:
            head.bias.mul_(alpha_head)


def is_identity_rescale(mode: str, S: float, alpha_body: float, alpha_head: float) -> bool:
    if mode == "none":
        return True
    if mode in ("all", "weights"):
        return S == 1.0
    return alpha_body == 1.0 and alpha_head == 1.0


@torch.no_grad()
def recalibrate_bn(model: nn.Module, x: torch.Tensor, batch_size: int = 500) -> None:
    """Recompute BatchNorm running statistics as an exact average over `x`."""
    bns = [m for m in model.modules() if isinstance(m, _BatchNorm)]
    momenta = [bn.momentum for bn in bns]
    for bn in bns:
        bn.reset_running_stats()
        bn.momentum = None  # cumulative average
    model.train()
    for start in range(0, len(x), batch_size):
        model(x[start:start + batch_size])
    for bn, momentum in zip(bns, momenta):
        bn.momentum = momentum
    model.eval()


# --------------------------------------------------------------------------- diagnostics

@torch.no_grad()
def weight_norms(model: nn.Module) -> Dict[str, float]:
    """Frobenius norm of every conv and linear weight, keyed by module name."""
    return {name: module.weight.norm().item() for name, module in model.named_modules()
            if isinstance(module, (nn.Conv2d, nn.Linear))}


@torch.no_grad()
def mean_logit_norm(model: nn.Module, x: torch.Tensor, batch_size: int = 1000) -> float:
    model.eval()
    norms = [model(x[i:i + batch_size]).norm(dim=1) for i in range(0, len(x), batch_size)]
    return torch.cat(norms).mean().item()


def state_dict_fp16(model: nn.Module) -> Dict[str, torch.Tensor]:
    """State dict with floating tensors stored as fp16, to halve checkpoint size."""
    return {k: (v.detach().half().cpu() if v.is_floating_point() else v.detach().cpu())
            for k, v in model.state_dict().items()}


def load_state(model: nn.Module, state: Dict[str, torch.Tensor]) -> nn.Module:
    """Load a (possibly fp16) state dict into an fp32 model."""
    reference = model.state_dict()
    model.load_state_dict({k: v.to(reference[k].dtype) for k, v in state.items()})
    return model
