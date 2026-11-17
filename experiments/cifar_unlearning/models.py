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


class PlainNet(nn.Module):
    """A positively homogeneous CNN: no BatchNorm, no biases, ReLU and max pooling only.

    Every path from input to output passes through every weight layer once, so multiplying every weight by c
    multiplies the logits by c^L (L = number of weight layers). A ResNet has no such property: its identity skips
    bypass the residual branch's weights. This is the network in which the toy's rescaling (function times S) is
    exact: `rescale_(mode="function")` multiplies each weight by S^(1/L).
    `widths` lists conv widths, "M" a 2x2 max pool; the classifier is one bias-free Linear named `fc`.
    """

    def __init__(self, widths, num_classes: int = 10):
        super().__init__()
        layers, c = [], 3
        for w in widths:
            if w == "M":
                layers.append(nn.MaxPool2d(2))
            else:
                conv = nn.Conv2d(c, w, 3, padding=1, bias=False)
                nn.init.kaiming_normal_(conv.weight, mode="fan_in", nonlinearity="relu")
                layers += [conv, nn.ReLU()]
                c = w
        layers += [nn.AdaptiveMaxPool2d(1), nn.Flatten()]
        self.features = nn.Sequential(*layers)
        self.fc = nn.Linear(c, num_classes, bias=False)
        nn.init.kaiming_normal_(self.fc.weight, mode="fan_in", nonlinearity="linear")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.features(x))


VGG11 = [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]


def build_model(arch: str, num_classes: int) -> nn.Module:
    if arch == "resnet18":
        return resnet18_cifar(num_classes)
    if arch == "tiny":
        return TinyNet(num_classes)
    if arch == "vgg11_plain":  # homogeneous arm (no BatchNorm, no biases)
        return PlainNet(VGG11, num_classes)
    if arch == "tiny_plain":  # homogeneous, tests only
        return PlainNet([16, "M", 32], num_classes)
    raise ValueError(f"unknown arch {arch!r}")


def homogeneity_degree(model: nn.Module) -> int:
    """L such that scaling every weight by c scales the output by c^L; raises if the model is not homogeneous
    (BatchNorm, biases or residual connections)."""
    if not isinstance(model, PlainNet):
        raise ValueError("only PlainNet is positively homogeneous (no BatchNorm, biases or skips)")
    return sum(isinstance(m, (nn.Conv2d, nn.Linear)) for m in model.modules())


# --------------------------------------------------------------------------- feature taps

def feature_taps(model: nn.Module) -> Dict[str, nn.Module]:
    """Modules whose outputs the layer-wise probes read, in depth order. Each output is global-average-pooled
    over space (or used as is when already flat). ResNet-18: the stem (after its ReLU) and the four residual
    stages; the last stage pooled is the penultimate representation. PlainNet / TinyNet: after every pooling."""
    from torchvision.models.resnet import ResNet

    if isinstance(model, ResNet):
        return {"stem": model.relu, "layer1": model.layer1, "layer2": model.layer2, "layer3": model.layer3,
                "layer4": model.layer4}
    taps, k = {}, 0
    for module in model.features:
        if isinstance(module, (nn.MaxPool2d, nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d)):
            k += 1
            taps[f"block{k}"] = module
    return taps


def set_train_mode(model: nn.Module, freeze_bn: bool) -> None:
    """Training mode; with `freeze_bn`, BatchNorm layers use and keep their running
    statistics while their affine parameters still receive gradients."""
    model.train()
    if freeze_bn:
        for module in model.modules():
            if isinstance(module, _BatchNorm):
                module.eval()


# --------------------------------------------------------------------------- rescaling

RESCALE_MODES = ("none", "all", "weights", "body_head", "function")


def _head(model: nn.Module) -> nn.Linear:
    head = getattr(model, "fc", None)
    if not isinstance(head, nn.Linear):
        raise ValueError("rescaling expects the final layer to be a Linear module named `fc`")
    return head


@torch.no_grad()
def rescale_(model: nn.Module, mode: str = "none", S: float = 1.0,
             alpha_body: float = 1.0, alpha_head: float = 1.0, scale_bn_stats: bool = False) -> None:
    """Rescale a trained model in place, before unlearning.

    none       no change
    all        every parameter times S, including biases and BatchNorm affine
               parameters (the `finetune_scaling` convention of Lippl & Lindsey)
    weights    every conv and linear weight times S; biases and BatchNorm untouched
    body_head  every conv weight times alpha_body and the final linear layer
               (weight and bias) times alpha_head; the analogue of the toy's
               (alpha_in, alpha_out) layer rescaling
    function   the network's function times S, the toy's rescaling (beta -> S beta): in a homogeneous PlainNet
               every weight times S^(1/L), spread over all layers like the toy's sqrt(S) on both layers; in
               any other network (BatchNorm, residual) only the head (weight and bias) times S, the one way to
               scale the logits exactly

    BatchNorm running statistics are buffers and are never touched here. With frozen
    BatchNorm they are fixed, so rescaling conv weights changes the network's
    function; use `recalibrate_bn` to recompute them from data, or `scale_bn_stats` to scale them
    exactly (mean times c, variance times c^2, with c the factor on the BN's pre-activation: the conv's factor, and
    for "all" an extra S on every BN but the stem's, because every BN output is scaled by S there; the BN affine
    parameters are scaled as well, as always).
    """
    if mode not in RESCALE_MODES:
        raise ValueError(f"unknown rescale mode {mode!r}; expected one of {RESCALE_MODES}")
    if scale_bn_stats:
        # c = the factor on a BN's pre-activation: the conv weights' factor times that of the conv's input. In `all` every BN
        # output is scaled by S too (gamma, beta), so the input of every conv but the first (the image) is scaled by S.
        conv = {"all": S, "weights": S, "body_head": alpha_body}.get(mode, 1.0)
        inputs = S if mode == "all" else 1.0
        stem = True
        for module in model.modules():
            if isinstance(module, _BatchNorm) and module.running_mean is not None:
                c = conv * (1.0 if stem else inputs)
                module.running_mean.mul_(c)
                module.running_var.mul_(c * c)
                stem = False
    if mode == "function":
        if isinstance(model, PlainNet):
            factor = S ** (1.0 / homogeneity_degree(model))
            for module in model.modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    module.weight.mul_(factor)
        else:
            head = _head(model)
            head.weight.mul_(S)
            if head.bias is not None:
                head.bias.mul_(S)
        return
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
    if mode in ("all", "weights", "function"):
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
