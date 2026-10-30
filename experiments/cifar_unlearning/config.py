"""Hyperparameters for the CIFAR unlearning and relearning experiments.

Each dataclass groups one stage. `add_config_args` turns every field into a
command-line flag with the group's prefix (for example `--ul_lr`), and every run
saves its resolved configs in result.json, so changing a default here never loses
track of what an earlier run used.
"""
from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass
from typing import Any, Dict

DEFAULT_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_FRACTIONS = (0.01, 0.03, 0.1, 0.3, 1.0)
DEFAULT_MU_F = (0.25, 0.5, 0.75)


@dataclass
class DataConfig:
    data_root: str = "data/cifar10"
    # Random split only. The class split removes every training image of the pair.
    n_forget: int = 10000
    # Removed images held out of unlearning and relearning; used for evaluation only.
    n_probe: int = 2000
    # Tests only: replace CIFAR-10 by a small, learnable synthetic set.
    synthetic: bool = False
    synthetic_train_per_class: int = 40
    synthetic_test_per_class: int = 20


@dataclass
class ModelConfig:
    arch: str = "resnet18"  # "resnet18" (CIFAR stem) or "tiny" (tests only)
    num_classes: int = 10


@dataclass
class ScratchConfig:
    """Training from random init: Pretrain and Retrain."""
    epochs: int = 60
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    batch_size: int = 128
    # The run is marked failed (non-zero exit) if the final train loss is above this.
    max_final_train_loss: float = 1e-2


@dataclass
class UnlearnConfig:
    mu_r: float = 1.0
    mu_f: float = 0.5
    lr: float = 0.01
    momentum: float = 0.9
    batch_size: int = 128
    max_steps: int = 5000
    # Ascent can blow up within tens of steps, so the stop rule is checked often.
    eval_every: int = 10
    # The stop rule's retain loss is measured on at most this many of the n_r images
    # (a stratified prefix), to keep frequent checks cheap at large n_r.
    retain_eval_max: int = 2000
    retain_loss_threshold: float = 1e-2
    forget_acc_threshold: float = 0.01
    # "both": stop when forget accuracy <= forget_acc_threshold AND retain loss <
    # retain_loss_threshold (the agreed rule). "forget": stop as soon as forget
    # accuracy <= forget_acc_threshold; retain damage is then a measured outcome.
    stop_rule: str = "both"
    # Safety net for plain -CE, which can blow up: if the retain loss at a check exceeds
    # this (well above chance, ln 10 = 2.3), roll back to the previous check and stop
    # with status "collapsed".
    retain_loss_abort: float = 5.0
    freeze_bn: bool = True


@dataclass
class RelearnConfig:
    lr: float = 0.01
    momentum: float = 0.9
    batch_size: int = 128
    max_steps: int = 5000
    eval_every: int = 25
    loss_threshold: float = 1e-2
    # Scratch ignores this and always trains BatchNorm normally.
    freeze_bn: bool = True


@dataclass
class RescaleConfig:
    """Rescaling applied to Pretrained before unlearning. Identity by default."""
    mode: str = "none"  # none | all | weights | body_head
    S: float = 1.0  # factor for "all" and "weights"
    alpha_body: float = 1.0  # "body_head": factor on every conv weight
    alpha_head: float = 1.0  # "body_head": factor on the final linear layer
    # Recompute BatchNorm running statistics on the n_r retain images after rescaling.
    recalibrate_bn: bool = False


@dataclass
class RepresentationConfig:
    """Post-hoc representation analysis (the `represent` stage)."""
    knn_k: int = 20
    # k-NN support for the kept classes: at most this many kept training images
    # (a stratified prefix of D_r).
    max_support_retain: int = 5000


# group name -> (dataclass, flag prefix)
GROUPS = {
    "data": (DataConfig, ""),
    "model": (ModelConfig, ""),
    "pt": (ScratchConfig, "pt_"),
    "ul": (UnlearnConfig, "ul_"),
    "rl": (RelearnConfig, "rl_"),
    "rs": (RescaleConfig, "rs_"),
    "rep": (RepresentationConfig, "rep_"),
}


def _str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in ("1", "true", "yes", "y"):
        return True
    if lowered in ("0", "false", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {value!r}")


def add_config_args(parser: argparse.ArgumentParser) -> None:
    for cls, prefix in GROUPS.values():
        for field in dataclasses.fields(cls):
            kind = _str2bool if isinstance(field.default, bool) else type(field.default)
            parser.add_argument(f"--{prefix}{field.name}", type=kind, default=field.default)


def build_configs(args: argparse.Namespace) -> Dict[str, Any]:
    configs = {}
    for name, (cls, prefix) in GROUPS.items():
        values = {f.name: getattr(args, f"{prefix}{f.name}") for f in dataclasses.fields(cls)}
        configs[name] = cls(**values)
    return configs


def configs_to_dict(configs: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {name: dataclasses.asdict(cfg) for name, cfg in configs.items()}
