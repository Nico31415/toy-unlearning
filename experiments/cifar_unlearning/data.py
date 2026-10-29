"""CIFAR-10 loading, the class and random splits, and nested stratified subsamples.

Notation (see the plan doc):
    D    all 50,000 training images (pretraining)
    D_f  removed images: every image of the forgotten pair (class split) or
         n_forget random images, stratified by class (random split)
    D_r  kept images, D minus D_f
    P_f  probe: n_probe images of D_f that unlearning and relearning never see;
         used only to measure memorisation of removed images
    forget pool  D_f minus P_f; unlearning and relearning subsample from it
    T_r, T_f  test images of the kept and forgotten classes (class split);
         in the random split T_r is the whole test set and T_f is empty
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CACHE_NAME = "cifar10_uint8.pt"


@dataclass
class Dataset:
    x_train: torch.Tensor  # uint8, (N, 3, 32, 32)
    y_train: torch.Tensor  # int64, (N,)
    x_test: torch.Tensor
    y_test: torch.Tensor
    num_classes: int


def load_cifar10(root: str, download: bool = False) -> Dataset:
    """Load CIFAR-10 as uint8 tensors, caching them in `root` after the first call."""
    cache = os.path.join(root, CACHE_NAME)
    if os.path.exists(cache):
        blob = torch.load(cache)
        return Dataset(blob["x_train"], blob["y_train"], blob["x_test"], blob["y_test"], 10)
    from torchvision import datasets  # imported lazily: tests never need it

    tensors = {}
    for split, train in (("train", True), ("test", False)):
        ds = datasets.CIFAR10(root=root, train=train, download=download)
        tensors[f"x_{split}"] = torch.from_numpy(np.asarray(ds.data)).permute(0, 3, 1, 2).contiguous()
        tensors[f"y_{split}"] = torch.as_tensor(ds.targets, dtype=torch.int64)
    os.makedirs(root, exist_ok=True)
    torch.save(tensors, cache)
    return Dataset(tensors["x_train"], tensors["y_train"], tensors["x_test"], tensors["y_test"], 10)


def make_synthetic(train_per_class: int, test_per_class: int, num_classes: int = 10,
                   seed: int = 0, noise: float = 40.0) -> Dataset:
    """Small learnable stand-in for CIFAR-10: one random prototype image per class plus noise."""
    gen = torch.Generator().manual_seed(seed)
    # Per-pixel pattern plus a strong per-class colour, so that even a small network
    # with global pooling can separate the classes.
    colours = torch.rand((num_classes, 3, 1, 1), generator=gen) * 120 - 60
    protos = (torch.randint(60, 196, (num_classes, 3, 32, 32), generator=gen).float() + colours)

    def sample(per_class: int) -> Tuple[torch.Tensor, torch.Tensor]:
        y = torch.arange(num_classes).repeat_interleave(per_class)
        x = protos[y] + noise * torch.randn((len(y), 3, 32, 32), generator=gen)
        return x.clamp(0, 255).to(torch.uint8), y

    x_train, y_train = sample(train_per_class)
    x_test, y_test = sample(test_per_class)
    return Dataset(x_train, y_train, x_test, y_test, num_classes)


def normalize(x_uint8: torch.Tensor, device: torch.device) -> torch.Tensor:
    """uint8 images -> normalised float32 on `device`."""
    mean = torch.tensor(CIFAR10_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD, device=device).view(1, 3, 1, 1)
    return (x_uint8.to(device).float().div_(255.0) - mean) / std


def stratified_order(indices: np.ndarray, labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Order `indices` so that every prefix is as class-balanced as possible.

    Each class is shuffled, then classes are interleaved round-robin (in a random
    class order). Taking the first n of the result gives a stratified sample, and
    the sample for a smaller n is always contained in the sample for a larger n.
    """
    indices = np.asarray(indices)
    classes = np.unique(labels[indices])
    rng.shuffle(classes)
    per_class = []
    for c in classes:
        members = indices[labels[indices] == c].copy()
        rng.shuffle(members)
        per_class.append(members)
    longest = max(len(m) for m in per_class)
    out = [m[k] for k in range(longest) for m in per_class if k < len(m)]
    return np.asarray(out, dtype=np.int64)


def n_for_fraction(fraction: float, total: int) -> int:
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    return max(1, int(round(fraction * total)))


@dataclass
class Split:
    kind: str  # "class" or "random"
    pair: Optional[Tuple[int, int]]
    forget_all: np.ndarray  # D_f
    probe: np.ndarray  # P_f
    forget_pool_order: np.ndarray  # D_f minus P_f, nested stratified order
    retain_order: np.ndarray  # D_r, nested stratified order
    test_retain: np.ndarray  # T_r (whole test set in the random split)
    test_forget: np.ndarray  # T_f (empty in the random split)

    def retain_subset(self, fraction: float) -> np.ndarray:
        return self.retain_order[: n_for_fraction(fraction, len(self.retain_order))]

    def forget_subset(self, fraction: float) -> np.ndarray:
        return self.forget_pool_order[: n_for_fraction(fraction, len(self.forget_pool_order))]


SPLIT_ARRAYS = ("forget_all", "probe", "forget_pool_order", "retain_order", "test_retain", "test_forget")


def save_split(split: Split, path: str) -> None:
    """Write a split atomically, so every stage of every job uses identical indices."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp.npz"
    np.savez(tmp, kind=split.kind, pair=np.asarray(split.pair if split.pair else (-1, -1)),
             **{name: getattr(split, name) for name in SPLIT_ARRAYS})
    os.replace(tmp, path)


def load_split(path: str) -> Split:
    with np.load(path) as blob:
        pair = tuple(int(v) for v in blob["pair"])
        return Split(kind=str(blob["kind"]), pair=None if pair == (-1, -1) else pair,
                     **{name: blob[name] for name in SPLIT_ARRAYS})


def experiment_name(kind: str, pair: Optional[Tuple[int, int]] = None) -> str:
    if kind == "class":
        a, b = sorted(pair)
        return f"pair-{a}-{b}"
    if kind == "random":
        return "random"
    raise ValueError(f"unknown split kind {kind!r}")


def parse_experiment(name: str) -> Tuple[str, Optional[Tuple[int, int]]]:
    if name == "random":
        return "random", None
    if name.startswith("pair-"):
        a, b = (int(v) for v in name[len("pair-"):].split("-"))
        if a == b:
            raise ValueError(f"a pair needs two different classes, got {name!r}")
        return "class", (min(a, b), max(a, b))
    raise ValueError(f"unknown experiment {name!r}; expected 'random' or 'pair-A-B'")


def make_split(ds: Dataset, experiment: str, seed: int, n_forget: int = 10000,
               n_probe: int = 2000) -> Split:
    """Build the split for one experiment and seed.

    The forgotten pair is fixed by `experiment`; the seed changes the probe set and
    the subsample orders, and in the random split it also redraws D_f.
    """
    kind, pair = parse_experiment(experiment)
    y_train = ds.y_train.numpy()
    y_test = ds.y_test.numpy()
    all_train = np.arange(len(y_train))
    all_test = np.arange(len(y_test))
    rng = np.random.default_rng([seed, 0 if kind == "class" else 1])

    if kind == "class":
        for c in pair:
            if not 0 <= c < ds.num_classes:
                raise ValueError(f"class {c} out of range for {ds.num_classes} classes")
        forget_all = np.sort(all_train[np.isin(y_train, pair)])
        test_forget = all_test[np.isin(y_test, pair)]
        test_retain = all_test[~np.isin(y_test, pair)]
    else:
        if n_forget >= len(all_train):
            raise ValueError("n_forget must be smaller than the training set")
        forget_all = np.sort(stratified_order(all_train, y_train, rng)[:n_forget])
        test_forget = np.empty(0, dtype=np.int64)
        test_retain = all_test

    if not 0 < n_probe < len(forget_all):
        raise ValueError(f"n_probe={n_probe} must be between 1 and |D_f|-1={len(forget_all) - 1}")
    probe = np.sort(stratified_order(forget_all, y_train, rng)[:n_probe])
    pool = np.setdiff1d(forget_all, probe)
    retain = np.setdiff1d(all_train, forget_all)

    return Split(
        kind=kind,
        pair=pair,
        forget_all=forget_all,
        probe=probe,
        forget_pool_order=stratified_order(pool, y_train, rng),
        retain_order=stratified_order(retain, y_train, rng),
        test_retain=test_retain,
        test_forget=test_forget,
    )
