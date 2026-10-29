"""load_cifar10 against files in torchvision's on-disk CIFAR-10 format.

The real dataset cannot be downloaded in the test environment, so this builds the
same pickled batch files torchvision reads (with known pixel values) and bypasses
only the MD5 checks.
"""
import os
import pickle

import numpy as np
import pytest
import torch

from experiments.cifar_unlearning.data import CACHE_NAME, CIFAR10_MEAN, CIFAR10_STD, load_cifar10, normalize

torchvision = pytest.importorskip("torchvision")


def _fake_cifar(root, n_per_batch=20):
    folder = os.path.join(root, "cifar-10-batches-py")
    os.makedirs(folder)
    rng = np.random.default_rng(0)
    expected = {"x_train": [], "y_train": [], "x_test": [], "y_test": []}
    names = [f"data_batch_{i}" for i in range(1, 6)] + ["test_batch"]
    for name in names:
        images = rng.integers(0, 256, size=(n_per_batch, 3, 32, 32), dtype=np.uint8)
        labels = rng.integers(0, 10, size=n_per_batch).tolist()
        # torchvision stores each image as 3072 bytes: all R, then all G, then all B, row-major.
        blob = {b"data": images.reshape(n_per_batch, 3072), b"labels": labels}
        with open(os.path.join(folder, name), "wb") as fh:
            pickle.dump({k.decode(): v for k, v in blob.items()}, fh)
        split = "test" if name == "test_batch" else "train"
        expected[f"x_{split}"].append(images)
        expected[f"y_{split}"] += labels
    with open(os.path.join(folder, "batches.meta"), "wb") as fh:
        pickle.dump({"label_names": [f"c{i}" for i in range(10)]}, fh)
    return {k: (np.concatenate(v) if k.startswith("x") else np.asarray(v)) for k, v in expected.items()}


def test_load_cifar10_layout_and_cache(tmp_path, monkeypatch):
    expected = _fake_cifar(str(tmp_path))
    monkeypatch.setattr(torchvision.datasets.CIFAR10, "_check_integrity", lambda self: True)
    monkeypatch.setattr(torchvision.datasets.cifar, "check_integrity", lambda *a, **k: True)

    ds = load_cifar10(str(tmp_path))
    assert ds.x_train.dtype == torch.uint8 and ds.x_train.shape == (100, 3, 32, 32)
    assert ds.x_test.shape == (20, 3, 32, 32) and ds.num_classes == 10
    assert np.array_equal(ds.x_train.numpy(), expected["x_train"])  # channel and pixel order preserved
    assert np.array_equal(ds.x_test.numpy(), expected["x_test"])
    assert np.array_equal(ds.y_train.numpy(), expected["y_train"]) and ds.y_train.dtype == torch.int64
    assert os.path.exists(tmp_path / CACHE_NAME)

    def no_torchvision(*a, **k):
        raise AssertionError("the cache should have been used")

    monkeypatch.setattr(torchvision.datasets, "CIFAR10", no_torchvision)
    cached = load_cifar10(str(tmp_path))
    assert torch.equal(cached.x_train, ds.x_train) and torch.equal(cached.y_test, ds.y_test)


def test_missing_dataset_is_a_clear_error(tmp_path):
    with pytest.raises(RuntimeError, match="not found"):
        load_cifar10(str(tmp_path), download=False)


def test_normalize():
    x = torch.full((1, 3, 32, 32), 255, dtype=torch.uint8)
    out = normalize(x, torch.device("cpu"))
    for c in range(3):
        assert out[0, c, 0, 0].item() == pytest.approx((1 - CIFAR10_MEAN[c]) / CIFAR10_STD[c], rel=1e-6)
