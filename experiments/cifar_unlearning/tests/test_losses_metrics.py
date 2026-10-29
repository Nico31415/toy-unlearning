import pytest
import torch
import torch.nn.functional as F

from experiments.cifar_unlearning.losses import (
    classification_loss, signed_unlearning_loss, unlearning_retain_loss)
from experiments.cifar_unlearning.metrics import accuracy, mia_auc, two_way_accuracy


def test_two_way_accuracy_ignores_other_classes():
    logits = torch.tensor([[0.0, 5.0, 1.0], [0.0, 1.0, 5.0], [9.0, 1.0, 2.0]])
    y = torch.tensor([1, 2, 2])
    assert accuracy(logits, y) == pytest.approx(2 / 3)  # row 3 predicts class 0 10-way
    assert two_way_accuracy(logits, y, (1, 2)) == 1.0  # but picks class 2 among {1, 2}


def test_pair_loss_uses_only_pair_logits():
    logits = torch.tensor([[100.0, 0.0, 2.0], [-50.0, 3.0, 0.0]])
    y = torch.tensor([2, 1])
    expected = F.cross_entropy(logits[:, [1, 2]], torch.tensor([1, 0]))
    assert torch.allclose(classification_loss(logits, y, "ce", pair=(1, 2)), expected)


def test_signed_ce():
    torch.manual_seed(0)
    lr, lf = torch.randn(5, 10), torch.randn(4, 10)
    yr, yf = torch.randint(0, 10, (5,)), torch.randint(0, 10, (4,))
    expected = 1.0 * F.cross_entropy(lr, yr) - 0.25 * F.cross_entropy(lf, yf)
    assert torch.allclose(signed_unlearning_loss(lr, yr, lf, yf, 1.0, 0.25, "ce"), expected)


def test_mse_effective_targets():
    yr, yf = torch.tensor([0, 3]), torch.tensor([5])
    # mu_r = 1, mu_f = 0.5 -> rho = 1: retain targets 2 * onehot, forget targets -1 * onehot
    lr = 2.0 * F.one_hot(yr, 10).float()
    lf = -1.0 * F.one_hot(yf, 10).float()
    assert signed_unlearning_loss(lr, yr, lf, yf, 1.0, 0.5, "mse").item() == pytest.approx(0.0)
    assert signed_unlearning_loss(lr + 0.1, yr, lf, yf, 1.0, 0.5, "mse").item() > 0
    with pytest.raises(ValueError):
        signed_unlearning_loss(lr, yr, lf, yf, 0.5, 0.5, "mse")


def test_mia_auc():
    low, high = torch.tensor([0.1, 0.2]), torch.tensor([0.5, 0.6, 0.7])
    assert mia_auc(low, high) == 1.0
    assert mia_auc(high, low) == 0.0
    assert mia_auc(low, low) == 0.5
    torch.manual_seed(0)
    a, b = torch.rand(3000), torch.rand(3000)
    assert abs(mia_auc(a, b) - 0.5) < 0.03


def test_unlearning_retain_loss_matches_effective_target():
    y = torch.tensor([1, 4])
    # mu_r = 1, mu_f = 0.5 -> rho = 1: the retain target is 2 * onehot
    assert unlearning_retain_loss(2.0 * F.one_hot(y, 10).float(), y, 1.0, 0.5, "mse").item() == 0.0
    logits = torch.randn(2, 10)
    assert torch.allclose(unlearning_retain_loss(logits, y, 1.0, 0.5, "ce"), F.cross_entropy(logits, y))
