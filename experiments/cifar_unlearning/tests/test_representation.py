import pytest
import torch

from experiments.cifar_unlearning.models import TinyNet, resnet18_cifar
from experiments.cifar_unlearning.representation import (
    RepSets, knn_accuracy, linear_cka, penultimate_features, reference_features, representation_metrics)


def test_linear_cka_properties():
    torch.manual_seed(0)
    x = torch.randn(500, 32)
    q, _ = torch.linalg.qr(torch.randn(32, 32))
    assert linear_cka(x, x) == pytest.approx(1.0)
    assert linear_cka(x, 3.0 * x @ q + 5.0) == pytest.approx(1.0, abs=1e-9)  # rotation, scale, shift invariant
    assert linear_cka(x, torch.randn(500, 64)) < 0.2  # independent features
    assert 0.0 <= linear_cka(x, x[:, :16]) <= 1.0
    with pytest.raises(ValueError):
        linear_cka(x, x[:10])
    # two well separated clusters: plain CKA is near 1 for unrelated within-class structure,
    # within-class CKA is not
    g = torch.arange(2).repeat_interleave(250)
    a = torch.randn(500, 32) + 50 * g[:, None]
    b = torch.randn(500, 32) + 50 * g[:, None]
    assert linear_cka(a, b) > 0.9 and linear_cka(a, b, groups=g) < 0.2
    assert linear_cka(a, a, groups=g) == pytest.approx(1.0)


def test_knn_accuracy():
    torch.manual_seed(0)
    centres = torch.randn(3, 16) * 5
    ys = torch.arange(3).repeat_interleave(50)
    support = centres[ys] + torch.randn(150, 16) * 0.1
    yq = torch.arange(3).repeat_interleave(20)
    query = centres[yq] + torch.randn(60, 16) * 0.1
    assert knn_accuracy(support, ys, query, yq, k=5) == 1.0
    assert knn_accuracy(support, ys[torch.randperm(150)], query, yq, k=5) < 0.7
    assert knn_accuracy(support[:0], ys[:0], query, yq) != knn_accuracy(support[:0], ys[:0], query, yq)  # nan


def test_penultimate_features_shapes():
    x = torch.randn(7, 3, 32, 32)
    assert penultimate_features(resnet18_cifar(10), x, batch_size=3).shape == (7, 512)
    model = TinyNet()
    feats = penultimate_features(model, x)
    assert feats.shape == (7, 32)
    assert torch.allclose(model.fc(feats), model.eval()(x), atol=1e-5)  # features are exactly fc's input
    assert not model.fc._forward_hooks  # hook removed


def test_representation_metrics_keys_and_self_similarity():
    torch.manual_seed(0)
    x, y = torch.randn(40, 3, 32, 32), torch.randint(0, 10, (40,))
    rs = RepSets(x[:10], torch.tensor([2, 7] * 5), x[10:30], y[10:30], x[30:], torch.tensor([2, 7] * 5),
                 x[10:30], y[10:30])
    model, other = TinyNet().eval(), TinyNet().eval()
    refs = {"retrain": reference_features(model, rs), "pretrained": reference_features(other, rs)}
    out = representation_metrics(model, rs, refs, pair=(2, 7), k=3)
    assert set(out) == {"cka_retrain_r", "cka_retrain_f", "cka_retrain_f_within", "cka_pretrained_r",
                        "cka_pretrained_f", "cka_pretrained_f_within", "knn_r", "knn_f"}
    assert out["cka_retrain_r"] == pytest.approx(1.0) and out["cka_retrain_f"] == pytest.approx(1.0)
    assert representation_metrics(model, rs, refs, pair=(2, 7), k=1)["knn_r"] == 1.0  # query set = support set
    random_split = representation_metrics(model, rs, refs, pair=None, k=3)
    assert set(random_split) == {"cka_retrain_r", "cka_pretrained_r", "knn_r"}
