import copy

import pytest
import torch
import torch.nn as nn

from experiments.cifar_unlearning.models import (
    TinyNet, is_identity_rescale, load_state, recalibrate_bn, rescale_, resnet18_cifar, set_train_mode,
    state_dict_fp16)


def test_resnet18_cifar_stem():
    model = resnet18_cifar(10).eval()
    assert model.conv1.kernel_size == (3, 3) and model.conv1.stride == (1, 1)
    assert isinstance(model.maxpool, nn.Identity)
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 10)


def test_frozen_bn_keeps_running_stats_but_trains_affine():
    torch.manual_seed(0)
    model = TinyNet()
    bn = model.features[1]
    before = bn.running_mean.clone()
    set_train_mode(model, freeze_bn=True)
    model(torch.randn(8, 3, 32, 32)).sum().backward()
    assert torch.equal(bn.running_mean, before)
    assert bn.weight.grad is not None
    set_train_mode(model, freeze_bn=False)
    model(torch.randn(8, 3, 32, 32))
    assert not torch.equal(bn.running_mean, before)


def _logits(model, x):
    model.eval()
    with torch.no_grad():
        return model(x)


def test_rescale_modes():
    torch.manual_seed(0)
    base_model = TinyNet()
    x = torch.randn(4, 3, 32, 32)
    base = _logits(base_model, x)

    for mode, kwargs in (("none", {}), ("all", {"S": 1.0}), ("body_head", {})):
        m = copy.deepcopy(base_model)
        rescale_(m, mode, **kwargs)
        assert torch.allclose(_logits(m, x), base)

    m = copy.deepcopy(base_model)
    rescale_(m, "body_head", alpha_head=2.0)  # head weight and bias doubled -> logits doubled
    assert torch.allclose(_logits(m, x), 2 * base, atol=1e-6)

    for mode in ("all", "weights"):
        m = copy.deepcopy(base_model)
        rescale_(m, mode, S=2.0)
        assert not torch.allclose(_logits(m, x), base)

    with pytest.raises(ValueError):
        rescale_(copy.deepcopy(base_model), "sideways")
    assert is_identity_rescale("weights", 1.0, 5.0, 5.0) and not is_identity_rescale("body_head", 1.0, 1.0, 2.0)


def test_recalibrate_bn_restores_momentum():
    torch.manual_seed(0)
    model = TinyNet()
    bn = model.features[1]
    rescale_(model, "weights", S=3.0)
    var_before = bn.running_var.clone()
    recalibrate_bn(model, torch.randn(64, 3, 32, 32), batch_size=16)
    assert not torch.allclose(bn.running_var, var_before)
    assert bn.momentum == 0.1 and not model.training


def test_fp16_checkpoint_roundtrip():
    torch.manual_seed(0)
    model = TinyNet().eval()
    x = torch.randn(4, 3, 32, 32)
    restored = load_state(TinyNet(), state_dict_fp16(model))
    assert next(restored.parameters()).dtype == torch.float32
    assert torch.allclose(_logits(restored, x), _logits(model, x), atol=5e-2)
