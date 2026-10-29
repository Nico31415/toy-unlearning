import copy

import pytest
import torch

from experiments.cifar_unlearning.config import RelearnConfig, ScratchConfig, UnlearnConfig
from experiments.cifar_unlearning.data import make_synthetic, normalize
from experiments.cifar_unlearning.models import TinyNet
from experiments.cifar_unlearning.train import _index_stream, relearn, train_from_scratch, unlearn

DEVICE = torch.device("cpu")
DS = make_synthetic(train_per_class=30, test_per_class=10)
X = normalize(DS.x_train, DEVICE)
Y = DS.y_train
FORGET = (Y == 0) | (Y == 1)
quiet = lambda *a, **k: None  # noqa: E731


def pretrained():
    torch.manual_seed(0)
    model = TinyNet()
    cfg = ScratchConfig(epochs=8, lr=0.05, batch_size=32, max_final_train_loss=10.0)
    train_from_scratch(model, X, Y, cfg, seed=0, log=quiet)
    return model


MODEL = pretrained()


def test_index_stream_gives_full_batches():
    stream = _index_stream(5, 3, torch.Generator().manual_seed(0), DEVICE)
    batches = [next(stream) for _ in range(5)]
    assert all(len(b) == 3 for b in batches)
    assert set(torch.cat(batches[:2]).tolist()) >= set(range(4))


def test_unlearn_stop_rule():
    cfg = UnlearnConfig(mu_f=0.5, lr=0.01, batch_size=32, max_steps=300, eval_every=2,
                        retain_loss_threshold=10.0, forget_acc_threshold=0.0)
    model = copy.deepcopy(MODEL)
    out = unlearn(model, X[~FORGET], Y[~FORGET], X[FORGET], Y[FORGET], cfg, seed=0, log=quiet)
    assert out["status"] == "success" and out["final_forget_acc"] == 0.0
    assert out["steps"] % 2 == 0 and out["history"][0]["step"] == 0

    impossible = UnlearnConfig(mu_f=0.5, lr=0.001, batch_size=32, max_steps=30, eval_every=10,
                               retain_loss_threshold=0.0)
    out = unlearn(copy.deepcopy(MODEL), X[~FORGET], Y[~FORGET], X[FORGET], Y[FORGET], impossible,
                  seed=0, log=quiet)
    assert out["status"] == "step_cap" and out["steps"] == 30


def test_relearn_two_way_and_curve():
    cfg = RelearnConfig(lr=0.05, batch_size=16, max_steps=400, eval_every=5, loss_threshold=0.05)
    torch.manual_seed(1)
    out = relearn(TinyNet(), X[FORGET], Y[FORGET], cfg, seed=0, freeze_bn=False, pair=(0, 1),
                  eval_fn=lambda m: {"marker": 1.0}, log=quiet)
    assert out["status"] == "converged" and out["final_train_loss"] < 0.05
    steps = [p["step"] for p in out["curve"]]
    assert steps[0] == 0 and steps[-1] == out["steps"] and steps == sorted(steps)

    easy = RelearnConfig(loss_threshold=100.0)
    out = relearn(copy.deepcopy(MODEL), X[FORGET], Y[FORGET], easy, seed=0, freeze_bn=True, pair=(0, 1),
                  log=quiet)
    assert out["status"] == "converged" and out["steps"] == 0


def test_unlearn_divergence_rolls_back():
    cfg = UnlearnConfig(mu_f=0.9, lr=0.5, batch_size=32, max_steps=500, eval_every=1,
                        retain_loss_threshold=0.0)
    model = copy.deepcopy(MODEL)
    out = unlearn(model, X[~FORGET], Y[~FORGET], X[FORGET], Y[FORGET], cfg, seed=0, log=quiet)
    assert out["status"] in ("collapsed", "diverged") and out["rollback_step"] is not None
    assert all(torch.isfinite(p).all() for p in model.parameters())
    # the model is left in the state of the last accepted check
    kept = [p for p in out["history"] if not p.get("rejected")][-1]
    assert kept["step"] == out["rollback_step"] and kept["retain_loss"] <= cfg.retain_loss_abort
    assert out["final_retain_loss"] == kept["retain_loss"]


def test_collapse_guard_and_forget_only_rule():
    base = dict(mu_f=0.75, lr=0.01, batch_size=32, max_steps=500, eval_every=2)
    out = unlearn(copy.deepcopy(MODEL), X[~FORGET], Y[~FORGET], X[FORGET], Y[FORGET],
                  UnlearnConfig(**base, retain_loss_abort=2.5), seed=0, log=quiet)
    assert out["status"] in ("collapsed", "success")
    if out["status"] == "collapsed":
        assert out["history"][-1]["rejected"] and out["history"][-1]["retain_loss"] > 2.5
    fast = unlearn(copy.deepcopy(MODEL), X[~FORGET], Y[~FORGET], X[FORGET], Y[FORGET],
                   UnlearnConfig(**base, stop_rule="forget", retain_loss_abort=1e9), seed=0, log=quiet)
    assert fast["status"] == "success" and fast["final_forget_acc"] <= 0.01
    assert fast["steps"] == fast["forget_reached_step"]
    with pytest.raises(ValueError):
        unlearn(copy.deepcopy(MODEL), X[~FORGET], Y[~FORGET], X[FORGET], Y[FORGET],
                UnlearnConfig(stop_rule="sometimes"), seed=0, log=quiet)
