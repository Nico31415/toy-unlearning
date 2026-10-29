import numpy as np
import pytest

from experiments.cifar_unlearning.data import (
    experiment_name, make_split, make_synthetic, n_for_fraction, parse_experiment)

DS = make_synthetic(train_per_class=40, test_per_class=20)
Y = DS.y_train.numpy()
YT = DS.y_test.numpy()


def counts(idx, labels=Y):
    return dict(zip(*np.unique(labels[idx], return_counts=True)))


def test_class_split_sets():
    s = make_split(DS, "pair-2-7", seed=0, n_probe=20)
    assert s.kind == "class" and s.pair == (2, 7)
    assert set(Y[s.forget_all]) == {2, 7} and len(s.forget_all) == 80
    assert counts(s.probe) == {2: 10, 7: 10}
    pool = s.forget_pool_order
    assert len(pool) == 60 and len(set(pool)) == 60
    assert not set(pool) & set(s.probe)
    assert set(pool) | set(s.probe) == set(s.forget_all)
    retain = s.retain_order
    assert len(retain) == 320 and not set(Y[retain]) & {2, 7}
    assert set(retain) | set(s.forget_all) == set(range(len(Y)))
    assert set(YT[s.test_forget]) == {2, 7} and not set(YT[s.test_retain]) & {2, 7}


def test_random_split_is_stratified():
    s = make_split(DS, "random", seed=0, n_forget=100, n_probe=20)
    assert counts(s.forget_all) == {c: 10 for c in range(10)}
    assert counts(s.probe) == {c: 2 for c in range(10)}
    assert len(s.test_forget) == 0 and len(s.test_retain) == len(YT)
    assert not set(s.retain_order) & set(s.forget_all)


def test_subsets_are_nested_and_balanced():
    s = make_split(DS, "pair-0-1", seed=3, n_probe=20)
    fractions = [0.1, 0.3, 0.5, 1.0]
    for small, large in zip(fractions, fractions[1:]):
        assert set(s.retain_subset(small)) <= set(s.retain_subset(large))
        assert set(s.forget_subset(small)) <= set(s.forget_subset(large))
    assert set(counts(s.retain_subset(0.1)).values()) == {4}  # 32 of 320, 8 classes
    assert set(counts(s.forget_subset(0.5)).values()) == {15}  # 30 of 60, 2 classes
    assert len(s.forget_subset(1.0)) == 60


def test_seed_controls_split():
    a = make_split(DS, "random", seed=1, n_forget=100, n_probe=20)
    b = make_split(DS, "random", seed=1, n_forget=100, n_probe=20)
    c = make_split(DS, "random", seed=2, n_forget=100, n_probe=20)
    assert np.array_equal(a.forget_all, b.forget_all) and np.array_equal(a.retain_order, b.retain_order)
    assert not np.array_equal(a.forget_all, c.forget_all)
    p1 = make_split(DS, "pair-0-1", seed=1, n_probe=20)
    p2 = make_split(DS, "pair-0-1", seed=2, n_probe=20)
    assert np.array_equal(p1.forget_all, p2.forget_all)  # the pair fixes D_f
    assert not np.array_equal(p1.probe, p2.probe)


def test_names_and_validation():
    assert parse_experiment("pair-7-2") == ("class", (2, 7))
    assert experiment_name("class", (7, 2)) == "pair-2-7"
    assert parse_experiment("random") == ("random", None)
    for bad in ("pair-3-3", "cats"):
        with pytest.raises(ValueError):
            parse_experiment(bad)
    with pytest.raises(ValueError):
        make_split(DS, "pair-0-1", seed=0, n_probe=80)
    assert n_for_fraction(0.01, 8000) == 80 and n_for_fraction(1e-6, 10) == 1
    with pytest.raises(ValueError):
        n_for_fraction(0.0, 10)
