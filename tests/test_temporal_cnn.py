"""CPU numeric gradients, deterministic fitting, masks, grouping and inference safety."""

from __future__ import annotations

import json

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from data.audio_io import write_audio
from data.invariants import ARCHIVE_SR
from detectors.temporal_cnn import (
    FIELDS,
    GEOMETRY,
    PARAMETERS,
    SmallCNN,
    extract_logmel,
    load_cnn,
    loss_weights,
    predict_cnn_file,
    save_cnn,
    train_cnn,
)
from detectors.temporal_features import assign_window_labels, guarded_labels, sample_labels
from experiments.learned_temporal import extract_split, inner_groups, learned_plan, sampled_training


def fixture_arrays():
    rng = np.random.default_rng(2)
    y = np.tile(np.repeat([0, 1], 4), 4)
    groups = np.repeat(["a", "b", "c", "d"], 8)
    x = rng.normal(size=(32, 24, 7)) + (2 * y[:, None, None] - 1)
    return x, y, groups


def test_every_trainable_gradient_matches_independent_finite_difference():
    rng = np.random.default_rng(44)
    x = rng.normal(size=(4, 24, 7))
    model = SmallCNN.initialize(x, 9)
    y, weights = np.array([0, 1, 1, 0]), np.array([0.5, 1, 1.5, 1])
    mask = (rng.random((4, 440)) >= 0.1) / 0.9
    with threadpool_limits(limits=1):
        _, gradient = model.loss_grad(x, y, weights, dropout_mask=mask)
        assert sum(getattr(model, k).size for k in PARAMETERS) == 521
        for key in PARAMETERS:
            parameter = getattr(model, key)
            for index in np.ndindex(parameter.shape):
                original = parameter[index]
                parameter[index] = original + 1e-6
                positive = model.loss_grad(x, y, weights, dropout_mask=mask)[0]
                parameter[index] = original - 1e-6
                negative = model.loss_grad(x, y, weights, dropout_mask=mask)[0]
                parameter[index] = original
                assert gradient[key][index] == pytest.approx(
                    (positive - negative) / 2e-6, abs=3e-8, rel=1e-4
                ), (key, index)


def test_loss_weighting_equalizes_each_source_and_class():
    y = np.array([0, 0, 1, 0, 1, 1, 1])
    groups = np.array(["a"] * 3 + ["b"] * 4)
    w = loss_weights(y, groups)
    assert w.mean() == pytest.approx(1)
    for group in ("a", "b"):
        for label in (0, 1):
            assert w[(groups == group) & (y == label)].sum() == pytest.approx(len(y) / 4)
    with pytest.raises(ValueError):
        loss_weights([0, 0], ["a", "b"])


def test_optimizer_normalization_determinism_and_inner_group_isolation():
    x, y, groups = fixture_arrays()
    with threadpool_limits(limits=1):
        model, first = train_cnn(x, y, groups, 4, epochs=3)
        other, second = train_cnn(x, y, groups, 4, epochs=3)
        assert first == second and first["selected_epoch"] == 3
        for key in FIELDS:
            np.testing.assert_array_equal(getattr(model, key), getattr(other, key))
        np.testing.assert_allclose(model.mean, x.mean(axis=(0, 2)), atol=1e-14)
        assert (
            first["history"][-1]["training_dropout_objective"]
            < first["history"][0]["training_dropout_objective"]
        )
        with pytest.raises(ValueError, match="overlap"):
            train_cnn(x, y, groups, 4, validation=(x, y, groups))
        take = groups != "d"
        _, fit = train_cnn(
            x[take], y[take], groups[take], 4, validation=(x[~take], y[~take], groups[~take])
        )
        assert 1 <= fit["selected_epoch"] <= len(fit["history"]) <= 20
    inner_fit, held = inner_groups([f"g{i}" for i in range(82)])
    assert len(inner_fit) == 65 and len(held) == 17 and not inner_fit & held
    assert (inner_fit, held) == inner_groups(list(reversed([f"g{i}" for i in range(82)])))


@pytest.mark.parametrize(
    "x",
    [np.zeros((2, 24)), np.ones((1, 24, 8)), np.full((1, 24, 7), np.nan), np.ones((1, 24, 7)) * 1j],
)
def test_invalid_cnn_inputs_rejected(x):
    model = SmallCNN.initialize(np.zeros((2, 24, 7)), 1)
    with pytest.raises(ValueError):
        model.margins(x)


def test_logmel_is_waveform_only_with_declared_geometry_and_padding():
    rng = np.random.default_rng(2)
    wav = rng.normal(0, 0.03, 4 * ARCHIVE_SR)
    starts, x = extract_logmel(wav)
    assert x.shape == (len(starts), 24, 7) and x.dtype == np.float32
    assert starts[0] == 0 and starts[-1] + GEOMETRY.window_samples == len(wav)
    again, other = extract_logmel(wav.copy())
    np.testing.assert_array_equal(starts, again)
    np.testing.assert_array_equal(x, other)
    _, gain = extract_logmel(wav * 2)
    np.testing.assert_allclose(gain - x, 2 * np.log10(2), atol=2e-6)
    assert extract_logmel([])[1].shape == (0, 24, 7)
    short, short_x = extract_logmel(wav[:200])
    assert short_x.shape == (1, 24, 7)
    truth = guarded_labels(np.zeros(200, dtype=np.int8), 3308)
    assert assign_window_labels(truth, short, GEOMETRY.window_samples).tolist() == [-1]
    with pytest.raises(ValueError):
        extract_logmel([1j])


def test_sampling_never_uses_ignore_or_dev_and_preserves_source():
    records = []
    for i, group in enumerate(("a", "b")):
        labels = np.r_[np.zeros(40), np.ones(40), np.full(40, -1)]
        records.append(
            {
                "row": {"split": "train", "group_id": group, "example_id": str(i)},
                "window_y": labels,
                "features": np.arange(120)[:, None, None] * np.ones((120, 24, 7)),
            }
        )
    x, y, groups = sampled_training(records)
    assert x.shape == (96, 24, 7) and len(groups) == 96
    assert np.all(x[:, 0, 0] < 80)
    for group in ("a", "b"):
        assert np.sum((groups == group) & (y == 0)) == 24
        assert np.sum((groups == group) & (y == 1)) == 24
    for a, b in zip((x, y, groups), sampled_training(list(reversed(records))), strict=True):
        # Content of each per-record sample is order-independent; overall order follows records.
        np.testing.assert_array_equal(a[:48], b[48:])
    records[0]["row"]["split"] = "dev"
    with pytest.raises(ValueError, match="train"):
        sampled_training(records)
    with pytest.raises(ValueError, match="Eval"):
        extract_split(None, [], None, None, "eval")


def test_bundle_roundtrip_safe_inventory_and_label_free_prediction(tmp_path):
    x, y, groups = fixture_arrays()
    model, _ = train_cnn(x, y, groups, 7, epochs=1)
    bundle = tmp_path / "bundle"
    save_cnn(model, bundle, {"fpr_05": 0.6})
    loaded, meta = load_cnn(bundle)
    np.testing.assert_array_equal(model.margins(x), loaded.margins(x))
    assert meta["development_only"] and not meta["citable"]
    wav = 0.1 * np.sin(2 * np.pi * 300 * np.arange(ARCHIVE_SR) / ARCHIVE_SR)
    path = tmp_path / "no-labels-here.wav"
    write_audio(path, wav.astype(np.float32), ARCHIVE_SR)
    prediction = predict_cnn_file(bundle, path, tmp_path / "work")
    assert prediction["ground_truth_used"] is False
    assert len(prediction["synthetic_margins"]) == len(prediction["window_start_samples"])
    assert set(prediction) & {"labels", "group_id", "start_sample", "end_sample"} == set()
    unexpected = bundle / "unrelated.txt"
    unexpected.write_text("not allowed")
    with pytest.raises(ValueError, match="inventory"):
        load_cnn(bundle)
    unexpected.unlink()
    state = bundle / "model.npz"
    state.write_bytes(state.read_bytes() + b"modified")
    with pytest.raises(ValueError, match="integrity"):
        load_cnn(bundle)
    assert learned_plan()["trainable_parameters"] == 521
    assert json.dumps(prediction, allow_nan=False)


def test_transition_and_filter_support_are_not_supervised_as_real():
    n = ARCHIVE_SR * 3
    intervals = [
        {"start_sample": 0, "end_sample": 20000, "label": 0},
        {"start_sample": 20000, "end_sample": 20220, "label": -1},
        {"start_sample": 20220, "end_sample": 40000, "label": 1},
        {"start_sample": 40000, "end_sample": n, "label": 0},
    ]
    truth = guarded_labels(sample_labels(intervals, n), 3308)
    starts, _ = extract_logmel(np.zeros(n))
    labels = assign_window_labels(truth, starts, GEOMETRY.window_samples)
    for start, label in zip(starts, labels, strict=True):
        if label >= 0:
            assert np.all(truth[start : start + GEOMETRY.window_samples] == label)
        if start < 23528 and start + GEOMETRY.window_samples > 16692:
            assert label == -1
