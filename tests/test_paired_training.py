"""CPU regressions for exact pairing, controlled mass, frozen normalization and inference."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from test_paired_signal import paired_dataset as paired_dataset

from data.partial_audio import VOCODERS
from detectors.temporal_context import load_context_bundle
from detectors.temporal_features import (
    analysis_comparison,
    assign_window_labels,
    extract_windows,
    guarded_labels,
    sample_labels,
)
from detectors.temporal_operating import grouped_folds
from experiments.paired_signal import reference_activity_threshold
from experiments.paired_training import (
    CONFIG,
    build_pair_masks,
    choose_candidate,
    controlled_weights,
    fit_weighted,
    paired_improvements,
    roc_control_recall,
    run,
)
from experiments.partial_audio import artifact_inventory, read_input
from experiments.temporal_ablation import cross_validate, save_features
from experiments.temporal_localization import fit_temporal_head
from experiments.temporal_review import read_json, review_rows


def pair_records():
    rows = []
    for kind in ("null_control", "real_to_real_control", *(f"mixed_{v}" for v in VOCODERS)):
        synthetic = kind.startswith("mixed_")
        y = np.r_[np.zeros(3, dtype=int), np.full(10, 1 if synthetic else 0), np.full(2, -1)]
        rows.append(
            {
                "row": {
                    "example_id": kind,
                    "example_class": kind,
                    "group_id": "g",
                    "split": "train",
                    "source_real_id": "original",
                    "has_synthetic": synthetic,
                },
                "starts": np.arange(15) * 353,
                "window_y": y,
            }
        )
    return rows, {
        "example_ids": np.array([r["row"]["example_id"] for r in rows]),
        "offsets": np.arange(6) * 15,
        "thresholds": np.full(5, 0.01),
        "own_rms": np.full(75, 0.02),
    }


def test_pair_coordinates_reference_identity_active_intersection_and_deduplication():
    rows, rms = pair_records()
    masks, evidence = build_pair_masks(rows, rms)
    assert sum(m.sum() for m in masks) == 10 and len(evidence["pairs"]) == 30
    for pair in evidence["pairs"]:
        assert pair["start_sample"] in rows[0]["starts"][3:13]
        assert pair["end_sample"] - pair["start_sample"] == CONFIG.window_samples
        assert pair["real_example_id"] == "null_control"
        assert pair["negative_global_window"] in np.flatnonzero(masks[0])
    # One quiet vocoder position removes that coordinate for every vocoder.
    rms["own_rms"][2 * 15 + 3] = 0
    masks, evidence = build_pair_masks(rows, rms)
    assert masks[0].sum() == 9 and len(evidence["pairs"]) == 27
    assert not masks[1].any()


@pytest.mark.parametrize(
    "change", ["coordinate", "identity", "split", "threshold", "activity", "duplicate"]
)
def test_unverified_or_ambiguous_pairs_fail(change):
    rows, rms = pair_records()
    if change == "coordinate":
        rows[2]["starts"][0] += 1
    if change == "identity":
        rows[2]["row"]["source_real_id"] = "different"
    if change == "split":
        rows[2]["row"]["split"] = "dev"
    if change == "threshold":
        rms["thresholds"][2] = 0.02
    if change == "activity":
        rms["own_rms"][2 * 15 : 3 * 15] = 0
    if change == "duplicate":
        rms["example_ids"][2] = rms["example_ids"][1]
    with pytest.raises((ValueError, AssertionError)):
        build_pair_masks(rows, rms)


@pytest.mark.parametrize("share", [None, 0.25, 0.5])
def test_negative_redistribution_preserves_all_source_class_masses_and_positive_weights(share):
    groups = np.array(["short"] * 12 + ["long"] * 24)
    y = np.r_[
        np.zeros(8, dtype=int),
        np.ones(4, dtype=int),
        np.zeros(18, dtype=int),
        np.ones(6, dtype=int),
    ]
    paired = np.zeros(36, dtype=bool)
    paired[[0, 1, 12, 13, 14]] = True
    base, base_info = controlled_weights(y, groups, paired, None)
    weights, info = controlled_weights(y, groups, paired, share)
    assert (weights > 0).all() and np.isfinite(weights).all()
    np.testing.assert_array_equal(weights[y == 1], base[y == 1])
    for group in ["short", "long"]:
        for label in [0, 1]:
            mask = (groups == group) & (y == label)
            assert weights[mask].sum() == pytest.approx(base[mask].sum(), abs=1e-12)
        if share is not None:
            assert weights[(groups == group) & paired].sum() / weights[
                (groups == group) & (y == 0)
            ].sum() == pytest.approx(share)
    assert info["class_mass"]["0"] == pytest.approx(len(y) / 2)
    assert info["class_mass"]["1"] == pytest.approx(len(y) / 2)
    assert info == controlled_weights(y, groups, paired, share)[1]


@pytest.mark.parametrize("share", [-0.1, 0, 0.1, 0.75, 1, True, np.nan])
def test_undeclared_weights_rejected(share):
    with pytest.raises(ValueError):
        controlled_weights(
            np.array([0, 0, 1]), np.array(["g"] * 3), np.array([True, False, False]), share
        )


def test_real_controls_and_both_classes_required_for_every_source():
    with pytest.raises(ValueError):
        controlled_weights(np.array([0, 1]), np.array(["g", "g"]), np.array([True, False]), 0.25)
    with pytest.raises(ValueError):
        controlled_weights(
            np.array([0, 0, 1]), np.array(["g"] * 3), np.array([False, False, True]), 0.25
        )


def test_control_ROC_moves_whole_ties_and_is_calibration_invariant():
    p = np.array([0.9, 0.8, 0.4, 0.3, 0.1, 0.05])
    positive = np.array([1, 1, 0, 0, 0, 0])
    null = np.array([0, 0, 1, 0, 1, 0])
    r2r = np.array([0, 0, 0, 1, 0, 1])
    result = roc_control_recall(p, positive, null, r2r, targets=(0, 0.5))
    assert result["fpr_00"]["recall"] == 1 and result["fpr_50"]["recall"] == 1
    assert result == roc_control_recall(3 * p + 2, positive, null, r2r, targets=(0, 0.5))
    ties = roc_control_recall(np.ones(6), positive, null, r2r, targets=(0, 0.5, 1))
    assert ties["fpr_00"]["recall"] == ties["fpr_50"]["recall"] == 0
    assert ties["fpr_100"]["recall"] == 1
    assert all("threshold" not in entry for entry in result.values())


def metric(tp, fp, fn, tn, group_f1=None):
    f1 = 2 * tp / (2 * tp + fp + fn)
    return {
        "sample_micro": {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "recall": tp / (tp + fn)},
        "source_group_macro_f1": f1 if group_f1 is None else group_f1,
        "sample_by_source_group": {
            f"g{i}": {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "f1": f1} for i in range(12)
        },
        "controls": {
            "null_control": {"sample_false_positive_rate": fp / (fp + tn)},
            "real_to_real_control": {"sample_false_positive_rate": fp / (fp + tn)},
        },
    }


def test_paired_source_uncertainty_and_selection_cannot_reward_extra_control_errors():
    a = metric(1, 1, 9, 99)
    b = metric(2, 1, 8, 99)
    c = metric(3, 1, 7, 99)
    result = paired_improvements(a, b)
    assert result["absolute_sample_recall_change"] == pytest.approx(0.1)
    assert result["relative_sample_recall_change"] == pytest.approx(1)
    np.testing.assert_allclose(result["paired_source_recall_change_ci95"], [0.1, 0.1])
    assert choose_candidate({"A_original": a, "B_paired25": b, "C_paired50": c}) == "C_paired50"
    c["controls"]["null_control"]["sample_false_positive_rate"] = 0.051
    with pytest.raises(ValueError):
        choose_candidate({"A_original": a, "B_paired25": b, "C_paired50": c})


def test_weighted_fit_is_deterministic_and_uses_only_supplied_train_scaler():
    rng = np.random.default_rng(4)
    records = []
    for i in range(4):
        y = np.tile([0, 0, 0, 1, 1], 8)
        records.append(
            {
                "row": {"split": "train", "group_id": f"g{i}"},
                "starts": np.arange(len(y)),
                "features": rng.normal(size=(len(y), 31)) + np.array(y)[:, None] / 5,
                "window_y": y,
                "paired_negative": np.arange(len(y)) % 5 == 0,
            }
        )
    reference, _ = fit_temporal_head(records, 1, 20260907)
    first, _, _, _ = fit_weighted(records, 0.25, reference)
    second, _, _, _ = fit_weighted(records, 0.25, reference)
    for key in ("mean", "scale", "coefficients", "intercept"):
        np.testing.assert_array_equal(getattr(first, key), getattr(second, key))
    np.testing.assert_array_equal(first.mean, reference.mean)
    np.testing.assert_array_equal(first.scale, reference.scale)
    records[0]["row"]["split"] = "dev"
    with pytest.raises(ValueError):
        fit_weighted(records, 0.25, reference)


def test_research_decision_is_separate_from_locked_configuration_selection():
    from experiments.paired_training import research_recommendation

    primary = {"A_original": metric(1, 1, 9, 99), "B_paired25": metric(3, 1, 7, 99)}
    changes = {name: paired_improvements(primary["A_original"], m) for name, m in primary.items()}
    assert research_recommendation("B_paired25", primary, changes, primary)["recommendation"] == "A"
    bad = deepcopy(primary)
    bad["B_paired25"]["controls"]["null_control"]["sample_false_positive_rate"] = 0.06
    outcome = research_recommendation("B_paired25", primary, changes, bad)
    assert (
        outcome["recommendation"] == "B"
        and outcome["preferred_configuration_unchanged"] == "B_paired25"
    )
    assert research_recommendation("A_original", primary, changes, primary)["recommendation"] == "C"


def test_complete_controlled_experiment_locks_before_dev_and_preserves_inputs(
    paired_dataset, monkeypatch
):
    import experiments.paired_training as module
    from detectors.forensic_pilot import write_json
    from experiments.acoustic_diagnostics import window_rms

    root, source, dataset, baseline = paired_dataset
    before = {p: artifact_inventory(p) for p in (source.parent, dataset, baseline)}
    selected, identity = review_rows(dataset, source, baseline)
    ablation = root / "paired-training-reference"
    review = root / "paired-review"
    review.mkdir()
    all_records = {}
    rms_rows = []
    work = root / "paired-cache-work"
    work.mkdir()
    for split in ["train", "dev"]:
        records = []
        for group in sorted({r["group_id"] for r in selected[split]}):
            members = [r for r in selected[split] if r["group_id"] == group]
            wavs = {
                ("real" if r["example_class"] == "null_control" else r["example_id"]): read_input(
                    dataset / r["output_filename"]
                )
                for r in members
            }
            processed = analysis_comparison(wavs, work)
            threshold = reference_activity_threshold(processed["real"])
            for row in members:
                key = "real" if row["example_class"] == "null_control" else row["example_id"]
                starts, features = extract_windows(processed[key], CONFIG)
                truth = guarded_labels(
                    sample_labels(row["interval_labels"], row["num_samples"]), 3308
                )
                records.append(
                    {
                        "row": row,
                        "starts": starts,
                        "features": features,
                        "window_y": assign_window_labels(truth, starts, CONFIG.window_samples),
                    }
                )
                if split == "train":
                    rms_rows.append(
                        (
                            row["example_id"],
                            window_rms(processed[key], starts),
                            window_rms(processed["real"], starts),
                            threshold,
                        )
                    )
        save_features(ablation / f"features/A64/{split}", records)
        all_records[split] = records
    assignments = grouped_folds([r["row"]["group_id"] for r in all_records["train"]], 20260907, 3)
    cross_validate(all_records["train"], CONFIG, assignments, ablation / "candidates/A64")
    np.savez_compressed(
        review / "window_rms.npz",
        example_ids=np.array([r[0] for r in rms_rows]),
        offsets=np.r_[0, np.cumsum([len(r[1]) for r in rms_rows])],
        own_rms=np.concatenate([r[1] for r in rms_rows]),
        reference_rms=np.concatenate([r[2] for r in rms_rows]),
        thresholds=np.array([r[3] for r in rms_rows]),
    )
    write_json(review / "artifacts.sha256.json", {"files": artifact_inventory(review)})
    # The inherited review receipt is a fixture boundary, not a fresh INV-16 experiment.
    monkeypatch.setattr(module, "admitted_inputs", lambda *args: (selected, identity, {}))
    out = root / "paired-training-experiment"
    staging = out.with_name(out.name + ".staging")
    old_load = module.load_features

    def guarded_load(path, rows):
        if Path(path).name == "dev":
            lock = read_json(staging / "selection_locked.json")
            assert lock["selection_completed_before_dev_features"]
            for name, inventory in lock["bundle_inventories"].items():
                assert artifact_inventory(staging / "candidates" / name / "bundle") == inventory
        return old_load(path, rows)

    monkeypatch.setattr(module, "load_features", guarded_load)
    forbidden = {"u6", "u7"} | {
        r["example_id"]
        for r in read_json(dataset / "manifest.json")["examples"]
        if r["split"] != "train"
    }

    def guarded_read(path):
        assert Path(path).stem not in forbidden
        return read_input(path)

    monkeypatch.setattr(module, "read_input", guarded_read)
    result = run(dataset, source, baseline, ablation, review, out)
    assert result["preferred"] in {"A_original", "B_paired25", "C_paired50"}
    v = read_json(out / "validation.json")
    assert v["eval_audio_bytes_opened"] == 0 and v["baseline_all_numeric_arrays_exact"]
    assert v["train_groups"] == 6 and v["dev_benchmark_groups"] == 1
    for name in ("A_original", "B_paired25", "C_paired50"):
        for fold in range(3):
            info = read_json(out / f"candidates/{name}/fold-{fold}/fit.json")
            assert not set(info["fit_groups"]) & set(info["held_out_groups"])
            bundle = load_context_bundle(out / f"candidates/{name}/fold-{fold}/bundle")
            training = [
                r for r in all_records["train"] if r["row"]["group_id"] in info["fit_groups"]
            ]
            x = np.vstack([r["features"][r["window_y"] >= 0] for r in training])
            np.testing.assert_allclose(bundle.head.mean, x.mean(axis=0), atol=1e-12)
        distribution = read_json(out / f"candidates/{name}/training_distribution.json")
        assert distribution["weighting"]["class_mass"]["0"] == pytest.approx(
            distribution["weighting"]["class_mass"]["1"]
        )
    assert read_json(out / "label_free_inference.json")["ground_truth_used"] is False
    for p, inventory in before.items():
        assert artifact_inventory(p) == inventory
    assert artifact_inventory(out) == read_json(out / "artifacts.sha256.json")["files"]
    with pytest.raises(FileExistsError):
        run(dataset, source, baseline, ablation, review, out)
