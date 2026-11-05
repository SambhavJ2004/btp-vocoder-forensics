"""End-to-end conditional CNN experiment on paired synthetic CPU fixtures."""

from pathlib import Path

import numpy as np
import pytest
from test_paired_signal import paired_dataset as paired_dataset

from detectors.temporal_cnn import GEOMETRY, load_cnn
from experiments.learned_temporal import run_learned
from experiments.paired_signal import run_paired_diagnostics
from experiments.partial_audio import artifact_inventory
from experiments.temporal_ablation import load_features
from experiments.temporal_review import read_json


def test_conditional_run_folds_masks_dev_lock_numeric_inference_and_immutable_inputs(
    paired_dataset, monkeypatch
):
    import experiments.learned_temporal as module

    root, source, dataset, baseline = paired_dataset
    originals = {path: artifact_inventory(path) for path in (source.parent, dataset, baseline)}
    diagnostic = root / "cnn-prerequisite"
    run_paired_diagnostics(dataset, source, baseline, diagnostic)
    out = root / "cnn-experiment"
    staging = out.with_name(out.name + ".staging")
    rows = read_json(dataset / "manifest.json")["examples"]
    by_id = {r["example_id"]: r for r in rows}
    original = module.read_input
    seen = []

    def checked_decode(path):
        row = by_id[Path(path).stem]
        assert row["split"] != "eval"
        if row["split"] == "dev":
            lock = read_json(staging / "selection_locked.json")
            assert lock["dev_decode_started"] is False
            assert artifact_inventory(staging / "bundle") == lock["bundle_inventory"]
        seen.append(row["split"])
        return original(path)

    monkeypatch.setattr(module, "read_input", checked_decode)
    # Six fixture groups cannot tolerate even one negative error at a 5% cap.
    # Exercise the rejection before stubbing ONLY this conditional prerequisite
    # for the orchestration test. Numeric gradients/fitting are tested separately;
    # the real-data run must satisfy the actual sealed diagnostic without a stub.
    with pytest.raises(ValueError, match="signal criterion"):
        run_learned(dataset, source, baseline, diagnostic, out)
    actual_read = module.read_json

    def passing_prerequisite(path):
        result = actual_read(path)
        if Path(path) == diagnostic / "oracle_summary.json":
            result["learned_model_signal_decision"]["proceed"] = True
            result["variants"]["rms_active"].update(raw_auroc=1.0, recall=1.0)
        return result

    monkeypatch.setattr(module, "read_json", passing_prerequisite)
    result = run_learned(dataset, source, baseline, diagnostic, out)
    assert result["status"] == "passed" and seen == ["train"] * 30 + ["dev"] * 5
    receipt = read_json(out / "validation.json")
    assert receipt["eval_waveforms_opened"] == 0 and receipt["label_free_inference_exact"]
    assert receipt["trainable_parameters"] == 521
    assignments = read_json(out / "folds.json")["assignments"]
    cv = read_json(out / "cross_validation/cross_validation.json")
    for fold in cv["folds"]:
        fit, held = set(fold["fit_groups"]), set(fold["held_out_groups"])
        inner, validation = set(fold["inner_fit_groups"]), set(fold["inner_validation_groups"])
        assert not fit & held and inner | validation == fit and not inner & validation
        assert fit | held == set(assignments) == {f"u{i}" for i in range(6)}
        assert set(assignments[g] for g in held) == {fold["fold"]}
    model, meta = load_cnn(out / "bundle")
    assert set(meta["fit_groups"]) == set(assignments)
    assert receipt["final_epochs"] == int(
        np.median([f["early_stopping"]["selected_epoch"] for f in cv["folds"]])
    )
    train_rows = sorted(
        [r for r in rows if r["split"] == "train"],
        key=lambda r: (r["group_id"], r["example_class"], r["example_id"]),
    )
    training = load_features(out / "features/train", train_rows)
    x, y, groups = module.sampled_training(training)
    np.testing.assert_allclose(model.mean, x.astype(float).mean(axis=(0, 2)), atol=1e-13)
    from detectors.temporal_features import guarded_labels, sample_labels

    for record in training:
        truth = guarded_labels(
            sample_labels(record["row"]["interval_labels"], record["row"]["num_samples"]), 3308
        )
        for start, label in zip(record["starts"], record["window_y"], strict=True):
            if label >= 0:
                assert np.all(truth[start : start + GEOMETRY.window_samples] == label)
    for point in cv["points"].values():
        if "target_sample_fpr" in point:
            assert all(
                fpr <= point["target_sample_fpr"]
                for fpr in point["oof_control_sample_fpr"].values()
            )
    for path, expected in originals.items():
        assert artifact_inventory(path) == expected
    assert artifact_inventory(out) == read_json(out / "artifacts.sha256.json")["files"]
    with pytest.raises(FileExistsError):
        run_learned(dataset, source, baseline, diagnostic, out)
