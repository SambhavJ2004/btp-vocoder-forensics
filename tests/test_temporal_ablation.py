"""Deterministic CPU contracts for Phase 3B; no production data are loaded."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from data.audio_io import write_audio
from data.invariants import ARCHIVE_SR, GLOBAL_SEED
from data.manifest import ManifestRow, write_manifest
from data.partial_audio import VOCODERS, PartialAudioConfig
from detectors.forensic_pilot import LinearHead, sha256_file, write_json
from detectors.temporal_context import (
    CANDIDATES,
    REJECT_ALL_THRESHOLD,
    ContextConfig,
    context_intervals,
    extract_context,
    load_context_bundle,
    predict_context_file,
    save_context_bundle,
)
from detectors.temporal_features import WindowConfig, extract_windows
from detectors.temporal_operating import (
    common_mask_ranking,
    control_operating_points,
    grouped_folds,
    weighted_control_cutoff,
    weighted_roc,
)
from experiments.partial_audio import artifact_inventory, build_partial_dataset
from experiments.temporal_ablation import (
    bundle_for,
    fixed_plan,
    load_features,
    run_ablation,
    save_features,
)
from experiments.temporal_localization import TemporalExperimentConfig, train_temporal
from experiments.temporal_review import (
    assert_current_guards,
    audit_phase3a,
    center_edges,
    read_json,
)


@pytest.mark.parametrize("name", CANDIDATES)
def test_context_feature_geometry_finiteness_and_local_data_access(name):
    cfg = CANDIDATES[name]
    time = np.arange(14000) / ARCHIVE_SR
    wav = 0.12 * np.sin(2 * np.pi * 1000 * time)
    original = wav.copy()
    starts, x = extract_context(wav, cfg)
    assert x.shape == (len(starts), len(cfg.feature_names))
    assert starts[0] == 0 and starts[-1] + cfg.window_samples == len(wav)
    assert np.isfinite(x).all()
    np.testing.assert_array_equal(wav, original)
    changed = wav.copy()
    changed[cfg.window_samples :] *= -0.23
    np.testing.assert_array_equal(x[0], extract_context(changed, cfg)[1][0])
    if name == "A64":
        np.testing.assert_array_equal(
            x, extract_windows(wav, WindowConfig(window_ms=64, hop_ms=16))[1]
        )


@pytest.mark.parametrize("name", CANDIDATES)
@pytest.mark.parametrize("n", [0, 1, 85])
def test_context_empty_short_silent_policy(name, n):
    cfg = CANDIDATES[name]
    starts, features = extract_context(np.zeros(n), cfg)
    assert features.shape == (len(starts), len(cfg.feature_names))
    assert np.isfinite(features).all()
    if n and name == "Dmel256":
        np.testing.assert_allclose(features[:, :24], -12, atol=1e-14)
        np.testing.assert_allclose(features[:, 24:], 0, atol=1e-14)


@pytest.mark.parametrize(
    "changes",
    [
        {"window_ms": 128},
        {"window_ms": 1024},
        {"hop_ms": 8},
        {"representation": "id"},
        {"window_ms": True},
        {"window_ms": float("nan")},
        {"band_limit_hz": 10000},
    ],
)
def test_undeclared_feature_settings_rejected(changes):
    with pytest.raises(ValueError):
        replace(ContextConfig(), **changes).validate()


def test_folds_group_variants_reproducible_and_order_independent():
    ids = [f"g{i}" for i in range(12)]
    folds = grouped_folds(ids * 5, GLOBAL_SEED)
    assert folds == grouped_folds(ids[::-1], GLOBAL_SEED)
    assert sorted(list(folds.values()).count(i) for i in range(3)) == [4, 4, 4]
    assert folds != grouped_folds(ids, GLOBAL_SEED + 1)
    for fold in range(3):
        train = {g for g in ids if folds[g] != fold}
        held = {g for g in ids if folds[g] == fold}
        assert not train & held and train | held == set(ids)


@pytest.mark.parametrize("kwargs", [{"seed": True}, {"seed": -1}, {"n_folds": 1}, {"n_folds": 5}])
def test_invalid_grouped_folds(kwargs):
    options = {"seed": 1, **kwargs}
    with pytest.raises(ValueError):
        grouped_folds(["a", "b", "c"], **options)


@pytest.mark.parametrize(
    "score,weight,target",
    [
        ([0.2, 0.5, 0.5, 0.9], [20, 30, 40, 10], 0.1),
        ([1, 1], [2, 8], 0.01),
        ([0, 0], [1, 9], 0),
        ([0.1, 0.2, 0.3], [2, 2, 6], 0.5),
        ([0.1, 0.2], [99, 1], 0.01),
        ([0.4, 0.5], [0, 10], 1),
    ],
)
def test_weighted_cutoff_moves_ties_together_and_is_minimal(score, weight, target):
    score, weight = np.array(score), np.array(weight)
    threshold = weighted_control_cutoff(score, weight, target)
    assert weight[score >= threshold].sum() / weight.sum() <= target
    candidates = sorted({0.0, *np.nextafter(score[weight > 0], np.inf).tolist()})
    expected = min(t for t in candidates if weight[score >= t].sum() / weight.sum() <= target)
    assert threshold == expected
    assert weighted_control_cutoff(score[::-1], weight[::-1], target) == threshold


@pytest.mark.parametrize(
    "score,weight,target",
    [
        ([], [], 0.1),
        ([0.1], [0], 0.1),
        ([np.nan], [1], 0.1),
        ([1.1], [1], 0.1),
        ([0.1], [-1], 0.1),
        ([0.1], [1], 2),
    ],
)
def test_invalid_weighted_threshold_inputs(score, weight, target):
    with pytest.raises(ValueError):
        weighted_control_cutoff(score, weight, target)


def tiny_record(kind, group="g", margin=None):
    n, start, end = 20000, 8000, 12000
    synthetic = kind.startswith("mixed_")
    intervals = [
        {"start_sample": 0, "end_sample": start, "label": 0},
        {"start_sample": start, "end_sample": end, "label": int(synthetic)},
        {"start_sample": end, "end_sample": n, "label": 0},
    ]
    return {
        "row": {
            "num_samples": n,
            "start_sample": start,
            "end_sample": end,
            "group_id": group,
            "example_class": kind,
            "interval_labels": intervals,
            "has_synthetic": synthetic,
            "example_id": kind,
            "split": "train",
            "pure_synthetic_ranges": [[start, end]] if synthetic else [],
            "transition_ignore_ranges": [],
        },
        "starts": np.array([0, 9000, 18589]),
        "window_y": np.array([0, int(synthetic), 0]),
        "margin": np.array([0.0, 1.0, 0.0]) if margin is None else margin,
    }


def test_control_targets_apply_to_each_control_separately():
    records = [tiny_record("null_control"), tiny_record("real_to_real_control")]
    margin = [np.array([-3.0, -2.0, -3.0]), np.array([1.0, 4.0, 1.0])]
    points = control_operating_points(records, margin, ContextConfig(), 0)
    for point in points.values():
        assert all(
            x <= point["target_sample_fpr"] for x in point["oof_control_sample_fpr"].values()
        )
        assert point["threshold"] == max(point["per_control_cutoff"].values())
    with pytest.raises(ValueError, match="Both real control"):
        control_operating_points(records[:1], margin[:1], ContextConfig(), 0)


def test_weighted_common_mask_ranking_equals_expanded_sample_roc():
    from sklearn.metrics import roc_auc_score

    cfg = ContextConfig()
    record = tiny_record("mixed_vocos")
    scores = np.array([-0.2, 0.7, 0.1])
    actual = common_mask_ranking([record], [scores], cfg, 0)
    edges = center_edges(record["starts"], record["row"]["num_samples"], cfg.window_samples)
    dense = np.repeat(scores, np.diff(edges))
    y = np.zeros(len(dense))
    y[8000:12000] = 1
    assert actual["auroc"] == pytest.approx(roc_auc_score(y, dense))
    expected = weighted_roc(y, dense, np.ones(len(y)))
    assert actual == pytest.approx(expected)
    assert weighted_roc([0, 1], [1, 0], [1, 1]) == {"auroc": 0.0, "directional_eer": 1.0}
    assert weighted_roc([0, 1], [0, 0], [1, 1]) == {"auroc": 0.5, "directional_eer": 0.5}


def test_independent_guard_oracle_includes_hard_transitions_and_edges():
    row = tiny_record("mixed_vocos")["row"]
    assert_current_guards(row)


def test_feature_cache_decompresses_each_member_once(tmp_path, monkeypatch):
    import experiments.temporal_ablation as module

    records = [tiny_record("mixed_vocos"), tiny_record("null_control")]
    for i, record in enumerate(records):
        record["features"] = np.arange(93, dtype=float).reshape(3, 31) + i
    root = tmp_path / "cache"
    save_features(root, records)
    original = np.load
    reads = Counter()

    class Counted:
        def __init__(self, *args, **kwargs):
            self.inner = original(*args, **kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.inner.close()

        def __getitem__(self, name):
            reads[name] += 1
            return self.inner[name]

    monkeypatch.setattr(module.np, "load", Counted)
    loaded = load_features(root, [r["row"] for r in records])
    assert reads == {"features": 1, "starts": 1, "window_y": 1, "offsets": 1}
    for expected, actual in zip(records, loaded, strict=True):
        np.testing.assert_array_equal(actual["features"], expected["features"])


@pytest.mark.parametrize("name", CANDIDATES)
def test_numeric_context_bundle_roundtrip_and_all_negative_saturation(tmp_path, name):
    cfg = CANDIDATES[name]
    d = len(cfg.feature_names)
    head = LinearHead(np.zeros(d), np.ones(d), np.zeros((1, d)), np.ones(1))
    bundle = bundle_for(head, cfg, {"fpr_05": REJECT_ALL_THRESHOLD})
    root = tmp_path / "bundle"
    save_context_bundle(bundle, root)
    restored = load_context_bundle(root)
    np.testing.assert_array_equal(
        bundle.margins(np.ones((2, d))), restored.margins(np.ones((2, d)))
    )
    assert context_intervals(np.array([0]), np.ones(1), 20000, cfg, REJECT_ALL_THRESHOLD) == []
    (root / "unrelated.txt").write_text("x")
    with pytest.raises(ValueError, match="Unexpected"):
        load_context_bundle(root)


@pytest.fixture(scope="module")
def controlled_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("context3b")
    n = 4 * ARCHIVE_SR
    time = np.arange(n) / ARCHIVE_SR
    rows = []
    for i, split in enumerate(["train"] * 6 + ["dev"] * 2 + ["eval"]):
        for j, condition in enumerate(("real", *VOCODERS)):
            path = root / "source" / "archive" / condition / f"u{i}.wav"
            wave = (0.1 * np.sin(2 * np.pi * (211 + 11 * i + 503 * j) * time)).astype(np.float32)
            write_audio(path, wave, ARCHIVE_SR)
            rows.append(
                ManifestRow(
                    utt_id=f"u{i}",
                    condition=condition,
                    path=str(path),
                    speaker="fixture",
                    split=split,
                    duration_s=4,
                    lufs_out=-27,
                    trim_start=0,
                    trim_end=n,
                    trim_ref_length=n,
                    lufs_in=-27,
                    lufs_in_band=-27,
                    lufs_out_band=-27,
                    loudness_gain_db=0,
                    loudness_target_lufs=-27,
                    alignment_checked=True,
                    alignment_peak_lag=0,
                    high_band_fraction=0,
                )
            )
    source = write_manifest(rows, root / "source" / "manifest.csv")
    dataset = root / "dataset"
    build_partial_dataset(
        source,
        dataset,
        PartialAudioConfig(
            n_utterances=9,
            all_utterances=True,
            real_to_real_controls=True,
            segment_seconds=1,
            boundary_margin_seconds=0.1,
            demo_examples=0,
        ),
    )
    baseline = root / "baseline"
    train_temporal(
        dataset,
        source,
        baseline,
        TemporalExperimentConfig(
            settings=((64.0, 16.0),),
            c_grid=(1.0,),
            thresholds=(0.5,),
            smoothing_widths=(3,),
            merge_ms=(0.0,),
        ),
    )
    audit = root / "audit"
    audit_phase3a(dataset, source, baseline, audit)
    return root, source, dataset, baseline, audit


def test_grouped_ablation_locked_selection_eval_byte_isolation_and_inference(
    controlled_run, monkeypatch
):
    import experiments.temporal_ablation as module
    import experiments.temporal_review as review

    root, source, dataset, baseline, audit = controlled_run
    before = artifact_inventory(dataset)
    old = artifact_inventory(baseline)
    eval_paths = {
        str((dataset / r["output_filename"]).resolve())
        for r in read_json(dataset / "manifest.json")["examples"]
        if r["split"] == "eval"
    }
    hasher = review.sha256_file
    reader = module.read_input
    original_extract = module.extract_split
    fit = module.fit_temporal_head
    calls = []

    def hash_without_eval(path):
        assert str(Path(path).resolve()) not in eval_paths
        return hasher(path)

    def decode_without_eval(path):
        assert str(Path(path).resolve()) not in eval_paths
        return reader(path)

    def assert_dev_locked(dataset, rows, target, work, split):
        if split == "dev":
            assert (target.parent / "selection.json").is_file()
            assert read_json(target.parent / "selection.json")["dev_used_for_selection"] is False
        return original_extract(dataset, rows, target, work, split)

    def train_only(records, c, seed):
        assert all(r["row"]["split"] == "train" for r in records)
        calls.append({r["row"]["group_id"] for r in records})
        head, info = fit(records, c, seed)
        x = np.vstack([r["features"][r["window_y"] >= 0] for r in records])
        np.testing.assert_allclose(head.mean, x.mean(axis=0), rtol=0, atol=1e-14)
        return head, info

    monkeypatch.setattr(review, "sha256_file", hash_without_eval)
    monkeypatch.setattr(module, "read_input", decode_without_eval)
    monkeypatch.setattr(module, "extract_split", assert_dev_locked)
    monkeypatch.setattr(module, "fit_temporal_head", train_only)
    output = root / "ablation"
    result = run_ablation(dataset, source, baseline, audit, output)
    assert result["selected_candidate"] in CANDIDATES
    assert (
        len(calls) == 16
        and sum(len(v) == 4 for v in calls) == 12
        and sum(len(v) == 6 for v in calls) == 4
    )
    assert read_json(output / "validation.json")["status"] == "passed"
    assert artifact_inventory(dataset) == before and artifact_inventory(baseline) == old
    assert not (output / "eval").exists()
    row = next(r for r in read_json(dataset / "manifest.json")["examples"] if r["split"] == "dev")
    work = root / "inference"
    work.mkdir()
    for name in CANDIDATES:
        b = load_context_bundle(output / "candidates" / name / "bundle")
        first = predict_context_file(b, dataset / row["output_filename"], work)
        second = predict_context_file(b, dataset / row["output_filename"], work)
        assert first == second and first["ground_truth_used"] is False
        with np.load(
            output / "candidates" / name / "dev" / "window_predictions.npz", allow_pickle=False
        ) as arrays:
            ids = arrays["record_example_ids"].tolist()
            index = ids.index(row["example_id"])
            begin, end = arrays["record_offsets"][index : index + 2]
            np.testing.assert_allclose(
                first["synthetic_margins"],
                arrays["synthetic_margins"][begin:end],
                rtol=0,
                atol=1e-12,
            )
        saved = read_json(output / "candidates" / name / "dev" / "fpr_05.recordings.json")[
            "recordings"
        ]
        intervals = next(
            r["model_predicted_intervals"] for r in saved if r["example_id"] == row["example_id"]
        )
        assert first["model_predicted_intervals"] == intervals
    with pytest.raises(FileExistsError):
        run_ablation(dataset, source, baseline, audit, output)
    # Reuse verified train-only bytes into a fresh run. Numeric results and
    # all selected parameters must match a complete extraction/fitting run.
    snapshot = root / "cache-driver.py"
    snapshot.write_bytes(Path(module.__file__).read_bytes())
    repeated = root / "cached-repeat"
    run_ablation(dataset, source, baseline, audit, repeated, output, snapshot)
    assert sha256_file(output / "selection.json") == sha256_file(repeated / "selection.json")
    for name in CANDIDATES:
        assert artifact_inventory(output / "candidates" / name / "bundle") == artifact_inventory(
            repeated / "candidates" / name / "bundle"
        )
        for split in ("oof", "dev"):
            assert sha256_file(
                output / "candidates" / name / split / "window_predictions.npz"
            ) == sha256_file(repeated / "candidates" / name / split / "window_predictions.npz")


def test_changed_predeclaration_rejected_before_fitting(controlled_run, tmp_path):
    _, source, dataset, baseline, audit = controlled_run
    newaudit = tmp_path / "audit"
    newaudit.mkdir()
    plan = fixed_plan()
    plan["smoothing"] = 1
    write_json(newaudit / "predeclared.json", plan)
    write_json(newaudit / "correctness.json", read_json(audit / "correctness.json"))
    with pytest.raises(ValueError, match="fixed predeclared"):
        run_ablation(dataset, source, baseline, newaudit, tmp_path / "out")
    assert not (tmp_path / "out.staging").exists()
