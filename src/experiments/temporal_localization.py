"""Train/dev-only CPU temporal localization pilot on immutable VocoPatch examples."""

from __future__ import annotations

import json
import platform
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits

from data.invariants import ARCHIVE_SR, GLOBAL_SEED
from data.partial_audio import VOCODERS, interval_labels
from detectors.forensic_pilot import LinearHead, runtime_versions, sha256_file, write_json
from detectors.temporal_features import (
    WindowConfig,
    analysis_comparison,
    assign_window_labels,
    extract_windows,
    filter_support_receipt,
    guarded_labels,
    sample_labels,
    smooth_probabilities,
)
from detectors.temporal_metrics import evaluate_localization
from detectors.temporal_model import (
    PROTOCOL,
    TemporalBundle,
    load_temporal_bundle,
    save_temporal_bundle,
)
from experiments.forensic_pilot import fit_head
from experiments.partial_audio import FLAGS, artifact_inventory, read_input, source_frame
from experiments.partial_audio_upload import _validate_package_files

CLASSES = {"null_control", "real_to_real_control", *(f"mixed_{v}" for v in VOCODERS)}


@dataclass(frozen=True)
class TemporalExperimentConfig:
    settings: tuple[tuple[float, float], ...] = ((32.0, 8.0), (64.0, 16.0))
    c_grid: tuple[float, ...] = (0.01, 0.1, 1.0)
    thresholds: tuple[float, ...] = (0.2, 0.35, 0.5, 0.65, 0.8)
    smoothing_widths: tuple[int, ...] = (1, 3)
    merge_ms: tuple[float, ...] = (0.0, 32.0)
    seed: int = GLOBAL_SEED

    def validate(self):
        if (
            not isinstance(self.seed, int)
            or isinstance(self.seed, bool)
            or not 0 <= self.seed < 2**31
            or not 1 <= len(self.settings) <= 8
        ):
            raise ValueError("Require a bounded, seeded temporal development search.")
        for window, hop in self.settings:
            WindowConfig(window_ms=window, hop_ms=hop).validate()
        if len({tuple(pair) for pair in self.settings}) != len(self.settings):
            raise ValueError("Duplicate window settings are not independent candidates.")
        for values, lo, hi, name in (
            (self.c_grid, 0, 100, "C"),
            (self.thresholds, 0, 1, "threshold"),
            (self.merge_ms, 0, 100, "merge"),
        ):
            if (
                not values
                or len(values) > 10
                or any(
                    isinstance(v, bool) or not np.isfinite(v) or not lo <= v <= hi for v in values
                )
            ):
                raise ValueError(f"Invalid bounded {name} grid.")
        if (
            min(self.c_grid) <= 0
            or not self.smoothing_widths
            or any(x not in (1, 3) or isinstance(x, bool) for x in self.smoothing_widths)
        ):
            raise ValueError("Require positive regularization and smoothing widths 1 or 3.")

    @property
    def score_guard_samples(self):
        configs = [WindowConfig(window_ms=w, hop_ms=h) for w, h in self.settings]
        # Constant across the whole predeclared search, so masks cannot improve
        # candidate scores merely by removing a larger/different denominator.
        return (
            configs[0].guard_samples
            + max((c.window_samples + 1) // 2 for c in configs)
            + max(c.hop_samples for c in configs) * (max(self.smoothing_widths) // 2)
            + max((c.hop_samples + 1) // 2 for c in configs)
        )


DEFAULT_TEMPORAL_CONFIG = TemporalExperimentConfig()


def load_temporal_rows(dataset_dir: str | Path, source_manifest: str | Path) -> tuple[dict, dict]:
    """Hash all inventory bytes, but never decode/featurize/score eval audio.

    Frozen split and group metadata are checked globally. Ground-truth interval
    semantics are consumed only after filtering to train/dev.
    """
    root = Path(dataset_dir).absolute()
    _validate_package_files(root, prepared=False)
    provenance = json.loads((root / "provenance.json").read_text())
    if provenance["source_manifest_sha256"] != sha256_file(source_manifest):
        raise ValueError("Original source manifest identity changed.")
    source = source_frame(source_manifest)
    frozen = source.groupby("utt_id").split.first().to_dict()
    rows = json.loads((root / "manifest.json").read_text())["examples"]
    catalog = json.loads((root / "sources.json").read_text())["sources"]
    labels = json.loads((root / "labels.json").read_text())["examples"]
    file_hashes = json.loads((root / "artifacts.sha256.json").read_text())["files"]
    groups, seen_files, seen_ids = {}, set(), set()
    hashes = {}
    selected = {"train": [], "dev": []}
    for row in rows:
        group, split = row["group_id"], row["split"]
        if (
            split not in ("train", "dev", "eval")
            or frozen.get(group) != split
            or row["original_utterance_id"] != group
            or row["example_class"] not in CLASSES
            or row["variant"] != 1
            or row["example_id"] in seen_ids
            or row["output_filename"] in seen_files
            or row["output_filename"] != f"audio/{row['example_id']}.wav"
            or row["development_only"] is not True
            or row["citable"] is not False
        ):
            raise ValueError("Invalid temporal source grouping, identity or frozen split.")
        seen_files.add(row["output_filename"])
        seen_ids.add(row["example_id"])
        actual_hash = file_hashes[row["output_filename"]]
        if row["generated_sha256"] != actual_hash:
            raise ValueError("Temporal recording hash disagrees with its validated inventory.")
        previous_split = hashes.setdefault(actual_hash, split)
        if previous_split != split:
            raise ValueError("Identical recordings cross frozen temporal splits.")
        groups.setdefault(group, []).append(row)
        for key in ("source_real_id", "source_donor_id"):
            record = catalog[row[key]]
            if record["utt_id"] != group or record["split"] != split:
                raise ValueError("A temporal donor/source crosses its frozen utterance group.")
            old = hashes.setdefault(record["sha256"], split)
            if old != split:
                raise ValueError("Identical source bytes cross temporal dataset splits.")
        if split == "eval":
            continue
        synthetic = row["example_class"].startswith("mixed_")
        donor_condition = row["example_class"].removeprefix("mixed_") if synthetic else "real"
        if (
            row["has_synthetic"] is not synthetic
            or catalog[row["source_real_id"]]["condition"] != "real"
            or catalog[row["source_donor_id"]]["condition"] != donor_condition
            or row["source_condition"] != donor_condition
            or row["source_synthetic_id"] != (row["source_donor_id"] if synthetic else None)
        ):
            raise ValueError("Control/synthetic annotation disagreement.")
        expected = interval_labels(
            row["num_samples"],
            row["start_sample"],
            row["end_sample"],
            row["crossfade_samples_per_boundary"],
            synthetic=synthetic,
        )
        if row["interval_labels"] != expected or labels.get(row["example_id"]) != expected:
            raise ValueError(
                "Sample-accurate temporal labels disagree with manipulation semantics."
            )
        selected[split].append(row)
    if any(
        len(group) != 5
        or {r["example_class"] for r in group} != CLASSES
        or len({r["split"] for r in group}) != 1
        for group in groups.values()
    ):
        raise ValueError("Temporal groups must contain exactly five frozen paired examples.")
    if not all(selected.values()):
        raise ValueError("Both frozen train and dev partitions must be nonempty.")
    for split in selected:
        selected[split].sort(key=lambda r: (r["group_id"], r["example_class"], r["example_id"]))
    return selected, {
        "dataset_manifest_sha256": sha256_file(root / "manifest.json"),
        "dataset_inventory_sha256": sha256_file(root / "artifacts.sha256.json"),
        "source_manifest_sha256": sha256_file(source_manifest),
        "frozen_group_counts": dict(Counter(g[0]["split"] for g in groups.values())),
        "consumed_group_ids": {
            s: sorted({r["group_id"] for r in rr}) for s, rr in selected.items()
        },
        "consumed_recordings": {s: len(rr) for s, rr in selected.items()},
        "eval_features": 0,
        "eval_predictions": 0,
        "eval_waveforms_decoded": 0,
        "eval_inventory_hashing_only": True,
        "source_provenance": provenance,
        "research_gate_admission": False,
        **FLAGS,
    }


def extract_temporal_dataset(
    root: Path, selected: dict, config: TemporalExperimentConfig, work: Path
) -> dict:
    """Decode/filter each paired group once; extract both predeclared acoustic grids."""
    result = {(w, h): {"train": [], "dev": []} for w, h in config.settings}
    for split in ("train", "dev"):
        rows = selected[split]
        group_ids = sorted({r["group_id"] for r in rows})
        for group in group_ids:
            members = [r for r in rows if r["group_id"] == group]
            keys = {
                r["example_id"]: (
                    "real" if r["example_class"] == "null_control" else r["example_id"]
                )
                for r in members
            }
            wavs = {}
            for row in members:
                wav = read_input(root / row["output_filename"])
                if len(wav) != row["num_samples"] or len(wav) > 60 * ARCHIVE_SR:
                    raise ValueError("Temporal waveform duration mismatch or exceeds 60 s.")
                wavs[keys[row["example_id"]]] = wav
            analysis = analysis_comparison(wavs, work)
            for row in members:
                raw_labels = sample_labels(row["interval_labels"], row["num_samples"])
                for window, hop in config.settings:
                    cfg = WindowConfig(window_ms=window, hop_ms=hop)
                    starts, features = extract_windows(analysis[keys[row["example_id"]]], cfg)
                    labels = guarded_labels(raw_labels, cfg.guard_samples)
                    yy = assign_window_labels(labels, starts, cfg.window_samples)
                    item = {"row": row, "starts": starts, "features": features, "window_y": yy}
                    if split == "dev":
                        item["score_labels"] = guarded_labels(
                            raw_labels, config.score_guard_samples
                        )
                    result[(window, hop)][split].append(item)
    return result


def fit_temporal_head(records: list[dict], c: float, seed: int) -> tuple[LinearHead, dict]:
    x = np.vstack([r["features"] for r in records])
    y = np.concatenate([r["window_y"] for r in records])
    take = y >= 0
    counts = np.bincount(y[take], minlength=2)
    if (counts == 0).any():
        raise ValueError("Training needs pure windows of both real and synthetic origin.")
    with threadpool_limits(limits=1):
        pipeline = fit_head(x[take], y[take], c, seed, balanced=True)
    return LinearHead.from_pipeline(pipeline), {
        "real_windows": int(counts[0]),
        "synthetic_windows": int(counts[1]),
        "ignored_windows": int(np.sum(~take)),
        "scaler_fit_split": "train_only",
        "class_weights": {str(i): float(counts.sum() / (2 * counts[i])) for i in (0, 1)},
        "weighting": "equal_labeled_windows_then_balanced_classes_no_independent_window_CI",
    }


def code_provenance():
    root = Path(__file__).resolve().parents[2]
    paths = [
        "src/detectors/temporal_features.py",
        "src/detectors/temporal_metrics.py",
        "src/detectors/temporal_model.py",
        "src/experiments/temporal_localization.py",
        "src/experiments/cli.py",
        "src/experiments/forensic_pilot.py",
        "src/experiments/partial_audio.py",
        "src/experiments/partial_audio_upload.py",
        "src/data/preprocess.py",
        "src/data/audio_io.py",
        "src/data/invariants.py",
        "src/detectors/base.py",
        "src/detectors/eer.py",
        "CLAUDE.md",
        "pyproject.toml",
    ]
    head = subprocess.run(
        ["git", "--no-optional-locks", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return {
        "head": head,
        "source_sha256": {p: sha256_file(root / p) for p in paths},
        "environment": runtime_versions(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "blas_thread_limit": 1,
        **FLAGS,
    }


def _metadata(config: WindowConfig):
    return {
        "protocol": PROTOCOL,
        "feature_names": config.feature_names,
        "score_direction": "higher_is_synthetic",
        "probability_calibration": "uncalibrated_class_balanced_logistic_output",
        **FLAGS,
    }


def _window_counts(records):
    labels = np.concatenate([r["window_y"] for r in records])
    return {
        "real": int(np.sum(labels == 0)),
        "synthetic": int(np.sum(labels == 1)),
        "ignored": int(np.sum(labels == -1)),
        "all": len(labels),
        "recordings": len(records),
        "source_groups": len({r["row"]["group_id"] for r in records}),
    }


def _write_predictions(root, records, probabilities, bundle):
    root.mkdir()
    margins = [record["margin"] for record in records]
    arrays = {
        "record_example_ids": np.asarray([r["row"]["example_id"] for r in records]),
        "record_source_groups": np.asarray([r["row"]["group_id"] for r in records]),
        "record_offsets": np.r_[0, np.cumsum([len(r["starts"]) for r in records])],
        "window_start_samples": np.concatenate([r["starts"] for r in records]),
        "window_end_samples": np.concatenate(
            [r["starts"] + bundle.config.window_samples for r in records]
        ),
        "window_labels": np.concatenate([r["window_y"] for r in records]),
        "synthetic_margins": np.concatenate(margins),
        "uncalibrated_probabilities": np.concatenate(probabilities),
        "smoothed_probabilities": np.concatenate(
            [smooth_probabilities(p, bundle.smoothing) for p in probabilities]
        ),
    }
    np.savez_compressed(root / "window_predictions.npz", **arrays)


def train_temporal(
    dataset_dir: str | Path,
    source_manifest: str | Path,
    out_dir: str | Path,
    config: TemporalExperimentConfig = DEFAULT_TEMPORAL_CONFIG,
) -> dict:
    config.validate()
    dataset, original = Path(dataset_dir).resolve(), Path(source_manifest).resolve()
    target = Path(out_dir).resolve()
    staging = target.with_name(target.name + ".staging")
    for protected in (dataset, original.parent):
        if (
            target.is_relative_to(protected)
            or protected.is_relative_to(target)
            or staging.is_relative_to(protected)
        ):
            raise ValueError("Temporal run must be separate from all source dataset paths.")
    if target.exists() or staging.exists():
        raise FileExistsError("Temporal output/staging exists; never overwrite or resume.")
    selected, identity = load_temporal_rows(dataset, original)
    before = artifact_inventory(dataset)
    support = filter_support_receipt()
    staging.mkdir(parents=True)
    work = staging / "_analysis"
    work.mkdir()
    try:
        write_json(
            staging / "config.json",
            {
                "search": asdict(config),
                "protocol": PROTOCOL,
                "score_guard_samples": config.score_guard_samples,
                "selection_rule": "dev_source_macro_sample_f1_then_control_fpr_then_simplicity",
                "window_label_policy": "100_percent_single_class_no_ignore_mixture_or_padding",
                "filter_support": support,
                **FLAGS,
            },
        )
        write_json(
            staging / "provenance.json",
            {
                "code": code_provenance(),
                "data": identity,
                "scope": "train_dev_only_exploratory",
                **FLAGS,
            },
        )
        extracted = extract_temporal_dataset(dataset, selected, config, work)
        work.rmdir()  # Only the now-empty temporary-analysis parent created by this run.
        best, candidates, fits, feature_counts = None, [], {}, {}
        with threadpool_limits(limits=1):
            for (window, hop), splits in sorted(extracted.items()):
                cfg = WindowConfig(window_ms=window, hop_ms=hop)
                feature_counts[f"{window}:{hop}"] = {
                    s: _window_counts(rr) for s, rr in splits.items()
                }
                if not (np.concatenate([r["score_labels"] for r in splits["dev"]]) == 1).any():
                    raise ValueError("No scoreable dev positives under the fixed support policy.")
                for c in sorted(set(config.c_grid)):
                    head, fit = fit_temporal_head(splits["train"], c, config.seed)
                    fits[f"{window}:{hop}:{c}"] = fit
                    margins = [head.decision_function(r["features"])[:, 0] for r in splits["dev"]]
                    probabilities = [expit(m) for m in margins]
                    for width in sorted(set(config.smoothing_widths)):
                        for merge_ms in sorted(set(config.merge_ms)):
                            merge = round(merge_ms * ARCHIVE_SR / 1000)
                            for threshold in sorted(set(config.thresholds)):
                                metrics = evaluate_localization(
                                    splits["dev"],
                                    probabilities,
                                    cfg,
                                    threshold,
                                    width,
                                    merge,
                                    config.score_guard_samples,
                                    detailed=False,
                                )
                                score = metrics["source_group_macro_f1"]
                                if score is None:
                                    raise ValueError("Dev source-group selection metric undefined.")
                                candidate = {
                                    "window_ms": window,
                                    "hop_ms": hop,
                                    "C": c,
                                    "threshold": threshold,
                                    "smoothing": width,
                                    "merge_ms": merge_ms,
                                    "dev_source_macro_sample_f1": score,
                                    "dev_control_sample_fpr": metrics[
                                        "control_sample_false_positive_rate"
                                    ],
                                }
                                candidates.append(candidate)
                                key = (
                                    -score,
                                    metrics["control_sample_false_positive_rate"],
                                    c,
                                    window,
                                    hop,
                                    width,
                                    merge_ms,
                                    -threshold,
                                )
                                if best is None or key < best[0]:
                                    best = (key, head, cfg, threshold, width, merge, candidate, fit)
        _, head, cfg, threshold, width, merge, chosen, fit = best
        bundle = TemporalBundle(head, cfg, threshold, width, merge, _metadata(cfg))
        save_temporal_bundle(bundle, staging / "bundle")
        restored = load_temporal_bundle(staging / "bundle")
        write_json(
            staging / "selection.json",
            {
                "candidates": candidates,
                "selected": chosen,
                "train_fit": fit,
                "fit_diagnostics": fits,
                "window_counts_by_setting": feature_counts,
                "dev_refit": False,
                "eval_used": False,
                **FLAGS,
            },
        )
        result = {
            "selected": chosen,
            "window_counts": {},
            "metrics": {},
            "filter_support": support,
            "score_guard_samples": config.score_guard_samples,
            "eval_used": False,
            "scope": "development_train_dev_only",
            **FLAGS,
        }
        with threadpool_limits(limits=1):
            for split, records in extracted[(cfg.window_ms, cfg.hop_ms)].items():
                probabilities = []
                for record in records:
                    margin = bundle.margins(record["features"])
                    if not np.array_equal(margin, restored.margins(record["features"])):
                        raise ValueError("Numeric bundle round-trip changed inference.")
                    record["margin"] = margin
                    probabilities.append(expit(margin))
                metrics = evaluate_localization(
                    records, probabilities, cfg, threshold, width, merge, config.score_guard_samples
                )
                outputs = metrics.pop("recordings")
                _write_predictions(staging / split, records, probabilities, bundle)
                write_json(staging / split / "recordings.json", {"recordings": outputs, **FLAGS})
                write_json(staging / split / "metrics.json", {**metrics, **FLAGS})
                result["window_counts"][split] = _window_counts(records)
                result["metrics"][split] = metrics
        if (
            artifact_inventory(dataset) != before
            or sha256_file(original) != identity["source_manifest_sha256"]
        ):
            raise ValueError("Immutable dataset/source identity changed during temporal training.")
        write_json(staging / "metrics.json", result)
        write_json(
            staging / "validation.json",
            {
                "status": "passed",
                "dataset_bytes_unchanged": True,
                "numeric_bundle_roundtrip": "exact",
                "eval_predictions": 0,
                "eval_waveforms_decoded": 0,
                "all_features_finite": True,
                **FLAGS,
            },
        )
        write_json(
            staging / "artifacts.sha256.json", {"files": artifact_inventory(staging), **FLAGS}
        )
        staging.rename(target)
        return result
    except Exception as exc:
        write_json(staging / "failure.json", {"status": "failed", "error": str(exc), **FLAGS})
        raise
