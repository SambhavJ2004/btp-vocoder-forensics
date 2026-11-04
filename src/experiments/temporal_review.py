"""Read-only Phase 3A audit and train/dev-only Phase 3B input contracts.

The previously sealed Phase 3A run is the dataset-identity trust anchor. Full
package admission is historical; current consumed bytes are verified without
opening eval WAVs or using eval annotation records. No package gate is changed.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score, roc_curve

from data.invariants import ARCHIVE_SR, GLOBAL_SEED
from data.partial_audio import interval_labels
from detectors.forensic_pilot import sha256_file
from detectors.temporal_features import guarded_labels, sample_labels
from detectors.temporal_metrics import confusion, rates
from experiments.partial_audio import FLAGS, source_frame
from experiments.partial_audio_upload import _package_name
from experiments.temporal_localization import CLASSES

SCORE_GUARD = 4544  # Fixed Phase 3A denominator, independent of candidate support.


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def checked_inventory(root: Path, expected: dict, *, skip_wavs=frozenset()):
    """Reject additions/links and check all in-scope bytes, with explicit scope."""
    root = root.resolve()
    files, directories = set(), set()
    for path in root.rglob("*"):
        if (
            path.is_symlink()
            or getattr(path, "is_junction", lambda: False)()
            or path.resolve() != path
            or not path.resolve().is_relative_to(root)
        ):
            raise ValueError("Unsafe link or escaped review input path.")
        name = _package_name(path.relative_to(root).as_posix())
        if path.is_dir():
            directories.add(name)
        elif path.is_file():
            if path.stat().st_nlink != 1 or path.stat().st_size <= 0:
                raise ValueError("Unsafe or empty review input file.")
            files.add(name)
        else:
            raise ValueError("Non-regular review input.")
    parents = {p.as_posix() for name in expected for p in Path(name).parents if p != Path(".")}
    if files != set(expected) or directories != parents:
        raise ValueError("Review input inventory has missing/unexpected files or directories.")
    for name, digest in expected.items():
        if name not in skip_wavs and sha256_file(root / name) != digest:
            raise ValueError(f"Review input hash mismatch: {name}")


def review_rows(dataset_dir, source_manifest, baseline_dir):
    """Reuse frozen manifests; exclude eval before consuming annotation fields."""
    root, baseline = Path(dataset_dir).resolve(), Path(baseline_dir).resolve()
    seal = read_json(baseline / "artifacts.sha256.json")
    checked_inventory(
        baseline,
        {**seal["files"], "artifacts.sha256.json": sha256_file(baseline / "artifacts.sha256.json")},
    )
    receipt = read_json(baseline / "validation.json")
    identity = read_json(baseline / "provenance.json")["data"]
    if (
        receipt.get("status") != "passed"
        or receipt.get("eval_predictions") != 0
        or identity["dataset_manifest_sha256"] != sha256_file(root / "manifest.json")
        or identity["dataset_inventory_sha256"] != sha256_file(root / "artifacts.sha256.json")
        or identity["source_manifest_sha256"] != sha256_file(source_manifest)
    ):
        raise ValueError("Phase 3A/data/source identity or validation trust anchor differs.")
    selected = {"train": [], "dev": []}
    for row in read_json(root / "manifest.json")["examples"]:
        if row["split"] in selected:
            selected[row["split"]].append(row)
    required = {r["output_filename"] for rows in selected.values() for r in rows}
    inventory = read_json(root / "artifacts.sha256.json")["files"]
    skip = {name for name in inventory if name.lower().endswith(".wav") and name not in required}
    checked_inventory(
        root,
        {**inventory, "artifacts.sha256.json": identity["dataset_inventory_sha256"]},
        skip_wavs=skip,
    )
    source = source_frame(source_manifest)
    frozen = source.groupby("utt_id").split.first().to_dict()
    catalog = read_json(root / "sources.json")["sources"]
    labels = read_json(root / "labels.json")["examples"]
    for split, rows in selected.items():
        groups, seen = defaultdict(list), set()
        for row in rows:
            group = row["group_id"]
            synthetic = row["example_class"].startswith("mixed_")
            condition = row["example_class"].removeprefix("mixed_") if synthetic else "real"
            expected = interval_labels(
                row["num_samples"],
                row["start_sample"],
                row["end_sample"],
                row["crossfade_samples_per_boundary"],
                synthetic=synthetic,
            )
            if (
                frozen.get(group) != split
                or row["original_utterance_id"] != group
                or row["example_class"] not in CLASSES
                or row["has_synthetic"] is not synthetic
                or row["example_id"] in seen
                or row["variant"] != 1
                or row["output_filename"] != f"audio/{row['example_id']}.wav"
                or row["generated_sha256"] != inventory[row["output_filename"]]
                or row["interval_labels"] != expected
                or labels[row["example_id"]] != expected
                or row["development_only"] is not True
                or row["citable"] is not False
            ):
                raise ValueError("Train/dev grouping, sample labels or provenance disagree.")
            for key, expected_condition in (
                ("source_real_id", "real"),
                ("source_donor_id", condition),
            ):
                item = catalog[row[key]]
                if (
                    item["utt_id"] != group
                    or item["split"] != split
                    or item["condition"] != expected_condition
                ):
                    raise ValueError("Train/dev source/donor identity or grouping disagrees.")
            groups[group].append(row)
            seen.add(row["example_id"])
        if not rows or any(
            len(g) != 5 or {r["example_class"] for r in g} != CLASSES for g in groups.values()
        ):
            raise ValueError("Every consumed source must retain its five conditions.")
        rows.sort(key=lambda r: (r["group_id"], r["example_class"], r["example_id"]))
    if {r["group_id"] for r in selected["train"]} & {r["group_id"] for r in selected["dev"]}:
        raise ValueError("Source group leakage.")
    return selected, {
        "dataset_identity": identity,
        "consumed_wav_hashes": {n: inventory[n] for n in sorted(required)},
        "waveform_hashes_skipped": len(skip),
        "eval_waveforms_opened": 0,
        "eval_annotation_records_used": 0,
        "scope": "current_train_dev_plus_metadata_historical_full_package_admission",
        **FLAGS,
    }


def independent_guard(row, radius):
    """Separate interval/index oracle, without the production prefix-sum path."""
    labels = np.full(row["num_samples"], -99, dtype=np.int8)
    for interval in row["interval_labels"]:
        labels[interval["start_sample"] : interval["end_sample"]] = interval["label"]
    if np.any(labels == -99):
        raise ValueError("Incomplete raw sample truth.")
    result = labels.copy()
    for interval in row["interval_labels"]:
        if interval["label"] == -1:
            result[
                max(0, interval["start_sample"] - radius) : min(
                    len(labels), interval["end_sample"] + radius
                )
            ] = -1
    if radius:
        for point in np.flatnonzero(labels[1:] != labels[:-1]) + 1:
            result[max(0, point - radius) : min(len(labels), point + radius + 1)] = -1
        result[:radius] = result[max(0, len(labels) - radius) :] = -1
    return result


def center_edges(starts, n, length):
    centers = np.minimum(np.asarray(starts) + length / 2, n)
    return np.r_[0, np.floor((centers[:-1] + centers[1:]) / 2).astype(np.int64), n]


def saved_records(baseline, selected, split):
    rows = {r["example_id"]: r for r in selected[split]}
    with np.load(Path(baseline) / split / "window_predictions.npz", allow_pickle=False) as arrays:
        arrays = {name: arrays[name] for name in arrays.files}
        ids = arrays["record_example_ids"].tolist()
        if set(ids) != set(rows) or len(ids) != len(rows):
            raise ValueError("Saved predictions differ from frozen train/dev IDs.")
        records = []
        for i, key in enumerate(ids):
            begin, end = arrays["record_offsets"][i : i + 2]
            records.append(
                {
                    "row": rows[key],
                    "starts": arrays["window_start_samples"][begin:end].copy(),
                    "window_y": arrays["window_labels"][begin:end].copy(),
                    "margin": arrays["synthetic_margins"][begin:end].copy(),
                    "probability": arrays["uncalibrated_probabilities"][begin:end].copy(),
                    "smoothed": arrays["smoothed_probabilities"][begin:end].copy(),
                }
            )
    return records


def distribution(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {
            "count": 0,
            "min": None,
            "p05": None,
            "median": None,
            "p95": None,
            "max": None,
            "mean": None,
        }
    q = np.quantile(values, [0, 0.05, 0.5, 0.95, 1])
    return dict(
        zip(("min", "p05", "median", "p95", "max"), q.tolist(), strict=True),
        count=len(values),
        mean=float(values.mean()),
    )


def diagnostic_predictions(records, config, threshold, score_guard=SCORE_GUARD):
    """Sample-weighted scores, boundary errors and false-alarm durations; no fitting."""
    summaries = defaultdict(lambda: defaultdict(list))
    by_class, by_source, by_record = {}, {}, {}
    bins = ((0, 0.05), (0.05, 0.15), (0.15, 0.25), (0.25, 0.5), (0.5, 1), (1, float("inf")))
    distances = {f"{lo}:{hi}": {} for lo, hi in bins}
    interval_durations, fp_run_durations = [], []
    sample_score_hist = {
        kind: np.zeros(20, dtype=np.int64) for kind in ("real", "synthetic", "ignore")
    }
    edges = np.linspace(0, 1, 21)
    for record in records:
        row, p = record["row"], record["smoothed"]
        bounds = center_edges(record["starts"], row["num_samples"], config.window_samples)
        dense_p = np.repeat(p, np.diff(bounds))
        truth = independent_guard(row, score_guard)
        pred = dense_p >= threshold
        counts = confusion(truth, pred)
        rate = rates(counts)
        by_record[row["example_id"]] = {
            "group_id": row["group_id"],
            "example_class": row["example_class"],
            **rate,
        }
        for table, key in ((by_class, row["example_class"]), (by_source, row["group_id"])):
            target = table.setdefault(key, Counter())
            target.update(counts)
        for label, name in ((0, "real"), (1, "synthetic"), (-1, "ignore")):
            # Histogram counts are samples, including projected endpoint cells.
            sample_score_hist[name] += np.histogram(dense_p[truth == label], bins=edges)[0]
            summaries[row["example_class"]][name].extend(
                dense_p[truth == label][:: max(1, len(dense_p) // 2000)].tolist()
            )
        distance = (
            np.minimum(
                abs(np.arange(len(truth)) - row["start_sample"]),
                abs(np.arange(len(truth)) - row["end_sample"]),
            )
            / ARCHIVE_SR
        )
        for lo, hi in bins:
            mask = (distance >= lo) & (distance < hi)
            target = distances[f"{lo}:{hi}"]
            for key, value in confusion(truth[mask], pred[mask]).items():
                target[key] = target.get(key, 0) + value
        for mask, target in ((pred, interval_durations), (pred & (truth == 0), fp_run_durations)):
            runs = np.flatnonzero(np.diff(np.r_[False, mask, False])).reshape(-1, 2)
            target.extend(((runs[:, 1] - runs[:, 0]) / ARCHIVE_SR).tolist())
    return {
        "threshold": threshold,
        "score_histogram_edges": edges.tolist(),
        "sample_score_histogram": {k: v.tolist() for k, v in sample_score_hist.items()},
        "score_distributions": {
            k: {name: distribution(v) for name, v in values.items()}
            for k, values in summaries.items()
        },
        "score_quantile_scope": (
            "deterministic_per_record_stride_sample_for_diagnostics_only_histograms_exact"
        ),
        "by_class": {k: rates(v) for k, v in sorted(by_class.items())},
        "by_source": {k: rates(v) for k, v in sorted(by_source.items())},
        "by_recording": by_record,
        "distance_from_edit_seconds": {k: rates(v) for k, v in distances.items()},
        "predicted_interval_seconds": distribution(interval_durations),
        "valid_real_false_positive_run_seconds": distribution(fp_run_durations),
        **FLAGS,
    }


def audit_saved_predictions(baseline, selected):
    """Recompute labels/metrics from saved scores independently of metric driver."""
    main = read_json(Path(baseline) / "metrics.json")
    length = round(main["selected"]["window_ms"] * ARCHIVE_SR / 1000)
    guard = main["score_guard_samples"]
    result = {}
    for split in ("train", "dev"):
        records = saved_records(baseline, selected, split)
        outputs = {
            r["example_id"]: r
            for r in read_json(Path(baseline) / split / "recordings.json")["recordings"]
        }
        total, groups, ys, ms = Counter(), defaultdict(Counter), [], []
        for record in records:
            row, starts, margin = record["row"], record["starts"], record["margin"]
            n = row["num_samples"]
            expected_starts = list(
                range(
                    0, max(0, n - length) + 1, round(main["selected"]["hop_ms"] * ARCHIVE_SR / 1000)
                )
            )
            if expected_starts[-1] != max(0, n - length):
                expected_starts.append(max(0, n - length))
            np.testing.assert_array_equal(starts, expected_starts)
            p = expit(margin)
            np.testing.assert_allclose(p, record["probability"], rtol=0, atol=1e-15)
            expected_p = (
                (np.r_[p[0], p[:-1]] + p + np.r_[p[1:], p[-1]]) / 3
                if main["selected"]["smoothing"] == 3
                else p
            )
            # Direct three-term arithmetic and convolution can differ by a few
            # float64 ulps, especially for probabilities close to one.
            np.testing.assert_allclose(
                expected_p, record["smoothed"], rtol=0, atol=4 * np.finfo(float).eps
            )
            truth = independent_guard(row, 3308)
            prefix_real, prefix_syn = (np.r_[0, np.cumsum(truth == y)] for y in (0, 1))
            yy = np.full(len(starts), -1, dtype=np.int8)
            for y, prefix in enumerate((prefix_real, prefix_syn)):
                full = np.flatnonzero(starts + length <= n)
                take = full[prefix[starts[full] + length] - prefix[starts[full]] == length]
                yy[take] = y
            np.testing.assert_array_equal(yy, record["window_y"])
            dense = np.repeat(
                record["smoothed"] >= main["selected"]["threshold"],
                np.diff(center_edges(starts, n, length)),
            )
            # This saved baseline has zero merge gap; refuse an unsupported oracle recipe.
            if main["selected"]["merge_ms"] != 0:
                raise ValueError("Independent baseline audit requires the recorded zero-gap model.")
            predicted = np.flatnonzero(np.diff(np.r_[False, dense, False])).reshape(-1, 2).tolist()
            saved = outputs[row["example_id"]]
            assert predicted == [
                [r["start_sample"], r["end_sample"]] for r in saved["model_predicted_intervals"]
            ]
            for interval in saved["model_predicted_intervals"]:
                assert interval["start_seconds"] == interval["start_sample"] / ARCHIVE_SR
                assert interval["end_seconds"] == interval["end_sample"] / ARCHIVE_SR
            counts = confusion(independent_guard(row, guard), dense)
            assert counts == {k: saved["sample_metrics"][k] for k in counts}
            total.update(counts)
            groups[row["group_id"]].update(counts)
            ys.append(yy[yy >= 0])
            ms.append(margin[yy >= 0])
        saved = main["metrics"][split]
        y, margin = np.concatenate(ys), np.concatenate(ms)
        fpr, tpr, _ = roc_curve(y, margin, drop_intermediate=False)
        d = fpr - (1 - tpr)
        i = int(np.searchsorted(d, 0))
        eer = float(
            fpr[i]
            if d[i] == 0
            else fpr[i - 1] + (-d[i - 1]) / (d[i] - d[i - 1]) * (fpr[i] - fpr[i - 1])
        )
        auc = float(roc_auc_score(y, margin))
        macro = float(np.mean([rates(c)["f1"] for c in groups.values()]))
        assert abs(eer - saved["window_micro"]["raw_directional_eer"]) < 1e-12
        assert abs(auc - saved["window_micro"]["raw_auroc"]) < 1e-14
        assert abs(macro - saved["source_group_macro_f1"]) < 1e-14
        assert dict(total) == {k: saved["sample_micro"][k] for k in total}
        result[split] = {
            "recordings": len(records),
            "groups": len(groups),
            "auroc": auc,
            "eer": eer,
            "source_macro_sample_f1": macro,
            "sample_counts": dict(total),
        }
    return {"status": "passed", "splits": result, "correctness_defects_found": [], **FLAGS}


def assert_current_guards(row):
    """Expose production/oracle agreement to CPU tests without reading audio."""
    for guard in (0, 3308, SCORE_GUARD):
        np.testing.assert_array_equal(
            independent_guard(row, guard),
            guarded_labels(sample_labels(row["interval_labels"], row["num_samples"]), guard),
        )


def audit_phase3a(dataset_dir, source_manifest, baseline_dir, out_dir, protected_dirs=()):
    """Fresh, reproducible audit without rewriting any previous run/data file."""
    import subprocess

    from threadpoolctl import threadpool_limits

    from detectors.forensic_pilot import write_json
    from detectors.temporal_features import WindowConfig, filter_support_receipt
    from detectors.temporal_model import load_temporal_bundle
    from experiments.partial_audio import artifact_inventory, read_input, validate_written_example
    from experiments.temporal_ablation import fixed_plan
    from experiments.temporal_localization import (
        TemporalExperimentConfig,
        code_provenance,
        extract_temporal_dataset,
        fit_temporal_head,
    )

    data, source, base, output = (
        Path(p).resolve() for p in (dataset_dir, source_manifest, baseline_dir, out_dir)
    )
    roots = list(
        dict.fromkeys((data, source.parent, base, *(Path(p).resolve() for p in protected_dirs)))
    )
    if output.exists():
        raise FileExistsError("Audit output must be fresh.")
    if any(output.is_relative_to(root) or root.is_relative_to(output) for root in roots):
        raise ValueError("Audit output must be separate from all protected inputs.")
    selected, identity = review_rows(data, source, base)
    output.mkdir(parents=True)
    stats = {
        str(root): {
            p.relative_to(root).as_posix(): [p.stat().st_size, p.stat().st_mtime_ns]
            for p in root.rglob("*")
            if p.is_file()
        }
        for root in roots
    }
    write_json(
        output / "before.json",
        {
            "roots_stat": stats,
            "baseline_sha256": artifact_inventory(base),
            "head": subprocess.check_output(
                ["git", "--no-optional-locks", "rev-parse", "HEAD"], text=True
            ).strip(),
            **FLAGS,
        },
    )
    write_json(output / "predeclared.json", fixed_plan())
    write_json(output / "input_scope.json", identity)
    code = code_provenance()
    code["source_sha256"]["src/experiments/temporal_review.py"] = sha256_file(__file__)
    write_json(
        output / "provenance.json",
        {
            "code": code,
            "baseline_inventory_sha256": sha256_file(base / "artifacts.sha256.json"),
            **FLAGS,
        },
    )
    evidence = audit_saved_predictions(base, selected)
    catalog = read_json(data / "sources.json")["sources"]
    frame = source_frame(source)
    paths = {(str(r.utt_id), str(r.condition)): Path(r.path) for r in frame.itertuples()}
    source_hashes, validated = {}, 0
    for split, rows in selected.items():
        for group in sorted({r["group_id"] for r in rows}):
            members = [r for r in rows if r["group_id"] == group]
            needed = {r[key] for r in members for key in ("source_real_id", "source_donor_id")}
            wavs = {}
            for key in needed:
                item = catalog[key]
                if item["split"] != split or item["utt_id"] != group:
                    raise ValueError("Source reconstruction cannot cross frozen groups.")
                path = paths[(group, item["condition"])]
                if sha256_file(path) != item["sha256"]:
                    raise ValueError("Original source byte identity changed.")
                source_hashes[str(path)] = item["sha256"]
                wavs[key] = read_input(path)
            for row in members:
                validate_written_example(
                    data, row, wavs[row["source_real_id"]], wavs[row["source_donor_id"]]
                )
                validated += 1
    evidence.update(
        independent_pcm_oracle_recordings=validated,
        original_train_dev_source_hashes=source_hashes,
        source_alignment_scope="exact_declared_sample_pairing_and_splice_not_new_legacy_acoustic_alignment_certification",
        filter_support=filter_support_receipt(),
    )
    dev = saved_records(base, selected, "dev")
    cfg = WindowConfig(window_ms=64, hop_ms=16)
    for threshold in fixed_plan()["fixed_dev_diagnostic_thresholds"]:
        write_json(
            output / f"dev-diagnostic-{threshold}.json", diagnostic_predictions(dev, cfg, threshold)
        )
    work = output / "_analysis"
    work.mkdir()
    with threadpool_limits(limits=1):
        extracted = extract_temporal_dataset(
            data, selected, TemporalExperimentConfig(settings=((64.0, 16.0),)), work
        )[(64.0, 16.0)]
        work.rmdir()
        old = load_temporal_bundle(base / "bundle")
        head, fit = fit_temporal_head(extracted["train"], 1.0, GLOBAL_SEED)
        for field in ("mean", "scale", "coefficients", "intercept"):
            np.testing.assert_array_equal(getattr(head, field), getattr(old.head, field))
        for split, records in extracted.items():
            saved = {r["row"]["example_id"]: r for r in saved_records(base, selected, split)}
            for record in records:
                ref = saved[record["row"]["example_id"]]
                np.testing.assert_array_equal(record["starts"], ref["starts"])
                np.testing.assert_array_equal(record["window_y"], ref["window_y"])
                np.testing.assert_array_equal(
                    head.decision_function(record["features"])[:, 0], ref["margin"]
                )
    evidence.update(
        original_numeric_model_reproduction="all_four_arrays_exact",
        original_margin_reproduction=f"all_{validated}_recordings_exact",
        train_only_scaler_refit=fit,
    )
    _, after = review_rows(data, source, base)
    if after != identity:
        raise ValueError("Input identity changed during the audit.")
    write_json(output / "correctness.json", evidence)
    return {"status": "passed", "recordings": validated, "output": str(output), **FLAGS}


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Independent Phase 3A audit; train/dev only.")
    for name in ("dataset-dir", "source-manifest", "baseline-dir", "out-dir"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--protected-dir", action="append", default=[])
    args = parser.parse_args(argv)
    print(
        audit_phase3a(
            args.dataset_dir,
            args.source_manifest,
            args.baseline_dir,
            args.out_dir,
            args.protected_dir,
        )
    )


if __name__ == "__main__":
    main()
