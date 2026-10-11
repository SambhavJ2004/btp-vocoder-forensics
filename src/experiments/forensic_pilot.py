"""Train/evaluate the explicitly non-citable multi-vocoder CPU development pilot."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import warnings
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data.invariants import (
    ARCHIVE_TIER,
    GLOBAL_SEED,
    PRIMARY_ANALYSIS,
    REAL_CONDITION,
    InvariantViolation,
)
from data.manifest import (
    analysis_frame,
    read_manifest,
    require_primary,
    select_tier,
    validate_manifest,
)
from detectors.eer import compute_eer
from detectors.forensic_features import (
    FeatureConfig,
    extract_comparison_features,
    read_archive_input,
)
from detectors.forensic_pilot import (
    BINARY_CLASSES,
    BUNDLE_VERSION,
    KNOWN_SOURCES,
    PILOT_PROTOCOL,
    SOURCE_CLASSES,
    LinearHead,
    PilotBundle,
    load_bundle,
    preprocessing_identity,
    runtime_versions,
    save_bundle,
    sha256_file,
    write_json,
)

C_GRID = (0.01, 0.1, 1.0, 10.0)


def binary_labels(conditions) -> np.ndarray:
    values = np.asarray(conditions, dtype=str)
    if not set(values) <= {REAL_CONDITION, *KNOWN_SOURCES}:
        raise ValueError("Unknown pilot condition.")
    return (values == REAL_CONDITION).astype(int)


def attribution_labels(conditions) -> np.ndarray:
    mapping = {name: i for i, name in enumerate(KNOWN_SOURCES)}
    if not set(conditions) <= set(mapping):
        raise ValueError("Attribution requires known synthetic conditions only.")
    return np.array([mapping[str(name)] for name in conditions], dtype=int)


def validate_pilot_frame(frame: pd.DataFrame) -> None:
    """Additional loader guards, without changing any existing dataset gate."""
    required = ["utt_id", "condition", "tier", "split", "speaker", "path"]
    if frame[required].isna().any().any() or frame.empty:
        raise InvariantViolation("Pilot manifest has missing identities or no archive rows.")
    if set(frame.condition) != {REAL_CONDITION, *KNOWN_SOURCES}:
        raise InvariantViolation("Pilot requires real, bigvgan_112m, griffin_lim and vocos.")
    if set(frame.split) != {"train", "dev", "eval"}:
        raise InvariantViolation("Pilot requires the existing train/dev/eval splits.")
    if (frame.duplicated(["utt_id", "condition", "tier"]).any()
            or frame.path.duplicated().any()):
        raise InvariantViolation("Duplicate pilot recordings; do not duplicate real rows.")
    grouped = frame.groupby("utt_id")
    if ((grouped.split.nunique() != 1).any() or (grouped.speaker.nunique() != 1).any()
            or (grouped.condition.nunique() != len(KNOWN_SOURCES) + 1).any()):
        raise InvariantViolation("Pilot conditions must share utterance/speaker/split groups.")
    if frame.speaker.nunique() > 1 and (frame.groupby("speaker").split.nunique() > 1).any():
        raise InvariantViolation("INV-09: multi-speaker pilot splits must be speaker-disjoint.")


def load_pilot_dataset(manifest: str | Path) -> tuple[pd.DataFrame, dict]:
    """Resolve existing paths, validate frozen groups and hash actual archive bytes.

    Saved provenance/gate artifacts are identified but never promoted to current
    admission evidence. This development workflow always records citable=false.
    """
    manifest = Path(manifest).resolve()
    full = read_manifest(manifest)
    validate_manifest(full)
    frame = require_primary(select_tier(full, ARCHIVE_TIER), why="forensic pilot")
    frame = analysis_frame(frame, PRIMARY_ANALYSIS, why="forensic pilot")
    validate_pilot_frame(frame)
    frame = frame.sort_values(["utt_id", "condition"]).reset_index(drop=True).copy()
    frame["audio_sha256"] = [sha256_file(path) for path in frame.path]
    # Identical recordings in different splits are content leakage even if IDs differ.
    if (frame.groupby("audio_sha256").split.nunique() > 1).any():
        raise InvariantViolation("Identical audio bytes occur across frozen splits.")
    groups = frame[["utt_id", "speaker", "split"]].drop_duplicates().to_dict("records")
    audio = frame[["utt_id", "condition", "audio_sha256"]].to_dict("records")
    def canonical(value):
        return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    identity = {
        "manifest_sha256": sha256_file(manifest),
        "split_sha256": hashlib.sha256(canonical(groups)).hexdigest(),
        "archive_audio_sha256": hashlib.sha256(canonical(audio)).hexdigest(),
        "utterance_groups": groups,
        "split_counts": frame.groupby("split").utt_id.nunique().to_dict(),
        "row_counts": frame.groupby(["split", "condition"]).size().unstack().to_dict(),
        "n_speakers": int(frame.speaker.nunique()),
        "source_artifacts": {
            path.name: sha256_file(path) for path in sorted(manifest.parent.iterdir())
            if path.is_file() and path.name in {
                "manifest.provenance.json", "manifest.alignment.json", "gates.json", "sanity.csv",
            }
        },
        "provenance_status": "legacy_or_unverified_development_only",
        "historical_gate_bytes_proven": False,
        "signal_gates_rerun": False,
        "orientation_gate_activated": False,
        "citable": False,
    }
    return frame, identity


def extract_dataset_features(
    frame: pd.DataFrame, config: FeatureConfig, work_dir: str | Path,
) -> np.ndarray:
    """Read one utterance's four waveforms at a time; retain only small vectors."""
    features = np.empty((len(frame), len(config.feature_names)), dtype=np.float64)
    # Positions are local, independent of a sliced frame's original index labels.
    frame = frame.reset_index(drop=True)
    with TemporaryDirectory(prefix="pilot-analysis-", dir=work_dir) as tmp:
        for _, group in frame.groupby("utt_id", sort=True):
            wavs = {str(row.condition): read_archive_input(row.path, config)
                    for row in group.itertuples()}
            vectors = extract_comparison_features(wavs, tmp, config)
            for index, row in group.iterrows():
                features[index] = vectors[str(row.condition)]
    return features


def fit_head(x: np.ndarray, y: np.ndarray, c: float, seed: int, *, balanced: bool) -> Pipeline:
    """Only the caller's training rows reach scaler.fit or classifier.fit."""
    pipeline = Pipeline([
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(
            C=c, solver="lbfgs", class_weight="balanced" if balanced else None,
            max_iter=2000, tol=1e-6, random_state=seed,
        )),
    ])
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        pipeline.fit(x, y)
    return pipeline


def select_operating_threshold(real: np.ndarray, synthetic: np.ndarray) -> tuple[float, dict]:
    """Dev-only minimum balanced error at attainable, grouped > boundaries.

    Unlike the EER crossing boundary, this deterministic threshold has a measured
    operating point. Equal-cost ties prefer lower |FAR-FRR|, then lower boundary.
    """
    compute_eer(real, synthetic)  # shared validation and score direction
    values = np.unique(np.r_[real, synthetic])
    below = np.nextafter(values[0], -np.inf)
    if not np.isfinite(below):
        raise ValueError("Dev scores too extreme for a finite accept-all boundary.")
    candidates = np.r_[below, values]
    points = []
    for threshold in candidates:
        frr = float(np.mean(real <= threshold))
        far = float(np.mean(synthetic > threshold))
        points.append(((frr + far) / 2, abs(frr - far), float(threshold), frr, far))
    ber, _, threshold, frr, far = min(points)
    return threshold, {"balanced_error": ber, "frr": frr, "far": far,
                       "selection": "dev_min_balanced_error_then_gap_then_lower_boundary"}


def select_heads(
    train_x: np.ndarray, train_conditions, dev_x: np.ndarray, dev_conditions,
    *, seed: int = GLOBAL_SEED, c_grid: tuple[float, ...] = C_GRID,
) -> tuple[LinearHead, LinearHead, float, dict]:
    if not c_grid or any(not np.isfinite(c) or c <= 0 for c in c_grid):
        raise ValueError("C grid must contain positive finite values.")
    train_y, dev_y = binary_labels(train_conditions), binary_labels(dev_conditions)
    train_syn, dev_syn = train_y == 0, dev_y == 0
    train_attr = attribution_labels(np.asarray(train_conditions)[train_syn])
    dev_attr = attribution_labels(np.asarray(dev_conditions)[dev_syn])
    if (set(train_y) != {0, 1} or set(dev_y) != {0, 1}
            or set(train_attr) != {0, 1, 2} or set(dev_attr) != {0, 1, 2}):
        raise ValueError("Each frozen split must contain real and every known source.")
    candidates, binary_best, attribution_best = [], None, None
    for c in sorted(set(c_grid)):
        binary = fit_head(train_x, train_y, c, seed, balanced=True)
        scores = binary.decision_function(dev_x)
        eer = compute_eer(scores[dev_y == 1], scores[dev_y == 0])[0]
        auc = float(roc_auc_score(dev_y, scores))
        attr = fit_head(train_x[train_syn], train_attr, c, seed, balanced=False)
        macro_f1 = float(f1_score(dev_attr, attr.predict(dev_x[dev_syn]), average="macro"))
        candidates.append({"C": c, "binary_dev_eer": eer, "binary_dev_auroc": auc,
                           "attribution_dev_macro_f1": macro_f1})
        binary_key, attr_key = (eer, -auc, c), (-macro_f1, c)
        if binary_best is None or binary_key < binary_best[0]:
            binary_best = (binary_key, binary)
        if attribution_best is None or attr_key < attribution_best[0]:
            attribution_best = (attr_key, attr)
    binary, attr = binary_best[1], attribution_best[1]
    scores = binary.decision_function(dev_x)
    threshold, operating_point = select_operating_threshold(scores[dev_y == 1], scores[dev_y == 0])
    counts = np.bincount(train_y, minlength=2)
    selection = {
        "candidates": candidates,
        "binary_C": binary_best[0][-1], "attribution_C": attribution_best[0][-1],
        "binary_selection": "dev_eer_then_auroc_then_lower_C",
        "attribution_selection": "dev_macro_f1_then_lower_C",
        "operating_point": operating_point,
        "train_binary_counts": dict(zip(("synthetic", "real"), map(int, counts), strict=True)),
        "binary_class_weights": dict(zip(
            ("synthetic", "real"), map(float, len(train_y) / (2 * counts)), strict=True,
        )),
        "scaler_fit_split": "train_only",
        "refit_on_dev": False,
    }
    return LinearHead.from_pipeline(binary), LinearHead.from_pipeline(attr), threshold, selection


def prediction_frame(bundle: PilotBundle, frame: pd.DataFrame, x: np.ndarray) -> pd.DataFrame:
    bundle.validate()
    result = frame[["utt_id", "condition", "split", "speaker", "tier", "path",
                    "audio_sha256"]].copy().reset_index(drop=True)
    scores = bundle.binary.decision_function(x)[:, 0]
    logits = bundle.attribution.decision_function(x)
    result["binary_label"] = binary_labels(frame.condition)
    result["real_score"] = scores
    result["binary_prediction"] = np.where(scores > bundle.metadata["operating_threshold"],
                                          "real", "synthetic")
    result["known_source_prediction"] = np.asarray(KNOWN_SOURCES)[logits.argmax(axis=1)]
    result["routed_source"] = result.known_source_prediction.where(
        result.binary_prediction == "synthetic", "",
    )
    for i, source in enumerate(KNOWN_SOURCES):
        result[f"source_logit_{source}"] = logits[:, i]
    result["protocol"] = PILOT_PROTOCOL
    result["band_limit_hz"] = bundle.config.band_limit_hz
    result["operating_threshold"] = bundle.metadata["operating_threshold"]
    result["model_sha256"] = bundle.metadata["model_sha256"]
    result["feature_version"] = bundle.config.version
    result["citable"] = False
    return result


def evaluate_predictions(predictions: pd.DataFrame) -> dict:
    y = predictions.binary_label.to_numpy(dtype=int)
    scores = predictions.real_score.to_numpy(dtype=float)
    binary_pred = (predictions.binary_prediction == "real").to_numpy(dtype=int)
    real, synthetic = y == 1, y == 0
    eer, boundary = compute_eer(scores[real], scores[synthetic])
    per_vocoder = {}
    for source in KNOWN_SOURCES:
        selected = predictions.condition.eq(source).to_numpy()
        per_vocoder[source] = {
            "eer": compute_eer(scores[real], scores[selected])[0],
            "n_real": int(real.sum()), "n_synthetic": int(selected.sum()),
        }
    attr_y = attribution_labels(predictions.loc[synthetic, "condition"])
    attr_pred = attribution_labels(predictions.loc[synthetic, "known_source_prediction"])
    precision, recall, f1, support = precision_recall_fscore_support(
        attr_y, attr_pred, labels=range(len(KNOWN_SOURCES)), zero_division=0,
    )
    return {
        "binary": {
            "auroc": float(roc_auc_score(y, scores)), "eer": eer,
            "eer_crossing_boundary": boundary,
            "eer_boundary_is_operating_threshold": False,
            "confusion_matrix": confusion_matrix(y, binary_pred, labels=[0, 1]).tolist(),
            "confusion_labels": ["synthetic", "real"],
            "frr_at_dev_threshold": float(np.mean(binary_pred[real] == 0)),
            "far_at_dev_threshold": float(np.mean(binary_pred[synthetic] == 1)),
            "n_real": int(real.sum()), "n_synthetic": int(synthetic.sum()),
            "per_vocoder": per_vocoder,
        },
        "attribution": {
            "evaluation_population": "all_ground_truth_synthetic_independent_of_binary_routing",
            "macro_f1": float(f1_score(attr_y, attr_pred, average="macro")),
            "confusion_matrix": confusion_matrix(
                attr_y, attr_pred, labels=range(len(KNOWN_SOURCES)),
            ).tolist(),
            "confusion_labels": list(KNOWN_SOURCES),
            "per_class": {source: {
                "precision": float(precision[i]), "recall": float(recall[i]),
                "f1": float(f1[i]), "support": int(support[i]),
            } for i, source in enumerate(KNOWN_SOURCES)},
            "cascade_synthetic_routing_recall": float(np.mean(binary_pred[synthetic] == 0)),
            "cascade_correct_source_fraction": float(np.mean(
                (binary_pred[synthetic] == 0) & (attr_pred == attr_y),
            )),
            "scores_calibrated": False,
            "unknown_source_rejection": False,
        },
        "protocol": PILOT_PROTOCOL,
        "citable": False,
        "uncertainty": "not_estimated_paired_bootstrap_is_a_separate_task",
    }


def code_provenance() -> dict:
    root = Path(__file__).resolve().parents[2]
    files = [*sorted((root / "src").rglob("*.py")), root / "pyproject.toml", root / "CLAUDE.md"]
    digests = {path.relative_to(root).as_posix(): sha256_file(path) for path in files}
    def git_read(*args):
        result = subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else "unavailable"
    return {
        "head": git_read("rev-parse", "HEAD"),
        "working_tree_status": git_read("status", "--porcelain"),
        "source_sha256": digests,
        "source_tree_sha256": hashlib.sha256(
            json.dumps(digests, sort_keys=True).encode(),
        ).hexdigest(),
        "python": platform.python_version(), "platform": platform.platform(),
    }


def _write_outputs(out_dir: Path, predictions: pd.DataFrame, metrics: dict) -> None:
    predictions.to_csv(out_dir / "predictions.csv", index=False)
    write_json(out_dir / "metrics.json", metrics)
    write_json(out_dir / "artifacts.sha256.json", {
        path.relative_to(out_dir).as_posix(): sha256_file(path)
        for path in sorted(out_dir.rglob("*")) if path.is_file()
        and path.name != "artifacts.sha256.json"
    })


def train_pilot(
    manifest: str | Path, out_dir: str | Path, *, dataset_id: str = "btpvf-v0",
    seed: int = GLOBAL_SEED,
) -> dict:
    """Train on train, select on dev, freeze, then evaluate eval once."""
    out_dir = Path(out_dir).resolve()
    if out_dir.exists():
        raise FileExistsError("Pilot output must be a new directory; preserve previous artifacts.")
    frame, identity = load_pilot_dataset(manifest)
    out_dir.mkdir(parents=True)
    config = FeatureConfig()
    fit_frame = frame[frame.split.isin(["train", "dev"])].reset_index(drop=True)
    fit_x = extract_dataset_features(fit_frame, config, out_dir)
    train, dev = (fit_frame.split.eq(split).to_numpy() for split in ("train", "dev"))
    binary, attribution, threshold, selection = select_heads(
        fit_x[train], fit_frame.loc[train, "condition"].to_numpy(),
        fit_x[dev], fit_frame.loc[dev, "condition"].to_numpy(), seed=seed,
    )
    metadata = {
        "bundle_version": BUNDLE_VERSION, "protocol": PILOT_PROTOCOL, "citable": False,
        "dataset_id": dataset_id, "dataset_identity": identity, "seed": seed,
        "features": asdict(config), "feature_names": config.feature_names,
        "binary_classes": BINARY_CLASSES, "source_classes": SOURCE_CLASSES,
        "score_direction": "higher_is_real", "decision_rule": "real_if_score_gt_threshold",
        "operating_threshold": threshold, "selection": selection,
        "estimators": "StandardScaler + L2 LogisticRegression(lbfgs) for each head",
        "environment": runtime_versions(), "code": code_provenance(),
        "preprocessing_identity": preprocessing_identity(),
        "research_eligibility": "development_only_legacy_provenance_unreviewed_gate_orientation",
    }
    bundle = PilotBundle(binary, attribution, metadata, config)
    save_bundle(bundle, out_dir / "bundle")  # freeze before accessing eval features
    eval_frame = frame[frame.split.eq("eval")].reset_index(drop=True)
    eval_x = extract_dataset_features(eval_frame, config, out_dir)
    np.savez_compressed(out_dir / "features.npz", train_dev=fit_x, eval=eval_x)
    feature_rows = pd.concat([fit_frame, eval_frame], ignore_index=True)
    feature_rows[["utt_id", "condition", "split", "tier", "audio_sha256"]].to_csv(
        out_dir / "feature_rows.csv", index=False,
    )
    write_json(out_dir / "dataset_identity.json", identity)
    write_json(out_dir / "selection.json", selection)
    predictions = prediction_frame(bundle, eval_frame, eval_x)
    metrics = evaluate_predictions(predictions)
    metrics["dataset_id"] = dataset_id
    metrics["dataset_identity"] = {key: identity[key] for key in (
        "manifest_sha256", "split_sha256", "archive_audio_sha256",
    )}
    metrics["model_sha256"] = sha256_file(out_dir / "bundle" / "models.npz")
    metrics["evaluation_source_tree_sha256"] = metadata["code"]["source_tree_sha256"]
    _write_outputs(out_dir, predictions, metrics)
    return metrics


def evaluate_pilot(bundle_path: str | Path, manifest: str | Path, out_dir: str | Path) -> dict:
    """Reproduce eval without fitting anything; require the saved dataset identity."""
    bundle = load_bundle(bundle_path)
    frame, identity = load_pilot_dataset(manifest)
    if identity != bundle.metadata["dataset_identity"]:
        raise ValueError("Evaluation dataset/split/audio identity differs from training.")
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=False)
    eval_frame = frame[frame.split.eq("eval")].reset_index(drop=True)
    x = extract_dataset_features(eval_frame, bundle.config, out_dir)
    metrics = evaluate_predictions(prediction_frame(bundle, eval_frame, x))
    metrics["dataset_id"] = bundle.metadata["dataset_id"]
    metrics["dataset_identity"] = {key: identity[key] for key in (
        "manifest_sha256", "split_sha256", "archive_audio_sha256",
    )}
    metrics["model_sha256"] = sha256_file(Path(bundle_path) / "models.npz")
    metrics["evaluation_source_tree_sha256"] = code_provenance()["source_tree_sha256"]
    _write_outputs(out_dir, prediction_frame(bundle, eval_frame, x), metrics)
    return metrics
