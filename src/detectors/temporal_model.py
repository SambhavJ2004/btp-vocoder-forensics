"""Explicit numerical temporal logistic model; never deserializes pickled objects."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.special import expit

from detectors.forensic_pilot import LinearHead, runtime_versions, sha256_file, write_json
from detectors.temporal_features import (
    WindowConfig,
    analysis_comparison,
    extract_windows,
    predicted_intervals,
    smooth_probabilities,
)

PROTOCOL = "vocopatch_temporal_logistic_archive8k_v1"


def implementation_identity():
    import data.audio_io
    import data.invariants
    import data.preprocess
    import detectors.base
    import detectors.temporal_features
    import detectors.temporal_model

    return {
        m.__name__: sha256_file(m.__file__)
        for m in (
            data.audio_io,
            data.invariants,
            data.preprocess,
            detectors.base,
            detectors.temporal_features,
            detectors.temporal_model,
        )
    }


@dataclass
class TemporalBundle:
    head: LinearHead
    config: WindowConfig
    threshold: float
    smoothing: int
    merge_samples: int
    metadata: dict

    def validate(self):
        self.config.validate()
        self.head.validate(len(self.config.feature_names), 1)
        if (
            not np.isfinite(self.threshold)
            or not 0 <= self.threshold <= 1
            or self.smoothing not in (1, 3)
            or not isinstance(self.merge_samples, int)
            or isinstance(self.merge_samples, bool)
            or self.merge_samples < 0
        ):
            raise ValueError("Invalid temporal operating parameters.")
        if (
            self.metadata.get("protocol") != PROTOCOL
            or self.metadata.get("development_only") is not True
            or self.metadata.get("citable") is not False
            or self.metadata.get("score_direction") != "higher_is_synthetic"
            or self.metadata.get("feature_names") != self.config.feature_names
        ):
            raise ValueError("Unsupported temporal bundle schema or disclosures.")

    def margins(self, x):
        self.validate()
        return self.head.decision_function(x)[:, 0]


def save_temporal_bundle(bundle: TemporalBundle, root: str | Path):
    bundle.validate()
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    fields = ("mean", "scale", "coefficients", "intercept")
    np.savez_compressed(root / "model.npz", **{k: getattr(bundle.head, k) for k in fields})
    write_json(
        root / "bundle.json",
        {
            **bundle.metadata,
            "config": asdict(bundle.config),
            "threshold": bundle.threshold,
            "smoothing": bundle.smoothing,
            "merge_samples": bundle.merge_samples,
            "environment": runtime_versions(),
            "implementation_identity": implementation_identity(),
            "format_version": 1,
        },
    )
    write_json(
        root / "checksums.json",
        {p.name: sha256_file(p) for p in (root / "model.npz", root / "bundle.json")},
    )


def load_temporal_bundle(root: str | Path) -> TemporalBundle:
    root = Path(root).resolve()
    if {p.name for p in root.iterdir()} != {"model.npz", "bundle.json", "checksums.json"}:
        raise ValueError("Unexpected temporal model bundle contents.")
    checksums = json.loads((root / "checksums.json").read_text())
    for name in ("model.npz", "bundle.json"):
        if (root / name).is_symlink() or checksums.get(name) != sha256_file(root / name):
            raise ValueError("Temporal bundle integrity failure.")
    metadata = json.loads((root / "bundle.json").read_text())
    if (
        metadata.get("format_version") != 1
        or metadata.get("environment") != runtime_versions()
        or metadata.get("implementation_identity") != implementation_identity()
    ):
        raise ValueError("Temporal bundle environment or implementation differs.")
    fields = ("mean", "scale", "coefficients", "intercept")
    with np.load(root / "model.npz", allow_pickle=False) as state:
        if set(state.files) != set(fields):
            raise ValueError("Unexpected numerical model fields.")
        head = LinearHead(*(state[k].copy() for k in fields))
    bundle = TemporalBundle(
        head,
        WindowConfig(**metadata["config"]),
        metadata["threshold"],
        metadata["smoothing"],
        metadata["merge_samples"],
        metadata,
    )
    bundle.validate()
    return bundle


def predict_temporal_file(
    bundle: TemporalBundle | str | Path, wav_path: str | Path, work_dir: str | Path
) -> dict:
    from experiments.partial_audio import read_input

    if not isinstance(bundle, TemporalBundle):
        bundle = load_temporal_bundle(bundle)
    wav = read_input(Path(wav_path))
    if len(wav) > 60 * bundle.config.sample_rate:
        raise ValueError("Temporal development inference is bounded to 60 seconds.")
    analysis = analysis_comparison({"real": wav}, work_dir)["real"]
    starts, x = extract_windows(analysis, bundle.config)
    margin = bundle.margins(x)
    probability = expit(margin)
    smoothed = smooth_probabilities(probability, bundle.smoothing)
    intervals = predicted_intervals(
        starts, smoothed, len(wav), bundle.config, bundle.threshold, bundle.merge_samples
    )
    return {
        "window_start_samples": starts.tolist(),
        "synthetic_margins": margin.tolist(),
        "uncalibrated_synthetic_probabilities": probability.tolist(),
        "smoothed_probabilities": smoothed.tolist(),
        "model_predicted_intervals": [
            {
                "start_sample": s,
                "end_sample": e,
                "start_seconds": s / bundle.config.sample_rate,
                "end_seconds": e / bundle.config.sample_rate,
            }
            for s, e in intervals
        ],
        "sample_rate": bundle.config.sample_rate,
        "num_samples": len(wav),
        "ground_truth_used": False,
        "development_only": True,
        "citable": False,
    }
