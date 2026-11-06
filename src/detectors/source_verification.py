"""Versioned numeric source verifier. Scores are evidence, not authenticity probabilities."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.special import softmax

from detectors.attribution import energy_score
from detectors.forensic_features import FeatureConfig, extract_file_features
from detectors.forensic_pilot import (
    KNOWN_SOURCES,
    LinearHead,
    preprocessing_identity,
    runtime_versions,
    sha256_file,
    write_json,
)

SCORES = ("distance", "msp", "negative_energy")
LEVELS = {"retain95": 0.05, "retain80": 0.20}
FLAGS = {"development_only": True, "citable": False}
PROTOCOL = "vocotrace_whole_archive8k_v1"


def implementation_identity():
    import detectors.attribution

    return {
        **preprocessing_identity(),
        "detectors.source_verification": sha256_file(__file__),
        "detectors.attribution": sha256_file(detectors.attribution.__file__),
    }


def confidence_cutoff(values, rejection_budget):
    """Largest >= cutoff rejecting at most budget of known calibration rows; ties whole."""
    values = np.asarray(values, dtype=float)
    if (
        values.ndim != 1
        or not len(values)
        or not np.isfinite(values).all()
        or not 0 <= rejection_budget < 1
    ):
        raise ValueError("Invalid known calibration scores or rejection budget.")
    return float(np.sort(values)[int(np.floor(rejection_budget * len(values)))])


def genuine_gate(values, false_attribution_budget=0.05):
    """Largest <= real-margin boundary accepting at most budget genuine calibration rows."""
    values = np.asarray(values, dtype=float)
    if (
        values.ndim != 1
        or not len(values)
        or not np.isfinite(values).all()
        or not 0 <= false_attribution_budget < 1
    ):
        raise ValueError("Invalid genuine calibration scores or budget.")
    first_rejected = np.sort(values)[int(np.floor(false_attribution_budget * len(values)))]
    below = float(np.nextafter(first_rejected, -np.inf))
    if not np.isfinite(below):
        raise ValueError("Cannot represent a reject-all genuine boundary.")
    return below


@dataclass
class SourceVerifier:
    binary: LinearHead
    attribution: LinearHead
    centroids: np.ndarray
    precision: np.ndarray
    metadata: dict

    @property
    def known(self):
        return tuple(self.metadata["known_classes"])

    def validate(self):
        m = self.metadata
        known = tuple(m.get("known_classes", ()))
        if (
            m.get("format_version") != 1
            or m.get("protocol") != PROTOCOL
            or m.get("citable") is not False
            or m.get("development_only") is not True
            or len(known) not in (2, 3)
            or len(set(known)) != len(known)
            or known != tuple(k for k in KNOWN_SOURCES if k in known)
            or m.get("selected_score") not in (*SCORES, "reject_all", None)
            or m.get("features") != asdict(FeatureConfig())
        ):
            raise ValueError("Invalid verifier protocol, classes or selection.")
        self.binary.validate(64, 1)
        self.attribution.validate(64, 1 if len(known) == 2 else 3)
        if (
            self.centroids.shape != (len(known), 64)
            or self.precision.shape != (64, 64)
            or not np.isfinite(self.centroids).all()
            or not np.isfinite(self.precision).all()
            or not np.allclose(self.precision, self.precision.T, atol=1e-12, rtol=0)
            or np.linalg.eigvalsh(self.precision).min() <= 0
        ):
            raise ValueError("Invalid class-support state.")
        if not np.isfinite(m.get("binary_threshold", np.nan)):
            raise ValueError("Invalid genuine gate.")
        thresholds = m.get("confidence_thresholds", {})
        if set(thresholds) != set(SCORES) or any(
            set(v) != set(LEVELS) or any(not np.isfinite(t) for t in v.values())
            for v in thresholds.values()
        ):
            raise ValueError("Invalid confidence cutoffs.")

    def raw_scores(self, x):
        x = np.asarray(x, dtype=float)
        real = self.binary.decision_function(x)[:, 0]
        logits = self.attribution.decision_function(x)
        # sklearn binary LR has one log-odds margin. These two logits reproduce sigmoid(m).
        if len(self.known) == 2:
            logits = np.column_stack((-logits[:, 0] / 2, logits[:, 0] / 2))
        if not np.isfinite(real).all() or not np.isfinite(logits).all():
            raise ValueError("Non-finite verifier margins.")
        prediction = logits.argmax(axis=1)
        standardized = (x - self.attribution.mean) / self.attribution.scale
        residual = standardized - self.centroids[prediction]
        distances = np.einsum("ni,ij,nj->n", residual, self.precision, residual)
        with np.errstate(over="ignore"):
            msp = softmax(logits, axis=1).max(axis=1)
        result = {
            "real_margin": real,
            "source_logits": logits,
            "source_index": prediction,
            "distance": -distances,
            "msp": msp,
            "negative_energy": -energy_score(logits, 1.0),
        }
        if any(not np.isfinite(a).all() for a in result.values()):
            raise ValueError("Non-finite confidence evidence.")
        return result

    def decisions(self, scores, score_name=None, level="retain95"):
        self.validate()
        method = self.metadata["selected_score"] if score_name is None else score_name
        if method not in (*SCORES, "reject_all") or level not in LEVELS:
            raise ValueError("Select a locked score and declared operating point.")
        syn_gate = scores["real_margin"] <= self.metadata["binary_threshold"]
        support = (
            scores[method] >= self.metadata["confidence_thresholds"][method][level]
            if method != "reject_all"
            else np.zeros(len(syn_gate), dtype=bool)
        )
        accepted = syn_gate & support
        return np.where(accepted, scores["source_index"], -1), np.where(
            accepted,
            "known_source",
            np.where(syn_gate, "unknown_or_inconclusive", "no_supported_synthetic_source"),
        )

    def predict_features(self, x, *, score_name=None, level="retain95"):
        x = np.asarray(x, dtype=float).reshape(1, -1)
        raw = self.raw_scores(x)
        label, state = self.decisions(raw, score_name, level)
        return {
            "decision": str(state[0]),
            "known_source": self.known[label[0]] if label[0] >= 0 else None,
            "genuine_claim_supported": False,
            "unknown_generator_identity_supported": False,
            "score_name": score_name or self.metadata["selected_score"],
            "operating_point": level,
            "real_margin_higher_is_real": float(raw["real_margin"][0]),
            "confidence_scores_higher_is_known_like": {k: float(raw[k][0]) for k in SCORES},
            "source_logits_uncalibrated": dict(
                zip(self.known, raw["source_logits"][0].tolist(), strict=True)
            ),
            "probabilities_calibrated": False,
            "ground_truth_used": False,
            **FLAGS,
        }


def save_verifier(bundle, directory):
    bundle.validate()
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    state = {
        f"{name}_{field}": getattr(head, field)
        for name, head in (("binary", bundle.binary), ("attribution", bundle.attribution))
        for field in ("mean", "scale", "coefficients", "intercept")
    }
    np.savez_compressed(
        root / "model.npz", **state, centroids=bundle.centroids, precision=bundle.precision
    )
    write_json(
        root / "bundle.json",
        {
            **bundle.metadata,
            "environment": runtime_versions(),
            "implementation_identity": implementation_identity(),
        },
    )
    write_json(root / "checksums.json", {p.name: sha256_file(p) for p in root.iterdir()})


def load_verifier(directory):
    root = Path(directory)
    if root.is_symlink() or getattr(root, "is_junction", lambda: False)():
        raise ValueError("Unsafe verifier directory.")
    if {p.name for p in root.iterdir()} != {"bundle.json", "model.npz", "checksums.json"}:
        raise ValueError("Unexpected verifier contents.")
    for p in root.iterdir():
        if (
            not p.is_file()
            or p.is_symlink()
            or p.stat().st_nlink != 1
            or getattr(p, "is_junction", lambda: False)()
        ):
            raise ValueError("Unsafe verifier file.")
    checks = json.loads((root / "checksums.json").read_text())
    if set(checks) != {"bundle.json", "model.npz"} or any(
        sha256_file(root / k) != v for k, v in checks.items()
    ):
        raise ValueError("Verifier integrity mismatch.")
    meta = json.loads((root / "bundle.json").read_text())
    if (
        meta.get("environment") != runtime_versions()
        or meta.get("implementation_identity") != implementation_identity()
    ):
        raise ValueError("Verifier implementation/environment mismatch.")
    fields = ("mean", "scale", "coefficients", "intercept")
    with np.load(root / "model.npz", allow_pickle=False) as a:
        expected = {f"{h}_{f}" for h in ("binary", "attribution") for f in fields}
        if set(a.files) != expected | {"centroids", "precision"}:
            raise ValueError("Invalid numeric verifier schema.")
        heads = [
            LinearHead(*(a[f"{h}_{f}"].copy() for f in fields)) for h in ("binary", "attribution")
        ]
        bundle = SourceVerifier(*heads, a["centroids"].copy(), a["precision"].copy(), meta)
    bundle.validate()
    return bundle


def predict_file(bundle, wav, *, work_dir=None):
    if not isinstance(bundle, SourceVerifier):
        bundle = load_verifier(bundle)
    return bundle.predict_features(extract_file_features(wav, work_dir=work_dir))
