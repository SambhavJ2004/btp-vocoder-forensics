"""Numeric model bundles and reusable single-file inference for the CPU pilot.

Only linear coefficients/scaler arrays are persisted (NPZ without pickle).
Bundle hashes establish integrity, not authenticity; use trusted local bundles.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np

from detectors.forensic_features import FeatureConfig, extract_file_features

KNOWN_SOURCES = ("bigvgan_112m", "griffin_lim", "vocos")
BINARY_CLASSES = {"0": "synthetic", "1": "real"}
SOURCE_CLASSES = {str(i): name for i, name in enumerate(KNOWN_SOURCES)}
PILOT_PROTOCOL = "multi_vocoder_pilot_archive_8khz"
BUNDLE_VERSION = 1


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_versions() -> dict[str, str]:
    return {name: version(name) for name in (
        "numpy", "scipy", "soundfile", "librosa", "pyloudnorm", "scikit-learn",
    )}


def preprocessing_identity() -> dict[str, str]:
    import data.audio_io
    import data.invariants
    import data.preprocess
    import detectors.base
    import detectors.forensic_features

    return {module.__name__: sha256_file(module.__file__) for module in (
        data.audio_io, data.invariants, data.preprocess,
        detectors.base, detectors.forensic_features,
    )}


def write_json(path: str | Path, payload: dict) -> None:
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8",
    )


@dataclass
class LinearHead:
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    intercept: np.ndarray

    @classmethod
    def from_pipeline(cls, pipeline) -> LinearHead:
        scaler, classifier = pipeline.named_steps["scale"], pipeline.named_steps["classifier"]
        return cls(scaler.mean_.copy(), scaler.scale_.copy(),
                   classifier.coef_.copy(), classifier.intercept_.copy())

    def validate(self, n_features: int, n_outputs: int) -> None:
        expected = [(n_features,), (n_features,), (n_outputs, n_features), (n_outputs,)]
        for array, shape in zip(
            (self.mean, self.scale, self.coefficients, self.intercept), expected, strict=True,
        ):
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError("Incompatible or non-finite linear model state.")
        if (self.scale <= 0).any():
            raise ValueError("Model scaler must have positive scales.")

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float64)
        if (features.ndim != 2 or features.shape[1] != self.mean.size
                or not np.isfinite(features).all()):
            raise ValueError("Invalid pilot feature matrix.")
        return ((features - self.mean) / self.scale) @ self.coefficients.T + self.intercept


@dataclass
class PilotBundle:
    binary: LinearHead
    attribution: LinearHead
    metadata: dict
    config: FeatureConfig = FeatureConfig()

    def validate(self) -> None:
        self.config.validate()
        if (self.metadata.get("bundle_version") != BUNDLE_VERSION
                or self.metadata.get("protocol") != PILOT_PROTOCOL
                or self.metadata.get("citable") is not False
                or self.metadata.get("binary_classes") != BINARY_CLASSES
                or self.metadata.get("source_classes") != SOURCE_CLASSES
                or self.metadata.get("features") != asdict(self.config)
                or self.metadata.get("feature_names") != self.config.feature_names
                or self.metadata.get("score_direction") != "higher_is_real"
                or self.metadata.get("decision_rule") != "real_if_score_gt_threshold"):
            raise ValueError("Incompatible pilot bundle schema, protocol or class mapping.")
        threshold = self.metadata.get("operating_threshold")
        if not isinstance(threshold, (int, float)) or not np.isfinite(threshold):
            raise ValueError("Bundle operating threshold must be finite.")
        self.binary.validate(len(self.config.feature_names), 1)
        self.attribution.validate(len(self.config.feature_names), len(KNOWN_SOURCES))

    def predict_features(self, features: np.ndarray) -> dict:
        self.validate()
        x = np.asarray(features, dtype=np.float64).reshape(1, -1)
        score = float(self.binary.decision_function(x)[0, 0])
        is_real = score > self.metadata["operating_threshold"]
        result = {
            "prediction": "real" if is_real else "synthetic",
            "real_score": score,
            "score_kind": "uncalibrated_logistic_decision_margin",
            "operating_threshold": self.metadata["operating_threshold"],
            "known_source": None,
            "known_source_scores": None,
            "source_score_kind": "uncalibrated_multiclass_decision_logits",
            "unknown_source_rejection": False,
            "protocol": PILOT_PROTOCOL,
            "citable": False,
        }
        if not is_real:
            scores = self.attribution.decision_function(x)[0]
            result["known_source"] = KNOWN_SOURCES[int(np.argmax(scores))]
            result["known_source_scores"] = dict(
                zip(KNOWN_SOURCES, map(float, scores), strict=True),
            )
        return result


def save_bundle(bundle: PilotBundle, out_dir: str | Path) -> Path:
    bundle.validate()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    arrays = {}
    for name, head in (("binary", bundle.binary), ("attribution", bundle.attribution)):
        for field in ("mean", "scale", "coefficients", "intercept"):
            arrays[f"{name}_{field}"] = getattr(head, field)
    np.savez_compressed(out_dir / "models.npz", **arrays)
    metadata = dict(bundle.metadata, model_sha256=sha256_file(out_dir / "models.npz"))
    write_json(out_dir / "bundle.json", metadata)
    write_json(out_dir / "bundle.sha256.json", {
        name: sha256_file(out_dir / name) for name in ("models.npz", "bundle.json")
    })
    bundle.metadata.update(metadata)
    return out_dir


def load_bundle(path: str | Path) -> PilotBundle:
    """Load a trusted local numerical bundle; validate before any scoring."""
    root = Path(path).expanduser().resolve()
    receipt = json.loads((root / "bundle.sha256.json").read_text(encoding="utf-8"))
    for name in ("models.npz", "bundle.json"):
        if receipt.get(name) != sha256_file(root / name):
            raise ValueError(f"Pilot bundle integrity mismatch: {name}")
    metadata = json.loads((root / "bundle.json").read_text(encoding="utf-8"))
    if metadata.get("environment") != runtime_versions():
        raise ValueError("Pilot dependency versions differ; reproduce the saved environment.")
    if metadata.get("preprocessing_identity") != preprocessing_identity():
        raise ValueError("Pilot preprocessing code differs; reproduce the saved implementation.")
    if metadata.get("model_sha256") != receipt["models.npz"]:
        raise ValueError("Model digest disagrees with bundle metadata.")
    config = FeatureConfig(**metadata["features"])
    fields = ("mean", "scale", "coefficients", "intercept")
    with np.load(root / "models.npz", allow_pickle=False) as state:
        expected = {f"{name}_{field}" for name in ("binary", "attribution") for field in fields}
        if set(state.files) != expected:
            raise ValueError("Unexpected model arrays in pilot bundle.")
        heads = [LinearHead(*(state[f"{name}_{field}"].copy() for field in fields))
                 for name in ("binary", "attribution")]
    bundle = PilotBundle(*heads, metadata, config)
    bundle.validate()
    return bundle


def predict_file(
    bundle: PilotBundle | str | Path, wav_path: str | Path,
    *, work_dir: str | Path | None = None,
) -> dict:
    """Reusable inference API. Paths must point to a trusted local bundle/WAV."""
    if not isinstance(bundle, PilotBundle):
        bundle = load_bundle(bundle)
    bundle.validate()
    features = extract_file_features(wav_path, bundle.config, work_dir=work_dir)
    return bundle.predict_features(features)
