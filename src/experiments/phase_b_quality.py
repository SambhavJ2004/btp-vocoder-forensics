"""Phase B, X-axis driver — quality metrics over a manifest. Person B."""

from __future__ import annotations

from pathlib import Path

from data.manifest import read_manifest, validate_manifest
from metrics.runner import evaluate_manifest, save


def run(manifest_path: str | Path, out_dir: str | Path, *, device: str = "cpu", **kwargs):
    """Validate the manifest first, then score. Refusing to measure an invalid
    dataset is cheaper than discovering the invalidity after the plot exists.
    """
    df = read_manifest(manifest_path)
    validate_manifest(df)
    per_utt = evaluate_manifest(df, device=device, **kwargs)
    return save(per_utt, out_dir)
