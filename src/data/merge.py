"""Merging per-condition Phase A runs into one dataset. INV-13's other half.

INV-13 says "one isolated Kaggle session per condition, output persisted to a
Kaggle Dataset, **manifests merged afterwards**". The merge was the half that did
not exist, so every multi-condition run so far had to fit in one environment --
which is why only the three reported-to-coexist conditions could ever run
together.

**The merge is also a measurement.** Each run regenerates the `real` condition
from the same corpus with the same seed, so if the environments did not disturb
the shared pipeline, every run's `real` audio is byte-identical. That is exactly
what INV-13 is worried about -- pip resolving a different librosa for one
condition and changing STFT edge behaviour -- and it is checkable rather than
assumed. :func:`merge_runs` checks it on every file and refuses a merge where it
does not hold, naming the utterances that differ.

So a successful merge carries evidence that the separate environments produced
the same pipeline, which one big shared environment could never have provided.

What it does NOT do is relax anything. The merged table goes through
``validate_manifest`` unchanged: pairing (INV-06), one split per utterance
(INV-09), reference-derived trim spans (INV-03), the full-band archive (INV-17),
the loudness record (INV-04) and the alignment records (INV-16) all still have to
hold across the union.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from .invariants import REAL_CONDITION, InvariantViolation
from .manifest import MANIFEST_COLUMNS, manifest_root, read_manifest, validate_manifest


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _real_index(df: pd.DataFrame) -> dict[tuple[str, str], Path]:
    real = df[df["condition"] == REAL_CONDITION]
    return {(str(r.tier), str(r.utt_id)): Path(str(r.path)) for r in real.itertuples()}


def compare_real(runs: dict[str, pd.DataFrame]) -> dict:
    """Is every run's `real` condition byte-identical? Returns the evidence.

    The reference is the first run in ``runs``. A difference here is the failure
    INV-13 exists to catch, so it is reported per utterance rather than as a
    count.
    """
    names = list(runs)
    if len(names) < 2:
        return {"runs": names, "compared": 0, "identical": True, "differing": []}
    base_name, *others = names
    base = _real_index(runs[base_name])
    differing: list[dict] = []
    compared = 0
    for name in others:
        other = _real_index(runs[name])
        shared = sorted(set(base) & set(other))
        missing = sorted(set(base) ^ set(other))
        if missing:
            raise InvariantViolation(
                f"INV-06: run {name!r} and {base_name!r} do not hold the same real "
                f"utterances; {len(missing)} differ (e.g. {missing[:3]}). A merge over "
                "different content is not a merge."
            )
        for key in shared:
            compared += 1
            if _digest(base[key]) != _digest(other[key]):
                differing.append({"run": name, "tier": key[0], "utt_id": key[1]})
    return {
        "runs": names,
        "reference": base_name,
        "compared": compared,
        "identical": not differing,
        "differing": differing[:20],
        "n_differing": len(differing),
    }


def merge_runs(
    manifests: Iterable[str | Path],
    out_dir: str | Path,
    *,
    require_identical_real: bool = True,
) -> dict:
    """Combine per-condition runs into one dataset directory with one manifest.

    Copies each run's audio into ``out_dir`` under the layout Phase A writes,
    takes `real` from the first run only, writes a merged manifest with paths
    relative to it, and validates the result. Returns a record for the sidecar.

    ``require_identical_real=False`` is for inspecting a mismatch, never for
    shipping one: it still reports the differences, it just does not raise.
    """
    paths = [Path(str(m)) for m in manifests]
    if len(paths) < 1:
        raise InvariantViolation("nothing to merge.")
    out_dir = Path(str(out_dir))
    if out_dir.exists():
        raise InvariantViolation(
            f"{out_dir} already exists. A dataset with two merges in it is not a "
            "dataset; choose a new directory."
        )

    runs = {str(p.parent.name): read_manifest(p) for p in paths}
    if len(runs) != len(paths):
        raise InvariantViolation(
            "two manifests share a parent directory name; the merge record would not "
            "say which run a row came from."
        )
    real_check = compare_real(runs)
    if require_identical_real and not real_check["identical"]:
        raise InvariantViolation(
            f"INV-13: `real` is not byte-identical across runs -- "
            f"{real_check['n_differing']} of {real_check['compared']} files differ "
            f"(e.g. {real_check['differing'][:3]}). The separate environments did not "
            "produce the same pipeline, so the conditions are not comparable. Do not "
            "merge: find out which run is the odd one out."
        )

    conditions: dict[str, str] = {}
    rows: list[pd.DataFrame] = []
    first = next(iter(runs))
    for name, df in runs.items():
        for cond in sorted(set(df["condition"])):
            if cond == REAL_CONDITION and name != first:
                continue  # one real, from the reference run
            if cond in conditions:
                raise InvariantViolation(
                    f"INV-06: condition {cond!r} appears in both {conditions[cond]!r} "
                    f"and {name!r}. Merging two versions of one condition would put "
                    "two different vocoder runs under one label."
                )
            conditions[cond] = name
            rows.append(df[df["condition"] == cond])

    merged = pd.concat(rows, ignore_index=True)
    out_dir.mkdir(parents=True)
    relocated = []
    for row in merged.itertuples():
        source = Path(str(row.path))
        target = out_dir / str(row.tier) / str(row.condition) / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        relocated.append(str(target.relative_to(out_dir).as_posix()))
    merged["path"] = relocated

    manifest_path = out_dir / "manifest.csv"
    merged = merged.reindex(columns=list(MANIFEST_COLUMNS))
    merged.sort_values(["tier", "condition", "utt_id"], inplace=True)
    merged.to_csv(manifest_path, index=False)

    # The same gate a single run passes. Read back from disk so the paths are
    # resolved the way every consumer will resolve them.
    validate_manifest(read_manifest(manifest_path), require_zeroshot=True)

    record = {
        "out_dir": str(out_dir),
        "manifest": str(manifest_path),
        "sources": {name: str(p) for name, p in zip(runs, paths, strict=True)},
        "conditions": conditions,
        "real_from": first,
        "real_identical_across_runs": real_check,
        "n_rows": int(len(merged)),
        "n_files": len(relocated),
        "code_snapshots": _collect_snapshots(paths),
    }
    (out_dir / "manifest.merge.json").write_text(
        json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
    )
    return record


def _collect_snapshots(paths: list[Path]) -> dict:
    """Each run's code provenance, kept per run rather than flattened.

    A merged dataset can be built from runs of different code, and pretending
    otherwise would lose the one thing that says whether the whole table is
    citable: it is only as citable as its least citable run.
    """
    out: dict[str, dict] = {}
    for path in paths:
        sidecar = manifest_root(path) / "manifest.provenance.json"
        if not sidecar.is_file():
            out[path.parent.name] = {"provenance": "missing"}
            continue
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        snapshot = payload.get("code_snapshot", {})
        out[path.parent.name] = {
            "mode": snapshot.get("mode"),
            "sha": snapshot.get("sha"),
            "tree_sha256": snapshot.get("tree_sha256"),
            "citable": snapshot.get("citable"),
        }
    return out


def merge_is_citable(record: dict) -> bool:
    """A merged dataset is citable only if every run in it is."""
    snapshots = record.get("code_snapshots", {})
    return bool(snapshots) and all(s.get("citable") for s in snapshots.values())


def summarise(record: dict) -> str:
    """A short human summary, for the CLI."""
    real = record["real_identical_across_runs"]
    lines = [
        f"merged {len(record['sources'])} run(s) -> {record['out_dir']}",
        f"  conditions: {', '.join(f'{c} ({r})' for c, r in sorted(record['conditions'].items()))}",
        f"  real taken from {record['real_from']!r}; byte-identical across runs: "
        f"{real['identical']} ({real['compared']} files compared)",
        f"  rows {record['n_rows']}, files {record['n_files']}",
        f"  citable: {merge_is_citable(record)}",
    ]
    for name, snap in sorted(record["code_snapshots"].items()):
        lines.append(f"    {name}: {snap.get('mode')} {str(snap.get('sha'))[:12]} "
                     f"tree {str(snap.get('tree_sha256'))[:12]}")
    return "\n".join(lines)
