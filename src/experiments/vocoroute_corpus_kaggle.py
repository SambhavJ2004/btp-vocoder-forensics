"""Stage an isolated private Kaggle CPU job for the VocoRoute-P corpus."""

import base64
import io
import json
import re
import shutil
import tempfile
import time
import traceback
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from vocoroute.vctk_inventory import ACQUISITION_REF, ACQUISITION_VERSION, sha256

from .kaggle_harness import (
    ITERATION,
    KAGGLE_INPUT,
    KAGGLE_WORKING,
    RECORD,
    KernelSpec,
    _archive_bytes,
    _git,
    _tree_sha256,
    check_slug,
    kernel_metadata,
)
from .vocoroute_kaggle import DATASET_SCHEMA

PLAN_SCHEMA = "vocoroute-vctk092-preparation-kaggle-v1"
MAX_AUDIO_BYTES = {"train-dev": 12_000_000_000, "held-out": 3_000_000_000}
STORAGE_MARGIN_BYTES = 1_000_000_000
SOURCES = [
    "src/data/__init__.py",
    "src/data/alignment.py",
    "src/data/audio_io.py",
    "src/data/invariants.py",
    "src/experiments/__init__.py",
    "src/experiments/kaggle_harness.py",
    "src/experiments/vocoroute_corpus_kaggle.py",
    "src/experiments/vocoroute_kaggle.py",
    "src/vocoroute/__init__.py",
    "src/vocoroute/spectral.py",
    "src/vocoroute/training_config.py",
    "src/vocoroute/training_data.py",
    "src/vocoroute/vctk_inventory.py",
    "src/vocoroute/vctk_prepare.py",
]


def stage_corpus_kernel(
    output: Path,
    *,
    sha: str | None = None,
    freeze_source: str | None = None,
    freeze_sha256: str | None = None,
) -> dict:
    """Render a standalone script with only the preparation source allowlist."""
    if output.exists():
        raise ValueError("use a fresh local staging directory")
    if (freeze_source is None) != (freeze_sha256 is None):
        raise ValueError("held-out release needs both a versioned source and freeze SHA-256")
    owner = ACQUISITION_REF.split("/")[0]
    training_ref = f"{owner}/vocoroute-vctk092-n2b0-train-dev"
    if freeze_source is not None and (
        not re.fullmatch(rf"{re.escape(training_ref)}/[1-9][0-9]*", freeze_source)
        or not re.fullmatch(r"[0-9a-f]{64}", freeze_sha256 or "")
    ):
        raise ValueError("held-out release requires an exact train/dev notebook version and digest")
    if sha is not None and not re.fullmatch(r"[0-9a-f]{7,40}", sha):
        raise ValueError("record staging requires an existing user-created commit SHA")
    commit = _git("rev-parse", "--verify", f"{sha}^{{commit}}").stdout.strip() if sha else None
    mode = RECORD if commit else ITERATION
    contents = _archive_bytes(mode, commit, SOURCES)
    payload = _fixed_zip(contents)
    kind = "held-out" if freeze_source else "train-dev"
    plan = {
        "schema": PLAN_SCHEMA,
        "kind": kind,
        "mode": mode,
        "commit": commit,
        "source": {"ref": ACQUISITION_REF, "version": ACQUISITION_VERSION},
        "freeze": {"source": freeze_source, "sha256": freeze_sha256} if freeze_source else None,
        "source_tree_sha256": _tree_sha256(contents),
        "payload_sha256": sha256(payload),
        "source_files": sorted(contents),
        "max_audio_bytes": MAX_AUDIO_BYTES[kind],
    }
    spec = KernelSpec(
        f"vocoroute-vctk092-n2b0-{kind}", f"VocoRoute VCTK092 N2B0 {kind}", (), 0, gpu=False
    )
    check_slug(spec)
    metadata = kernel_metadata(spec, username=owner, code_file="prepare.py")
    sources = [ACQUISITION_REF]
    if freeze_source:
        sources.append(freeze_source.rsplit("/", 1)[0])
    metadata.update(
        kernel_type="script", enable_internet="false", dataset_sources=[], kernel_sources=sources
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / "kernel-metadata.json").write_bytes(_json_bytes(metadata))
    script = f"PLAN = {plan!r}\nPAYLOAD_B64 = {base64.b64encode(payload).decode()!r}\n" + _BOOTSTRAP
    (output / "prepare.py").write_text(script, encoding="utf-8", newline="\n")
    return {
        "kernel": metadata["id"],
        "plan": plan,
        "staged_script_sha256": sha256(script.encode()),
        "metadata_sha256": sha256(_json_bytes(metadata)),
    }


def _fixed_zip(contents: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with ZipFile(buffer, "w") as zipped:
        for name, data in sorted(contents.items()):
            info = ZipInfo(name)
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            zipped.writestr(info, data, compress_type=ZIP_DEFLATED, compresslevel=9)
    return buffer.getvalue()


def _json_bytes(doc: dict) -> bytes:
    return (json.dumps(doc, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _single_file(inputs: Path, name: str, *, parent_name: str) -> Path:
    candidates = [
        path for path in inputs.rglob(name) if path.parent.name == parent_name and path.is_file()
    ]
    if len(candidates) != 1:
        raise ValueError(f"expected one mounted {parent_name}/{name}, got {len(candidates)}")
    return candidates[0]


def run_corpus_kernel(plan: dict, inputs: Path, working: Path) -> dict:
    """Verify input identities and publish only a complete, checked release."""
    from importlib.metadata import version

    import torch

    from vocoroute.vctk_inventory import SplitProtocol, inspect_archive
    from vocoroute.vctk_prepare import (
        ARCHIVE_NAME,
        prepare_held_out,
        prepare_train_dev,
        verify_acquisition,
        write_json,
    )

    kind = plan["kind"]
    if (
        plan["schema"] != PLAN_SCHEMA
        or plan["source"]
        != {
            "ref": ACQUISITION_REF,
            "version": ACQUISITION_VERSION,
        }
        or kind not in MAX_AUDIO_BYTES
        or plan["max_audio_bytes"] != MAX_AUDIO_BYTES[kind]
        or (plan["freeze"] is None) != (kind == "train-dev")
    ):
        raise ValueError("unknown corpus preparation plan")
    start = time.monotonic()
    report_dir = working / "n2b0-reports"
    report_dir.mkdir(exist_ok=False)
    final = working / f"n2b0-{kind}"
    result = {"schema": "vocoroute-vctk092-preparation-receipt-v1", "plan": plan}
    try:
        if final.exists() or torch.cuda.is_available() or torch.cuda.device_count():
            raise RuntimeError("corpus staging needs a fresh CPU-only job with no visible GPUs")
        torch.set_num_threads(1)
        free = shutil.disk_usage(working).free
        if free < plan["max_audio_bytes"] + STORAGE_MARGIN_BYTES:
            raise ValueError(f"insufficient scratch space for the approved audio cap: {free} bytes")
        result["runtime"] = {
            "cpu_only": True,
            "torch": str(torch.__version__),
            "torchaudio": version("torchaudio"),
            "free_bytes_before": free,
        }
        archive = _single_file(inputs, ARCHIVE_NAME, parent_name="vctk-0.92")
        print("Verifying mounted VCTK archive size, MD5, SHA-256 and attribution", flush=True)
        attribution = verify_acquisition(archive)
        protocol = SplitProtocol()
        with (
            tempfile.TemporaryDirectory(prefix=".n2b0-", dir=working) as scratch,
            ZipFile(archive) as zipped,
        ):
            payload = Path(scratch) / "payload"
            inventory = inspect_archive(zipped, protocol.microphone)
            write_json(report_dir / "inventory.json", inventory.report)
            print("Archive verified. Preparing the approved role partitions", flush=True)
            if plan["freeze"] is None:
                release = prepare_train_dev(
                    zipped,
                    inventory,
                    protocol,
                    payload,
                    report_dir,
                    attribution,
                    max_audio_bytes=plan["max_audio_bytes"],
                )
                owner = ACQUISITION_REF.split("/")[0]
                release["dataset_receipts"] = {}
                for name in ("expanded", "pilot"):
                    release["dataset_receipts"][name] = write_json(
                        payload / name / "dataset-receipt.json",
                        {
                            "schema": DATASET_SCHEMA,
                            "ref": f"{owner}/vocoroute-vctk092-n2b0-{name}",
                            "version": 1,
                            "manifest_sha256": release["manifests"][name],
                            "audio_roles": ["train", "dev"],
                        },
                    )
            else:
                freeze = _single_file(inputs, "freeze.json", parent_name="n2b0-train-dev")
                release = prepare_held_out(
                    zipped,
                    inventory,
                    protocol,
                    freeze.parent,
                    plan["freeze"]["sha256"],
                    payload,
                    attribution,
                    max_audio_bytes=plan["max_audio_bytes"],
                )
            release["saved_tree_bytes"] = sum(
                p.stat().st_size for p in payload.rglob("*") if p.is_file()
            )
            payload.rename(final)
        result.update(status="verified", release=release, output=final.name)
    except Exception as exc:
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        (report_dir / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        result["elapsed_seconds"] = time.monotonic() - start
        write_json(report_dir / "prep-receipt.json", result)
    print("VocoRoute corpus preparation status:", result["status"], flush=True)
    return result


_BOOTSTRAP = f"""\
import base64
import hashlib
import io
import sys
import tempfile
import zipfile
from pathlib import Path

payload = base64.b64decode(PAYLOAD_B64, validate=True)
if hashlib.sha256(payload).hexdigest() != PLAN["payload_sha256"]:
    raise ValueError("staged corpus source payload checksum mismatch")
with tempfile.TemporaryDirectory(prefix="vocoroute-corpus-source-") as scratch:
    root = Path(scratch)
    with zipfile.ZipFile(io.BytesIO(payload)) as zipped:
        if sorted(zipped.namelist()) != PLAN["source_files"]:
            raise ValueError("unexpected staged corpus source files")
        digest = hashlib.sha256()
        for name in PLAN["source_files"]:
            relative = Path(name)
            if relative.is_absolute() or relative.parts[0] != "src" or ".." in relative.parts:
                raise ValueError("invalid staged source path")
            data = zipped.read(name)
            digest.update(name.encode() + b"\\0" + hashlib.sha256(data).digest())
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        if digest.hexdigest() != PLAN["source_tree_sha256"]:
            raise ValueError("staged corpus source tree checksum mismatch")
    sys.path.insert(0, str(root / "src"))
    from experiments.vocoroute_corpus_kaggle import run_corpus_kernel

    run_corpus_kernel(PLAN, Path({KAGGLE_INPUT!r}), Path({KAGGLE_WORKING!r}))
"""
