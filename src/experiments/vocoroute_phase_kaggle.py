"""Stage the N2B.1-P private two-arm diagnostic in RECORD mode."""

import base64
import hashlib
import re
import traceback
from dataclasses import asdict
from pathlib import Path
from time import monotonic

from .kaggle_harness import (
    RECORD,
    REPO_ROOT,
    KernelSpec,
    _archive_bytes,
    _git,
    _tree_sha256,
    check_slug,
    kernel_metadata,
)
from .vocoroute_corpus_kaggle import _fixed_zip, _json_bytes, embedded_kernel_bootstrap
from .vocoroute_overfit_kaggle import ACCELERATOR, DATA_SLUG, OVERFIT_SOURCES, OWNER

NOTEBOOK_SLUG = "vocoroute-vctk092-n2b1p-phase-ablation"
OUTPUT_ROOT = "vocoroute-n2b1p"
PLAN_SCHEMA = "vocoroute-phase-ablation-kaggle-v1"
DATASET = {"ref": f"{OWNER}/{DATA_SLUG}", "id": 13671242, "version": 1}
PHASE_SOURCES = sorted(
    OVERFIT_SOURCES
    + [
        "src/experiments/vocoroute_phase_kaggle.py",
        "src/vocoroute/phase_ablation.py",
        "src/vocoroute/phase_diagnostic.py",
    ]
)


def draft_protocol(baseline_receipt: Path) -> dict:
    from vocoroute.overfit import N2B1_CORPUS
    from vocoroute.phase_ablation import phase_protocol
    from vocoroute.phase_diagnostic import baseline_record
    from vocoroute.training_config import canonical_json

    protocol = phase_protocol()
    return {
        "schema": PLAN_SCHEMA,
        "accelerator": ACCELERATOR,
        "dataset": DATASET.copy(),
        "corpus": asdict(N2B1_CORPUS),
        "protocol": protocol,
        "protocol_sha256": hashlib.sha256(canonical_json(protocol)).hexdigest(),
        "baseline": baseline_record(baseline_receipt),
    }


def stage_phase_kernel(sha: str, baseline_receipt: Path, output: Path) -> dict:
    """Package a user-created commit. This function does not contact Kaggle."""
    if output.exists() or not re.fullmatch(r"[0-9a-f]{7,40}", sha):
        raise ValueError("use a fresh stage path and an explicit user-created commit SHA")
    commit = _git("rev-parse", "--verify", f"{sha}^{{commit}}").stdout.strip()
    contents = _archive_bytes(RECORD, commit, PHASE_SOURCES)
    for name in PHASE_SOURCES:
        if (REPO_ROOT / name).read_bytes().replace(b"\r\n", b"\n") != contents[name].replace(
            b"\r\n", b"\n"
        ):
            raise ValueError(f"checkout differs from the committed phase ablation source: {name}")
    packed = _fixed_zip(contents)
    plan = {
        **draft_protocol(baseline_receipt),
        "mode": RECORD,
        "commit": commit,
        "source_files": PHASE_SOURCES,
        "source_tree_sha256": _tree_sha256(contents),
        "payload_sha256": hashlib.sha256(packed).hexdigest(),
    }
    spec = KernelSpec(NOTEBOOK_SLUG, "VocoRoute VCTK092 N2B1P Phase Ablation", (), 0, gpu=True)
    check_slug(spec)
    metadata = kernel_metadata(spec, username=OWNER, code_file="phase_ablation.py")
    metadata.update(
        kernel_type="script",
        enable_internet="false",
        machine_shape=ACCELERATOR,
        dataset_sources=[DATASET["ref"]],
    )
    bootstrap = embedded_kernel_bootstrap(__name__, "run_phase_kernel")
    script = f"PLAN = {plan!r}\nPAYLOAD_B64 = {base64.b64encode(packed).decode()!r}\n{bootstrap}"
    kernel = output / "kernel"
    kernel.mkdir(parents=True, exist_ok=False)
    script_bytes = script.encode("utf-8")
    metadata_bytes = _json_bytes(metadata)
    (kernel / "phase_ablation.py").write_bytes(script_bytes)
    (kernel / "kernel-metadata.json").write_bytes(metadata_bytes)
    result = {
        "notebook": metadata["id"],
        "plan": plan,
        "script_sha256": hashlib.sha256(script_bytes).hexdigest(),
        "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
    }
    (output / "plan.json").write_bytes(_json_bytes(result))
    return result


def run_phase_kernel(plan: dict, inputs: Path, working: Path) -> None:
    started_at = monotonic()
    output = working / OUTPUT_ROOT
    try:
        from vocoroute.overfit import N2B1_CORPUS
        from vocoroute.phase_ablation import phase_protocol, run_phase_ablation
        from vocoroute.phase_diagnostic import BASELINE_RECEIPT_SHA256
        from vocoroute.training_config import canonical_json, is_sha256

        protocol = phase_protocol()
        original = plan["baseline"]
        if (
            plan["schema"] != PLAN_SCHEMA
            or plan["mode"] != RECORD
            or not re.fullmatch(r"[0-9a-f]{40}", plan["commit"])
            or plan["dataset"] != DATASET
            or plan["accelerator"] != ACCELERATOR
            or plan["corpus"] != asdict(N2B1_CORPUS)
            or plan["protocol"] != protocol
            or plan["protocol_sha256"] != hashlib.sha256(canonical_json(protocol)).hexdigest()
            or original["receipt_sha256"] != BASELINE_RECEIPT_SHA256
            or not is_sha256(original["initial_fp32_sha256"])
            or not is_sha256(original["step_1000_raw_sha256"])
        ):
            raise ValueError("invalid pinned phase ablation RECORD plan")
        mounts = [path for path in inputs.rglob(DATA_SLUG) if path.is_dir()]
        if len(mounts) != 1:
            raise ValueError("expected exactly one private single-utterance Kaggle mount")
        result = run_phase_ablation(
            inputs, mounts[0], output, original=original, code=plan, started_at=started_at
        )
        print("N2B.1-P diagnostic:", result["status"], result["stop_reason"], flush=True)
    except Exception as exc:
        output.mkdir(exist_ok=True)
        (output / "failure.log").write_text(traceback.format_exc(), encoding="utf-8")
        receipt = output / "ablation-receipt.json"
        if not receipt.exists():
            receipt.write_bytes(
                _json_bytes(
                    {
                        "schema": "vocoroute-n2b1p-phase-ablation-receipt-v1",
                        "status": "failed",
                        "code": plan,
                        "matched_updates": 0,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "worker_seconds": monotonic() - started_at,
                    }
                )
            )
        raise
