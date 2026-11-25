"""Stage the fixed N2B.1 diagnostic without contacting Kaggle."""

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
from .vocoroute_corpus_kaggle import SOURCES, _fixed_zip, _json_bytes, embedded_kernel_bootstrap

OWNER = "sambhavjainsj"
DATA_SLUG = "vocoroute-vctk092-n2b1-p229-128"
NOTEBOOK_SLUG = "vocoroute-vctk092-n2b1-overfit"
ACCELERATOR = "NvidiaTeslaT4"
PLAN_SCHEMA = "vocoroute-overfit-kaggle-v1"
OVERFIT_SOURCES = sorted(
    SOURCES
    + [
        "src/experiments/vocoroute_overfit_kaggle.py",
        "src/vocoroute/losses.py",
        "src/vocoroute/model.py",
        "src/vocoroute/overfit.py",
        "src/vocoroute/refinement.py",
        "src/vocoroute/train_base.py",
        "src/vocoroute/training_checkpoint.py",
        "src/vocoroute/training_device.py",
        "src/vocoroute/training_resume.py",
    ]
)


def stage_overfit_kernel(sha: str, output: Path) -> dict:
    """Render a GPU script exclusively from a user-created commit containing N2B.1."""
    from vocoroute.overfit import (
        GPU_SESSION_LIMIT,
        N2B1_CORPUS,
        PLANNED_GPU_SECONDS,
        diagnostic_config,
    )
    from vocoroute.training_config import canonical_json

    if output.exists() or not re.fullmatch(r"[0-9a-f]{7,40}", sha):
        raise ValueError("use a fresh stage path and an explicit user-created commit SHA")
    commit = _git("rev-parse", "--verify", f"{sha}^{{commit}}").stdout.strip()
    contents = _archive_bytes(RECORD, commit, OVERFIT_SOURCES)
    # Also pin the stager contract, which constructs the metadata and run configuration.
    for name in OVERFIT_SOURCES:
        checked_out = (REPO_ROOT / name).read_bytes().replace(b"\r\n", b"\n")
        if checked_out != contents[name].replace(b"\r\n", b"\n"):
            raise ValueError(f"checkout differs from the committed diagnostic source: {name}")
    payload = _fixed_zip(contents)
    config = diagnostic_config()
    plan = {
        "schema": PLAN_SCHEMA,
        "commit": commit,
        "mode": RECORD,
        "corpus": asdict(N2B1_CORPUS),
        "dataset": {"ref": f"{OWNER}/{DATA_SLUG}", "version": 1},
        "config": config,
        "config_sha256": hashlib.sha256(canonical_json(config)).hexdigest(),
        "accelerator": ACCELERATOR,
        "session_timeout_seconds": GPU_SESSION_LIMIT,
        "planned_gpu_seconds": PLANNED_GPU_SECONDS,
        "source_files": OVERFIT_SOURCES,
        "source_tree_sha256": _tree_sha256(contents),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
    }
    spec = KernelSpec(NOTEBOOK_SLUG, "VocoRoute VCTK092 N2B1 Overfit", (), 0, gpu=True)
    check_slug(spec)
    metadata = kernel_metadata(spec, username=OWNER, code_file="overfit.py")
    metadata.update(
        kernel_type="script",
        enable_internet="false",
        machine_shape=ACCELERATOR,
        dataset_sources=[plan["dataset"]["ref"]],
    )
    bootstrap = embedded_kernel_bootstrap(
        "experiments.vocoroute_overfit_kaggle", "run_overfit_kernel"
    )
    script = f"PLAN = {plan!r}\nPAYLOAD_B64 = {base64.b64encode(payload).decode()!r}\n" + bootstrap
    output.mkdir(parents=True, exist_ok=False)
    kernel = output / "kernel"
    kernel.mkdir()
    (kernel / "kernel-metadata.json").write_bytes(_json_bytes(metadata))
    (kernel / "overfit.py").write_bytes(script.encode())
    result = {
        "notebook": metadata["id"],
        "plan": plan,
        "script_sha256": hashlib.sha256(script.encode()).hexdigest(),
        "metadata_sha256": hashlib.sha256(_json_bytes(metadata)).hexdigest(),
    }
    (output / "plan.json").write_bytes(_json_bytes(result))
    return result


def run_overfit_kernel(plan: dict, inputs: Path, working: Path) -> None:
    started_at = monotonic()
    output = working / "vocoroute-n2b1"
    try:
        from vocoroute.overfit import (
            GPU_SESSION_LIMIT,
            N2B1_CORPUS,
            PLANNED_GPU_SECONDS,
            diagnostic_config,
            run_overfit,
        )
        from vocoroute.training_config import canonical_json

        config = diagnostic_config()
        if (
            plan["schema"] != PLAN_SCHEMA
            or plan["mode"] != RECORD
            or not re.fullmatch(r"[0-9a-f]{40}", plan["commit"])
            or plan["corpus"] != asdict(N2B1_CORPUS)
            or plan["config"] != config
            or plan["config_sha256"] != hashlib.sha256(canonical_json(config)).hexdigest()
            or plan["dataset"] != {"ref": f"{OWNER}/{DATA_SLUG}", "version": 1}
            or plan["accelerator"] != ACCELERATOR
            or plan["session_timeout_seconds"] != GPU_SESSION_LIMIT
            or plan["planned_gpu_seconds"] != PLANNED_GPU_SECONDS
        ):
            raise ValueError("invalid pinned N2B.1 record plan")
        mounts = [path for path in inputs.rglob(DATA_SLUG) if path.is_dir()]
        if len(mounts) != 1:
            raise ValueError("expected exactly one private single-utterance Kaggle mount")
        result = run_overfit(inputs, mounts[0], output, worker_started_at=started_at, code=plan)
        print("N2B.1 diagnostic:", result["status"], result["stop_reason"], flush=True)
    except Exception as exc:
        output.mkdir(exist_ok=True)
        (output / "failure.log").write_text(traceback.format_exc(), encoding="utf-8")
        receipt = output / "diagnostic-receipt.json"
        if not receipt.exists():
            receipt.write_bytes(
                _json_bytes(
                    {
                        "schema": "vocoroute-single-utterance-receipt-v1",
                        "status": "failed",
                        "code": plan,
                        "updates": 0,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "elapsed_seconds": monotonic() - started_at,
                    }
                )
            )
        print("N2B.1 diagnostic failed:", type(exc).__name__, str(exc), flush=True)
