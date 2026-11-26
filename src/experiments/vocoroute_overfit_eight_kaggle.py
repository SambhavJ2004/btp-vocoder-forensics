"""RECORD staging for the isolated eight-utterance VocoRoute-P notebook."""

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
from .vocoroute_overfit_kaggle import ACCELERATOR, OWNER
from .vocoroute_phase_kaggle import PHASE_SOURCES

SCHEMA = "vocoroute-eight-overfit-kaggle-v1"
NOTEBOOK_SLUG = "vocoroute-vctk092-n2b2-overfit"
DATASET_SLUG = "vocoroute-vctk092-n2b2-overfit-eight"
DATASET = {"ref": f"{OWNER}/{DATASET_SLUG}", "version": 1}
OUTPUT_ROOT = "vocoroute-n2b2"
SOURCES = sorted(
    PHASE_SOURCES
    + [
        "src/experiments/vocoroute_overfit_eight_kaggle.py",
        "src/vocoroute/overfit_eight.py",
    ]
)


def _order_digest(ids: tuple[str, ...], settings: dict) -> str:
    from vocoroute.overfit_eight import round_order

    order = hashlib.sha256()
    for index in range(settings["max_updates"] // settings["round_size"]):
        for uid in round_order(ids, settings["seed"], index):
            order.update((uid + "\n").encode())
    return order.hexdigest()


def draft_plan(source_metadata: Path) -> dict:
    from vocoroute.overfit_eight import N2B2_CORPUS, protocol, source_inventory
    from vocoroute.training_config import canonical_json

    config = protocol()
    inventory = source_inventory(source_metadata)
    ids = tuple(row["utterance_id"] for row in inventory["recordings"])
    return {
        "schema": SCHEMA,
        "accelerator": ACCELERATOR,
        "dataset": DATASET.copy(),
        "corpus": asdict(N2B2_CORPUS),
        "protocol": config,
        "input_inventory": inventory,
        "protocol_sha256": hashlib.sha256(canonical_json(config)).hexdigest(),
        "input_inventory_sha256": hashlib.sha256(canonical_json(inventory)).hexdigest(),
        "expected_full_order_sha256": _order_digest(ids, config),
    }


def stage_eight_kernel(sha: str, source_metadata: Path, output: Path) -> dict:
    if output.exists() or not re.fullmatch(r"[0-9a-f]{7,40}", sha):
        raise ValueError("use a fresh stage directory and an explicit user-created commit SHA")
    commit = _git("rev-parse", "--verify", f"{sha}^{{commit}}").stdout.strip()
    contents = _archive_bytes(RECORD, commit, SOURCES)
    for name, committed in contents.items():
        checked_out = (REPO_ROOT / name).read_bytes().replace(b"\r\n", b"\n")
        if checked_out != committed.replace(b"\r\n", b"\n"):
            raise ValueError(f"checkout differs from committed diagnostic source: {name}")
    packed = _fixed_zip(contents)
    plan = {
        **draft_plan(source_metadata),
        "mode": RECORD,
        "commit": commit,
        "source_files": SOURCES,
        "source_tree_sha256": _tree_sha256(contents),
        "payload_sha256": hashlib.sha256(packed).hexdigest(),
    }
    spec = KernelSpec(NOTEBOOK_SLUG, "VocoRoute VCTK092 N2B2 Overfit", (), 0, gpu=True)
    check_slug(spec)
    metadata = kernel_metadata(spec, username=OWNER, code_file="overfit_eight.py")
    metadata.update(
        kernel_type="script",
        enable_internet="false",
        machine_shape=ACCELERATOR,
        dataset_sources=[DATASET["ref"]],
    )
    bootstrap = embedded_kernel_bootstrap(__name__, "run_eight_kernel")
    script = f"PLAN = {plan!r}\nPAYLOAD_B64 = {base64.b64encode(packed).decode()!r}\n{bootstrap}"
    script_bytes, metadata_bytes = script.encode("utf-8"), _json_bytes(metadata)
    kernel = output / "kernel"
    kernel.mkdir(parents=True, exist_ok=False)
    (kernel / "overfit_eight.py").write_bytes(script_bytes)
    (kernel / "kernel-metadata.json").write_bytes(metadata_bytes)
    result = {
        "notebook": metadata["id"],
        "plan": plan,
        "script_sha256": hashlib.sha256(script_bytes).hexdigest(),
        "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
    }
    (output / "plan.json").write_bytes(_json_bytes(result))
    return result


def run_eight_kernel(plan: dict, inputs: Path, working: Path) -> None:
    started_at = monotonic()
    output = working / OUTPUT_ROOT
    try:
        from vocoroute.overfit_eight import N2B2_CORPUS, protocol, run_diagnostic
        from vocoroute.training_config import canonical_json

        inventory = plan["input_inventory"]
        ids = tuple(row["utterance_id"] for row in inventory["recordings"])
        settings = protocol()
        if (
            plan["schema"] != SCHEMA
            or plan["mode"] != RECORD
            or not re.fullmatch(r"[0-9a-f]{40}", plan["commit"])
            or plan["accelerator"] != ACCELERATOR
            or plan["dataset"] != DATASET
            or plan["corpus"] != asdict(N2B2_CORPUS)
            or plan["protocol"] != settings
            or hashlib.sha256(canonical_json(settings)).hexdigest() != plan["protocol_sha256"]
            or hashlib.sha256(canonical_json(inventory)).hexdigest()
            != plan["input_inventory_sha256"]
            or len(set(ids)) != settings["round_size"]
            or _order_digest(ids, settings) != plan["expected_full_order_sha256"]
        ):
            raise ValueError("invalid pinned eight-utterance RECORD plan")
        mounts = [path for path in inputs.rglob(DATASET_SLUG) if path.is_dir()]
        if len(mounts) != 1:
            raise ValueError("expected one private eight-utterance Kaggle mount")
        report = run_diagnostic(inputs, mounts[0], output, code=plan, started_at=started_at)
        print(
            "N2B.2 diagnostic:",
            report["status"],
            report["stop_reason"],
            report["updates"],
            flush=True,
        )
    except Exception as exc:
        output.mkdir(exist_ok=True)
        (output / "failure.log").write_text(traceback.format_exc(), encoding="utf-8")
        receipt = output / "diagnostic-receipt.json"
        if not receipt.exists():
            receipt.write_bytes(
                _json_bytes(
                    {
                        "schema": "vocoroute-n2b2-overfit-receipt-v1",
                        "status": "failed",
                        "code": plan,
                        "updates": 0,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "worker_seconds": monotonic() - started_at,
                    }
                )
            )
        raise
