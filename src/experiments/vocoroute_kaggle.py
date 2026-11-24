"""Local-only staging and a version-pinned Kaggle kernel for VocoRoute-P."""

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import zipfile
from importlib.metadata import version
from pathlib import Path, PurePosixPath

from .kaggle_harness import (
    CODE_DATASET_SLUG,
    RECORD,
    SNAPSHOT_ROOT_IN_ZIP,
    KernelSpec,
    kernel_metadata,
    slugify,
    stage_snapshot,
)

SPEC_SCHEMA = "vocoroute-kaggle-spec-v1"
DATASET_SCHEMA = "vocoroute-kaggle-train-dev-v1"
_RUNTIME_KEYS = {"python", "torch", "torchaudio", "cuda_build", "soundfile", "numpy", "librosa"}


def _relative(value: str) -> str:
    if (
        not isinstance(value, str)
        or "\\" in value
        or ":" in value
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise ValueError("expected a portable relative Kaggle mount path")
    return value


def _sha(value: str) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _mount_spec(document: dict, fields: set[str]) -> None:
    if not isinstance(document, dict) or set(document) != {"ref", "version", "mount"} | fields:
        raise ValueError("invalid versioned Kaggle dataset mount")
    if not isinstance(document["ref"], str) or not re.fullmatch(
        r"[a-z0-9][a-z0-9-]*/[a-z0-9][a-z0-9-]*", document["ref"]
    ):
        raise ValueError("dataset refs must use owner/slug")
    if type(document["version"]) is not int or document["version"] < 1:
        raise ValueError("an explicit positive Kaggle dataset version is required")
    _relative(document["mount"])
    if "receipt_sha256" in fields and not _sha(document["receipt_sha256"]):
        raise ValueError("dataset or resume receipt requires a SHA-256 pin")
    if "checkpoint" in fields:
        _relative(document["checkpoint"])


def _validate_spec(spec: dict) -> None:
    if (
        not isinstance(spec, dict)
        or set(spec)
        != {
            "schema",
            "title",
            "code",
            "data",
            "resume",
            "runtime",
        }
        or spec["schema"] != SPEC_SCHEMA
    ):
        raise ValueError("invalid VocoRoute-P Kaggle specification")
    if not isinstance(spec["title"], str) or not slugify(spec["title"]).startswith("vocoroute-"):
        raise ValueError("use a separate vocoroute-* kernel title")
    _mount_spec(spec["code"], set())
    _mount_spec(spec["data"], {"receipt_sha256"})
    mounts = [spec["code"], spec["data"]]
    if spec["resume"] is not None:
        _mount_spec(spec["resume"], {"checkpoint", "receipt_sha256"})
        mounts.append(spec["resume"])
    for key in ("ref", "mount"):
        if len({mount[key] for mount in mounts}) != len(mounts):
            raise ValueError("code, train/dev, and resume must have distinct sources")
    if spec["code"]["ref"].split("/")[1] == CODE_DATASET_SLUG:
        raise ValueError("use a separate code dataset for VocoRoute-P")
    runtime = spec["runtime"]
    if (
        not isinstance(runtime, dict)
        or set(runtime) != _RUNTIME_KEYS
        or not all(isinstance(value, str) and value.strip() for value in runtime.values())
    ):
        raise ValueError("pin Python, audio libraries, Torch builds and CUDA build")


def stage_vocoroute_plan(sha: str, spec_path: Path, config_path: Path, output: Path) -> dict:
    """Prepare a private kernel and code snapshot locally. Make no Kaggle calls."""
    from vocoroute.training_config import BUDGET_CONFIG_SCHEMA, load_training_config, parse_json

    if output.exists() or not re.fullmatch(r"[0-9a-f]{7,40}", sha):
        raise ValueError("use a fresh output directory and an explicit existing commit SHA")
    spec = parse_json(spec_path.read_bytes())
    _validate_spec(spec)
    config = load_training_config(config_path)
    if (
        config.schema != BUDGET_CONFIG_SCHEMA
        or config.device != "cuda"
        or not config.manifest_sha256
    ):
        raise ValueError("Kaggle training requires a CUDA v2 config and frozen manifest pin")
    output.mkdir(parents=True, exist_ok=False)
    snapshot = stage_snapshot(output / "code", mode=RECORD, sha=sha)
    archive_path = output / "code" / snapshot.archive
    archive = archive_path.read_bytes()
    with zipfile.ZipFile(archive_path) as source:
        required = {
            f"{SNAPSHOT_ROOT_IN_ZIP}/src/{name}"
            for name in (
                "experiments/vocoroute_kaggle.py",
                "vocoroute/training_resume.py",
            )
        }
        if not required <= set(source.namelist()):
            raise ValueError("the specified commit does not contain N2K training support")
    plan = {
        "spec": spec,
        "config": config.document(),
        "config_sha256": config.sha256,
        "snapshot": snapshot.as_record(),
        "archive_sha256": hashlib.sha256(archive).hexdigest(),
    }
    kernel = output / "kernel"
    kernel.mkdir()
    slug, username = slugify(spec["title"]), spec["code"]["ref"].split("/")[0]
    metadata = kernel_metadata(
        KernelSpec(slug, spec["title"], (), 0, gpu=True), username=username, code_file="train.py"
    )
    sources = [spec[label]["ref"] for label in ("code", "data", "resume") if spec[label]]
    metadata.update(kernel_type="script", enable_internet="false", dataset_sources=sources)
    _write_json(kernel / "kernel-metadata.json", metadata)
    (kernel / "train.py").write_text(
        f"PLAN_JSON = {json.dumps(plan, sort_keys=True)!r}\n" + _KERNEL, encoding="utf-8"
    )
    _write_json(output / "plan.json", plan)
    return plan


def _write_json(path: Path, data: dict) -> None:
    path.write_text(
        json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def _under(root: Path, name: str) -> Path:
    path = root.joinpath(*PurePosixPath(_relative(name)).parts).resolve(strict=True)
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Kaggle mount path escapes the approved input root")
    return path


def _dataset_manifest(plan: dict, root: Path) -> tuple[Path, Path]:
    expected = plan["spec"]["data"]
    mount = _under(root, expected["mount"])
    raw = _under(mount, "dataset-receipt.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected["receipt_sha256"]:
        raise ValueError("mounted train/dev dataset receipt SHA-256 mismatch")
    manifest = _under(mount, "manifest.json")
    wanted = plan["config"]["manifest_sha256"]
    if (
        json.loads(raw)
        != {
            "schema": DATASET_SCHEMA,
            "ref": expected["ref"],
            "version": expected["version"],
            "manifest_sha256": wanted,
            "audio_roles": ["train", "dev"],
        }
        or hashlib.sha256(manifest.read_bytes()).hexdigest() != wanted
    ):
        raise ValueError("mounted train/dev dataset identity disagrees with the plan")
    # The frozen manifest retains eval metadata. Eval recordings must be absent.
    for row in json.loads(manifest.read_bytes())["rows"]:
        if row["role"] == "eval" and any(
            (mount / _relative(row[key])).exists() for key in ("native_path", "prepared_path")
        ):
            raise ValueError("held-out evaluation audio is present in the training mount")
    return mount, manifest


def run_kernel_plan(plan: dict, source: Path, inputs: Path, output: Path) -> dict:
    """Called only by the staged, verified kernel; preserve failures as outputs."""
    import torch

    from vocoroute.training_config import TrainingConfig

    config = TrainingConfig.from_document(plan["config"])
    if config.sha256 != plan["config_sha256"] or config.device != "cuda":
        raise ValueError("kernel config digest or device disagrees with the plan")
    actual = {package: version(package) for package in ("soundfile", "numpy", "librosa")}
    import torchaudio

    actual.update(
        python=platform.python_version(),
        torch=str(torch.__version__),
        torchaudio=torchaudio.__version__,
        cuda_build=torch.version.cuda,
    )
    if actual != plan["spec"]["runtime"]:
        raise ValueError(f"Kaggle runtime disagrees with the approved pins: {actual}")
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        raise RuntimeError("Kaggle training requires visible CUDA device 0")
    _write_json(output / "runtime.json", actual)
    smi = shutil.which("nvidia-smi")
    if smi:
        driver = subprocess.run(
            [smi, "--query-gpu=name,driver_version,memory.total", "--format=csv"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        (output / "gpu.txt").write_text(driver.stdout + driver.stderr, encoding="utf-8")
    mount, manifest = _dataset_manifest(plan, inputs)
    config_path = output / "training-config.json"
    _write_json(config_path, plan["config"])
    command = [
        sys.executable,
        "-B",
        "-m",
        "vocoroute.train_base",
        "--train",
        "--device",
        "cuda",
        "--config",
        str(config_path),
        "--manifest",
        str(manifest),
        "--data-root",
        str(mount),
        "--out-dir",
        str(output / "training"),
    ]
    resume = plan["spec"]["resume"]
    if resume:
        checkpoint = _under(_under(inputs, resume["mount"]), resume["checkpoint"])
        command.extend(
            ("--resume", str(checkpoint), "--resume-receipt-sha256", resume["receipt_sha256"])
        )
    env = dict(os.environ, PYTHONPATH=str(source / "src"))
    with (output / "training.log").open("w", encoding="utf-8") as log:
        subprocess.run(
            command, env=env, cwd=source, stdout=log, stderr=subprocess.STDOUT, check=True
        )
    result = json.loads((output / "training" / "run.json").read_bytes())
    return {key: result[key] for key in ("status", "reason", "updates", "latest_resume")}


# Bootstrap uses only stdlib until the attached commit's contents are verified.
_KERNEL = """\
import hashlib
import io
import json
import sys
import tempfile
import traceback
import zipfile
from pathlib import Path

plan = json.loads(PLAN_JSON)
inputs = Path("/kaggle/input")
output = Path("/kaggle/working/vocoroute-run")
output.mkdir(exist_ok=False)
receipt = {"schema": "vocoroute-kaggle-receipt-v1", "status": "running", "plan": plan}
try:
    snapshot = plan["snapshot"]
    mounted = (inputs / plan["spec"]["code"]["mount"]).resolve(strict=True)
    if not mounted.is_relative_to(inputs.resolve()):
        raise ValueError("code mount escapes Kaggle inputs")
    actual = json.loads((mounted / "btpvf_snapshot.json").read_bytes())
    if actual != snapshot or snapshot["mode"] != "record":
        raise ValueError("mounted snapshot identity disagrees with the plan")
    with tempfile.TemporaryDirectory(prefix="vocoroute-code-") as scratch:
        archive = mounted / snapshot["archive"]
        if archive.is_file():
            payload = archive.read_bytes()
            if hashlib.sha256(payload).hexdigest() != plan["archive_sha256"]:
                raise ValueError("code archive SHA-256 mismatch")
            with zipfile.ZipFile(io.BytesIO(payload)) as zipped:
                zipped.extractall(scratch)
            source = Path(scratch) / "__SNAPSHOT_ROOT__"
        else:
            candidates = (mounted / "btpvf_code" / "__SNAPSHOT_ROOT__",
                          mounted / "__SNAPSHOT_ROOT__", mounted / "btpvf_code")
            roots = [path for path in candidates if (path / "pyproject.toml").is_file()]
            if len(roots) != 1:
                raise ValueError("expected one expanded code snapshot")
            source = roots[0]
        digest = hashlib.sha256()
        for path in sorted(source.rglob("*"), key=lambda p: p.relative_to(source).as_posix()):
            if path.is_symlink():
                raise ValueError("symlinks are not accepted in the mounted code snapshot")
            if path.is_file():
                digest.update(path.relative_to(source).as_posix().encode("utf-8"))
                digest.update(b"\\0")
                digest.update(hashlib.sha256(path.read_bytes()).digest())
        if digest.hexdigest() != snapshot["tree_sha256"]:
            raise ValueError("attached source tree digest mismatch")
        sys.path.insert(0, str(source / "src"))
        from experiments.vocoroute_kaggle import run_kernel_plan
        receipt.update(run_kernel_plan(plan, source, inputs, output))
except Exception as exc:
    receipt.update(status="failed", error_type=type(exc).__name__, error=str(exc))
    (output / "failure.log").write_text(traceback.format_exc(), encoding="utf-8")
finally:
    (output / "kernel-receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
print("VocoRoute-P training status:", receipt["status"])
""".replace("__SNAPSHOT_ROOT__", SNAPSHOT_ROOT_IN_ZIP)
