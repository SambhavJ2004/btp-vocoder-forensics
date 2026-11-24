"""Instrumentation for the synthetic N2K.1 acceptance scripts."""

import hashlib
import json
import math
import os
import platform
import random
import shutil
import subprocess
import time
from importlib.metadata import version

ACCEPTANCE_STARTED = time.monotonic()


def acceptance_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def acceptance_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def acceptance_mount(root, descriptor):
    root = root.resolve()
    slug = descriptor["ref"].split("/")[1]
    options = {
        path.resolve() for path in (root / descriptor["mount"], root / slug) if path.is_dir()
    }
    if len(options) != 1 or not next(iter(options)).is_relative_to(root):
        raise ValueError(f"missing, ambiguous, or escaping dataset mount: {descriptor['ref']}")
    return options.pop()


def acceptance_state(root, pin):
    import torch

    receipt = root / "resume.json"
    payload = root / "training.pt"
    if acceptance_sha(receipt) != pin or (
        acceptance_sha(payload) != json.loads(receipt.read_bytes())["state_sha256"]
    ):
        raise ValueError("acceptance checkpoint digest mismatch")
    return torch.load(payload, map_location="cpu", weights_only=True)


def acceptance_compare(control, candidate, left_updates, right_updates):
    import torch

    report = {"exact": True, "within_tolerance": True, "max_abs": {}, "mismatches": []}

    def check(left, right, path, numeric=True):
        if type(left) is not type(right):
            exact = close = False
        elif isinstance(left, dict) and left.keys() == right.keys():
            for key in left:
                check(left[key], right[key], f"{path}.{key}", numeric)
            return
        elif isinstance(left, (list, tuple)) and len(left) == len(right):
            for index, (a, b) in enumerate(zip(left, right, strict=True)):
                check(a, b, f"{path}.{index}", numeric)
            return
        elif isinstance(left, torch.Tensor):
            same_shape = left.shape == right.shape and left.dtype == right.dtype
            exact = same_shape and torch.equal(left, right)
            close = exact
            if numeric and same_shape and left.is_floating_point() and left.numel():
                group = path.split(".")[0]
                gap = (left.double() - right.double()).abs().max().item()
                if math.isfinite(gap):
                    report["max_abs"][group] = max(gap, report["max_abs"].get(group, 0.0))
                if path.endswith(".step"):
                    close = exact
                else:
                    close = torch.allclose(left, right, atol=1e-7, rtol=1e-6)
        elif isinstance(left, float) and numeric:
            exact = left == right
            close = abs(left - right) <= 1e-7 + 1e-6 * abs(left)
            group = path.split(".")[0]
            report["max_abs"][group] = max(abs(left - right), report["max_abs"].get(group, 0.0))
        else:
            exact = close = left == right
        report["exact"] &= exact
        report["within_tolerance"] &= close
        if not exact:
            report["mismatches"].append({"field": path, "within_tolerance": close})

    for key in control:
        check(control[key], candidate[key], key, key not in {"rng", "scheduler"})
    check([row["uid"] for row in left_updates], [row["uid"] for row in right_updates], "order")
    check(
        [row["rng"] for row in left_updates], [row["rng"] for row in right_updates], "draws", False
    )
    check([row["loss"] for row in left_updates], [row["loss"] for row in right_updates], "losses")
    return report


def acceptance_run(output, mount, manifest, config, device, *, stop=None, resume=None, delay=0):
    import torch

    from vocoroute import train_base

    profile = {"updates": [], "checkpoint_saves": [], "checkpoint_loads": []}
    config_file = output.with_suffix(".json")
    acceptance_json(config_file, config)
    args = [
        "--train",
        "--device",
        device,
        "--config",
        str(config_file),
        "--manifest",
        str(manifest),
        "--data-root",
        str(mount),
        "--out-dir",
        str(output),
    ]
    if stop:
        args.extend(("--attempt-updates", str(stop)))
    if resume:
        args.extend(("--resume", str(resume[0]), "--resume-receipt-sha256", resume[1]))
    gpu = device == "cuda"
    if gpu:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)

    def sync():
        if gpu:
            torch.cuda.synchronize(0)

    update, save, load = train_base._train_example, train_base.write_resume, train_base.read_resume
    preflight = train_base._cuda_preflight

    def measure_update(model, optimizer, waveform, uid):
        draws = [random.random(), torch.rand(()).item()]
        if gpu:
            draws.append(torch.rand((), device="cuda:0").item())
        sync()
        start = time.monotonic()
        loss = update(model, optimizer, waveform, uid)
        sync()
        seconds = time.monotonic() - start
        profile["updates"].append(
            {
                "uid": uid,
                "loss": loss,
                "rng": draws,
                "seconds": seconds,
                "samples": waveform.numel(),
            }
        )
        if delay and len(profile["updates"]) == 1:
            time.sleep(delay)
        return loss

    def measure_checkpoint(function, records):
        def measured(path, *args, **kwargs):
            sync()
            start = time.monotonic()
            value = function(path, *args, **kwargs)
            sync()
            records.append(
                {
                    "seconds": time.monotonic() - start,
                    "bytes": sum(p.stat().st_size for p in path.iterdir() if p.is_file()),
                }
            )
            return value

        return measured

    def measure_preflight(*args):
        start = time.monotonic()
        preflight(*args)
        profile["preflight_seconds"] = time.monotonic() - start
        profile["cuda_fp32_forward_backward_passed"] = gpu

    train_base._train_example = measure_update
    train_base.write_resume = measure_checkpoint(save, profile["checkpoint_saves"])
    train_base.read_resume = measure_checkpoint(load, profile["checkpoint_loads"])
    train_base._cuda_preflight = measure_preflight
    started = time.monotonic()
    try:
        train_base.main(args)
    except SystemExit as exc:
        raise RuntimeError(f"N2K trainer exited with status {exc.code}") from exc
    finally:
        train_base._train_example, train_base.write_resume, train_base.read_resume = (
            update,
            save,
            load,
        )
        train_base._cuda_preflight = preflight
        profile["entrypoint_seconds"] = time.monotonic() - started
        if gpu:
            profile["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(0)
            profile["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(0)
        update_time = sum(row["seconds"] for row in profile["updates"])
        samples = sum(row["samples"] for row in profile["updates"])
        profile["audio_seconds_per_update_wall_second"] = (
            samples / 22050 / update_time if update_time else None
        )
        acceptance_json(output.with_suffix(".profile.json"), profile)
    result = json.loads((output / "run.json").read_bytes())
    period = config["budget"]["checkpoint_every_updates"]
    needed = {
        i
        for i in range(result["attempt_start_updates"] + 1, result["updates"] + 1)
        if i % period == 0
    }
    checkpoints = {int(path.name.rsplit("-", 1)[1]) for path in output.glob("resume-*")}
    if not needed <= checkpoints or result["convergence_claimed"]:
        raise ValueError("checkpoint cadence or convergence flag violated")
    return result, profile["updates"]


def acceptance_checkpoint(output, run):
    last = run["latest_resume"]
    return output / last["directory"], last["receipt_sha256"]


def run_acceptance(plan, source, inputs, output, settings):
    if not settings["local"]:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    import torchaudio

    from experiments.vocoroute_kaggle import _dataset_manifest, _under
    from vocoroute.train_base import _training_parts
    from vocoroute.training_checkpoint import run_provenance
    from vocoroute.training_config import TrainingConfig
    from vocoroute.training_data import TrainingAudio, load_manifest
    from vocoroute.training_device import training_runtime
    from vocoroute.training_resume import read_resume

    torch.set_num_threads(1)
    config = TrainingConfig.from_document(plan["config"])
    if config.sha256 != plan["config_sha256"]:
        raise ValueError("acceptance config digest mismatch")
    device = config.device
    if device != ("cpu" if settings["local"] else "cuda"):
        raise ValueError("acceptance device does not match execution mode")
    runtime = {name: version(name) for name in ("numpy", "soundfile", "librosa")}
    runtime.update(
        python=platform.python_version(),
        torch=str(torch.__version__),
        torchaudio=torchaudio.__version__,
        cuda_build=torch.version.cuda or "cpu",
    )
    acceptance_json(output / "runtime.json", runtime)
    pins = plan["spec"]["runtime"]
    if pins != runtime and set(pins.values()) != {"observe-in-session-a"}:
        raise ValueError("Kaggle image differs from the pinned runtime")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for GPU acceptance")
        binary = shutil.which("nvidia-smi")
        if binary:
            smi = subprocess.run(
                [
                    binary,
                    "--query-gpu=name,driver_version,memory.total,memory.free",
                    "--format=csv",
                ],
                capture_output=True,
                text=True,
                timeout=20,
                check=True,
            )
            (output / "gpu.txt").write_text(smi.stdout, encoding="utf-8")
        free, total = torch.cuda.mem_get_info(0)
        card = torch.cuda.get_device_properties(0)
        acceptance_json(
            output / "device.json",
            {
                "name": card.name,
                "capability": list(torch.cuda.get_device_capability(0)),
                "free_bytes": free,
                "total_bytes": total,
            },
        )
    inputs = inputs.resolve()
    data = plan["spec"]["data"]
    mounted = acceptance_mount(inputs, data).relative_to(inputs).as_posix()
    adjusted = dict(plan, spec={**plan["spec"], "data": dict(data, mount=mounted)})
    mount, manifest = _dataset_manifest(adjusted, inputs)
    acceptance_json(
        output / "setup.json", {"script_setup_seconds": time.monotonic() - ACCEPTANCE_STARTED}
    )

    def run(name, document=None, **kwargs):
        return acceptance_run(
            output / name, mount, manifest, document or plan["config"], device, **kwargs
        )

    def require(result, status, why, updates, epochs):
        if (result["status"], result["reason"], result["updates"], len(result["history"])) != (
            status,
            why,
            updates,
            epochs,
        ):
            raise ValueError(f"unexpected stop or progress: {result['status']}, {result['reason']}")

    def validate_resume(name, result, checkpoint_config, *, best_dev=False):
        path, pin = acceptance_checkpoint(output / name, result)
        frozen = load_manifest(manifest, checkpoint_config.manifest_sha256)
        train_rows, dev_rows = frozen.examples("train"), frozen.examples("dev")
        with training_runtime(device) as target:
            model, optimizer, scheduler = _training_parts(checkpoint_config, target)
            progress, best = read_resume(
                path,
                receipt_sha256=pin,
                run=run_provenance(checkpoint_config, frozen),
                config=checkpoint_config,
                train_rows=len(train_rows),
                dev_rows=len(dev_rows),
                model=model.backbone,
                optimizer=optimizer,
                scheduler=scheduler,
                device=target,
            )
            verified = {"receipt_sha256": pin, "cursor": progress.cursor()}
            if best_dev:
                if best is None:
                    raise ValueError("no validated best weights in cumulative-stop checkpoint")
                model.backbone.load_state_dict(best, strict=True)
                audio = TrainingAudio(frozen, mount)
                model.eval()
                with torch.no_grad():
                    loss = model(audio.waveform(dev_rows[0])).total.item()
                if not math.isfinite(loss):
                    raise ValueError("recovered best weights produced a nonfinite dev loss")
                verified["recovered_dev_example_loss"] = loss
        return verified

    if settings["session"] == "a":
        control, _ = run("control")
        require(control, "completed", "max_epochs", 6, 2)
        partial, _ = run("interrupted", stop=4)
        require(partial, "budget_limited", "attempt_updates", 4, 1)
        acceptance_json(
            output / "interrupted-validation.json", validate_resume("interrupted", partial, config)
        )
        cap = dict(plan["config"], budget={**plan["config"]["budget"], "max_updates": 4})
        capped, _ = run("capped", cap)
        require(capped, "budget_limited", "max_updates", 4, 1)
        if (output / "capped" / "inference").exists():
            raise ValueError("cumulative budget stop unexpectedly exported inference")
        cap_config = TrainingConfig.from_document(cap)
        verified = validate_resume("capped", capped, cap_config, best_dev=True)
        acceptance_json(
            output / "best-recovery.json",
            {
                "recoverable": True,
                "validated_epochs": 1,
                **verified,
                "convergence_claimed": False,
            },
        )
        wall = dict(plan["config"], budget={**plan["config"]["budget"], "wall_time_seconds": 5})
        timed, _ = run("wall", wall, delay=6)
        require(timed, "budget_limited", "wall_time", 1, 0)
        acceptance_json(
            output / "wall.json",
            {
                "target_seconds": 5,
                "injected_example_seconds": 6,
                "elapsed_seconds": timed["elapsed_seconds"],
                "overshoot_seconds": timed["elapsed_seconds"] - 5,
            },
        )
        return {"status": "passed", "session": "a", "runtime": runtime}

    spec = plan["spec"]["resume"]
    handoff = acceptance_mount(inputs, spec)
    transfer = _under(handoff, "handoff.json")
    if acceptance_sha(transfer) != settings["handoff_sha256"]:
        raise ValueError("Session A handoff digest mismatch")
    identity = json.loads(transfer.read_bytes())
    for name, digest in identity["sha256"].items():
        if acceptance_sha(_under(handoff, name)) != digest:
            raise ValueError(f"Session A output digest mismatch: {name}")
    recovery = output / "recovered"
    recovery.mkdir()
    for tag in ("control", "interrupted"):
        destination = recovery / tag
        destination.mkdir()
        for filename in ("resume.json", "training.pt"):
            shutil.copyfile(_under(handoff, f"{tag}-{filename}"), destination / filename)
    checkpoint = _under(recovery, spec["checkpoint"])
    resumed, later = run("resumed", resume=(checkpoint, spec["receipt_sha256"]))
    require(resumed, "completed", "max_epochs", 6, 2)
    if resumed["attempt_start_updates"] != 4 or len(later) != 2:
        raise ValueError("Session B must execute exactly two optimizer updates")
    left = acceptance_state(recovery / "control", identity["receipt_pins"]["control"])
    right = acceptance_state(*acceptance_checkpoint(output / "resumed", resumed))
    traces = json.loads((handoff / "updates.json").read_bytes())
    compared = acceptance_compare(left, right, traces["control"], traces["interrupted"] + later)
    acceptance_json(output / "comparison.json", compared)
    if not compared["within_tolerance"]:
        raise ValueError("GPU resume comparison exceeded the predeclared tolerance")
    return {
        "status": "passed",
        "session": "b",
        "agreement": "exact" if compared["exact"] else "tolerance",
    }
