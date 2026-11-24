"""Stage the two private N2K.1 acceptance jobs locally, with no Kaggle calls."""

import argparse
import hashlib
import json
import re
import shutil
import struct
import wave
from dataclasses import replace
from io import BytesIO
from pathlib import Path

from experiments.vocoroute_kaggle import DATASET_SCHEMA, SPEC_SCHEMA, stage_vocoroute_plan
from vocoroute.training_config import BUDGET_CONFIG_SCHEMA, TrainingConfig, canonical_json
from vocoroute.training_data import MANIFEST_SCHEMA, TrainingAudio, load_manifest

RUNTIME = ("python", "torch", "torchaudio", "cuda_build", "soundfile", "numpy", "librosa")


def write(path, value):
    path.write_bytes(canonical_json(value) + b"\n")


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dataset(root, ref):
    write(
        root / "dataset-metadata.json",
        {
            "title": ref.split("/")[1],
            "id": ref,
            "description": "Private synthetic acceptance artifacts under project control.",
            "licenses": [{"name": "other"}],
        },
    )
    return {"ref": ref, "version": 1, "mount": f"datasets/{ref}"}


def fixture(root, ref):
    root.mkdir()
    roles = ("train", "train", "train", "dev", "dev", "eval")
    rows = []
    for i, role in enumerate(roles):
        length = 1409 + 128 * i
        period = 37 + 6 * i
        values = [
            170 * (period - 2 * abs((n % (2 * period)) - period)) + (700 if n == 24 + i else 0)
            for n in range(length)
        ]
        stream = BytesIO()
        with wave.open(stream, "wb") as pcm:
            pcm.setparams((1, 2, 22050, length, "NONE", "not compressed"))
            pcm.writeframes(struct.pack(f"<{length}h", *values))
        payload = stream.getvalue()
        name = f"fixture-{i}.wav"
        digest = hashlib.sha256(payload).hexdigest()
        if role != "eval":
            (root / name).write_bytes(payload)
        rows.append(
            {
                "utterance_id": f"fixture-{i}",
                "speaker_id": f"synthetic-{role}-{i}",
                "source_group": f"integer-waveform-{i}",
                "role": role,
                "native_path": name,
                "native_sha256": digest,
                "native_rate": 22050,
                "native_samples": length,
                "prepared_path": name,
                "prepared_sha256": digest,
                "prepared_samples": length,
                "resample_steps": 0,
                "resampling_profile": "identity",
                "resampling_receipt_sha256": None,
            }
        )
    manifest = root / "manifest.json"
    write(
        manifest,
        {
            "schema": MANIFEST_SCHEMA,
            "dataset_id": "vocoroute-n2k1-fixture-v1",
            "source_kind": "native_generator",
            "target_rate": 22050,
            "rows": rows,
        },
    )
    manifest_pin = sha(manifest)
    parsed = load_manifest(manifest, manifest_pin)
    audio = TrainingAudio(parsed, root)
    audio.validate_headers()
    decoded = {
        row.utterance_id: audio.waveform(row).numel()
        for role in ("train", "dev")
        for row in parsed.examples(role)
    }
    assert decoded == {row["utterance_id"]: row["prepared_samples"] for row in rows[:-1]}
    write(
        root / "fixture-provenance.json",
        {
            "schema": "vocoroute-synthetic-fixture-v1",
            "generator_sha256": sha(Path(__file__)),
            "fixture_only": True,
            "signal": "integer triangle plus one impulse, no random draws",
            "format": "mono PCM16 WAV at 22050 Hz",
            "resample_steps": 0,
            "manifest_sha256": manifest_pin,
            "decoded_samples": decoded,
        },
    )
    spec = dataset(root, ref)
    receipt = root / "dataset-receipt.json"
    write(
        receipt,
        {
            "schema": DATASET_SCHEMA,
            "ref": ref,
            "version": spec["version"],
            "manifest_sha256": manifest_pin,
            "audio_roles": ["train", "dev"],
        },
    )
    return dict(spec, receipt_sha256=sha(receipt)), manifest_pin


def render(root, plan, metadata, settings):
    helper = Path(__file__).with_name("vocoroute_acceptance_kernel.py").read_text(encoding="utf-8")
    _, bootstrap = (root / "original-kernel.py").read_text(encoding="utf-8").split("\n", 1)
    old_call = (
        "    from experiments.vocoroute_kaggle import run_kernel_plan\n"
        "        receipt.update(run_kernel_plan(plan, source, inputs, output))"
    )
    if bootstrap.count(old_call) != 1:
        raise ValueError("the committed N2K bootstrap layout differs from the acceptance template")
    bootstrap = bootstrap.replace(
        old_call, "    receipt.update(run_acceptance(plan, source, inputs, output, ACCEPTANCE))"
    )
    mount_line = '(inputs / plan["spec"]["code"]["mount"]).resolve(strict=True)'
    if bootstrap.count(mount_line) != 1:
        raise ValueError("the committed N2K code mount layout differs from the acceptance template")
    bootstrap = bootstrap.replace(mount_line, 'acceptance_mount(inputs, plan["spec"]["code"])')
    if settings["local"]:
        inputs = root / "mock-inputs"
        output = root / f"local-output-{settings['session']}"
        bootstrap = bootstrap.replace('Path("/kaggle/input")', f"Path({inputs.as_posix()!r})")
        bootstrap = bootstrap.replace(
            'Path("/kaggle/working/vocoroute-run")', f"Path({output.as_posix()!r})"
        )
    entry = (
        f"{helper}\nPLAN_JSON = {json.dumps(plan, sort_keys=True)!r}\n"
        f"ACCEPTANCE = {settings!r}\n{bootstrap}"
    )
    compile(entry, "train.py", "exec")
    kernel = root / f"kernel-{settings['session']}"
    kernel.mkdir()
    (kernel / "train.py").write_text(entry, encoding="utf-8", newline="\n")
    write(kernel / "kernel-metadata.json", metadata)
    write(
        root / f"kernel-{settings['session']}-hashes.json",
        {path.name: sha(path) for path in kernel.iterdir()},
    )
    write(root / f"plan-{settings['session']}.json", plan)


def metadata(plan):
    owner = plan["spec"]["code"]["ref"].split("/")[0]
    from experiments.kaggle_harness import KernelSpec, kernel_metadata, slugify

    spec = plan["spec"]
    title = spec["title"]
    result = kernel_metadata(
        KernelSpec(slugify(title), title, (), 0, gpu=True), username=owner, code_file="train.py"
    )
    result.update(
        kernel_type="script",
        enable_internet="false",
        dataset_sources=[spec[name]["ref"] for name in ("code", "data", "resume") if spec[name]],
    )
    return result


def mock_mount(root, content, descriptor):
    dest = root / "mock-inputs" / descriptor["mount"]
    shutil.copytree(content, dest)


def prepare(root, owner, commit, local, runtime):
    root.mkdir(parents=True)
    training, manifest_pin = fixture(root / "fixture", f"{owner}/vocoroute-n2k1-fixture")
    code = {
        "ref": f"{owner}/vocoroute-n2k1-code",
        "version": 1,
        "mount": f"datasets/{owner}/vocoroute-n2k1-code",
    }
    spec = {
        "schema": SPEC_SCHEMA,
        "title": "vocoroute-n2k1-session-a",
        "code": code,
        "data": training,
        "resume": None,
        "runtime": runtime or dict.fromkeys(RUNTIME, "observe-in-session-a"),
    }
    config = TrainingConfig(
        schema=BUDGET_CONFIG_SCHEMA,
        device="cuda",
        epochs=2,
        max_updates=6,
        wall_time_seconds=120,
        checkpoint_every_updates=2,
        manifest_sha256=manifest_pin,
    )
    write(root / "spec-a.json", spec)
    write(root / "config-cuda.json", config.document())
    raw = root / "n2k-stage"
    plan = stage_vocoroute_plan(commit, root / "spec-a.json", root / "config-cuda.json", raw)
    (root / "original-kernel.py").write_bytes((raw / "kernel" / "train.py").read_bytes())
    dataset(raw / "code", code["ref"])
    if local:
        config = replace(config, device="cpu")
        plan.update(config=config.document(), config_sha256=config.sha256)
        mock_mount(root, raw / "code", code)
        mock_mount(root, root / "fixture", training)
    render(root, plan, metadata(plan), {"session": "a", "local": local})


def recover(root, saved, kernel_version):
    plan = read(root / "plan-a.json")
    if (saved / "vocoroute-run" / "kernel-receipt.json").is_file():
        saved = saved / "vocoroute-run"
    owner = plan["spec"]["code"]["ref"].split("/")[0]
    if not re.fullmatch(
        rf"{re.escape(owner)}/vocoroute-n2k1-session-a/[1-9][0-9]*", kernel_version
    ):
        raise ValueError("recover the explicit owner/vocoroute-n2k1-session-a/version output")
    receipt = read(saved / "kernel-receipt.json")
    if (receipt.get("status"), receipt.get("session"), receipt.get("plan")) != (
        "passed",
        "a",
        plan,
    ):
        raise ValueError("Session A must have passed against the exact staged plan")
    transfer = root / "handoff"
    transfer.mkdir()
    ref = f"{owner}/vocoroute-n2k1-handoff"
    resume = dataset(transfer, ref)
    identity = {
        "schema": "vocoroute-n2k1-handoff-v1",
        "source_kernel_version": kernel_version,
        "receipt_pins": {},
        "sha256": {},
    }
    updates = {}
    for name, expected in (("control", 6), ("interrupted", 4)):
        past = read(saved / name / "run.json")
        if (past["updates"], past["run"]["config_sha256"]) != (expected, plan["config_sha256"]):
            raise ValueError("saved run progress or config digest differs from the acceptance plan")
        last = past["latest_resume"]
        checkpoint = saved / name / last["directory"]
        if sha(checkpoint / "resume.json") != last["receipt_sha256"] or (
            sha(checkpoint / "training.pt") != read(checkpoint / "resume.json")["state_sha256"]
        ):
            raise ValueError("Session A checkpoint hash disagrees with its receipt")
        identity["receipt_pins"][name] = last["receipt_sha256"]
        for filename in ("resume.json", "training.pt"):
            shutil.copyfile(checkpoint / filename, transfer / f"{name}-{filename}")
        updates[name] = read(saved / f"{name}.profile.json")["updates"]
        if len(updates[name]) != expected:
            raise ValueError("Session A trace length disagrees with recorded progress")
    write(transfer / "updates.json", updates)
    for path in transfer.iterdir():
        if path.name != "dataset-metadata.json":
            identity["sha256"][path.name] = sha(path)
    identity["session_a_receipt_sha256"] = sha(saved / "kernel-receipt.json")
    write(transfer / "handoff.json", identity)
    plan["spec"].update(
        title="vocoroute-n2k1-session-b",
        runtime=receipt["runtime"],
        resume=dict(
            resume, checkpoint="interrupted", receipt_sha256=identity["receipt_pins"]["interrupted"]
        ),
    )
    local = plan["config"]["device"] == "cpu"
    if local:
        mock_mount(root, transfer, resume)
    render(
        root,
        plan,
        metadata(plan),
        {"session": "b", "local": local, "handoff_sha256": sha(transfer / "handoff.json")},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    action = actions.add_parser("prepare")
    action.add_argument("--root", type=Path, required=True)
    action.add_argument("--owner", required=True)
    action.add_argument("--commit", required=True)
    action.add_argument("--local", action="store_true", help="render CPU-only rehearsal fixtures")
    action.add_argument("--runtime", type=Path, help="optional exact observed GPU runtime JSON")
    action = actions.add_parser("recover")
    action.add_argument("--root", type=Path, required=True)
    action.add_argument("--saved", type=Path, required=True)
    action.add_argument("--kernel-version", required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(
            args.root.resolve(),
            args.owner,
            args.commit,
            args.local,
            read(args.runtime) if args.runtime else None,
        )
    else:
        recover(args.root.resolve(), args.saved.resolve(), args.kernel_version)


if __name__ == "__main__":
    main()
