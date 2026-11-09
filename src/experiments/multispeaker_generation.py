"""Explicit ingestion-backed pilot; never changes the legacy LJSpeech driver.

All artifacts are development-only/non-citable. Fixture mocks have their own
condition identity and cannot be used with real inputs or called neural results.
"""

from __future__ import annotations

import csv
import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
import pyloudnorm as pyln

from data.alignment import assert_aligned, check_vocoder_alignment
from data.audio_io import probe, read_native, read_processed, write_audio
from data.corpus import Utterance, assign_splits
from data.invariants import (
    ALIGNMENT_PROBE_FILES,
    ARCHIVE_SR,
    GLOBAL_SEED,
    LOUDNESS_TARGET_LUFS,
    LOUDNESS_TOLERANCE_LU,
    ZEROSHOT_SR,
    InvariantViolation,
)
from data.manifest import loudness_gain_report, read_manifest, validate_manifest, write_manifest
from data.preprocess import (
    derive_trim_span,
    derive_zeroshot,
    finalise_reference,
    measure_band_loudness,
    process_condition_output,
    resample_once,
    resample_with_spec,
    would_clip,
)
from data.source_ingestion import (
    DATASET,
    array_digest,
    digest,
    inspect_source,
    plain,
    plain_root,
    read_json,
    release_evidence,
    resample_source,
    resampling_record,
    safe_path,
    speaker_records,
    verify_ingestion,
    write_json,
)
from data.source_ingestion import (
    FILES as INTAKE_FILES,
)
from data.source_ingestion import (
    implementation as intake_implementation,
)
from experiments.phase_a_resynthesis import _environment, _rows_for, band_gate
from experiments.sanity_checks import run_all, silence_reference
from vocoders.registry import checkpoint_audit, get_spec, get_vocoder

FLAGS = {"development_only": True, "citable": False, "scientific_admission_complete": False}
CONDITIONS = ("fixture_mock", "griffin_lim", "bigvgan_112m", "vocos")
ACQUISITION_KEYS = {
    "dataset",
    "version",
    "microphone",
    "license",
    "local_processing_authorized",
    "reviewed_by",
    "reviewed_at",
    "source_inventory_sha256",
    "release_readme_sha256",
    "release_license_sha256",
    "source_url",
    "archive_sha256_reviewed",
    "ingestion_checksums_sha256",
    "source_manifest_sha256",
}
PLAN_FILES = {"plan.json", "checksums.json", *{f"ingestion/{n}" for n in INTAKE_FILES}}
OUTPUT_FILES = {
    "manifest.csv",
    "alignment.json",
    "lengths.json",
    "sanity.csv",
    "calibration.json",
    "loudness_gain.csv",
    "provenance.json",
    "validation.json",
    "checksums.json",
}
MOCK_SPEC = replace(
    get_spec("griffin_lim"),
    key="fixture_mock",
    display_name="SYNTHETIC FIXTURE IDENTITY MOCK",
    family="fixture_mock",
    year=2026,
    primary_ladder=False,
    notes="Identity on synthetic fixture input; no neural inference.",
)


@dataclass(frozen=True)
class NativeOutput:
    waveform: np.ndarray
    sample_rate: int


class FixtureIdentity:
    """Test generator, explicitly distinct from every research vocoder."""

    spec = MOCK_SPEC

    def load(self):
        pass

    def unload(self):
        pass

    def resynthesize(self, waveform, sample_rate):
        return NativeOutput(waveform.copy(), sample_rate)


def _spec(condition):
    if condition not in CONDITIONS:
        raise ValueError("Unsupported multi-speaker pilot condition.")
    return MOCK_SPEC if condition == "fixture_mock" else get_spec(condition)


def _implementation():
    import data.alignment
    import data.audio_io
    import data.corpus
    import data.invariants
    import data.manifest
    import data.measurements
    import data.mel
    import data.preprocess
    import data.source_ingestion
    import experiments.phase_a_resynthesis
    import experiments.sanity_checks
    import vocoders.base
    import vocoders.bigvgan
    import vocoders.checkpoints
    import vocoders.compat
    import vocoders.griffin_lim
    import vocoders.registry
    import vocoders.upstream
    import vocoders.vocos

    modules = (
        data.alignment,
        data.audio_io,
        data.corpus,
        data.invariants,
        data.manifest,
        data.measurements,
        data.mel,
        data.preprocess,
        data.source_ingestion,
        experiments.phase_a_resynthesis,
        experiments.sanity_checks,
        vocoders.base,
        vocoders.bigvgan,
        vocoders.checkpoints,
        vocoders.compat,
        vocoders.griffin_lim,
        vocoders.registry,
        vocoders.upstream,
        vocoders.vocos,
    )
    return {**{m.__name__: digest(m.__file__) for m in modules}, __name__: digest(__file__)}


def _files(root):
    root = plain_root(root)
    result = {}
    for p in sorted(root.rglob("*")):
        plain(p)
        if p.is_file():
            result[p.relative_to(root).as_posix()] = {
                "bytes": p.stat().st_size,
                "sha256": digest(p),
            }
    return result


def _seal(root):
    write_json(
        root / "checksums.json", {k: v for k, v in _files(root).items() if k != "checksums.json"}
    )


def _check_inventory(root, expected):
    found = _files(root)
    allowed_directories = {
        p.as_posix() for name in expected for p in Path(name).parents if p.as_posix() != "."
    }
    for p in root.rglob("*"):
        if p.is_dir() and p.relative_to(root).as_posix() not in allowed_directories:
            raise ValueError("Unexpected artifact directory.")
    if set(found) != set(expected):
        raise ValueError("Unexpected/missing pilot artifact; inventory cannot authorize new files.")
    checks = read_json(root / "checksums.json")
    if checks != {k: v for k, v in found.items() if k != "checksums.json"}:
        raise ValueError("Pilot artifact size/hash mismatch.")


def _fresh_out(out_dir, inputs):
    out = plain_root(out_dir)
    if out.exists():
        raise FileExistsError("Fresh output required; no existing artifacts are repaired/deleted.")
    for directory in inputs:
        p = plain_root(directory)
        if out.is_relative_to(p) or p.is_relative_to(out):
            raise ValueError("Output must be disjoint from source, evidence and input artifacts.")
    return out


def _acquisition(intake, source_kind, receipt):
    """A reviewed attestation bound to supplied byte identities, not a rights certificate."""
    if source_kind not in ("fixture", "local_dataset"):
        raise ValueError("Declare fixture or local_dataset explicitly.")
    claimed = read_json(intake / "plan.json")
    if claimed["source_kind"] != source_kind:
        raise ValueError("Source-kind mismatch; fixtures cannot be promoted to real speech.")
    if source_kind == "fixture":
        if receipt is not None:
            raise ValueError("Fixture intake cannot claim a real acquisition review.")
        return None
    if receipt is None:
        raise ValueError("Real source access requires an explicit reviewed acquisition receipt.")
    if set(receipt) != ACQUISITION_KEYS:
        raise ValueError("Acquisition receipt has missing or unexpected fields.")
    try:
        reviewed_at = datetime.fromisoformat(receipt["reviewed_at"])
    except (TypeError, ValueError) as exc:
        raise ValueError("Acquisition review must have an ISO datetime with timezone.") from exc
    if (
        receipt.get("dataset") != DATASET
        or receipt.get("version") != "0.92"
        or receipt.get("microphone") != "mic2"
        or receipt.get("license") != "CC-BY-4.0"
        or receipt.get("local_processing_authorized") is not True
        or not isinstance(receipt.get("reviewed_by"), str)
        or not receipt["reviewed_by"].strip()
        or not isinstance(receipt.get("reviewed_at"), str)
        or not receipt["reviewed_at"].strip()
        or reviewed_at.tzinfo is None
        or receipt["source_url"] != "https://datashare.ed.ac.uk/handle/10283/3443"
        or receipt["archive_sha256_reviewed"]
        != "f96258be9fdc2cbff6559541aae7ea4f59df3fcaf5cf963aae5ca647357e359c"
        or receipt.get("source_inventory_sha256") != digest(intake / "source_inventory.json")
        or receipt.get("release_readme_sha256") != digest(intake / "release_readme.txt")
        or receipt.get("release_license_sha256") != digest(intake / "release_license.txt")
        or receipt["ingestion_checksums_sha256"] != digest(intake / "checksums.json")
        or receipt["source_manifest_sha256"] != digest(intake / "source_manifest.csv")
    ):
        raise ValueError("Acquisition attestation is missing review or does not bind this intake.")
    return receipt


def _accepted_real_intake(intake, source_root, evidence_dir):
    """Check the externally bound completed intake without reading held-out audio.

    _acquisition has already checked the human-reviewed intake checksum digest.
    This is deliberately scoped revalidation, not a new full-intake pass.
    """
    _check_inventory(intake, INTAKE_FILES)
    plan = read_json(intake / "plan.json")
    provenance = read_json(intake / "provenance.json")
    summary = read_json(intake / "validation.json")
    if (
        plan["dataset"] != DATASET
        or plan["schema_version"] != 1
        or plan["source_kind"] != "local_dataset"
        or plan["microphone"] != "mic2"
        or plan["native_sr"] != 48000
        or summary["status"] != "passed"
        or summary["source_kind"] != "local_dataset"
        or summary["sources"] != 32
        or summary["speakers"] != 8
        or summary["source_files_unchanged"] is not True
        or summary["audio_outputs_generated"] != 0
        or summary["models_trained"] != 0
        or provenance["implementation_sha256"] != intake_implementation()
        or provenance["source_authenticity_verified"] is not False
        or provenance["redistribution_cleared"] is not False
        or read_json(intake / "resampling.json") != resampling_record()
        or any(d.get(k) != v for d in (plan, provenance, summary) for k, v in FLAGS.items())
    ):
        raise ValueError(
            "Bound completed-ingestion receipt is invalid or its implementation changed."
        )
    root = plain_root(source_root if source_root is not None else provenance["source_root"])
    evidence = plain_root(
        evidence_dir
        if evidence_dir is not None
        else (
            root
            if provenance["source_root"] == provenance["evidence_root"]
            else provenance["evidence_root"]
        )
    )
    texts = release_evidence(root, evidence, "local_dataset")  # Also rejects fixture markers.
    for name, copied in (
        ("README.txt", "release_readme.txt"),
        ("license_text.txt", "release_license.txt"),
    ):
        p = safe_path(evidence, name)
        if (
            provenance["evidence_files"][name] != {"bytes": p.stat().st_size, "sha256": digest(p)}
            or (intake / copied).read_text(encoding="utf-8") != texts[name]
        ):
            raise ValueError("Reviewed release evidence changed.")
    source_inventory = read_json(intake / "source_inventory.json")
    if (
        digest(safe_path(root, "speaker-info.txt"))
        != source_inventory["speaker-info.txt"]["sha256"]
    ):
        raise ValueError("Original speaker identity metadata changed.")
    return summary


def _verify_selected_sources(intake, root, chosen):
    """Remeasure ONLY chosen training sources; do not read any held-out waveform."""
    inventory = read_json(intake / "source_inventory.json")
    speakers = speaker_records(root)
    for row in chosen:
        if row["split"] != "train":
            raise ValueError("A held-out source reached training-only verification.")
        for field, hash_field in (
            ("source_path", "source_sha256"),
            ("transcript_path", "transcript_sha256"),
        ):
            if digest(safe_path(root, row[field])) != row[hash_field]:
                raise ValueError("Selected original source or transcript byte identity changed.")
        current = inspect_source(root, row["native_utterance_id"], speakers, inventory)
        if any(row[k] != str(value) for k, value in current.items()):
            raise ValueError("Selected source identity, native format or measurement changed.")


def _selection(intake, source_kind, condition, receipt, source_root=None, evidence_dir=None):
    # Check the explicit access gate BEFORE deep intake verification/decoding.
    _acquisition(intake, source_kind, receipt)
    spec = _spec(condition)
    if source_kind == "fixture" and condition in ("bigvgan_112m", "vocos"):
        raise ValueError("Fixture neural execution is disabled; use fixture_mock explicitly.")
    if source_kind == "local_dataset" and condition == "fixture_mock":
        raise ValueError("Fixture mocks cannot consume real-dataset intake.")
    verification = (
        verify_ingestion(intake, source_root=source_root, evidence_dir=evidence_dir)
        if source_kind == "fixture"
        else _accepted_real_intake(intake, source_root, evidence_dir)
    )
    original = read_json(intake / "plan.json")
    if original["policy"] != {"speakers": 8, "utterances_per_speaker": 4, "seed": GLOBAL_SEED}:
        raise ValueError(
            "The bounded pilot requires the predetermined eight-speaker/four-source intake."
        )
    with (intake / "source_manifest.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 32 or verification["speaker_counts_by_split"] != {
        "train": 6,
        "dev": 1,
        "eval": 1,
    }:
        raise ValueError("Unexpected frozen pilot population or speaker topology.")
    groups = {}
    for r in rows:
        groups.setdefault(r["speaker_id"], set()).add(r["split"])
    if any(len(v) != 1 for v in groups.values()):
        raise ValueError("Speaker crosses frozen splits.")
    splits = assign_splits(
        [Utterance(r["source_id"], Path(r["source_path"]), r["speaker_id"]) for r in rows],
        speaker_disjoint=True,
        seed=GLOBAL_SEED,
    )
    if (
        len(splits) != 32
        or any(splits[r["source_id"]] != r["split"] for r in rows)
        or any(
            len({r[key] for r in rows}) != 32
            for key in (
                "source_id",
                "source_sha256",
                "decoded_audio_sha256",
                "normalized_text_sha256",
            )
        )
    ):
        raise ValueError(
            "Frozen split metadata, unique source identities or duplicate receipts differ."
        )
    training = {}
    for row in rows:
        if row["split"] == "train":
            training.setdefault(row["speaker_id"], []).append(row)
    chosen = []
    for index in range(4):
        for speaker in sorted(training):
            chosen.append(sorted(training[speaker], key=lambda r: r["source_id"])[index])
    chosen = chosen[:ALIGNMENT_PROBE_FILES]
    if len(chosen) != 20 or any(r["split"] != "train" for r in chosen):
        raise ValueError("Insufficient training-only alignment cohort; do not resplit/backfill.")
    if source_kind == "local_dataset":
        prov = read_json(intake / "provenance.json")
        root = plain_root(source_root if source_root is not None else prov["source_root"])
        _verify_selected_sources(intake, root, chosen)
    return {
        "schema_version": 1,
        "dataset": "synthetic_fixture_vctk_layout" if source_kind == "fixture" else DATASET,
        "dataset_version": "0.92",
        "source_kind": source_kind,
        "microphone": "mic2",
        "reference_semantics": "synthetic_fixture_input_not_genuine"
        if source_kind == "fixture"
        else "acquired_source_reference_with_manual_review",
        "condition": condition,
        "generator_spec": asdict(spec),
        "selection_policy": "round_robin_sorted_train_speakers_then_sorted_source_ids_first20",
        "sources": chosen,
        "frozen_splits": {r["source_id"]: r["split"] for r in rows},
        "source_manifest_sha256": digest(intake / "source_manifest.csv"),
        "ingestion_inventory_sha256": digest(intake / "checksums.json"),
        "ingestion_verification_scope": "full_fixture_revalidation"
        if source_kind == "fixture"
        else "bound_completed_intake_receipt_and_selected_training_source_revalidation",
        "resampling": resampling_record(),
        "implementation_sha256": _implementation(),
        "acquisition_attestation": receipt,
        **FLAGS,
    }


def create_plan(
    ingestion_dir,
    out_dir,
    *,
    source_kind,
    condition,
    acquisition_receipt=None,
    source_root=None,
    evidence_dir=None,
):
    intake = plain_root(ingestion_dir)
    receipt = read_json(acquisition_receipt) if acquisition_receipt is not None else None
    plan = _selection(intake, source_kind, condition, receipt, source_root, evidence_dir)
    prov = read_json(intake / "provenance.json")
    out = _fresh_out(
        out_dir, [intake, source_root or prov["source_root"], evidence_dir or prov["evidence_root"]]
    )
    out.mkdir(parents=True)
    shutil.copytree(intake, out / "ingestion")
    write_json(out / "plan.json", plan)
    _seal(out)
    load_plan(out, source_root=source_root, evidence_dir=evidence_dir)
    return out / "plan.json"


def load_plan(plan_dir, *, source_root=None, evidence_dir=None):
    directory = plain_root(plan_dir)
    _check_inventory(directory, PLAN_FILES)
    plan = read_json(directory / "plan.json")
    expected = _selection(
        directory / "ingestion",
        plan["source_kind"],
        plan["condition"],
        plan["acquisition_attestation"],
        source_root,
        evidence_dir,
    )
    if plan != expected:
        raise ValueError(
            "Frozen generation plan, source identity, split or implementation changed."
        )
    prov = read_json(directory / "ingestion/provenance.json")
    root = plain_root(source_root if source_root is not None else prov["source_root"])
    return plan, root


def _generator_identity(generator, condition):
    cls = type(generator)
    if condition == "fixture_mock":
        import inspect

        if not isinstance(generator, FixtureIdentity) or asdict(generator.spec) != asdict(
            MOCK_SPEC
        ):
            raise ValueError("Only explicitly labeled fixture-mock generators may be injected.")
        return {
            "kind": "fixture_mock",
            "class": f"{cls.__module__}.{cls.__qualname__}",
            "implementation_sha256": digest(inspect.getfile(cls)),
            "neural_inference": False,
        }
    return {
        "kind": "griffin_lim" if condition == "griffin_lim" else "neural_vocoder",
        "class": f"{cls.__module__}.{cls.__qualname__}",
        "spec": asdict(generator.spec),
        "neural_inference": condition != "griffin_lim",
    }


def _assert_generator_label(plan, identity):
    condition = plan["condition"]
    expected_kind = (
        "fixture_mock"
        if condition == "fixture_mock"
        else "griffin_lim"
        if condition == "griffin_lim"
        else "neural_vocoder"
    )
    neural = condition in ("bigvgan_112m", "vocos")
    if identity["kind"] != expected_kind or identity["neural_inference"] is not neural:
        raise ValueError(
            "Generator label cannot promote fixture/Griffin-Lim results to neural evidence."
        )


def _audio_rel(source_kind, tier, condition, uid):
    prefix = "fixture__" if source_kind == "fixture" else ""
    return f"{tier}/{condition}/{prefix}{uid}.wav"


def _cached_models(condition):
    """No downloads, even when a neural path is used in a future authorized run."""
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise ValueError(
            "Neural execution requires HF_HUB_OFFLINE=1 and pre-existing pinned caches."
        )
    from huggingface_hub import constants, try_to_load_from_cache

    if not constants.HF_HUB_OFFLINE:
        raise ValueError("Hub was imported before offline mode; start a fresh offline process.")
    if condition == "bigvgan_112m":
        from vocoders.bigvgan import UPSTREAM_CODE_FILES, UPSTREAM_CONFIG, UPSTREAM_WEIGHTS

        names = [*UPSTREAM_CODE_FILES, UPSTREAM_CONFIG, UPSTREAM_WEIGHTS]
    else:
        names = ["config.yaml", "pytorch_model.bin"]
    spec, found = _spec(condition), {}
    for name in names:
        path = try_to_load_from_cache(spec.checkpoint, name, revision=spec.checkpoint_revision)
        if not isinstance(path, str) or not Path(path).is_file():
            raise ValueError(f"Missing pinned local checkpoint asset: {condition}/{name}")
        found[name] = {"bytes": Path(path).stat().st_size, "sha256": digest(path)}
    if condition == "vocos":
        import yaml

        config_path = try_to_load_from_cache(
            spec.checkpoint, "config.yaml", revision=spec.checkpoint_revision
        )
        config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        classes = {
            "feature_extractor": "vocos.feature_extractors.MelSpectrogramFeatures",
            "backbone": "vocos.models.VocosBackbone",
            "head": "vocos.heads.ISTFTHead",
        }
        if any(config[k]["class_path"] != v for k, v in classes.items()):
            raise ValueError("Unexpected Vocos component; never load/download an Encodec model.")
    return found


def _prepare_generator(plan, generator, device):
    condition = plan["condition"]
    if generator is not None and (plan["source_kind"] != "fixture" or condition != "fixture_mock"):
        raise ValueError("Injected generators are fixture_mock-only.")
    generator = generator or (
        FixtureIdentity() if condition == "fixture_mock" else get_vocoder(condition)
    )
    if asdict(generator.spec) != plan["generator_spec"]:
        raise ValueError("Loaded generator specification differs from the frozen plan.")
    identity = _generator_identity(generator, condition)
    _assert_generator_label(plan, identity)
    assets = {}
    if condition in ("bigvgan_112m", "vocos"):
        audit = next(r for r in checkpoint_audit() if r["condition"] == condition)
        if audit["blocked"]:
            raise ValueError("Checkpoint specification audit blocks this generator.")
        from importlib.util import find_spec

        missing = [m for m in generator.runtime_modules if find_spec(m) is None]
        if missing:
            raise ValueError(f"Missing isolated neural runtime: {missing}")
        assets = _cached_models(condition)
        if (
            condition == "vocos"
            and metadata.version("torch").split("+")[0]
            != metadata.version("torchaudio").split("+")[0]
        ):
            raise ValueError(
                "PyTorch/torchaudio versions must match; check the binary ABI separately."
            )
    generator.device = device
    return generator, identity, assets


def _input(source, spec):
    return (
        resample_source(source)
        if spec.resampler is None
        else resample_with_spec(source, 48_000, spec.mel.sample_rate, spec.resampler)
    )


def _native_output(value, rate):
    result = value if isinstance(value, NativeOutput) else NativeOutput(value, rate)
    wav = np.asarray(result.waveform)
    if result.sample_rate != rate or wav.ndim != 1 or not len(wav) or wav.dtype.kind != "f":
        raise ValueError("Generator output violates native sample-rate/mono/float contract.")
    if not np.isfinite(wav).all():
        raise ValueError("Generator output contains non-finite samples.")
    return wav.astype(np.float32)


def _emit(wav, loudness, out, condition, uid, source_kind):
    archive_path = out / _audio_rel(source_kind, "archive", condition, uid)
    write_audio(archive_path, wav, ARCHIVE_SR, measured_lufs=loudness.lufs_out_band)
    # New path derives from the finished PCM16 archive, never from native/source.
    archived = read_processed(archive_path, ARCHIVE_SR)
    derived, lufs = derive_zeroshot(archived, archive_lufs=loudness.lufs_out)
    if would_clip(derived):
        raise InvariantViolation("Derived peak failure; reject the entire paired cohort.")
    derived_path = out / _audio_rel(source_kind, "zeroshot", condition, uid)
    write_audio(derived_path, derived, ZEROSHOT_SR)
    return archive_path, derived_path, lufs, len(derived)


def _gate_outputs(out, expected_rows):
    stored = pd.read_csv(out / "manifest.csv")
    for r in stored.itertuples():
        if r.utt_id not in expected_rows or r.tier not in ("archive", "zeroshot"):
            raise ValueError("Unexpected output source or tier.")
        expected = _audio_rel(r.source_kind, r.tier, r.condition, r.utt_id)
        if r.path != expected:
            raise ValueError("Manifest must reference the exact relative packaged output path.")
        safe_path(out, r.path)
    frame = read_manifest(out / "manifest.csv")
    validate_manifest(frame, require_zeroshot=True)
    if set(frame["utt_id"]) != set(expected_rows) or set(frame["split"]) != {"train"}:
        raise ValueError("Output does not preserve the exact training source set.")
    for row in frame.itertuples():
        source = expected_rows[row.utt_id]
        if (
            row.speaker != source["speaker_id"]
            or row.split != source["split"]
            or row.source_sha256 != source["source_sha256"]
            or row.dataset_version != 0.92
        ):
            raise ValueError("Generated speaker/source grouping differs from intake.")
        with Path(row.path).open("rb") as stream:
            header = stream.read(12)
        if header[:4] != b"RIFF" or header[8:12] != b"WAVE":
            raise ValueError("Output must be an actual RIFF/WAVE container, not a renamed codec.")
        h = probe(row.path)
        samples = int(source["archive_trim_end"]) - int(source["archive_trim_start"])
        if row.tier == "zeroshot":
            samples = (samples * ZEROSHOT_SR + ARCHIVE_SR - 1) // ARCHIVE_SR
        if (
            h["sample_rate"] != row.sr_out
            or h["channels"] != 1
            or h["subtype"] != "PCM_16"
            or h["frames"] != samples
            or abs(row.duration_s - samples / row.sr_out) > 1e-12
        ):
            raise ValueError("Output audio format, samples or recorded duration mismatch.")
        decoded = read_processed(row.path, row.sr_out)
        actual_lufs = float(pyln.Meter(row.sr_out).integrated_loudness(decoded))
        if abs(actual_lufs - row.lufs_out) > LOUDNESS_TOLERANCE_LU:
            raise ValueError("Decoded loudness differs from the recorded output.")
        if (
            row.tier == "archive"
            and abs(measure_band_loudness(decoded) - LOUDNESS_TARGET_LUFS) > LOUDNESS_TOLERANCE_LU
        ):
            raise InvariantViolation(
                "INV-04: decoded archive misses the unchanged analysis-band target."
            )
    band = band_gate(frame)
    sanity = run_all(frame, split=None)
    return frame, band, sanity


def generate(
    plan_dir,
    out_dir,
    *,
    stage,
    compatibility_dir=None,
    generator=None,
    device="cpu",
    source_root=None,
    evidence_dir=None,
):
    if stage not in ("compatibility", "generation"):
        raise ValueError("Choose compatibility or generation explicitly.")
    plan_dir = plain_root(plan_dir)
    plan, root = load_plan(plan_dir, source_root=source_root, evidence_dir=evidence_dir)
    prov = read_json(plan_dir / "ingestion/provenance.json")
    inputs = [plan_dir, root, evidence_dir or prov["evidence_root"]]
    prerequisite = None
    if stage == "generation":
        if compatibility_dir is None:
            raise ValueError(
                "Limited generation requires a separately validated compatibility artifact."
            )
        prerequisite = validate(
            compatibility_dir, source_root=source_root, evidence_dir=evidence_dir
        )
        compatible = read_json(Path(compatibility_dir) / "provenance.json")
        if prerequisite["stage"] != "compatibility" or compatible["plan_sha256"] != digest(
            plan_dir / "plan.json"
        ):
            raise ValueError("Compatibility evidence does not belong to this immutable plan.")
        inputs.append(compatibility_dir)
    elif compatibility_dir is not None:
        raise ValueError("Compatibility stage does not consume another compatibility run.")
    out = _fresh_out(out_dir, inputs)
    vocoder, identity, assets = _prepare_generator(plan, generator, device)
    identity.update(
        device_requested=device,
        seed=GLOBAL_SEED,
        n_iter=getattr(vocoder, "n_iter", None),
        runtime_versions={n: metadata.version(n) for n in ("librosa", "numpy", "scipy")},
    )
    if prerequisite and compatible["generator"] != identity:
        raise ValueError("Compatibility generator identity differs from generation.")
    if prerequisite and compatible["cached_checkpoint_assets"] != assets:
        raise ValueError("Compatibility checkpoint asset identities differ from generation.")
    selected = {r["source_id"]: r for r in plan["sources"]}
    input_hashes = {r["source_path"]: r["source_sha256"] for r in selected.values()}
    out.mkdir(parents=True)
    shutil.copytree(plan_dir, out / "plan")
    write_json(out / "validation.json", {"status": "running", "stage": stage, **FLAGS})
    records, real, spans, lengths, manifest = {}, {}, {}, [], []
    loaded = False
    try:
        loaded = True
        vocoder.load()
        environment = _environment()
        if prerequisite and compatible["environment"] != environment:
            raise ValueError(
                "Compatibility environment differs; rerun compatibility independently."
            )
        for row in plan["sources"]:
            uid = row["source_id"]
            path = safe_path(root, row["source_path"])
            if digest(path) != row["source_sha256"]:
                raise ValueError("Source changed after plan verification.")
            native, rate = read_native(path)
            if rate != 48_000 or array_digest(native) != row["decoded_audio_sha256"]:
                raise ValueError("Source rate or decoded identity changed.")
            reference = resample_source(native)
            if array_digest(reference) != row["archive_preview_sha256"]:
                raise ValueError("Source archive preview differs from intake.")
            span = derive_trim_span(reference, uid)
            if (span.start, span.end) != (
                int(row["archive_trim_start"]),
                int(row["archive_trim_end"]),
            ):
                raise ValueError("Reference trim span changed.")
            ref, measured, high_band, loudness = finalise_reference(reference, span)
            if would_clip(ref):
                raise InvariantViolation(
                    "Real peak failure; no replacements or per-condition attenuation."
                )
            real[uid], spans[uid] = ref, span
            paths = _emit(ref, loudness, out, "real", uid, plan["source_kind"])
            manifest.extend(
                _rows_for(
                    uid,
                    "real",
                    row["speaker_id"],
                    "train",
                    span,
                    paths[0],
                    paths[1],
                    len(ref),
                    paths[3],
                    measured,
                    paths[2],
                    rate,
                    high_band=high_band,
                    loudness=loudness,
                )
            )
            spec = vocoder.spec
            generator_input = _input(native, spec)
            generated = _native_output(
                vocoder.resynthesize(generator_input, spec.mel.sample_rate), spec.mel.sample_rate
            )
            prealigned = (
                resample_with_spec(generated, spec.mel.sample_rate, ARCHIVE_SR, spec.resampler)
                if spec.resampler is not None
                else resample_once(generated, spec.mel.sample_rate)
            )
            # The shared entry point's archive->archive resample is a no-op.
            wav, measured, high_band, loudness = process_condition_output(
                prealigned, ARCHIVE_SR, span, condition=plan["condition"]
            )
            if would_clip(wav):
                raise InvariantViolation(
                    "Synthetic peak failure; reject all paired conditions, never attenuate."
                )
            records[uid] = (wav, measured, high_band, loudness)
            lengths.append(
                {
                    "source_id": uid,
                    "native_source_samples": len(native),
                    "generator_input_samples": len(generator_input),
                    "generator_input_sr": spec.mel.sample_rate,
                    "generator_output_samples": len(generated),
                    "generator_output_sr": spec.mel.sample_rate,
                    "generator_output_float32_sha256": array_digest(generated),
                    "prealignment_archive_samples": len(prealigned),
                    "reference_archive_samples": len(reference),
                    "length_adjustment_samples": len(reference) - len(prealigned),
                    "trim_start": span.start,
                    "trim_end": span.end,
                    "final_archive_samples": len(wav),
                }
            )
        report = check_vocoder_alignment(
            [(real[u], records[u][0]) for u in selected], plan["condition"]
        )
        write_json(
            out / "alignment.json",
            {"pre_encoding": report.to_dict(), "source_kind": plan["source_kind"], **FLAGS},
        )
        write_json(
            out / "lengths.json", {"records": lengths, "source_kind": plan["source_kind"], **FLAGS}
        )
        assert_aligned(report)
        for uid, (wav, measured, high_band, loudness) in records.items():
            row = selected[uid]
            paths = _emit(wav, loudness, out, plan["condition"], uid, plan["source_kind"])
            manifest.extend(
                _rows_for(
                    uid,
                    plan["condition"],
                    row["speaker_id"],
                    "train",
                    spans[uid],
                    paths[0],
                    paths[1],
                    len(wav),
                    paths[3],
                    measured,
                    paths[2],
                    vocoder.spec.mel.sample_rate,
                    mel_fields=vocoder.spec.mel.as_manifest_fields(),
                    checkpoint=vocoder.spec.checkpoint,
                    commit=vocoder.spec.checkpoint_revision,
                    alignment_lag=report.peak_lag,
                    high_band=high_band,
                    loudness=loudness,
                )
            )
        write_manifest(manifest, out / "manifest.csv")
        raw = pd.read_csv(out / "manifest.csv")
        raw["source_kind"] = plan["source_kind"]
        raw["dataset"] = plan["dataset"]
        raw["reference_semantics"] = plan["reference_semantics"]
        raw["generator_kind"] = identity["kind"]
        raw["source_sha256"] = raw["utt_id"].map(
            {u: r["source_sha256"] for u, r in selected.items()}
        )
        raw["dataset_version"] = plan["dataset_version"]
        for key, value in FLAGS.items():
            raw[key] = value
        raw.to_csv(out / "manifest.csv", index=False)
        frame, band, sanity = _gate_outputs(out, selected)
        sanity.to_csv(out / "sanity.csv", index=False)
        if sanity["leaked"].any():
            raise InvariantViolation("INV-12: confound leak; never relax the floor.")
        decoded_report = check_vocoder_alignment(
            [
                (
                    read_processed(
                        out / _audio_rel(plan["source_kind"], "archive", "real", uid), ARCHIVE_SR
                    ),
                    read_processed(
                        out / _audio_rel(plan["source_kind"], "archive", plan["condition"], uid),
                        ARCHIVE_SR,
                    ),
                )
                for uid in selected
            ],
            plan["condition"],
        )
        assert_aligned(decoded_report)
        write_json(
            out / "alignment.json",
            {
                "pre_encoding": report.to_dict(),
                "decoded": decoded_report.to_dict(),
                "source_kind": plan["source_kind"],
                **FLAGS,
            },
        )
        write_json(
            out / "calibration.json",
            {
                "records": {
                    t: silence_reference(frame, t, None).as_record()
                    for t in ("archive", "zeroshot")
                },
                "source_kind": plan["source_kind"],
                **FLAGS,
            },
        )
        loudness_gain_report(frame).to_csv(out / "loudness_gain.csv", index=False)
        for rel, sha in input_hashes.items():
            if digest(safe_path(root, rel)) != sha:
                raise ValueError("Original source changed during generation.")
        repo = Path(__file__).resolve().parents[2]
        write_json(
            out / "provenance.json",
            {
                "plan_sha256": digest(plan_dir / "plan.json"),
                "generator": identity,
                "cached_checkpoint_assets": assets,
                "checkpoint_spec": asdict(vocoder.spec),
                "environment": environment,
                "python": platform.python_version(),
                "git_head": subprocess.check_output(
                    ["git", "--no-optional-locks", "rev-parse", "HEAD"], cwd=repo, text=True
                ).strip(),
                "compatibility": None
                if prerequisite is None
                else {
                    "directory": str(plain_root(compatibility_dir)),
                    "checksums_sha256": digest(Path(compatibility_dir) / "checksums.json"),
                },
                "file_identities": {
                    p.relative_to(out).as_posix(): {
                        "sha256": digest(p),
                        "decoded_float32_sha256": array_digest(
                            read_processed(p, probe(p)["sample_rate"])
                        ),
                    }
                    for p in out.rglob("*.wav")
                },
                **FLAGS,
            },
        )
        write_json(
            out / "validation.json",
            {
                "status": "passed",
                "stage": stage,
                "source_kind": plan["source_kind"],
                "condition": plan["condition"],
                "sources": len(selected),
                "speakers": len({r["speaker_id"] for r in selected.values()}),
                "outputs": len(frame),
                "source_split": "train",
                "alignment_files": report.n_files,
                "band": band,
                "sanity_probes": len(sanity),
                "selected_source_files_unchanged": True,
                "real_vctk_sources": 0 if plan["source_kind"] == "fixture" else len(selected),
                "generator_kind": identity["kind"],
                **FLAGS,
            },
        )
        _seal(out)
        return validate(
            out,
            source_root=source_root,
            evidence_dir=evidence_dir,
            compatibility_dir=compatibility_dir,
        )
    except Exception as exc:
        write_json(
            out / "validation.json",
            {
                "status": "failed",
                "stage": stage,
                "error": str(exc),
                "source_kind": plan["source_kind"],
                **FLAGS,
            },
        )
        raise
    finally:
        if loaded:
            vocoder.unload()


def validate(directory, *, source_root=None, evidence_dir=None, compatibility_dir=None):
    out = plain_root(directory)
    # The schema, not an editable inventory, authorizes every file recursively.
    preliminary = read_json(out / "plan/plan.json")
    expected = set(OUTPUT_FILES) | {f"plan/{n}" for n in PLAN_FILES}
    for row in preliminary["sources"]:
        for tier in ("archive", "zeroshot"):
            for condition in ("real", preliminary["condition"]):
                expected.add(
                    _audio_rel(preliminary["source_kind"], tier, condition, row["source_id"])
                )
    _check_inventory(out, expected)
    provenance = read_json(out / "provenance.json")
    if set(provenance["file_identities"]) != {
        p.relative_to(out).as_posix() for p in out.rglob("*.wav")
    }:
        raise ValueError("Output waveform identity inventory differs.")
    for rel, saved in provenance["file_identities"].items():
        path = safe_path(out, rel)
        if digest(path) != saved["sha256"]:
            raise ValueError("Output waveform byte hash mismatch.")
    plan, root = load_plan(out / "plan", source_root=source_root, evidence_dir=evidence_dir)
    selected = {r["source_id"]: r for r in plan["sources"]}
    frame, band, sanity = _gate_outputs(out, selected)
    if sanity["leaked"].any():
        raise InvariantViolation("INV-12: confound leak; artifact is not admitted.")
    raw = pd.read_csv(out / "manifest.csv")
    for name, expected_value in {
        **FLAGS,
        "source_kind": plan["source_kind"],
        "dataset": plan["dataset"],
        "reference_semantics": plan["reference_semantics"],
    }.items():
        if name not in raw or not (raw[name] == expected_value).all():
            raise ValueError("Output source-kind or scientific status changed.")
    if len(frame) != 80 or set(frame["condition"]) != {"real", plan["condition"]}:
        raise ValueError("Unexpected pairing, duplicate rows or generated population.")
    if frame.duplicated(["utt_id", "condition", "tier"]).any():
        raise ValueError("Duplicate source/condition/tier row.")
    summary = read_json(out / "validation.json")
    _assert_generator_label(plan, provenance["generator"])
    if (
        provenance["plan_sha256"] != digest(out / "plan/plan.json")
        or provenance["checkpoint_spec"] != plan["generator_spec"]
        or any(provenance.get(k) != v or summary.get(k) != v for k, v in FLAGS.items())
    ):
        raise ValueError("Output provenance differs from its immutable plan.")
    if not (raw["generator_kind"] == provenance["generator"]["kind"]).all():
        raise ValueError("Generator-kind mismatch.")
    if plan["condition"] == "fixture_mock" and (
        provenance["generator"]["kind"] != "fixture_mock"
        or provenance["generator"]["neural_inference"]
    ):
        raise ValueError("A fixture mock cannot become neural-vocoder evidence.")
    if set(provenance["file_identities"]) != {
        p.relative_to(out).as_posix() for p in out.rglob("*.wav")
    }:
        raise ValueError("Output waveform identity inventory differs.")
    for rel, saved in provenance["file_identities"].items():
        path = safe_path(out, rel)
        wav = read_processed(path, probe(path)["sample_rate"])
        if saved != {"sha256": digest(path), "decoded_float32_sha256": array_digest(wav)}:
            raise ValueError("Output waveform byte/decoded hash mismatch.")
    pairs = []
    lengths_document = read_json(out / "lengths.json")
    lengths = lengths_document["records"]
    if lengths_document["source_kind"] != plan["source_kind"] or any(
        lengths_document.get(k) != v for k, v in FLAGS.items()
    ):
        raise ValueError("Length receipt scientific status changed.")
    if [r["source_id"] for r in lengths] != list(selected):
        raise ValueError("Length receipts differ from selected training sources.")
    for r, measured in zip(plan["sources"], lengths, strict=True):
        uid = r["source_id"]
        native, sr = read_native(safe_path(root, r["source_path"]))
        if sr != 48_000 or array_digest(native) != r["decoded_audio_sha256"]:
            raise ValueError("Source decoded identity mismatch.")
        preview = resample_source(native)
        span = derive_trim_span(preview, uid)
        reference = finalise_reference(preview, span)[0]
        actual = read_processed(
            out / _audio_rel(plan["source_kind"], "archive", "real", uid), ARCHIVE_SR
        )
        if np.max(np.abs(reference - actual)) > 1 / 32768 + 1e-7:
            raise ValueError("Reference reconstruction differs beyond PCM16 quantization.")
        condition_wav = read_processed(
            out / _audio_rel(plan["source_kind"], "archive", plan["condition"], uid), ARCHIVE_SR
        )
        pairs.append((actual, condition_wav))
        if (
            measured["native_source_samples"] != len(native)
            or measured["reference_archive_samples"] != len(preview)
            or measured["trim_start"] != span.start
            or measured["trim_end"] != span.end
            or measured["final_archive_samples"] != span.length
            or measured["length_adjustment_samples"]
            != len(preview) - measured["prealignment_archive_samples"]
            or abs(measured["length_adjustment_samples"]) > ARCHIVE_SR // 20
            or measured["generator_input_sr"] != _spec(plan["condition"]).mel.sample_rate
            or measured["generator_output_sr"] != _spec(plan["condition"]).mel.sample_rate
            or measured["generator_input_samples"]
            != (len(native) * measured["generator_input_sr"] + 47999) // 48000
            or type(measured["generator_output_samples"]) is not int
            or measured["generator_output_samples"] <= 0
            or measured["prealignment_archive_samples"]
            != (
                measured["generator_output_samples"] * ARCHIVE_SR
                + measured["generator_output_sr"]
                - 1
            )
            // measured["generator_output_sr"]
        ):
            raise ValueError("Native/archive length and processing receipt is inconsistent.")
        for condition in ("real", plan["condition"]):
            archive = read_processed(
                out / _audio_rel(plan["source_kind"], "archive", condition, uid), ARCHIVE_SR
            )
            parent_lufs = float(
                frame[
                    (frame["utt_id"] == uid)
                    & (frame["condition"] == condition)
                    & (frame["tier"] == "archive")
                ].iloc[0]["lufs_out"]
            )
            expected_derived = derive_zeroshot(archive, archive_lufs=parent_lufs)[0]
            derived = read_processed(
                out / _audio_rel(plan["source_kind"], "zeroshot", condition, uid), ZEROSHOT_SR
            )
            if np.max(np.abs(expected_derived - derived)) > 1 / 32768 + 1e-7:
                raise ValueError("Derived audio is not the common archive->16k path.")
    report = check_vocoder_alignment(pairs, plan["condition"])
    assert_aligned(report)
    saved_alignment = read_json(out / "alignment.json")
    saved_report = saved_alignment["pre_encoding"]
    if (
        not saved_report["passed"]
        or saved_report["n_files"] != 20
        or saved_report["peak_lag"] != 0
        or saved_report["condition"] != plan["condition"]
        or saved_alignment["decoded"] != report.to_dict()
        or saved_alignment["source_kind"] != plan["source_kind"]
        or any(saved_alignment.get(k) != v for k, v in FLAGS.items())
    ):
        raise ValueError("Alignment receipt is inconsistent.")
    saved_sanity = pd.read_csv(out / "sanity.csv")
    pd.testing.assert_frame_equal(saved_sanity, sanity, check_dtype=False, atol=1e-12, rtol=1e-12)
    calibration = {
        t: silence_reference(frame, t, None).as_record() for t in ("archive", "zeroshot")
    }
    if read_json(out / "calibration.json") != {
        "records": calibration,
        "source_kind": plan["source_kind"],
        **FLAGS,
    }:
        raise ValueError("Silence calibration receipt differs.")
    pd.testing.assert_frame_equal(
        pd.read_csv(out / "loudness_gain.csv"),
        loudness_gain_report(frame),
        check_dtype=False,
        atol=1e-12,
        rtol=1e-12,
    )
    expected_summary = {
        "status": "passed",
        "stage": summary["stage"],
        "source_kind": plan["source_kind"],
        "condition": plan["condition"],
        "sources": 20,
        "speakers": 6,
        "outputs": 80,
        "source_split": "train",
        "alignment_files": 20,
        "band": band,
        "sanity_probes": len(sanity),
        "selected_source_files_unchanged": True,
        "real_vctk_sources": 0 if plan["source_kind"] == "fixture" else 20,
        "generator_kind": provenance["generator"]["kind"],
        **FLAGS,
    }
    if summary != expected_summary or summary["stage"] not in ("compatibility", "generation"):
        raise ValueError("Validation summary disagrees with measured artifacts.")
    if summary["stage"] == "generation":
        prerequisite = provenance["compatibility"]
        prior = (
            plain_root(
                compatibility_dir if compatibility_dir is not None else prerequisite["directory"]
            )
            if prerequisite is not None
            else None
        )
        if (
            prerequisite is None
            or digest(prior / "checksums.json") != prerequisite["checksums_sha256"]
        ):
            raise ValueError("Compatibility prerequisite changed.")
        prior_prov = read_json(prior / "provenance.json")
        if (
            read_json(prior / "validation.json")["stage"] != "compatibility"
            or prior_prov["plan_sha256"] != provenance["plan_sha256"]
            or prior_prov["generator"] != provenance["generator"]
            or prior_prov["cached_checkpoint_assets"] != provenance["cached_checkpoint_assets"]
            or prior_prov["environment"] != provenance["environment"]
        ):
            raise ValueError("Compatibility belongs to a different plan, generator or environment.")
        result = validate(prior, source_root=source_root, evidence_dir=evidence_dir)
        if result["stage"] != "compatibility":
            raise ValueError("Not a compatibility prerequisite.")
    elif provenance["compatibility"] is not None:
        raise ValueError("Compatibility run has an unexpected prerequisite.")
    return summary
