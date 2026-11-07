"""Read-only VCTK 0.92 source intake, distinct from a gated Phase A archive.

Native FLAC is an acquired source container. No FLAC intermediates or processed
audio are written. Accepted rows are candidates for subsequent scientific gates.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import platform
import re
import subprocess
import unicodedata
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path, PurePosixPath

import librosa
import numpy as np
import soundfile as sf
import soxr

from data.audio_io import probe, read_native
from data.corpus import Corpus, Utterance, assign_splits
from data.invariants import ARCHIVE_SR, GLOBAL_SEED, RESAMPLE_METHOD, SPLIT_FRACTIONS, TRIM_TOP_DB
from data.preprocess import derive_trim_span

FLAGS = {"development_only": True, "citable": False, "scientific_admission_complete": False}
DATASET = "vctk_0.92_mic2"
NATIVE_SR = 48_000
SOURCE_URL = "https://datashare.ed.ac.uk/handle/10283/3443"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/legalcode"
PROFILE = "vctk48_archive_soxr_hq_v1"
PINNED = {
    "numpy": "2.5.3",
    "scipy": "1.18.1",
    "librosa": "1.0.0",
    "soxr": "1.1.0",
    "soundfile": "0.14.0",
}
LIBSOXR = "0.1.3-14-ga66f3ee"
LIBSNDFILE = "1.2.2"
FILES = {
    "plan.json",
    "source_inventory.json",
    "source_manifest.csv",
    "speakers.json",
    "exclusions.json",
    "selection.json",
    "resampling.json",
    "provenance.json",
    "validation.json",
    "release_readme.txt",
    "release_license.txt",
    "checksums.json",
}
ROOT_FILES = {
    "speaker-info.txt",
    "README",
    "README.txt",
    "COPYING",
    "license_text.txt",
    "update.txt",
    "fixture.json",
}
ID = re.compile(r"p[0-9]{3}_[0-9]+\Z")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for b in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def array_digest(wav):
    return hashlib.sha256(np.asarray(wav, dtype="<f4").tobytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(
        json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def plain(path):
    p = Path(path)
    if (
        p.is_symlink()
        or getattr(p, "is_junction", lambda: False)()
        or (p.is_file() and p.stat().st_nlink != 1)
    ):
        raise ValueError(f"Unsafe link in source/intake path: {p}")


def safe_path(root, relative):
    rel = PurePosixPath(relative)
    if (
        not relative
        or "\\" in relative
        or ":" in relative
        or rel.is_absolute()
        or any(p in ("", ".", "..") for p in relative.split("/"))
    ):
        raise ValueError("Unsafe relative source path.")
    p = Path(root).joinpath(*rel.parts)
    for node in [p, *p.parents]:
        plain(node)
        if node == Path(root):
            break
    if not p.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("Source path escapes root.")
    return p


def plain_root(path):
    p = Path(path).absolute()
    for node in [p, *p.parents]:
        plain(node)
    return p.resolve()


def inventory(root):
    """Complete source inventory. Unknown files are rejected, never ignored."""
    root = plain_root(root)
    if not root.is_dir():
        raise ValueError("Expected an extracted local VCTK directory.")
    result = {}
    for p in sorted(root.rglob("*")):
        plain(p)
        if p.is_dir():
            continue
        rel = p.relative_to(root).as_posix()
        parts = PurePosixPath(rel).parts
        allowed = (len(parts) == 1 and rel in ROOT_FILES) or (
            len(parts) == 3
            and (
                (parts[0] == "txt" and p.suffix == ".txt")
                or (parts[0] == "wav48_silence_trimmed" and p.suffix == ".flac")
            )
        )
        if not allowed:
            raise ValueError(f"Unexpected source file (choose the extracted corpus root): {rel}")
        result[rel] = {"bytes": p.stat().st_size, "sha256": digest(p)}
    if "speaker-info.txt" not in result:
        raise ValueError("Missing speaker-info.txt.")
    return result


def resampling_record():
    actual = {name: version(name) for name in PINNED}
    if (
        actual != PINNED
        or soxr.__libsoxr_version__ != LIBSOXR
        or sf.__libsndfile_version__ != LIBSNDFILE
    ):
        raise ValueError("Unsupported intake environment; review a new profile, never substitute.")
    if RESAMPLE_METHOD != "soxr_hq":
        raise ValueError("Frozen resampling method changed; review before source intake.")
    return {
        "profile": PROFILE,
        "libraries": actual,
        "libsoxr": LIBSOXR,
        "libsndfile": LIBSNDFILE,
        "implementation": "librosa.resample -> soxr HQ",
        "res_type": RESAMPLE_METHOD,
        "fix": True,
        "scale": False,
        "axis": -1,
        "input_dtype": "float32",
        "output_dtype": "float32",
        "input_sr": NATIVE_SR,
        "output_sr": ARCHIVE_SR,
        "length_policy": "ceil(N*22050/48000);right zero-pad/truncate only per librosa fix=True",
        "length_integer_formula": "(N*22050+47999)//48000",
        "alignment_compensation": False,
        "signal_gain_normalization": False,
        "test_vector_absolute_tolerance": 1e-6,
        "test_vector_relative_tolerance": 1e-6,
        "cross_platform_bitwise_identity_claimed": False,
    }


def resample_source(wav, sample_rate=NATIVE_SR):
    """New explicit intake profile; legacy resample functions/bundles are untouched."""
    resampling_record()
    wav = np.asarray(wav)
    if (
        sample_rate != NATIVE_SR
        or wav.ndim != 1
        or not len(wav)
        or wav.dtype.kind not in "fiu"
        or not np.isfinite(wav).all()
    ):
        raise ValueError("Expected finite nonempty mono native 48000-Hz source.")
    out = librosa.resample(
        wav.astype(np.float32),
        orig_sr=NATIVE_SR,
        target_sr=ARCHIVE_SR,
        res_type=RESAMPLE_METHOD,
        fix=True,
        scale=False,
        axis=-1,
    )
    expected = (len(wav) * ARCHIVE_SR + NATIVE_SR - 1) // NATIVE_SR
    if len(out) != expected or out.dtype != np.float32 or not np.isfinite(out).all():
        raise ValueError("Pinned resampling output geometry/integrity failed.")
    return out


@dataclass(frozen=True)
class IntakePolicy:
    speakers: int = 50
    utterances_per_speaker: int = 10
    seed: int = GLOBAL_SEED

    def validate(self):
        if (
            type(self.speakers) is not int
            or self.speakers < 6
            or type(self.utterances_per_speaker) is not int
            or self.utterances_per_speaker < 1
            or type(self.seed) is not int
            or self.seed != GLOBAL_SEED
        ):
            raise ValueError("Need >=6 speakers, positive utterance count and the frozen seed.")


def ranked(values, namespace):
    return sorted(
        values,
        key=lambda s: (
            hashlib.sha256(f"{GLOBAL_SEED}:vocotrace-ms-v1:{namespace}:{s}".encode()).hexdigest(),
            s,
        ),
    )


def speaker_records(root):
    records = {}
    for line in (root / "speaker-info.txt").read_text(encoding="utf-8-sig").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split(maxsplit=3)
        if parts[0].upper() == "ID":
            continue
        native = "p" + parts[0] if parts[0].isdigit() else parts[0]
        if not re.fullmatch(r"p[0-9]{3}", native) or len(parts) < 3 or native in records:
            raise ValueError("Malformed or duplicate speaker identity metadata.")
        records[native] = {
            "speaker_id": f"vctk092_{native}",
            "native_speaker_id": native,
            "age_text": parts[1],
            "gender_text": parts[2],
            "accent_region_text": parts[3] if len(parts) == 4 else "",
        }
    if not records:
        raise ValueError("No speaker identities found.")
    return records


def release_evidence(root, evidence_dir, source_kind):
    if source_kind not in ("fixture", "local_dataset"):
        raise ValueError("Declare fixture or local_dataset; no implicit evidence promotion.")
    marker = root / "fixture.json"
    if source_kind == "fixture":
        if not marker.is_file() or read_json(marker).get("kind") != "synthetic_audio_fixture":
            raise ValueError("Fixture mode requires its explicit synthetic_audio_fixture marker.")
    elif marker.exists():
        raise ValueError("Synthetic fixtures cannot be declared real dataset ingestion.")
    evidence_dir = plain_root(evidence_dir)
    texts = {}
    for name in ("README.txt", "license_text.txt"):
        p = safe_path(evidence_dir, name)
        if not p.is_file() or p.stat().st_size > 1024 * 1024:
            raise ValueError(f"Missing/oversized release evidence: {name}")
        texts[name] = p.read_text(encoding="utf-8-sig")
    if "0.92" not in texts["README.txt"] or "VCTK" not in texts["README.txt"]:
        raise ValueError("README does not identify VCTK 0.92.")
    if not any(
        s in texts["license_text.txt"]
        for s in ("creativecommons.org/licenses/by/4.0", "Attribution 4.0 International")
    ):
        raise ValueError("Release evidence does not identify CC BY 4.0.")
    return texts


class SourceRejection(ValueError):
    def __init__(self, reason, **observed):
        self.reason, self.observed = reason, observed
        super().__init__(reason)


def inspect_source(root, native_id, speakers, source_files):
    spk = native_id.split("_")[0]
    audio_rel = f"wav48_silence_trimmed/{spk}/{native_id}_mic2.flac"
    text_rel = f"txt/{spk}/{native_id}.txt"
    if spk not in speakers:
        raise SourceRejection("missing_speaker_metadata", native_speaker_id=spk)
    for rel in (audio_rel, text_rel):
        if rel not in source_files:
            raise SourceRejection(
                "missing_audio" if rel == audio_rel else "missing_transcript", missing_path=rel
            )
    audio, text_path = safe_path(root, audio_rel), safe_path(root, text_rel)
    if text_path.stat().st_size > 1024 * 1024:
        raise SourceRejection("oversized_transcript", bytes=text_path.stat().st_size)
    try:
        text = text_path.read_text(encoding="utf-8-sig").strip()
    except UnicodeError as exc:
        raise SourceRejection("invalid_transcript_encoding") from exc
    if not text:
        raise SourceRejection("empty_transcript")
    try:
        with audio.open("rb") as stream:
            magic = stream.read(4)
        header = probe(audio)
    except (RuntimeError, OSError, ValueError) as exc:
        raise SourceRejection("unreadable_audio", error_type=type(exc).__name__) from exc
    observed = {
        k: header[k] for k in ("sample_rate", "channels", "subtype", "frames", "duration_s")
    }
    if (
        magic != b"fLaC"
        or header["sample_rate"] != NATIVE_SR
        or header["channels"] != 1
        or header["subtype"] != "PCM_16"
    ):
        raise SourceRejection("unsupported_native_format", **observed)
    if not 0 < header["duration_s"] <= 60:
        raise SourceRejection("unsafe_native_duration", **observed)
    try:
        wav, rate = read_native(audio)
    except (RuntimeError, OSError, ValueError) as exc:
        raise SourceRejection(
            "audio_decode_failed", **observed, error_type=type(exc).__name__
        ) from exc
    if (
        rate != NATIVE_SR
        or wav.ndim != 1
        or len(wav) != header["frames"]
        or not np.isfinite(wav).all()
    ):
        raise SourceRejection("invalid_decoded_audio", **observed)
    peak = float(np.max(np.abs(wav)))
    if peak == 0:
        raise SourceRejection("silent_source", peak=peak, **observed)
    clipped = int(np.sum((wav <= -1) | (wav >= 32767 / 32768)))
    if clipped:
        raise SourceRejection("native_pcm_endpoint_samples", endpoint_samples=clipped, **observed)
    preview = resample_source(wav, rate)
    span = derive_trim_span(preview, native_id)  # Existing invariant errors propagate.
    seconds = span.length / ARCHIVE_SR
    if not 4.1 <= seconds <= 8.0:
        raise SourceRejection(
            "trimmed_duration_outside_cohort_bounds",
            trimmed_seconds=seconds,
            required_min_seconds=4.1,
            required_max_seconds=8.0,
            **observed,
        )
    normalized_text = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return {
        "source_id": f"vctk092_mic2_{native_id}",
        "speaker_id": speakers[spk]["speaker_id"],
        "native_utterance_id": native_id,
        "source_path": audio_rel,
        "transcript_path": text_rel,
        "source_sha256": source_files[audio_rel]["sha256"],
        "transcript_sha256": source_files[text_rel]["sha256"],
        "normalized_text_sha256": hashlib.sha256(normalized_text.encode()).hexdigest(),
        "decoded_audio_sha256": array_digest(wav),
        "native_sample_rate": rate,
        "native_channels": 1,
        "native_subtype": "PCM_16",
        "native_samples": len(wav),
        "native_seconds": len(wav) / rate,
        "native_peak": peak,
        "relative_quiet_sample_fraction": float(
            np.mean(np.abs(wav) <= peak * 10 ** (-TRIM_TOP_DB / 20))
        ),
        "archive_preview_samples": len(preview),
        "archive_preview_sha256": array_digest(preview),
        "archive_trim_start": span.start,
        "archive_trim_end": span.end,
        "archive_trim_seconds": seconds,
        "microphone": "mic2",
        "dataset_version": "0.92",
        "resampling_profile": PROFILE,
    }


def analyze(root, policy, source_files):
    """Pure deterministic selection/measurement. All duplicate members are excluded."""
    speakers = speaker_records(root)
    ids, malformed = set(), []
    for rel in source_files:
        p = PurePosixPath(rel)
        if p.parts[0] == "txt" and len(p.parts) == 3:
            native = p.stem
        elif p.parts[0] == "wav48_silence_trimmed" and p.name.endswith("_mic2.flac"):
            native = p.name.removesuffix("_mic2.flac")
        else:
            continue
        if not ID.fullmatch(native) or p.parts[1] != native.split("_")[0]:
            malformed.append(
                {
                    "candidate_id": rel,
                    "reason": "invalid_source_identity",
                    "observed": source_files[rel],
                }
            )
        else:
            ids.add(native)
    valid, excluded = [], list(malformed)
    for native in sorted(ids):
        try:
            valid.append(inspect_source(root, native, speakers, source_files))
        except SourceRejection as exc:
            spk = native.split("_")[0]
            path = f"wav48_silence_trimmed/{spk}/{native}_mic2.flac"
            excluded.append(
                {
                    "candidate_id": native,
                    "source_path": path,
                    "source_file_record": source_files.get(path),
                    "reason": exc.reason,
                    "observed": exc.observed,
                }
            )
    duplicates = {}
    for key in ("source_sha256", "decoded_audio_sha256", "normalized_text_sha256"):
        groups = {}
        for row in valid:
            groups.setdefault(row[key], []).append(row["native_utterance_id"])
        for value, members in groups.items():
            if len(members) > 1:
                for member in members:
                    duplicates.setdefault(member, []).append(
                        {"field": key, "sha256": value, "all_members": sorted(members)}
                    )
    eligible = []
    for row in valid:
        native = row["native_utterance_id"]
        if native in duplicates:
            excluded.append(
                {
                    "candidate_id": native,
                    "reason": "duplicate_source_or_content",
                    "observed": {"collisions": duplicates[native]},
                }
            )
        else:
            eligible.append(row)
    by_speaker = {}
    for row in eligible:
        by_speaker.setdefault(row["speaker_id"], []).append(row)
    qualified = ranked(
        [s for s, rows in by_speaker.items() if len(rows) >= policy.utterances_per_speaker],
        "speaker",
    )
    selection = {
        "requested_speakers": policy.speakers,
        "available_qualified_speakers": len(qualified),
        "qualified_speaker_rank": qualified,
        "candidates": len(ids) + len(malformed),
        "eligible_utterances": len(eligible),
        "decisions": [],
        "selected_speakers": [],
        "eligible_sources_by_speaker": {
            s: len(by_speaker.get(s, []))
            for s in sorted(r["speaker_id"] for r in speakers.values())
        },
        "requested_utterances_per_speaker": policy.utterances_per_speaker,
    }
    if len(qualified) < policy.speakers:
        selection["shortfall"] = policy.speakers - len(qualified)
        selection["decisions"] = [
            {
                "source_id": r["source_id"],
                "speaker_id": r["speaker_id"],
                "reason": "insufficient_qualified_speakers"
                if r["speaker_id"] in qualified
                else "insufficient_utterances_for_speaker",
            }
            for r in eligible
        ]
        return [], speakers, excluded, selection
    chosen = qualified[: policy.speakers]
    selected = []
    for speaker in chosen:
        rows = {r["source_id"]: r for r in by_speaker[speaker]}
        for sid in ranked(rows, "utterance")[: policy.utterances_per_speaker]:
            selected.append(rows[sid])
    selected_ids = {r["source_id"] for r in selected}
    for row in eligible:
        spk, sid = row["speaker_id"], row["source_id"]
        reason = (
            "selected"
            if sid in selected_ids
            else "insufficient_utterances_for_speaker"
            if spk not in qualified
            else "speaker_rank_limit"
            if spk not in chosen
            else "utterance_rank_limit"
        )
        selection["decisions"].append({"source_id": sid, "speaker_id": spk, "reason": reason})
    splits = assign_splits(
        [Utterance(r["source_id"], Path(r["source_path"]), r["speaker_id"]) for r in selected],
        speaker_disjoint=True,
        seed=policy.seed,
    )
    if set(splits.values()) != {"train", "dev", "eval"}:
        raise ValueError("Insufficient speakers for nonempty frozen partitions.")
    for row in selected:
        row.update(split=splits[row["source_id"]])
    selection["selected_speakers"] = chosen
    return sorted(selected, key=lambda r: r["source_id"]), speakers, excluded, selection


def csv_text(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def implementation():
    import data.audio_io
    import data.corpus
    import data.invariants
    import data.preprocess

    modules = [data.audio_io, data.corpus, data.invariants, data.preprocess]
    return {**{m.__name__: digest(m.__file__) for m in modules}, __name__: digest(__file__)}


def ingest_vctk(
    source_root, out_dir, *, policy=None, evidence_dir=None, source_kind="local_dataset"
):
    policy = IntakePolicy() if policy is None else policy
    policy.validate()
    root, out = plain_root(source_root), plain_root(out_dir)
    evidence = plain_root(evidence_dir) if evidence_dir is not None else root
    if out.exists():
        raise FileExistsError(
            "Fresh ingestion output required; existing artifacts are never repaired."
        )
    if any(out.is_relative_to(p) or p.is_relative_to(out) for p in (root, evidence)):
        raise ValueError(
            "Ingestion output must be separate from original source/evidence directories."
        )
    env = resampling_record()
    original = inventory(root)
    texts = release_evidence(root, evidence, source_kind)
    out.mkdir(parents=True)
    plan = {
        "schema_version": 1,
        "dataset": DATASET,
        "microphone": "mic2",
        "native_sr": NATIVE_SR,
        "policy": asdict(policy),
        "source_kind": source_kind,
        "source_url": SOURCE_URL,
        "license_claim": "CC-BY-4.0"
        if source_kind == "local_dataset"
        else "fixture_not_source_rights",
        "license_url": LICENSE_URL,
        "split_fractions": SPLIT_FRACTIONS,
        "trimmed_seconds_bounds": [4.1, 8.0],
        "native_max_seconds": 60,
        "relative_silence_reference_db": TRIM_TOP_DB,
        "source_stage": "native_intake_only_no_published_audio",
        **FLAGS,
    }
    write_json(out / "plan.json", plan)
    write_json(out / "source_inventory.json", original)
    try:
        rows, speakers, excluded, selection = analyze(root, policy, original)
        write_json(out / "speakers.json", speakers)
        write_json(out / "exclusions.json", excluded)
        write_json(out / "selection.json", selection)
        if not rows:
            raise ValueError(
                f"Insufficient qualified speakers: {selection['available_qualified_speakers']} "
                f"available, {policy.speakers} requested; no split fabricated."
            )
        for row in rows:
            row.update(source_kind=source_kind, development_only=True, citable=False)
        (out / "source_manifest.csv").write_text(csv_text(rows), encoding="utf-8", newline="\n")
        write_json(out / "resampling.json", env)
        repo = Path(__file__).resolve().parents[2]

        def git_read(*args):
            return subprocess.check_output(
                ["git", "--no-optional-locks", *args], cwd=repo, text=True
            ).strip()

        write_json(
            out / "provenance.json",
            {
                "source_root": str(root),
                "evidence_root": str(evidence),
                "git_head": git_read("rev-parse", "HEAD"),
                "working_tree_status": git_read("status", "--short"),
                "python": platform.python_version(),
                "platform": platform.platform(),
                "implementation_sha256": implementation(),
                "evidence_files": {
                    n: {"bytes": (evidence / n).stat().st_size, "sha256": digest(evidence / n)}
                    for n in texts
                },
                "redistribution_cleared": False,
                "source_authenticity_verified": False,
                **FLAGS,
            },
        )
        (out / "release_readme.txt").write_text(texts["README.txt"], encoding="utf-8", newline="\n")
        (out / "release_license.txt").write_text(
            texts["license_text.txt"], encoding="utf-8", newline="\n"
        )
        if inventory(root) != original or release_evidence(root, evidence, source_kind) != texts:
            raise ValueError("Original source/evidence changed during intake.")
        split_counts = {s: sum(r["split"] == s for r in rows) for s in ("train", "dev", "eval")}
        write_json(
            out / "validation.json",
            {
                "status": "passed",
                "source_kind": source_kind,
                "sources": len(rows),
                "speakers": len(selection["selected_speakers"]),
                "source_counts_by_split": split_counts,
                "speaker_counts_by_split": {
                    s: len({r["speaker_id"] for r in rows if r["split"] == s}) for s in split_counts
                },
                "rejected_candidates": len(excluded),
                "source_files_unchanged": True,
                "audio_outputs_generated": 0,
                "models_trained": 0,
                **FLAGS,
            },
        )
        write_json(
            out / "checksums.json",
            {
                p.name: {"bytes": p.stat().st_size, "sha256": digest(p)}
                for p in sorted(out.iterdir())
            },
        )
        return verify_ingestion(out)
    except Exception as exc:
        write_json(out / "validation.json", {"status": "failed", "error": str(exc), **FLAGS})
        raise


def verify_ingestion(directory, *, source_root=None, evidence_dir=None):
    """Verify exact schema, live inventory, measurements, selection and frozen splits."""
    out = plain_root(directory)
    if {p.name for p in out.iterdir()} != FILES:
        raise ValueError("Unexpected/missing ingestion artifacts.")
    for p in out.iterdir():
        plain(p)
        if not p.is_file():
            raise ValueError("Non-file ingestion artifact.")
    checks = read_json(out / "checksums.json")
    if set(checks) != FILES - {"checksums.json"} or any(
        checks[p.name] != {"bytes": p.stat().st_size, "sha256": digest(p)}
        for p in out.iterdir()
        if p.name != "checksums.json"
    ):
        raise ValueError("Ingestion integrity mismatch.")
    plan, provenance = read_json(out / "plan.json"), read_json(out / "provenance.json")
    policy = IntakePolicy(**plan["policy"])
    policy.validate()
    if (
        plan["dataset"] != DATASET
        or plan["schema_version"] != 1
        or plan["microphone"] != "mic2"
        or plan["native_sr"] != NATIVE_SR
        or plan["split_fractions"] != SPLIT_FRACTIONS
        or plan["trimmed_seconds_bounds"] != [4.1, 8.0]
        or plan["native_max_seconds"] != 60
        or plan["relative_silence_reference_db"] != TRIM_TOP_DB
        or plan["source_stage"] != "native_intake_only_no_published_audio"
        or plan["source_url"] != SOURCE_URL
        or plan["license_url"] != LICENSE_URL
        or plan["license_claim"]
        != ("CC-BY-4.0" if plan["source_kind"] == "local_dataset" else "fixture_not_source_rights")
        or any(plan.get(k) != v or provenance.get(k) != v for k, v in FLAGS.items())
        or provenance["implementation_sha256"] != implementation()
        or provenance["redistribution_cleared"] is not False
        or provenance["source_authenticity_verified"] is not False
    ):
        raise ValueError("Ingestion protocol/provenance mismatch.")
    root = plain_root(source_root if source_root is not None else provenance["source_root"])
    evidence = plain_root(
        evidence_dir
        if evidence_dir is not None
        else (
            root
            if provenance["evidence_root"] == provenance["source_root"]
            else provenance["evidence_root"]
        )
    )
    original = inventory(root)
    if original != read_json(out / "source_inventory.json"):
        raise ValueError("Original dataset inventory changed.")
    texts = release_evidence(root, evidence, plan["source_kind"])
    if provenance["evidence_files"] != {
        n: {"bytes": (evidence / n).stat().st_size, "sha256": digest(evidence / n)} for n in texts
    }:
        raise ValueError("Original release evidence byte identity changed.")
    for name, content in (
        ("release_readme.txt", texts["README.txt"]),
        ("release_license.txt", texts["license_text.txt"]),
    ):
        if (out / name).read_text(encoding="utf-8") != content:
            raise ValueError("Original release evidence changed.")
    if resampling_record() != read_json(out / "resampling.json"):
        raise ValueError("Resampling profile mismatch.")
    rows, speakers, excluded, selection = analyze(root, policy, original)
    if not rows:
        raise ValueError("Ingestion no longer has enough qualified speakers.")
    for row in rows:
        row.update(source_kind=plan["source_kind"], development_only=True, citable=False)
    if (
        (out / "source_manifest.csv").read_text(encoding="utf-8") != csv_text(rows)
        or read_json(out / "speakers.json") != speakers
        or read_json(out / "exclusions.json") != excluded
        or read_json(out / "selection.json") != selection
    ):
        raise ValueError("Source measurements, selection or split inheritance differ.")
    saved = read_json(out / "validation.json")
    expected_counts = {s: sum(r["split"] == s for r in rows) for s in ("train", "dev", "eval")}
    expected_speakers = {
        s: len({r["speaker_id"] for r in rows if r["split"] == s}) for s in expected_counts
    }
    if (
        saved["status"] != "passed"
        or saved["source_kind"] != plan["source_kind"]
        or saved["source_files_unchanged"] is not True
        or saved["sources"] != len(rows)
        or saved["speakers"] != policy.speakers
        or saved["source_counts_by_split"] != expected_counts
        or saved["speaker_counts_by_split"] != expected_speakers
        or saved["rejected_candidates"] != len(excluded)
        or any(saved.get(k) != v for k, v in FLAGS.items())
        or saved["audio_outputs_generated"] != 0
        or saved["models_trained"] != 0
    ):
        raise ValueError("Ingestion validation summary inconsistent.")
    if inventory(root) != original:
        raise ValueError("Original source changed during verification.")
    return saved


class IngestedVCTK(Corpus):
    """Readonly Corpus bridge. Future generation must preserve frozen_splits explicitly."""

    name = DATASET
    native_sample_rate = NATIVE_SR
    is_multi_speaker = True

    def __init__(self, directory, *, source_root=None, evidence_dir=None):
        self.directory = Path(directory)
        verify_ingestion(directory, source_root=source_root, evidence_dir=evidence_dir)
        prov = read_json(self.directory / "provenance.json")
        self.root = plain_root(source_root if source_root is not None else prov["source_root"])
        with (self.directory / "source_manifest.csv").open(encoding="utf-8", newline="") as f:
            self.rows = tuple(csv.DictReader(f))

    @property
    def frozen_splits(self):
        return {r["source_id"]: r["split"] for r in self.rows}

    def utterances(self):
        values = []
        for r in self.rows:
            path = safe_path(self.root, r["source_path"])
            text_path = safe_path(self.root, r["transcript_path"])
            if digest(path) != r["source_sha256"] or digest(text_path) != r["transcript_sha256"]:
                raise ValueError("Original selected source changed since admission.")
            values.append(
                Utterance(
                    r["source_id"],
                    path,
                    r["speaker_id"],
                    text_path.read_text(encoding="utf-8-sig").strip(),
                )
            )
        return values
