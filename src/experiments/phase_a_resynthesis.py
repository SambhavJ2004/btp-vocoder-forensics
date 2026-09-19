"""Phase A — controlled resynthesis. Person A.

Builds a dataset in which vocoder identity is the only free variable. One
vocoder per invocation, because dependency trees conflict and each condition
runs in its own isolated Kaggle session (INV-13); results are merged later at
the manifest level.

Two artifacts come out of this (INV-01):

  ARCHIVE (22.05 kHz, primary)
      Real audio -> vocoder's native mel -> waveform -> the fixed post-pipeline
      (INV-07). One resample, native -> archive. Because LJSpeech is already
      22.05 kHz, the real condition and most vocoders take no resampling filter
      at all.

  ZEROSHOT (16 kHz, derived)
      One further downsample of the finished archive file. Never from source.
      Exists because pretrained ASVspoof detectors, PESQ-WB and UTMOS are all
      hard-locked to 16 kHz.

Archive high, derive low, never the reverse. The real condition passes through
the identical treatment at both tiers, which is what makes "identical" true
rather than aspirational.

The archive is FULL-BAND (INV-17). Nothing is low-passed at generation: the
content above LADDER_FMAX is hallucinated by the vocoder, and it is the most
forensically interesting content the archive holds. What Phase A does record is
the measured `high_band_fraction` per file, which is the evidence separating a
genuine bandwidth zero from a vocoder tracking real.

Each new vocoder is also probed for sample alignment (INV-16) before its
condition is accepted.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyloudnorm as pyln
from tqdm import tqdm

from data.alignment import AlignmentReport, assert_aligned, check_vocoder_alignment
from data.audio_io import read_native, read_processed, write_audio
from data.corpus import Corpus, assign_splits
from data.invariants import (
    ALIGNMENT_PROBE_FILES,
    ANALYSES,
    ARCHIVE_SR,
    ARCHIVE_TIER,
    BAND_LIMIT_MAX_STOPBAND_ENERGY,
    GLOBAL_SEED,
    LADDER_FMAX,
    REAL_CONDITION,
    ZEROSHOT_SR,
    ZEROSHOT_TIER,
    InvariantViolation,
    exclusion_for,
    out_of_band_energy,
)
from data.manifest import ManifestRow, dump_provenance, select_tier, write_manifest
from data.preprocess import (
    LoudnessRecord,
    TrimSpan,
    archive_band_spec,
    band_limit_comparison_set,
    derive_trim_span,
    derive_zeroshot,
    finalise_reference,
    process_condition_output,
    resample_once,
    resample_with_spec,
    would_clip,
)
from vocoders.registry import get_spec, get_vocoder


def _emit_pair(
    wav_archive: np.ndarray,
    loudness: LoudnessRecord,
    out_dir: Path,
    condition: str,
    utt_id: str,
    zeroshot_meter: pyln.Meter,
) -> tuple[Path, Path, float, int]:
    """Write the archive artifact, then derive and write the zero-shot artifact.

    The write is verified against the ANALYSIS-BAND loudness, because that is
    where the target is defined (INV-04). The full-band value is what the derived
    tier's drift is measured against, since that tier is a full-band downsample.
    """
    archive_path = out_dir / ARCHIVE_TIER / condition / f"{utt_id}.wav"
    write_audio(
        archive_path, wav_archive, ARCHIVE_SR, measured_lufs=loudness.lufs_out_band
    )

    wav_zs, measured_zs = derive_zeroshot(
        wav_archive, archive_lufs=loudness.lufs_out, meter=zeroshot_meter
    )
    if would_clip(wav_zs):
        raise InvariantViolation(
            f"INV-04: derived zero-shot for {condition}/{utt_id} exceeds the peak "
            "ceiling. Resampling can lift true peak slightly; drop this utterance "
            "from every condition and both tiers rather than attenuating it here."
        )
    # measured_lufs is deliberately not asserted against the archive target: the
    # derived tier legitimately drifts because band-limiting removes energy, and
    # re-normalising would apply a gain proportional to high-band content --
    # which is the artifact under study. The value is recorded instead.
    zs_path = out_dir / ZEROSHOT_TIER / condition / f"{utt_id}.wav"
    write_audio(zs_path, wav_zs, ZEROSHOT_SR)
    return archive_path, zs_path, measured_zs, len(wav_zs)


def _rows_for(
    utt_id: str,
    condition: str,
    speaker: str,
    split: str,
    span: TrimSpan,
    archive_path: Path,
    zs_path: Path,
    n_archive: int,
    n_zs: int,
    lufs_archive: float,
    lufs_zs: float,
    sr_native: int,
    *,
    mel_fields: dict | None = None,
    checkpoint: str | None = None,
    commit: str | None = None,
    alignment_lag: int | None = None,
    high_band: float | None = None,
    loudness: LoudnessRecord | None = None,
) -> list[ManifestRow]:
    """One archive row plus its derived zero-shot row, both fully provenanced.

    The loudness record goes on the archive row only: the derived row was never
    normalised, and recording a gain there would claim a gain stage that did not
    happen.
    """
    gain_fields = (
        dict(
            lufs_in=loudness.lufs_in,
            loudness_gain_db=loudness.gain_db,
            loudness_target_lufs=loudness.target_lufs,
            lufs_in_band=loudness.lufs_in_band,
            lufs_out_band=loudness.lufs_out_band,
        )
        if loudness is not None
        else {}
    )
    band_hz, band_name = archive_band_spec()
    common = dict(
        utt_id=utt_id,
        condition=condition,
        speaker=speaker,
        split=split,
        trim_start=span.start,
        trim_end=span.end,
        trim_ref_length=span.ref_length,
        vocoder_checkpoint=checkpoint,
        vocoder_commit=commit,
        seed=GLOBAL_SEED,
        alignment_checked=alignment_lag is not None,
        alignment_peak_lag=alignment_lag,
        archive_band_hz=band_hz,
        archive_band=band_name,
        **(mel_fields or {}),
    )
    return [
        ManifestRow(
            path=str(archive_path),
            tier=ARCHIVE_TIER,
            derived=False,
            source_rate=sr_native,
            derivation=f"resample_once:{sr_native}->{ARCHIVE_SR}",
            resample_steps=1,
            sr_out=ARCHIVE_SR,
            duration_s=n_archive / ARCHIVE_SR,
            lufs_out=lufs_archive,
            high_band_fraction=high_band,
            **gain_fields,
            **common,
        ),
        ManifestRow(
            path=str(zs_path),
            tier=ZEROSHOT_TIER,
            derived=True,
            source_rate=ARCHIVE_SR,
            derivation=f"downsample_once:{ARCHIVE_SR}->{ZEROSHOT_SR}",
            resample_steps=1,
            sr_out=ZEROSHOT_SR,
            duration_s=n_zs / ZEROSHOT_SR,
            lufs_out=lufs_zs,
            **common,
        ),
    ]


def build_real_condition(
    corpus: Corpus, out_dir: Path, splits: dict[str, str]
) -> tuple[list[ManifestRow], dict[str, TrimSpan], dict[str, np.ndarray], set[str]]:
    """Emit the real condition at both tiers and derive the shared trim spans.

    Also returns the processed archive waveforms (needed for the INV-16
    alignment probe) and the set of utterances dropped for peak protection
    (INV-04). Drops apply to EVERY condition and both tiers, because a per-file
    gain change would break the identical-loudness guarantee.
    """
    archive_meter = pyln.Meter(ARCHIVE_SR)
    zs_meter = pyln.Meter(ZEROSHOT_SR)
    rows: list[ManifestRow] = []
    spans: dict[str, TrimSpan] = {}
    real_archive: dict[str, np.ndarray] = {}
    dropped: set[str] = set()

    for utt in tqdm(corpus.utterances(), desc="real"):
        wav_native, sr_native = read_native(utt.path)
        wav = resample_once(wav_native, sr_native, ARCHIVE_SR)
        span = derive_trim_span(wav, utt.utt_id)

        # INV-07. Real goes through the SAME post-trim implementation as every
        # vocoder condition. Real is the reference every span and pairing
        # derives from, so a divergence here would move every condition against
        # real at once. The equivalence is asserted by test, not assumed.
        norm, measured, high_band, loudness = finalise_reference(
            wav, span, meter=archive_meter
        )

        if would_clip(norm):
            dropped.add(utt.utt_id)
            continue

        archive_path, zs_path, lufs_zs, n_zs = _emit_pair(
            norm, loudness, out_dir, REAL_CONDITION, utt.utt_id, zs_meter
        )
        spans[utt.utt_id] = span
        real_archive[utt.utt_id] = norm
        rows.extend(
            _rows_for(
                utt.utt_id, REAL_CONDITION, utt.speaker, splits[utt.utt_id], span,
                archive_path, zs_path, len(norm), n_zs, measured, lufs_zs, sr_native,
                high_band=high_band, loudness=loudness,
            )
        )
    return rows, spans, real_archive, dropped


def build_vocoder_condition(
    condition: str,
    corpus: Corpus,
    out_dir: Path,
    spans: dict[str, TrimSpan],
    real_archive: dict[str, np.ndarray],
    splits: dict[str, str],
    dropped: set[str],
    *,
    device: str | None = None,
) -> tuple[list[ManifestRow], AlignmentReport]:
    """Resynthesize one condition at both tiers, reusing the reference trim spans.

    The INV-16 alignment probe runs on the first ``ALIGNMENT_PROBE_FILES``
    processed archive outputs. It gates the condition: a fixed vocoder delay
    would make the reference trim span land on the wrong span of speech and
    would register as an artifact on both axes at once.
    """
    spec = get_spec(condition)
    voc = get_vocoder(condition)
    if device is not None:
        voc.device = device
    voc.load()
    archive_meter = pyln.Meter(ARCHIVE_SR)
    zs_meter = pyln.Meter(ZEROSHOT_SR)

    for analysis in ANALYSES:
        exc = exclusion_for(analysis, condition)
        if exc is not None:
            print(f"[INV-18] generated, but excluded from {exc.render()}")
    staged: list[
        tuple[str, np.ndarray, float, int, str, float | None, LoudnessRecord]
    ] = []
    probe_pairs: list[tuple[np.ndarray, np.ndarray]] = []

    for utt in tqdm(corpus.utterances(), desc=condition):
        if utt.utt_id in dropped or utt.utt_id not in spans:
            continue

        src, sr_src = read_native(utt.path)
        # Source goes to the VOCODER's training rate for analysis. That is the
        # input-side path; the single sanctioned output resample (native ->
        # archive) happens below, in process_condition_output (INV-01).
        if sr_src != spec.mel.sample_rate:
            if spec.resampler is not None:
                # INV-18: the input-side resample uses the SAME pinned resampler
                # as the output side. Previously this called librosa.resample
                # with no res_type at all -- neither RESAMPLE_METHOD nor a pin.
                src = resample_with_spec(src, sr_src, spec.mel.sample_rate, spec.resampler)
            else:
                import librosa

                src = librosa.resample(src, orig_sr=sr_src, target_sr=spec.mel.sample_rate)

        wav_native = voc.resynthesize(src, spec.mel.sample_rate)
        wav, measured, high_band, loudness = process_condition_output(
            wav_native,
            spec.mel.sample_rate,
            spans[utt.utt_id],
            condition=condition,
            meter=archive_meter,
            resampler=spec.resampler,
        )
        if would_clip(wav):
            raise InvariantViolation(
                f"INV-04: '{condition}' clips on {utt.utt_id} where real did not. "
                "Add the utterance to the global drop list and regenerate every "
                "condition without it -- do not attenuate this one file."
            )
        staged.append(
            (utt.utt_id, wav, measured, len(wav), utt.speaker, high_band, loudness)
        )
        if len(probe_pairs) < ALIGNMENT_PROBE_FILES:
            probe_pairs.append((real_archive[utt.utt_id], wav))

    voc.unload()

    # INV-16 gate, before a single file of this condition is accepted.
    report = check_vocoder_alignment(probe_pairs, condition, sample_rate=ARCHIVE_SR)
    assert_aligned(report)

    rows: list[ManifestRow] = []
    for utt_id, wav, measured, n_archive, speaker, high_band, loudness in staged:
        archive_path, zs_path, lufs_zs, n_zs = _emit_pair(
            wav, loudness, out_dir, condition, utt_id, zs_meter
        )
        rows.extend(
            _rows_for(
                utt_id, condition, speaker, splits[utt_id], spans[utt_id],
                archive_path, zs_path, n_archive, n_zs, measured, lufs_zs,
                spec.mel.sample_rate,
                mel_fields=spec.mel.as_manifest_fields(),
                checkpoint=spec.checkpoint,
                commit=spec.upstream_commit,
                alignment_lag=report.peak_lag,
                high_band=high_band,
                loudness=loudness,
            )
        )
    return rows, report


def run(
    corpus: Corpus,
    conditions: list[str],
    out_dir: str | Path,
    manifest_path: str | Path,
    *,
    speaker_disjoint: bool | None = None,
    device: str | None = None,
) -> Path:
    """Generate the real condition plus ``conditions``, both tiers, and the manifest.

    ``speaker_disjoint`` has no default for a multi-speaker corpus (INV-09):
    assign_splits raises rather than silently splitting by utterance alone.
    """
    from data.alignment import write_alignment_report

    out_dir = Path(out_dir)
    np.random.seed(GLOBAL_SEED)
    utts = corpus.utterances()

    # INV-09: a corpus that misdeclares its own shape is worse than one that
    # says nothing, so the declaration is checked against what it yielded.
    observed_speakers = {u.speaker for u in utts}
    if corpus.is_multi_speaker != (len(observed_speakers) > 1):
        raise InvariantViolation(
            f"INV-09: corpus '{corpus.name}' declares is_multi_speaker="
            f"{corpus.is_multi_speaker} but yielded {len(observed_speakers)} "
            "speaker(s). Fix the declaration -- the split gate and the thesis "
            "both rely on it."
        )

    splits = assign_splits(utts, speaker_disjoint=speaker_disjoint)

    rows, spans, real_archive, dropped = build_real_condition(corpus, out_dir, splits)
    if dropped:
        print(f"[INV-04] dropped {len(dropped)} utterances for peak protection")

    reports: list[AlignmentReport] = []
    for cond in conditions:
        cond_rows, report = build_vocoder_condition(
            cond, corpus, out_dir, spans, real_archive, splits, dropped, device=device
        )
        rows.extend(cond_rows)
        reports.append(report)

    path = write_manifest(rows, manifest_path)
    if reports:
        write_alignment_report(reports, Path(manifest_path).with_suffix(".alignment.json"))
    dump_provenance(
        Path(manifest_path).with_suffix(".provenance.json"),
        {
            "git_commit": _git_commit(),
            "seed": GLOBAL_SEED,
            "conditions": [REAL_CONDITION, *conditions],
            "archive_sr": ARCHIVE_SR,
            "zeroshot_sr": ZEROSHOT_SR,
            "ladder_fmax": LADDER_FMAX,
            "archive_band": "full_band",
            "analysis_exclusions": {
                a: [c for c in conditions if exclusion_for(a, c) is not None]
                for a in ANALYSES
            },
            "n_utterances": len(spans),
            "dropped_for_peak": sorted(dropped),
            "corpus": corpus.name,
            "alignment": {r.condition: r.peak_lag for r in reports},
            # INV-13: one process, one environment, for every condition listed.
            "environment_shared_by": [REAL_CONDITION, *conditions],
            "environment": _environment(),
            "device": device or "auto",
        },
    )
    return path


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


# --- The gated run: every gate, then atomic output ----------------------------
#
# `run` writes as it goes. `run_gated` is what a real run uses: it generates into
# a staging directory beside the output, runs every gate against what was
# written, and only then renames staging into place. A failed gate leaves no
# output directory at all -- not a half-written one that a consumer could mistake
# for a dataset. A crash mid-run leaves `<out>.staging`, never `<out>`.

MANIFEST_NAME = "manifest.csv"
BAND_GATE_UTTERANCES = 5


@dataclass
class GatedRunResult:
    out_dir: Path
    manifest_path: Path
    sanity: pd.DataFrame
    loudness: pd.DataFrame
    gates: dict


def _sibling(out_dir: Path, suffix: str) -> Path:
    return out_dir.with_name(out_dir.name + suffix)


def preflight(
    corpus_root: str | Path, conditions: list[str], out_dir: str | Path, n_utterances: int
) -> None:
    """Everything that can be checked before a single file is written. Fails fast."""
    import importlib.util

    from vocoders.registry import (
        ALL_CONDITIONS,
        SHARED_ENV_REPORTED,
        checkpoint_audit,
        get_spec,
        get_vocoder,
    )

    conditions = list(conditions)
    if not conditions:
        raise InvariantViolation("Phase A: no conditions requested.")
    if len(set(conditions)) != len(conditions):
        raise InvariantViolation(f"Phase A: duplicate conditions in {conditions}.")
    unknown = [c for c in conditions if c not in ALL_CONDITIONS]
    if unknown:
        raise InvariantViolation(
            f"Phase A: {unknown} are not generable conditions ({sorted(ALL_CONDITIONS)}). "
            f"'{REAL_CONDITION}' is always generated and is not requested."
        )
    if n_utterances < ALIGNMENT_PROBE_FILES:
        raise InvariantViolation(
            f"INV-16: {n_utterances} utterances cannot run the alignment gate, which "
            f"needs {ALIGNMENT_PROBE_FILES}."
        )

    root = Path(corpus_root)
    if not (root / "metadata.csv").is_file() or not (root / "wavs").is_dir():
        raise InvariantViolation(
            f"Phase A: {root} is not an LJSpeech root (metadata.csv and wavs/ expected)."
        )

    out_dir = Path(out_dir)
    for path in (out_dir, _sibling(out_dir, ".staging"), _sibling(out_dir, ".failed")):
        if path.exists():
            raise InvariantViolation(
                f"Phase A: {path} already exists. A dataset with two runs in it is not a "
                "dataset; choose a new output directory or remove the old one "
                "deliberately."
            )

    blocked = {
        r["condition"]: r["missing_pins"] or r["pin"]
        for r in checkpoint_audit()
        if r["condition"] in conditions and r["blocked"]
    }
    if blocked:
        raise InvariantViolation(
            f"INV-08: {blocked} are blocked by the checkpoint audit (missing pins, an "
            "unpinned checkpoint, or an unverified mel config). Run `btpvf audit`."
        )

    unpinned = {
        c: [f for f in type(get_vocoder(c)).required_pins if not getattr(get_spec(c), f)]
        for c in conditions
    }
    unpinned = {c: f for c, f in unpinned.items() if f}
    if unpinned:
        raise InvariantViolation(
            f"INV-08: {unpinned} -- these adapters refuse to load without those pins. "
            "The audit may still list them as runnable; loading is stricter."
        )

    # INV-13, stated rather than assumed. More than one condition with its own
    # dependency tree may share a process only if every one of them is in the
    # set reported to have coexisted in testing.
    neural = [c for c in conditions if get_spec(c).family != "signal_processing"]
    if len(neural) > 1 and not set(neural) <= SHARED_ENV_REPORTED:
        raise InvariantViolation(
            f"INV-13: {neural} would share one environment, and only "
            f"{sorted(SHARED_ENV_REPORTED)} are reported to coexist. Run the others in "
            "their own sessions and merge manifests afterwards."
        )

    missing = sorted({
        m
        for c in conditions
        for m in type(get_vocoder(c)).runtime_modules
        if importlib.util.find_spec(m) is None
    })
    if missing:
        raise InvariantViolation(
            f"INV-13: {conditions} need {missing}, which are not importable here. They "
            "are deliberately not dependencies of this package; install them in the "
            "condition's environment (pip install -e .[resynth])."
        )


def band_gate(manifest: pd.DataFrame, n_utterances: int = BAND_GATE_UTTERANCES) -> dict:
    """INV-17 on real output: the analysis band limit must hold on what was written.

    Applies the sanctioned comparison-set transform, real included, to the first
    ``n_utterances`` archive utterances read back from their PCM_16 files. The
    transform gates its own output (check_band_limit) and raises on a residual
    above the allowance.
    """
    archive = select_tier(manifest, ARCHIVE_TIER)
    utts = sorted(archive["utt_id"].unique())[:n_utterances]
    worst = 0.0
    for utt in utts:
        rows = archive[archive["utt_id"] == utt]
        wavs = {
            str(c): read_processed(p, ARCHIVE_SR)
            for c, p in zip(rows["condition"], rows["path"], strict=True)
        }
        limited = band_limit_comparison_set(wavs, sr=ARCHIVE_SR, verify=True)
        worst = max(
            worst, max(out_of_band_energy(w, ARCHIVE_SR, LADDER_FMAX) for w in limited.values())
        )
    return {
        "utterances": len(utts),
        "worst_residual_above_ladder_fmax": worst,
        "allowance": BAND_LIMIT_MAX_STOPBAND_ENERGY,
    }


def run_gated(
    corpus_root: str | Path,
    conditions: list[str],
    out_dir: str | Path,
    n_utterances: int,
    *,
    device: str | None = None,
    keep_failed: bool = False,
) -> GatedRunResult:
    """Phase A on a real corpus with every gate, and output only if all of them pass.

    Gates, in order:

      preflight     conditions generable and pinned (INV-08), shared-environment
                    claim within what was reported (INV-13), runtime modules
                    importable, output location unused.
      alignment     INV-16, per condition, inside generation, before its files
                    are accepted.
      manifest      validate_manifest with the zero-shot tier required: rates,
                    derivations, pairing, trim spans, splits, full-band archive,
                    the loudness record, alignment records.
      band          the analysis band limit holds on the written audio (INV-17).
      sanity        every INV-12 probe on every condition and tier, over ALL
                    utterances rather than the eval split: these probes train
                    nothing, and at a few hundred utterances the eval split alone
                    is small enough to trip the floor by chance.

    Then the loudness report is written (a report, not a gate), the manifest's
    paths are pointed at the final location, and staging is renamed into place.
    On any failure staging is deleted -- or, with ``keep_failed``, renamed to
    ``<out>.failed`` for inspection -- and the error propagates.
    """
    from data.alignment import read_alignment_report
    from data.corpus import LJSpeech
    from data.manifest import loudness_gain_report, read_manifest, validate_manifest
    from experiments.sanity_checks import run_all

    out_dir = Path(out_dir)
    preflight(corpus_root, conditions, out_dir, n_utterances)
    staging = _sibling(out_dir, ".staging")
    staging.mkdir(parents=True)
    gates: dict = {"preflight": "passed"}
    try:
        corpus = LJSpeech(corpus_root, subset_size=n_utterances)
        staged_manifest = run(
            corpus, list(conditions), staging, staging / MANIFEST_NAME, device=device
        )
        gates["alignment"] = read_alignment_report(staged_manifest.with_suffix(".alignment.json"))

        df = read_manifest(staged_manifest)
        validate_manifest(df, require_zeroshot=True)
        gates["manifest"] = "passed"

        gates["band"] = band_gate(df)

        sanity = run_all(df, split=None)
        sanity.to_csv(staging / "sanity.csv", index=False)
        leaked = sanity[sanity["leaked"]]
        gates["sanity"] = {
            "split": "all",
            "probes": len(sanity),
            "leaked": len(leaked),
            "expected_by_design": int(sanity["expected_by_design"].sum()),
        }
        if not leaked.empty:
            raise InvariantViolation(
                "INV-12: confound leak -- fix the pipeline and regenerate; never relax "
                f"the floor.\n{leaked.to_string(index=False)}"
            )

        loudness = loudness_gain_report(df)
        loudness.to_csv(staging / "loudness_gain.csv", index=False)
        gates["loudness_report"] = "loudness_gain.csv"

        # Point the manifest at where the files WILL be, then move them there.
        df["path"] = [str(out_dir / Path(p).relative_to(staging)) for p in df["path"]]
        df.to_csv(staged_manifest, index=False)
        (staging / "gates.json").write_text(
            json.dumps(gates, indent=2, sort_keys=True, default=str), encoding="utf-8"
        )
    except BaseException:
        if keep_failed:
            failed = _sibling(out_dir, ".failed")
            staging.rename(failed)
            print(f"[Phase A] FAILED -- partial output kept for inspection at {failed}")
        else:
            shutil.rmtree(staging, ignore_errors=True)
            print("[Phase A] FAILED -- no output written")
        raise

    staging.rename(out_dir)
    manifest_path = out_dir / MANIFEST_NAME
    final = read_manifest(manifest_path)
    absent = [p for p in final["path"] if not Path(p).is_file()]
    if absent:
        raise InvariantViolation(
            f"Phase A: {len(absent)} manifest paths do not exist after the move "
            f"(e.g. {absent[:3]}). The output directory is not a usable dataset."
        )
    return GatedRunResult(out_dir, manifest_path, sanity, loudness, gates)


def _environment() -> dict:
    """INV-13: the environment each condition ran in, as a recorded fact."""
    import platform
    import sys
    from importlib import metadata

    packages = {}
    for name in ("numpy", "scipy", "librosa", "soundfile", "pyloudnorm", "pandas",
                 "torch", "torchaudio", "vocos", "huggingface_hub"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    env: dict = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
    }
    torch = sys.modules.get("torch")
    if torch is not None:
        env["cuda_available"] = bool(torch.cuda.is_available())
        if env["cuda_available"]:
            env["cuda_device"] = torch.cuda.get_device_name(0)
    return env
