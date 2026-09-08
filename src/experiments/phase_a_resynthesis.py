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

import subprocess
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
from tqdm import tqdm

from data.alignment import AlignmentReport, assert_aligned, check_vocoder_alignment
from data.audio_io import read_native, write_audio
from data.corpus import Corpus, assign_splits
from data.invariants import (
    ALIGNMENT_PROBE_FILES,
    ARCHIVE_SR,
    ARCHIVE_TIER,
    GLOBAL_SEED,
    LADDER_FMAX,
    REAL_CONDITION,
    ZEROSHOT_SR,
    ZEROSHOT_TIER,
    InvariantViolation,
    exclusion_reason,
    is_correlation_excluded,
)
from data.manifest import ManifestRow, dump_provenance, write_manifest
from data.preprocess import (
    TrimSpan,
    archive_band_spec,
    derive_trim_span,
    derive_zeroshot,
    finalise_reference,
    process_condition_output,
    resample_once,
    would_clip,
)
from vocoders.registry import get_spec, get_vocoder


def _emit_pair(
    wav_archive: np.ndarray,
    measured_archive: float,
    out_dir: Path,
    condition: str,
    utt_id: str,
    zeroshot_meter: pyln.Meter,
) -> tuple[Path, Path, float, int]:
    """Write the archive artifact, then derive and write the zero-shot artifact."""
    archive_path = out_dir / ARCHIVE_TIER / condition / f"{utt_id}.wav"
    write_audio(archive_path, wav_archive, ARCHIVE_SR, measured_lufs=measured_archive)

    wav_zs, measured_zs = derive_zeroshot(
        wav_archive, archive_lufs=measured_archive, meter=zeroshot_meter
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
) -> list[ManifestRow]:
    """One archive row plus its derived zero-shot row, both fully provenanced."""
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
        norm, measured, high_band = finalise_reference(wav, span, meter=archive_meter)

        if would_clip(norm):
            dropped.add(utt.utt_id)
            continue

        archive_path, zs_path, lufs_zs, n_zs = _emit_pair(
            norm, measured, out_dir, REAL_CONDITION, utt.utt_id, zs_meter
        )
        spans[utt.utt_id] = span
        real_archive[utt.utt_id] = norm
        rows.extend(
            _rows_for(
                utt.utt_id, REAL_CONDITION, utt.speaker, splits[utt.utt_id], span,
                archive_path, zs_path, len(norm), n_zs, measured, lufs_zs, sr_native,
                high_band=high_band,
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
) -> tuple[list[ManifestRow], AlignmentReport]:
    """Resynthesize one condition at both tiers, reusing the reference trim spans.

    The INV-16 alignment probe runs on the first ``ALIGNMENT_PROBE_FILES``
    processed archive outputs. It gates the condition: a fixed vocoder delay
    would make the reference trim span land on the wrong span of speech and
    would register as an artifact on both axes at once.
    """
    spec = get_spec(condition)
    voc = get_vocoder(condition)
    voc.load()
    archive_meter = pyln.Meter(ARCHIVE_SR)
    zs_meter = pyln.Meter(ZEROSHOT_SR)

    if is_correlation_excluded(condition):
        print(f"[INV-17] '{condition}': {exclusion_reason(condition)}")
    staged: list[tuple[str, np.ndarray, float, int, str, float | None]] = []
    probe_pairs: list[tuple[np.ndarray, np.ndarray]] = []

    for utt in tqdm(corpus.utterances(), desc=condition):
        if utt.utt_id in dropped or utt.utt_id not in spans:
            continue

        src, sr_src = read_native(utt.path)
        # Source goes to the VOCODER's training rate for analysis. That is the
        # input-side path; the single sanctioned output resample (native ->
        # archive) happens below, in process_condition_output (INV-01).
        if sr_src != spec.mel.sample_rate:
            import librosa

            src = librosa.resample(src, orig_sr=sr_src, target_sr=spec.mel.sample_rate)

        wav_native = voc.resynthesize(src, spec.mel.sample_rate)
        wav, measured, high_band = process_condition_output(
            wav_native,
            spec.mel.sample_rate,
            spans[utt.utt_id],
            condition=condition,
            meter=archive_meter,
        )
        if would_clip(wav):
            raise InvariantViolation(
                f"INV-04: '{condition}' clips on {utt.utt_id} where real did not. "
                "Add the utterance to the global drop list and regenerate every "
                "condition without it -- do not attenuate this one file."
            )
        staged.append((utt.utt_id, wav, measured, len(wav), utt.speaker, high_band))
        if len(probe_pairs) < ALIGNMENT_PROBE_FILES:
            probe_pairs.append((real_archive[utt.utt_id], wav))

    voc.unload()

    # INV-16 gate, before a single file of this condition is accepted.
    report = check_vocoder_alignment(probe_pairs, condition, sample_rate=ARCHIVE_SR)
    assert_aligned(report)

    rows: list[ManifestRow] = []
    for utt_id, wav, measured, n_archive, speaker, high_band in staged:
        archive_path, zs_path, lufs_zs, n_zs = _emit_pair(
            wav, measured, out_dir, condition, utt_id, zs_meter
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
            cond, corpus, out_dir, spans, real_archive, splits, dropped
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
            "correlation_excluded": [
                c for c in conditions if is_correlation_excluded(c)
            ],
            "n_utterances": len(spans),
            "dropped_for_peak": sorted(dropped),
            "corpus": corpus.name,
            "alignment": {r.condition: r.peak_lag for r in reports},
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
