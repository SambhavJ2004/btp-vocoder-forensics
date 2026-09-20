"""Confound-leakage sanity checks — the gate before any result is believed.

From the plan's risk table: "sanity check that a detector cannot separate
conditions on silence or bandwidth alone". These checks are cheap, they run on
CPU, and they are the difference between a defensible result and a retracted
one. Run them on every dataset version before B or C consume it.

Each check trains a deliberately weak classifier on a feature that carries NO
vocoder artifact. If such a classifier separates real from vocoded, the
separation is a confound, not a finding.

Probes run per tier (INV-01) and the tier is reported with every row. The
bandwidth probe is ARCHIVE-only: on the derived 16 kHz set every condition has
passed through the same downsampler, so the probe would report a reassuring 0.5
while saying nothing about the archive the study actually measures.

The silence probe's threshold is DERIVED from the corpus, per tier, and no
caller may supply a constant: see :func:`silence_reference`. INV-03 defines
silence relatively and this probe compares |sample| to an absolute level, so the
two agree only if the level is placed against the corpus's measured floor.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from data.audio_io import read_processed
from data.invariants import (
    ARCHIVE_SR,
    ARCHIVE_TIER,
    HIGH_BAND_ANALYSIS,
    REAL_CONDITION,
    SANITY_EER_FLOOR,
    SILENCE_FLOOR_FRAME_PERCENTILE,
    SILENCE_FLOOR_MARGIN_DB,
    SILENCE_FLOOR_PEAK_PERCENTILE,
    SILENCE_PAD_AGREEMENT,
    TIER_RATES,
    TIERS,
    TRIM_FRAME_LENGTH,
    TRIM_HOP_LENGTH,
    TRIM_PAD_MS,
    TRIM_TOP_DB,
    InvariantViolation,
    is_excluded,
)
from data.manifest import select_tier
from detectors.eer import compute_eer

# Below one least-significant bit of PCM_16 nothing is recorded, so such frames
# carry no information about the corpus's noise floor. Excluding them stops a
# run of digital zeros -- which is not a noise floor, it is the absence of one --
# from dragging the floor estimate to the quantiser and the threshold with it.
_PCM16_LSB = 1.0 / 32768.0


def _leading_trailing_silence(
    wav: np.ndarray, sr: int, *, threshold_db: float
) -> tuple[float, float]:
    """Seconds of silence at each end, at an EXPLICIT threshold.

    ``threshold_db`` has no default on purpose. It used to default to -50.0 dBFS,
    a value that appeared in no constants block and had no recorded derivation,
    and that on LJSpeech sat ~3 dB inside the corpus's own noise floor. A default
    here is how that went unexamined: every caller got a number without ever
    choosing one. Callers now take it from :func:`silence_reference`.
    """
    amp = np.abs(wav)
    thr = 10 ** (threshold_db / 20.0)
    voiced = np.where(amp > thr)[0]
    if voiced.size == 0:
        return len(wav) / sr, 0.0
    return voiced[0] / sr, (len(wav) - voiced[-1]) / sr


def _frame_rms(wav: np.ndarray, sr: int = ARCHIVE_SR) -> np.ndarray:
    """Frame RMS on INV-03's own frame/hop, so the floor and the trimmer agree."""
    import librosa

    return librosa.feature.rms(
        y=wav, frame_length=TRIM_FRAME_LENGTH, hop_length=TRIM_HOP_LENGTH, center=True
    )[0]


def _quiet_mask(wav: np.ndarray, sr: int = ARCHIVE_SR) -> np.ndarray:
    """Samples in the quietest ``SILENCE_FLOOR_FRAME_PERCENTILE`` of the file's frames.

    The quiet END of the level distribution, which is where a noise floor lives.
    Deliberately NOT :func:`reference_silent_mask`: that uses INV-03's relative
    criterion, ``TRIM_TOP_DB`` below the loudest frame, which on LJSpeech admits
    frames at -53.5 dBFS -- quiet speech, not floor. Calibrating on the loudest
    samples of a "quiet" band puts the threshold near the top of it.
    """
    rms = _frame_rms(wav, sr)
    recorded = rms > _PCM16_LSB
    if not recorded.any():
        return np.zeros(len(wav), dtype=bool)
    cut = float(np.percentile(rms[recorded], SILENCE_FLOOR_FRAME_PERCENTILE))
    quiet = recorded & (rms <= cut)
    mask = np.repeat(quiet, TRIM_HOP_LENGTH)[: len(wav)]
    if len(mask) < len(wav):
        mask = np.pad(mask, (0, len(wav) - len(mask)), constant_values=False)
    return mask


def silence_floor_dbfs(wavs: list[np.ndarray], sr: int = ARCHIVE_SR) -> tuple[float, float]:
    """The corpus's noise floor, as the probe measures levels: a SAMPLE PEAK.

    Returns ``(floor_peak_dbfs, frame_rms_percentile_dbfs)`` -- the pooled
    ``SILENCE_FLOOR_PEAK_PERCENTILE`` of |sample| inside every file's quietest
    frames, and the median across files of those frames' RMS cut. The second is
    reported, not used: it is what makes the first reproducible from
    ``btpvf silence-diag`` output rather than from a separate calculation.

    Peak against peak is the point. The probe thresholds |sample|; a floor given
    as an RMS would leave the margin absorbing a crest factor of sqrt(2 ln N),
    which scales with the region length and the sample rate, so the same constant
    would silently mean something else on another corpus or tier.
    """
    peaks: list[np.ndarray] = []
    cuts: list[float] = []
    for wav in wavs:
        mask = _quiet_mask(wav, sr)
        if not mask.any():
            continue
        peaks.append(np.abs(wav[mask].astype(np.float64)))
        rms = _frame_rms(wav, sr)
        recorded = rms > _PCM16_LSB
        cuts.append(float(np.percentile(rms[recorded], SILENCE_FLOOR_FRAME_PERCENTILE)))
    if not peaks:
        raise InvariantViolation(
            "INV-12: no frame of the real reference carries signal above the PCM_16 "
            "quantisation floor, so the corpus has no measurable noise floor and the "
            "silence threshold cannot be derived. Check the audio before the probe."
        )
    pooled = np.concatenate(peaks)
    floor = float(np.percentile(pooled, SILENCE_FLOOR_PEAK_PERCENTILE))
    return _dbfs_amp(floor), _dbfs_amp(float(np.median(cuts)))


def _dbfs_amp(amplitude: float) -> float:
    return float(20.0 * np.log10(max(amplitude, 1e-12)))


def _tier_rows(
    manifest: pd.DataFrame, tier: str, split: str | None
) -> tuple[pd.DataFrame, int]:
    """``split=None`` means every split. The probes train nothing -- each is a
    scalar feature scored directly -- so there is no held-out set to protect,
    and using every utterance is what keeps the EER's sampling noise below the
    floor at small dataset sizes."""
    sub = select_tier(manifest, tier)
    return (sub if split is None else sub[sub["split"] == split]), TIER_RATES[tier]


def _pad_level_dbfs(wav: np.ndarray, sr: int) -> float:
    """Level of the trim pad -- the TRIM_PAD_MS of context INV-03 keeps at each end.

    Not a noise floor: the pad holds the onset ramp, which is why it sits above
    the file's silent regions (LJSpeech: -47.3 dBFS against -53.5).
    """
    pad = int(round(TRIM_PAD_MS * 1e-3 * sr))
    return _dbfs(np.concatenate([wav[:pad], wav[-pad:]]))


@dataclass(frozen=True)
class SilenceReference:
    """The silence probe's threshold for one (dataset, tier), derived from real.

    Every field is reported with the EER so that a silence result can never be
    read without the threshold it was measured at.
    """

    tier: str
    n_files: int
    frame_rms_percentile_dbfs: float  # the quiet-frame cut, for reproducing the floor
    floor_dbfs: float                 # sample-peak floor of real's quietest frames
    threshold_dbfs: float             # floor + SILENCE_FLOOR_MARGIN_DB
    pad_expected_s: float             # 2 * TRIM_PAD_MS, fixed by INV-03
    pad_measured_s: float             # what real actually reads at the threshold
    pad_level_dbfs: float             # real's trim-pad level, for the ungated diagnostic
    real_feature: dict[str, float] = field(repr=False, default_factory=dict)

    @property
    def pad_error(self) -> float:
        return abs(self.pad_measured_s - self.pad_expected_s) / self.pad_expected_s

    @property
    def ok(self) -> bool:
        return self.pad_error <= SILENCE_PAD_AGREEMENT

    def as_record(self) -> dict[str, object]:
        """Everything needed to reproduce the threshold, for the sidecar and the CSV."""
        return {
            "tier": self.tier,
            "n_real_files": self.n_files,
            "frame_percentile": SILENCE_FLOOR_FRAME_PERCENTILE,
            "peak_percentile": SILENCE_FLOOR_PEAK_PERCENTILE,
            "quiet_frame_rms_cut_dbfs": round(self.frame_rms_percentile_dbfs, 3),
            "floor_dbfs": round(self.floor_dbfs, 3),
            "margin_db": SILENCE_FLOOR_MARGIN_DB,
            "threshold_dbfs": round(self.threshold_dbfs, 3),
            "pad_expected_s": round(self.pad_expected_s, 6),
            "pad_measured_s": round(self.pad_measured_s, 6),
            "pad_error": round(self.pad_error, 4),
            "pad_tolerance": SILENCE_PAD_AGREEMENT,
            "pad_agreement_ok": self.ok,
            "real_trim_pad_dbfs": round(self.pad_level_dbfs, 3),
        }


def silence_reference(
    manifest: pd.DataFrame,
    tier: str = ARCHIVE_TIER,
    split: str | None = "eval",
    *,
    strict: bool = True,
) -> SilenceReference:
    """Derive the silence threshold from the REAL reference of this dataset.

    ``threshold = floor + SILENCE_FLOOR_MARGIN_DB``, with the floor measured by
    :func:`silence_floor_dbfs`. Real alone, never the conditions: a threshold
    that saw the conditions would be fitted to the answer.

    **The pad-agreement check raises.** INV-03 writes exactly ``TRIM_PAD_MS`` of
    context at each end, so at a threshold that is reading silence the real
    reference must read ``2 * TRIM_PAD_MS``. Outside ``SILENCE_PAD_AGREEMENT``
    the threshold is not reading silence -- too low and it is counting the
    corpus's own noise as signal, too high and it is eating speech onsets -- and
    the EER it would produce is a number nobody can interpret. That is worse than
    no number, so the probe stops instead of reporting it.

    ``strict=False`` returns the same record unraised, for
    :func:`silence_diagnostic`: diagnosing a dataset whose threshold does not
    agree with its pad is exactly what the diagnostic is for.
    """
    sub, sr = _tier_rows(manifest, tier, split)
    rows = sub[sub["condition"] == REAL_CONDITION]
    if rows.empty:
        raise InvariantViolation(
            f"INV-12: tier '{tier}' has no real rows, so the silence threshold cannot "
            "be derived. It is defined against the real reference's own noise floor."
        )
    wavs = {str(u): read_processed(p, sr) for u, p in zip(rows["utt_id"], rows["path"],
                                                         strict=True)}
    floor_dbfs, frame_cut_dbfs = silence_floor_dbfs(list(wavs.values()), sr)
    threshold = floor_dbfs + SILENCE_FLOOR_MARGIN_DB

    feature = {
        utt: sum(_leading_trailing_silence(wav, sr, threshold_db=threshold))
        for utt, wav in wavs.items()
    }
    pad_expected = 2.0 * TRIM_PAD_MS * 1e-3
    ref = SilenceReference(
        tier=tier,
        n_files=len(wavs),
        frame_rms_percentile_dbfs=frame_cut_dbfs,
        floor_dbfs=floor_dbfs,
        threshold_dbfs=threshold,
        pad_expected_s=pad_expected,
        pad_measured_s=float(np.mean(list(feature.values()))),
        pad_level_dbfs=float(np.median([_pad_level_dbfs(w, sr) for w in wavs.values()])),
        real_feature=feature,
    )
    if strict and not ref.ok:
        raise InvariantViolation(
            f"INV-03/INV-12: the derived silence threshold does not read silence on "
            f"tier '{tier}'. Real reads {ref.pad_measured_s * 1e3:.1f} ms of leading + "
            f"trailing silence at {threshold:.1f} dBFS, against the "
            f"{pad_expected * 1e3:.1f} ms that INV-03's TRIM_PAD_MS fixes "
            f"({ref.pad_error:.0%} off, tolerance {SILENCE_PAD_AGREEMENT:.0%}). "
            f"Measured floor {floor_dbfs:.1f} dBFS (quiet-frame RMS cut "
            f"{frame_cut_dbfs:.1f} dBFS), margin {SILENCE_FLOOR_MARGIN_DB} dB. "
            "Below the pad means the threshold is counting the corpus's own noise as "
            "signal; above it means it is eating speech onsets. Either way the silence "
            "EER would not be a silence measurement. Diagnose with `btpvf silence-diag` "
            "-- do not widen the tolerance to get a number."
        )
    return ref


def silence_probe(
    manifest: pd.DataFrame, condition: str, split: str | None = "eval",
    tier: str = ARCHIVE_TIER, reference: SilenceReference | None = None,
) -> dict[str, object]:
    """INV-03 check. Can leading/trailing silence duration alone separate the
    condition from real? It must not: identical trim spans are applied to both.

    The threshold comes from :func:`silence_reference`, derived from real's own
    noise floor, and is reported on the row. Also reported, and NOT gated:
    ``trim_pad_above_real_db``, this condition's trim-pad level against real's.
    """
    sub, sr = _tier_rows(manifest, tier, split)
    ref = reference or silence_reference(manifest, tier, split)

    real_rows = sub[sub["condition"] == REAL_CONDITION]
    real_feature = np.asarray([ref.real_feature[str(u)] for u in real_rows["utt_id"]])

    rows = sub[sub["condition"] == condition]
    values, pads = [], []
    for p in rows["path"]:
        wav = read_processed(p, sr)
        values.append(sum(_leading_trailing_silence(wav, sr, threshold_db=ref.threshold_dbfs)))
        pads.append(_pad_level_dbfs(wav, sr))

    eer, _ = compute_eer(real_feature, np.asarray(values))
    verdict = _verdict("silence_duration", condition, eer, tier)
    verdict["silence_s_mean"] = float(np.mean(values))
    # INV-12 records this; nothing gates it. See `run_all` and docs/measurements.md.
    verdict["trim_pad_above_real_db"] = float(np.median(pads)) - ref.pad_level_dbfs
    return verdict


def loudness_probe(
    manifest: pd.DataFrame, condition: str, split: str | None = "eval",
    tier: str = ARCHIVE_TIER,
) -> dict[str, object]:
    """INV-04 check. Can broadband RMS alone separate the condition from real?

    Worth running on the derived tier too: the zero-shot set is deliberately not
    re-normalised, so this is the probe that would catch a condition whose
    high-band energy differed enough to shift its post-downsample level into a
    learnable cue.
    """
    sub, sr = _tier_rows(manifest, tier, split)
    feats: dict[str, np.ndarray] = {}
    for name in (REAL_CONDITION, condition):
        rows = sub[sub["condition"] == name]
        feats[name] = np.asarray(
            [float(np.sqrt(np.mean(read_processed(p, sr) ** 2))) for p in rows["path"]]
        )

    eer, _ = compute_eer(feats[REAL_CONDITION], feats[condition])
    return _verdict("rms_level", condition, eer, tier)


def duration_probe(
    manifest: pd.DataFrame, condition: str, split: str | None = "eval",
    tier: str = ARCHIVE_TIER,
) -> dict[str, object]:
    """INV-06 check. Can total duration alone separate the condition from real?
    Should be exactly at chance -- paired conditions have identical lengths.
    """
    sub, _ = _tier_rows(manifest, tier, split)
    real = sub[sub["condition"] == REAL_CONDITION]["duration_s"].to_numpy()
    other = sub[sub["condition"] == condition]["duration_s"].to_numpy()
    eer, _ = compute_eer(real, other)
    return _verdict("duration", condition, eer, tier)


def bandwidth_probe(
    manifest: pd.DataFrame, condition: str, split: str | None = "eval",
    tier: str = ARCHIVE_TIER,
) -> dict[str, object]:
    """INV-01 check. Can the highest frequency with meaningful energy separate
    the condition from real? A hit here usually means a resampling difference
    left a distinct anti-aliasing rolloff on one condition.

    ARCHIVE tier only. On the derived set every condition passed through the same
    downsampler, so the probe would measure that shared filter rather than the
    conditions and would report a falsely reassuring 0.5.

    Note this probe is expected to be less clean than the others: a genuine
    high-band vocoder artifact also moves effective bandwidth. Read a hit as
    "inspect the spectra", not automatically as "confound".
    """
    import librosa

    if tier != ARCHIVE_TIER:
        raise InvariantViolation(
            f"INV-01: the bandwidth probe is archive-only; '{tier}' has been "
            "low-passed by the shared downsampler and would always look clean."
        )
    sub, sr = _tier_rows(manifest, tier, split)
    feats: dict[str, np.ndarray] = {}
    for name in (REAL_CONDITION, condition):
        rows = sub[sub["condition"] == name]
        vals = []
        for p in rows["path"]:
            wav = read_processed(p, sr)
            rolloff = librosa.feature.spectral_rolloff(y=wav, sr=sr, roll_percent=0.995)
            vals.append(float(rolloff.mean()))
        feats[name] = np.asarray(vals)

    eer, _ = compute_eer(feats[REAL_CONDITION], feats[condition])
    return _verdict("bandwidth_rolloff", condition, eer, tier)


def _verdict(probe: str, condition: str, eer: float, tier: str) -> dict[str, object]:
    """One probe result.

    Some conditions are expected to separate on bandwidth *by design*, and
    flagging them as leaks would train the team to ignore the column. Those are
    exactly the conditions excluded from the high-band analysis (INV-18) --
    ``griffin_lim`` (mel-inversion cliff) and ``vocos`` (resampler edge) -- plus
    the paired control, trained at a wider fmax. They are reported as
    ``expected_by_design`` instead: visible, not a blocker.

    Every OTHER condition firing this probe is a genuine finding now that the
    archive is full-band: a time-domain vocoder should track real in the high
    band, and one that does not is telling you something.
    """
    from vocoders.registry import CONTROL_CONDITIONS

    by_design = probe == "bandwidth_rolloff" and (
        is_excluded(HIGH_BAND_ANALYSIS, condition) or condition in CONTROL_CONDITIONS
    )
    return {
        "probe": probe,
        "condition": condition,
        "tier": tier,
        "eer": eer,
        "leaked": bool(eer < SANITY_EER_FLOOR) and not by_design,
        "expected_by_design": by_design,
        "floor": SANITY_EER_FLOOR,
    }


def run_all(manifest: pd.DataFrame, split: str | None = "eval") -> pd.DataFrame:
    """Every probe against every non-real condition, on every tier present.

    A row with ``leaked == True`` blocks publication of that condition. The fix
    is to correct the pipeline and regenerate, never to relax SANITY_EER_FLOOR.

    Every row carries the silence threshold and the floor it was derived from,
    for its tier, so no EER in this table can be read without the calibration
    behind it. The threshold is derived ONCE per tier and shared by every
    condition -- deriving it per condition would give each a different ruler.

    ``trim_pad_above_real_db`` is **recorded and ungated**: it is a level
    difference in the 40 ms trim pad, which the derived threshold deliberately
    sits above, and which ``rms_level`` cannot see because whole-file RMS is
    dominated by speech. There is no basis in this repository for choosing a
    floor for it, and a gate without a justified floor is the failure this whole
    change is about. See docs/measurements.md for what gating it would need.
    """
    conditions = [c for c in manifest["condition"].unique() if c != REAL_CONDITION]
    tiers = [t for t in TIERS if t in set(manifest["tier"])]
    out: list[dict[str, object]] = []
    for tier in tiers:
        ref = silence_reference(manifest, tier, split)
        calibration = {
            "silence_threshold_dbfs": ref.threshold_dbfs,
            "silence_floor_dbfs": ref.floor_dbfs,
            "silence_frame_rms_cut_dbfs": ref.frame_rms_percentile_dbfs,
            "real_pad_s": ref.pad_measured_s,
            "real_trim_pad_dbfs": ref.pad_level_dbfs,
        }
        for cond in conditions:
            rows = [
                duration_probe(manifest, cond, split, tier),
                silence_probe(manifest, cond, split, tier, reference=ref),
                loudness_probe(manifest, cond, split, tier),
            ]
            if tier == ARCHIVE_TIER:
                rows.append(bandwidth_probe(manifest, cond, split, tier))
            for row in rows:
                row.update(calibration)
            out.extend(rows)
    return pd.DataFrame(out)


# --- Diagnosing a silence-probe hit ------------------------------------------


SILENCE_DIAGNOSTIC_THRESHOLDS: tuple[float, ...] = (-60.0, -50.0, -40.0, -30.0, -20.0)


def reference_silent_mask(real: np.ndarray, sr: int = ARCHIVE_SR) -> np.ndarray:
    """Samples the REAL reference considers silent, by INV-03's own criterion.

    Frames more than ``TRIM_TOP_DB`` below the file's loudest frame -- the rule
    ``librosa.effects.trim`` applies, evaluated over the whole file instead of
    only its ends. Derived from real and applied by index to every condition, so
    every condition is measured over exactly the same samples (INV-03).
    """
    import librosa

    rms = librosa.feature.rms(
        y=real, frame_length=TRIM_FRAME_LENGTH, hop_length=TRIM_HOP_LENGTH, center=True
    )[0]
    db = librosa.amplitude_to_db(rms, ref=np.max)
    silent_frames = db < -TRIM_TOP_DB
    mask = np.repeat(silent_frames, TRIM_HOP_LENGTH)[: len(real)]
    if len(mask) < len(real):
        mask = np.pad(mask, (0, len(real) - len(mask)), constant_values=False)
    return mask


def _dbfs(x: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    return float(20.0 * np.log10(max(float(np.sqrt(np.mean(x.astype(np.float64) ** 2))), 1e-12)))


def _quiet_stats(wav: np.ndarray, sr: int) -> tuple[float, float]:
    """One file's quiet-frame RMS cut and the peak level inside those frames.

    The two numbers the derived threshold is built from, per file. Reported by
    the diagnostic for every condition so the gate's calibration is reproducible
    from its output rather than from a separate calculation; the gate itself
    pools across real's files instead of taking a median of per-file values.
    """
    rms = _frame_rms(wav, sr)
    recorded = rms > _PCM16_LSB
    if not recorded.any():
        return float("nan"), float("nan")
    cut = float(np.percentile(rms[recorded], SILENCE_FLOOR_FRAME_PERCENTILE))
    mask = _quiet_mask(wav, sr)
    if not mask.any():
        return _dbfs_amp(cut), float("nan")
    peak = float(np.percentile(np.abs(wav[mask].astype(np.float64)),
                               SILENCE_FLOOR_PEAK_PERCENTILE))
    return _dbfs_amp(cut), _dbfs_amp(peak)


def silence_diagnostic(
    manifest: pd.DataFrame,
    *,
    tier: str = ARCHIVE_TIER,
    split: str | None = None,
    thresholds: tuple[float, ...] = SILENCE_DIAGNOSTIC_THRESHOLDS,
    limit: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, SilenceReference]:
    """Why a condition separates on silence duration. Diagnosis, not a gate.

    Returns three things:

      floors    per condition: the level inside the REAL reference's silent
                regions, the level in the trim pad, and the two quiet-frame
                statistics the threshold is derived from
                (``quiet_frame_rms_dbfs``, ``quiet_peak_dbfs``). A vocoder that
                reconstructs noise where the reference has near-silence reads
                higher here than real does.
      sweep     per condition and threshold: the silence probe's own feature
                (leading + trailing silence duration) recomputed at that
                threshold, and its EER against real. The derived threshold is
                swept alongside the fixed points and flagged ``gate_threshold``,
                so what the gate will read is visible next to what it would read
                elsewhere.
      reference the gate's own calibration for this dataset, derived NON-STRICTLY
                -- a dataset whose threshold disagrees with its pad is exactly
                what this function is for, so it reports the disagreement where
                :func:`silence_probe` raises on it.

    Nothing here decides anything. INV-12 says a leak blocks the condition, and
    whether a hit is a vocoder artifact or a pipeline fault is a judgement for a
    human to make on these numbers.

    ``limit`` caps the per-utterance tables only. The reference is always derived
    from every real file of the tier, because the threshold is a property of the
    dataset and a diagnostic that reported a different one would be describing a
    gate that does not exist.
    """
    sub, sr = _tier_rows(manifest, tier, split)
    reference = silence_reference(manifest, tier, split, strict=False)

    utts = sorted(sub[sub["condition"] == REAL_CONDITION]["utt_id"].unique())
    if limit is not None:
        utts = utts[:limit]
    conditions = [c for c in sorted(sub["condition"].unique()) if c != REAL_CONDITION]
    paths = {
        (str(r.condition), str(r.utt_id)): r.path
        for r in sub.itertuples()
        if str(r.utt_id) in set(utts)
    }
    swept = (*thresholds, reference.threshold_dbfs)

    floors: dict[str, list[float]] = {c: [] for c in (REAL_CONDITION, *conditions)}
    pads: dict[str, list[float]] = {c: [] for c in (REAL_CONDITION, *conditions)}
    cuts: dict[str, list[float]] = {c: [] for c in (REAL_CONDITION, *conditions)}
    peaks: dict[str, list[float]] = {c: [] for c in (REAL_CONDITION, *conditions)}
    silent_fraction: list[float] = []
    features: dict[float, dict[str, list[float]]] = {
        t: {c: [] for c in (REAL_CONDITION, *conditions)} for t in swept
    }

    for utt in utts:
        real = read_processed(paths[(REAL_CONDITION, utt)], sr)
        mask = reference_silent_mask(real, sr)
        silent_fraction.append(float(mask.mean()))
        for cond in (REAL_CONDITION, *conditions):
            wav = real if cond == REAL_CONDITION else read_processed(paths[(cond, utt)], sr)
            n = min(len(wav), len(mask))
            floors[cond].append(_dbfs(wav[:n][mask[:n]]))
            pads[cond].append(_pad_level_dbfs(wav, sr))
            cut, peak = _quiet_stats(wav, sr)
            cuts[cond].append(cut)
            peaks[cond].append(peak)
            for t in swept:
                lead, trail = _leading_trailing_silence(wav, sr, threshold_db=t)
                features[t][cond].append(lead + trail)

    floor_rows = [
        {
            "condition": cond,
            "n": len(values),
            "silent_region_dbfs_median": float(np.nanmedian(values)),
            "silent_region_dbfs_p90": float(np.nanpercentile(values, 90)),
            "above_real_db": float(np.nanmedian(values) - np.nanmedian(floors[REAL_CONDITION])),
            "trim_pad_dbfs_median": float(np.nanmedian(pads[cond])),
            "trim_pad_above_real_db": float(
                np.nanmedian(pads[cond]) - np.nanmedian(pads[REAL_CONDITION])
            ),
            # The estimator's own inputs, per condition. On `real` these are the
            # per-file medians of what the gate pools; the gate's exact pooled
            # values are in `reference`.
            "quiet_frame_rms_dbfs": float(np.nanmedian(cuts[cond])),
            "quiet_peak_dbfs": float(np.nanmedian(peaks[cond])),
            "reference_silent_fraction": float(np.mean(silent_fraction)),
        }
        for cond, values in floors.items()
    ]

    sweep_rows = []
    for t in swept:
        for cond in conditions:
            real_feature = np.asarray(features[t][REAL_CONDITION])
            cond_feature = np.asarray(features[t][cond])
            eer, _ = compute_eer(real_feature, cond_feature)
            sweep_rows.append(
                {
                    "threshold_dbfs": t,
                    "gate_threshold": t == reference.threshold_dbfs,
                    "condition": cond,
                    "eer": eer,
                    "leaked_at_floor": bool(eer < SANITY_EER_FLOOR),
                    "real_silence_s_mean": float(real_feature.mean()),
                    "condition_silence_s_mean": float(cond_feature.mean()),
                }
            )
    return pd.DataFrame(floor_rows), pd.DataFrame(sweep_rows), reference
