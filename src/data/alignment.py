"""INV-16 — vocoder sample-alignment probe.

Reference-derived trim spans (INV-03) apply the real reference's sample indices
to every condition. That is only meaningful if the vocoder's output is
sample-aligned with its input. A vocoder with a fixed algorithmic delay -- a
padding convention, a centre/non-centre STFT mismatch, a filterbank group
delay -- would shift its whole output by a constant number of samples.

The damage is specific and silent. A shifted condition would:
  - have the reference trim span land on the wrong part of the signal, so the
    retained audio is not the same span of speech as the reference;
  - score badly on MCD, PESQ and F0 error for a reason that has nothing to do
    with reconstruction quality, dragging the X-axis;
  - present a strong, perfectly learnable cross-correlation cue to the
    detector, dragging the Y-axis.

Both axes move together, in the direction that manufactures a correlation. A
time shift would be read as a vocoder "artifact" and would look exactly like the
headline result.

So: cross-correlate real against resynthesized when a vocoder is first
onboarded. The peak lag must be exactly zero. Anything else is a blocker -- fix
the adapter (usually a padding or centre-convention flag), do not compensate by
shifting the audio, because a compensating shift is another per-condition
operation and INV-10 exists to prevent those.


Two measurements, because one is not enough
-------------------------------------------

**Envelope lag** (primary gate). Cross-correlation of the short-time energy
envelope. Phase-blind, defined for every vocoder, and it is the quantity that
actually decides whether a trim span lands on the right span of speech.

The envelope window is 50 ms and that is load-bearing, not a default. A shorter
window leaves pitch ripple in the envelope, and correlating two rippled
envelopes lets the peak lock onto a pitch period instead of the true offset.
Measured on Griffin-Lim over 20 LJSpeech-rate utterances, the modal lag was 0 at
every window tested, but the agreement was: 6/20 files at 5 ms, 17/20 at 20 ms,
**20/20 at 50 ms**. Below ~20 ms the probe reports spurious lags of one pitch
period and would raise false blockers.

**Waveform lag** (corroborating gate, where it applies). Phase-sensitive, and a
sharper instrument where it works: for a phase-preserving vocoder it recovers a
fixed delay exactly, while the envelope peak is only as sharp as the onsets.

It is enforced only when it is unambiguous evidence, which takes two things: a
mean |correlation| above ``ALIGNMENT_PHASE_CORR_THRESHOLD`` (there is a phase
relationship at all) and lags that agree on at least
``ALIGNMENT_MODAL_SHARE_FLOOR`` of the files (a *fixed* delay is consistent by
definition). Griffin-Lim reconstructs phase iteratively from magnitude alone, so
its waveform correlation with the reference is near noise -- measured at
|r| ~ 0.05 on real speech -- and a partial correlation in between produces
scattered lags that are noise rather than a delay.

Nothing is lost by that leniency: a genuine fixed delay displaces the envelope
too, so the envelope gate catches it either way.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy.signal import correlate

from .invariants import (
    ALIGNMENT_ENVELOPE_MS,
    ALIGNMENT_MAX_LAG_SAMPLES,
    ALIGNMENT_MODAL_SHARE_FLOOR,
    ALIGNMENT_PHASE_CORR_THRESHOLD,
    ALIGNMENT_PROBE_FILES,
    ALIGNMENT_REQUIRED_LAG,
    ARCHIVE_SR,
    InvariantViolation,
)


@dataclass
class AlignmentReport:
    """Outcome of the INV-16 probe for one condition."""

    condition: str
    n_files: int
    peak_lag: int                    # the gated value: envelope modal lag
    envelope_lags: list[int] = field(default_factory=list)
    envelope_modal_share: float = 0.0
    mean_envelope_correlation: float = 0.0
    waveform_modal_lag: int = 0
    waveform_lags: list[int] = field(default_factory=list)
    waveform_modal_share: float = 0.0
    mean_waveform_correlation: float = 0.0
    phase_preserving: bool = False    # is there a phase relationship at all?
    waveform_gate_applied: bool = False
    sample_rate: int = ARCHIVE_SR
    passed: bool = False

    @property
    def lag_ms(self) -> float:
        return 1000.0 * self.peak_lag / self.sample_rate

    def to_dict(self) -> dict:
        d = asdict(self)
        d["lag_ms"] = self.lag_ms
        return d

    def summary(self) -> str:
        gate = "waveform gate applied" if self.waveform_gate_applied else "envelope gate only"
        return (
            f"{self.condition}: envelope lag {self.peak_lag} "
            f"({self.envelope_modal_share:.0%} of {self.n_files} files, "
            f"r={self.mean_envelope_correlation:.3f}); {gate} "
            f"(waveform r={self.mean_waveform_correlation:.3f}, "
            f"lag {self.waveform_modal_lag} on {self.waveform_modal_share:.0%})"
        )


def energy_envelope(wav: np.ndarray, sr: int, window_ms: float = ALIGNMENT_ENVELOPE_MS):
    """Short-time energy envelope at sample resolution.

    A boxcar moving average of the squared signal. Sample resolution (rather
    than a framed envelope) is deliberate: the invariant is about sample
    alignment, so the measurement must be able to express a lag of one sample.
    """
    win = max(1, int(round(window_ms * 1e-3 * sr)))
    kernel = np.ones(win, dtype=np.float64) / win
    return np.convolve(wav.astype(np.float64) ** 2, kernel, mode="same")


def cross_correlation_lag(
    reference: np.ndarray,
    degraded: np.ndarray,
    *,
    max_lag: int = ALIGNMENT_MAX_LAG_SAMPLES,
) -> tuple[int, float]:
    """Lag in samples that maximises normalised cross-correlation.

    Positive lag means ``degraded`` trails ``reference`` -- i.e. the vocoder
    introduced a delay. Searched over a bounded window because an unbounded
    argmax on speech will happily lock onto a pitch period a long way from zero.

    Note the sign flip below: ``np.correlate(a, b)`` peaks at ``k = -d`` when
    ``b`` lags ``a`` by ``d``, so the raw argmax has the opposite sign to the
    intuitive reading. The gate itself only tests against zero, but the number
    is reported in error messages and written to the manifest, so it has to mean
    what the docstring says it means.

    Both signals are mean-removed and energy-normalised so the returned peak is
    a correlation coefficient, comparable across files and conditions.

    FFT-based correlation, not the direct O(n^2) form: a 5-second utterance at
    22.05 kHz is 110k samples, and the direct method makes a 20-file probe take
    minutes per condition rather than under a second.
    """
    n = min(len(reference), len(degraded))
    if n == 0:
        raise InvariantViolation("INV-16: cannot correlate an empty signal.")
    a = np.asarray(reference[:n], dtype=np.float64)
    b = np.asarray(degraded[:n], dtype=np.float64)
    a -= a.mean()
    b -= b.mean()

    denom = np.sqrt(float(a @ a) * float(b @ b))
    if denom <= 0.0:
        raise InvariantViolation("INV-16: cannot correlate a silent signal.")

    max_lag = int(min(max_lag, n - 1))
    corr = correlate(a, b, mode="full", method="fft") / denom
    centre = n - 1
    window = corr[centre - max_lag : centre + max_lag + 1]
    idx = int(np.argmax(np.abs(window)))
    return -(idx - max_lag), float(window[idx])


def _mode(values: list[int]) -> tuple[int, float]:
    """Modal value and the fraction of samples that share it.

    The mode rather than the mean: a genuine fixed delay shows the same integer
    on nearly every file, while an occasional outlier is a file whose
    correlation locked onto a pitch period. A mean would blur a real 1-sample
    delay into something that looks like rounding.
    """
    uniq, counts = np.unique(np.asarray(values), return_counts=True)
    best = int(np.argmax(counts))
    return int(uniq[best]), float(counts[best] / len(values))


def check_vocoder_alignment(
    pairs: list[tuple[np.ndarray, np.ndarray]],
    condition: str,
    *,
    max_lag: int = ALIGNMENT_MAX_LAG_SAMPLES,
    sample_rate: int = ARCHIVE_SR,
) -> AlignmentReport:
    """Probe a condition for a fixed vocoder delay.

    ``pairs`` is a list of ``(real, resynthesized)`` waveforms at ``sample_rate``.
    """
    if not pairs:
        raise InvariantViolation(f"INV-16: no probe files supplied for '{condition}'.")
    if len(pairs) < ALIGNMENT_PROBE_FILES:
        raise InvariantViolation(
            f"INV-16: '{condition}' probed on {len(pairs)} files, "
            f"{ALIGNMENT_PROBE_FILES} required. A small sample cannot distinguish a "
            "fixed delay from a couple of unlucky correlations."
        )

    env_lags: list[int] = []
    env_corrs: list[float] = []
    wav_lags: list[int] = []
    wav_corrs: list[float] = []

    for ref, deg in pairs:
        lag, corr = cross_correlation_lag(ref, deg, max_lag=max_lag)
        wav_lags.append(lag)
        wav_corrs.append(corr)

        e_lag, e_corr = cross_correlation_lag(
            energy_envelope(ref, sample_rate),
            energy_envelope(deg, sample_rate),
            max_lag=max_lag,
        )
        env_lags.append(e_lag)
        env_corrs.append(e_corr)

    env_modal, env_share = _mode(env_lags)
    wav_modal, wav_share = _mode(wav_lags)
    mean_wav_corr = float(np.mean(np.abs(wav_corrs)))

    phase_preserving = mean_wav_corr >= ALIGNMENT_PHASE_CORR_THRESHOLD
    waveform_gate = phase_preserving and wav_share >= ALIGNMENT_MODAL_SHARE_FLOOR

    passed = env_modal == ALIGNMENT_REQUIRED_LAG
    if waveform_gate:
        passed = passed and wav_modal == ALIGNMENT_REQUIRED_LAG

    return AlignmentReport(
        condition=condition,
        n_files=len(pairs),
        peak_lag=env_modal,
        envelope_lags=env_lags,
        envelope_modal_share=env_share,
        mean_envelope_correlation=float(np.mean(env_corrs)),
        waveform_modal_lag=wav_modal,
        waveform_lags=wav_lags,
        waveform_modal_share=wav_share,
        mean_waveform_correlation=mean_wav_corr,
        phase_preserving=phase_preserving,
        waveform_gate_applied=waveform_gate,
        sample_rate=sample_rate,
        passed=passed,
    )


def assert_aligned(report: AlignmentReport) -> None:
    """INV-16 gate. Raise unless the condition is sample-aligned."""
    if report.passed:
        return

    if report.peak_lag != ALIGNMENT_REQUIRED_LAG:
        detail = (
            f"envelope lag {report.peak_lag} samples ({report.lag_ms:+.3f} ms), "
            f"shared by {report.envelope_modal_share:.0%} of {report.n_files} files"
        )
    else:
        detail = (
            f"envelope is aligned but the waveform is offset by "
            f"{report.waveform_modal_lag} samples on "
            f"{report.waveform_modal_share:.0%} of files, and this vocoder "
            f"preserves waveform phase (r={report.mean_waveform_correlation:.3f}), "
            "so that offset is real"
        )

    raise InvariantViolation(
        f"INV-16: '{report.condition}' is not sample-aligned with the real "
        f"reference -- {detail}. Reference-derived trim spans assume sample "
        "alignment, so this shift would be measured as a vocoder artifact on both "
        "axes at once. Fix the adapter's padding or centre convention -- do NOT "
        "shift the audio to compensate (INV-10)."
    )


def write_alignment_report(reports: list[AlignmentReport], path: str | Path) -> Path:
    """Persist the onboarding probe alongside the manifest."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {r.condition: r.to_dict() for r in reports}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def read_alignment_report(path: str | Path) -> dict[str, dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
