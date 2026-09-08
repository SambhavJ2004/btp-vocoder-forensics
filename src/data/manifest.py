"""Dataset manifest: the provenance record that makes the study auditable.

One row per (utterance, condition, tier). The manifest is small, committed to
git, and is the artefact a reviewer reads to confirm that the confound controls
were actually applied rather than merely intended.

Manifest columns exist to answer specific questions:
  - ``tier``/``derived``/``source_rate``/``derivation``/``resample_steps``
                     : INV-01. Which artifact this is, what it came from, and
                       how many filtering passes it has been through. One step,
                       always -- a chained resample is rejected here.
  - ``mel_*``        : INV-02. Mel configuration cannot be equalised across
                       vocoders, so it is recorded per condition and reported
                       as a stated limitation.
  - ``trim_*``       : INV-03. Proves every condition used the reference span.
  - ``lufs_out``     : INV-04. Proves loudness landed on target (archive), and
                       records the band-limiting drift (derived).
  - ``split``        : INV-09. Frozen once, shared across conditions and tiers.
  - ``alignment_*``  : INV-16. Proves the vocoder is sample-aligned.
  - ``archive_band_hz`` / ``archive_band``
                     : INV-17. The archive is FULL-BAND, so every row records
                       the same thing; a row that does not means someone
                       band-limited at generation, which is the mistake this
                       invariant was rewritten to undo.
  - ``high_band_fraction``
                     : INV-17. Measured energy above LADDER_FMAX. Evidence, not
                       a gate -- it is how Griffin-Lim's hard zero (0.00000) is
                       distinguishable from a neural vocoder tracking real
                       (~0.015 against real's ~0.018).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from .invariants import (
    ALIGNMENT_REQUIRED_LAG,
    ARCHIVE_SR,
    ARCHIVE_TIER,
    MAX_RESAMPLE_STEPS,
    REAL_CONDITION,
    SANCTIONED_RATES,
    SUBTYPE,
    TIER_RATES,
    TIERS,
    ZEROSHOT_SR,
    ZEROSHOT_TIER,
    InvariantViolation,
    check_pairing,
    is_correlation_excluded,
)

MANIFEST_COLUMNS: tuple[str, ...] = (
    "utt_id",
    "condition",
    "tier",
    "path",
    "speaker",
    "split",
    "derived",
    "source_rate",
    "derivation",
    "resample_steps",
    "sr_out",
    "subtype",
    "duration_s",
    "lufs_out",
    "trim_start",
    "trim_end",
    "trim_ref_length",
    "mel_n_fft",
    "mel_hop",
    "mel_win",
    "mel_n_mels",
    "mel_fmin",
    "mel_fmax",
    "alignment_checked",
    "alignment_peak_lag",
    "archive_band_hz",
    "archive_band",
    "high_band_fraction",
    "vocoder_checkpoint",
    "vocoder_commit",
    "seed",
)


@dataclass
class ManifestRow:
    utt_id: str
    condition: str
    path: str
    speaker: str
    split: str
    duration_s: float
    lufs_out: float
    trim_start: int
    trim_end: int
    trim_ref_length: int
    # INV-01 provenance. `source_rate` is what this record was derived FROM:
    # the vocoder's native rate for an archive record, ARCHIVE_SR for a derived
    # one. `derivation` names the single step that produced it.
    tier: str = ARCHIVE_TIER
    derived: bool = False
    source_rate: int = ARCHIVE_SR
    derivation: str = "resample_once"
    resample_steps: int = 1
    sr_out: int = ARCHIVE_SR
    mel_n_fft: int | None = None
    mel_hop: int | None = None
    mel_win: int | None = None
    mel_n_mels: int | None = None
    mel_fmin: float | None = None
    mel_fmax: float | None = None
    alignment_checked: bool = False
    alignment_peak_lag: int | None = None
    # INV-17. The archive is full-band for every condition, so these are the
    # same on every row; `high_band_fraction` is the per-file measurement.
    archive_band_hz: float = ARCHIVE_SR / 2.0
    archive_band: str = "full_band"
    high_band_fraction: float | None = None
    vocoder_checkpoint: str | None = None
    vocoder_commit: str | None = None
    seed: int | None = None
    subtype: str = SUBTYPE
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("extra", None)
        return d


def write_manifest(rows: list[ManifestRow], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([r.to_dict() for r in rows])
    df = df.reindex(columns=list(MANIFEST_COLUMNS))
    df.sort_values(["tier", "condition", "utt_id"], inplace=True)
    df.to_csv(path, index=False)
    return path


def read_manifest(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(MANIFEST_COLUMNS) - set(df.columns)
    if missing:
        raise InvariantViolation(f"manifest {path} missing columns: {sorted(missing)}")
    return df


def select_tier(df: pd.DataFrame, tier: str) -> pd.DataFrame:
    """Rows for one tier. The only sanctioned way to pick an artifact."""
    if tier not in TIERS:
        raise InvariantViolation(f"INV-01: unknown tier '{tier}'. Known: {list(TIERS)}")
    sub = df[df["tier"] == tier]
    if sub.empty:
        raise InvariantViolation(
            f"INV-01: manifest has no '{tier}' rows. The archive is primary; the "
            "zero-shot set is derived from it and must be generated separately."
        )
    return sub


def require_primary(df: pd.DataFrame, *, why: str) -> pd.DataFrame:
    """INV-01. Refuse derived records where a primary artifact is required.

    The derived 16 kHz set has had everything above 8 kHz filtered away. Any
    measurement of the high band -- the band-limit ablation, band-wise spectral
    error, effective-bandwidth checks -- run against it would be measuring the
    downsampler, and would return a clean, plausible, wrong answer.
    """
    if "derived" not in df.columns:
        raise InvariantViolation(
            f"INV-01: cannot verify provenance for {why}; the frame has no 'derived' "
            "column. Load it through data.manifest.read_manifest."
        )
    offending = df[df["derived"].astype(bool)]
    if not offending.empty:
        tiers = sorted(set(offending["tier"]))
        raise InvariantViolation(
            f"INV-01: {why} requires the primary archive artifact at {ARCHIVE_SR} Hz, "
            f"but {len(offending)} derived rows were passed (tiers={tiers}). The "
            f"derived set is low-passed at {ZEROSHOT_SR // 2} Hz -- measuring the high "
            "band on it measures the downsampler. Use select_tier(df, 'archive')."
        )
    return df


def validate_manifest(df: pd.DataFrame, *, require_zeroshot: bool = False) -> None:
    """Gate that Phase A must pass before B or C are allowed to consume the data.

    Checks the invariants that are visible at the table level. Signal-level
    checks live in :mod:`experiments.sanity_checks`; the alignment probe itself
    lives in :mod:`data.alignment` and lands here as two columns.
    """
    # --- INV-01: rates, tiers, and the one-step derivation rule ---
    bad_tier = set(df["tier"]) - set(TIERS)
    if bad_tier:
        raise InvariantViolation(f"INV-01: unknown tiers in manifest: {sorted(bad_tier)}")

    for tier, group in df.groupby("tier"):
        expected = TIER_RATES[str(tier)]
        if (group["sr_out"] != expected).any():
            raise InvariantViolation(
                f"INV-01: '{tier}' rows must be at {expected} Hz; found "
                f"{sorted(set(group['sr_out']))}."
            )

    unsanctioned = set(df["sr_out"]) - set(SANCTIONED_RATES)
    if unsanctioned:
        raise InvariantViolation(
            f"INV-01: unsanctioned output rates {sorted(unsanctioned)}. Only "
            f"{ARCHIVE_SR} (archive) and {ZEROSHOT_SR} (derived) exist."
        )

    chained = df[df["resample_steps"] > MAX_RESAMPLE_STEPS]
    if not chained.empty:
        worst = int(chained["resample_steps"].max())
        raise InvariantViolation(
            f"INV-01: {len(chained)} records carry {MAX_RESAMPLE_STEPS + 1}+ resample "
            f"steps (max {worst}). Each artifact is ONE filtering pass: native->archive, "
            "or archive->zeroshot. Chaining stacks a second anti-aliasing imprint on "
            "the signal, which is itself a per-condition label."
        )

    # Provenance must agree with the tier it claims.
    archive = df[df["tier"] == ARCHIVE_TIER]
    if archive["derived"].astype(bool).any():
        raise InvariantViolation(
            "INV-01: archive rows marked derived. The archive is the primary artifact."
        )
    zeroshot = df[df["tier"] == ZEROSHOT_TIER]
    if not zeroshot.empty:
        if not zeroshot["derived"].astype(bool).all():
            raise InvariantViolation(
                "INV-01: zero-shot rows must be marked derived; the 16 kHz set is "
                "never a primary artifact."
            )
        if (zeroshot["source_rate"] != ARCHIVE_SR).any():
            found = sorted(set(zeroshot["source_rate"]))
            raise InvariantViolation(
                f"INV-01: zero-shot rows derive from {found}, not from the "
                f"{ARCHIVE_SR} Hz archive. Deriving from source is a second "
                "independent resampling path -- archive high, derive low, never the "
                "reverse."
            )
    elif require_zeroshot:
        raise InvariantViolation(
            "INV-01: no zero-shot rows. The mismatched protocol, PESQ-WB and UTMOS "
            "all need the derived 16 kHz set."
        )

    # --- INV-05 ---
    if (df["subtype"] != SUBTYPE).any():
        raise InvariantViolation(f"INV-05: manifest contains rows not in {SUBTYPE}.")

    # --- INV-06: pairing, within each tier ---
    for _tier, group in df.groupby("tier"):
        check_pairing({c: set(g["utt_id"]) for c, g in group.groupby("condition")})
    if REAL_CONDITION not in set(df["condition"]):
        raise InvariantViolation("INV-06: manifest has no real condition.")

    # --- INV-03: every condition of an utterance used the reference trim span ---
    # Checked within a tier: spans are archive-sample indices, and the derived
    # tier records the same values for traceability at its own rate.
    span_cols = ["trim_start", "trim_end", "trim_ref_length"]
    for tier, group in df.groupby("tier"):
        spans = group.groupby("utt_id")[span_cols].nunique()
        offenders = spans[(spans > 1).any(axis=1)].index.tolist()
        if offenders:
            raise InvariantViolation(
                f"INV-03: {len(offenders)} utterances in tier '{tier}' have "
                f"per-condition trim spans (e.g. {offenders[:5]}). Trim boundaries "
                "come from the real reference and are applied by index to every "
                "condition."
            )

    # --- INV-09: split assignment is a property of the utterance ---
    split_counts = df.groupby("utt_id")["split"].nunique()
    if (split_counts > 1).any():
        bad = split_counts[split_counts > 1].index.tolist()[:5]
        raise InvariantViolation(
            f"INV-09: utterances assigned to different splits per condition or tier: {bad}"
        )

    # --- INV-17: the ladder band ---
    _validate_band(df)

    # --- INV-16: sample alignment, per vocoder condition ---
    for cond, group in df[df["condition"] != REAL_CONDITION].groupby("condition"):
        if not group["alignment_checked"].astype(bool).all():
            raise InvariantViolation(
                f"INV-16: condition '{cond}' has no alignment probe on record. Run "
                "data.alignment.check_vocoder_alignment when a vocoder is first "
                "onboarded -- reference-derived trim spans assume sample alignment."
            )
        lags = set(group["alignment_peak_lag"].dropna().astype(int))
        if lags != {ALIGNMENT_REQUIRED_LAG}:
            raise InvariantViolation(
                f"INV-16: condition '{cond}' records peak lag(s) {sorted(lags)}, "
                f"required {ALIGNMENT_REQUIRED_LAG}. A fixed vocoder delay would be "
                "measured as an artifact on both axes at once."
            )


def _validate_band(df: pd.DataFrame) -> None:
    """INV-17. The archive is full-band, uniformly, for every condition.

    The old version of this check asserted that every condition had been
    band-limited to LADDER_FMAX at generation. That was the mistake: `fmax`
    constrains a vocoder's analysis, not its synthesis, so filtering at
    generation destroyed real hallucinated high-band content. It now asserts the
    opposite -- that nobody filtered.
    """
    expected_hz = float(ARCHIVE_SR / 2.0)

    bands = set(df["archive_band"].astype(str))
    if bands != {"full_band"}:
        raise InvariantViolation(
            f"INV-17: archive rows record band(s) {sorted(bands)}, expected "
            "{'full_band'}. The archive is full-band; band-limiting is an "
            "analysis-time transform (data.preprocess.band_limit_comparison_set). "
            "A band-limited archive throws away the hallucinated high-band content "
            "that is the most forensically interesting thing it holds."
        )

    hz = set(df["archive_band_hz"].dropna().astype(float))
    if hz != {expected_hz}:
        raise InvariantViolation(
            f"INV-17: archive rows record archive_band_hz {sorted(hz)}, expected "
            f"{expected_hz} (Nyquist at the archive rate)."
        )

    # high_band_fraction is descriptive, so it is not gated on a value -- only
    # on being present for the archive tier, where it is measured.
    archive = df[df["tier"] == ARCHIVE_TIER]
    if not archive.empty and archive["high_band_fraction"].isna().any():
        missing = sorted(
            set(archive[archive["high_band_fraction"].isna()]["condition"])
        )
        raise InvariantViolation(
            f"INV-17: archive rows for {missing} have no high_band_fraction. It is "
            "the evidence that distinguishes a genuine bandwidth zero from a "
            "vocoder tracking real, and is measured during Phase A."
        )


def mel_config_table(df: pd.DataFrame) -> pd.DataFrame:
    """INV-02 reporting helper: the per-condition mel table for the limitations
    section. Vocoders are trained on their own front-ends; forcing a shared mel
    would take each model out of distribution and measure a broken vocoder
    rather than the real one. Documenting the difference is the honest control.
    """
    cols = ["mel_n_fft", "mel_hop", "mel_win", "mel_n_mels", "mel_fmin", "mel_fmax"]
    return df[df["tier"] == ARCHIVE_TIER].groupby("condition")[cols].first().reset_index()


def provenance_table(df: pd.DataFrame) -> pd.DataFrame:
    """INV-01/INV-17 reporting helper: one row per (tier, condition)."""
    cols = [
        "derived",
        "source_rate",
        "sr_out",
        "derivation",
        "resample_steps",
        "archive_band",
        "high_band_fraction",
    ]
    return df.groupby(["tier", "condition"])[cols].first().reset_index()


def band_report(df: pd.DataFrame) -> pd.DataFrame:
    """INV-17 reporting helper: the archive band and measured high-band content.

    ``high_band_*`` is the fraction of energy above LADDER_FMAX. It is the
    column that separates a genuine bandwidth zero (Griffin-Lim, 0.00000) from a
    vocoder producing high-band content (real ~0.018, BigVGAN ~0.015).
    """
    rows = []
    for (tier, cond), g in df.groupby(["tier", "condition"]):
        hb = g["high_band_fraction"].dropna().astype(float)
        rows.append(
            {
                "tier": tier,
                "condition": cond,
                "archive_band": g["archive_band"].iloc[0],
                "archive_band_hz": g["archive_band_hz"].iloc[0],
                "high_band_mean": f"{hb.mean():.5f}" if len(hb) else "n/a",
                "high_band_max": f"{hb.max():.5f}" if len(hb) else "n/a",
                "in_correlation": not is_correlation_excluded(str(cond)),
            }
        )
    return pd.DataFrame(rows)


def primary_ladder_frame(df: pd.DataFrame, *, why: str) -> pd.DataFrame:
    """INV-17. Drop correlation-excluded conditions before a ladder-wide series.

    Use this wherever conditions are pooled. Excluded conditions are generated,
    measured and reported like any other; what they are not is comparable on the
    headline axis. See `invariants.CORRELATION_EXCLUDED` for the reason attached
    to each.
    """
    if "condition" not in df.columns:
        raise InvariantViolation(f"INV-17: cannot filter conditions for {why}.")
    keep = ~df["condition"].map(is_correlation_excluded).astype(bool)
    return df[keep]


def dump_provenance(path: str | Path, payload: dict) -> Path:
    """Write the run-level provenance sidecar (git commit, env, seeds, versions)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path
