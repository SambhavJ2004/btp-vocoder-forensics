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
  - ``band_*``       : INV-17. The ladder band actually applied, the filter that
                       applied it, and the residual out-of-band energy. An
                       exempt condition records its exemption rather than a
                       cutoff, so "kept its band by design" is distinguishable
                       from "nobody filtered it".
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
    BAND_LIMIT_MAX_STOPBAND_ENERGY,
    LADDER_FMAX,
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
    is_band_exempt,
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
    "band_limit_hz",
    "band_filter",
    "band_exempt",
    "band_oob_energy",
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
    # INV-17. `band_limit_hz` is None only for exempt conditions, which must
    # also set band_exempt=True -- the two are checked against each other.
    band_limit_hz: float | None = LADDER_FMAX
    band_filter: str = ""
    band_exempt: bool = False
    band_oob_energy: float | None = None
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
    """INV-17. Every non-exempt condition shares one band; exempt ones say so."""
    for cond, group in df.groupby("condition"):
        exempt_flags = set(group["band_exempt"].astype(bool))
        if len(exempt_flags) > 1:
            raise InvariantViolation(
                f"INV-17: condition '{cond}' is marked exempt on some rows and not "
                "others. Exemption is a property of the condition."
            )
        declared_exempt = exempt_flags.pop()
        if declared_exempt != is_band_exempt(str(cond)):
            raise InvariantViolation(
                f"INV-17: condition '{cond}' records band_exempt={declared_exempt}, "
                f"but invariants.BAND_EXEMPT_CONDITIONS says "
                f"{is_band_exempt(str(cond))}. The manifest and the code must agree "
                "about which conditions keep their full band."
            )

        if declared_exempt:
            continue

        cutoffs = set(group["band_limit_hz"].dropna().astype(float))
        if cutoffs != {float(LADDER_FMAX)}:
            raise InvariantViolation(
                f"INV-17: condition '{cond}' was band-limited to {sorted(cutoffs)}, "
                f"not to the ladder band {LADDER_FMAX}. Every non-exempt condition, "
                "REAL included, shares one cutoff -- a condition with a different "
                "band is separable from the rest on high-band energy alone."
            )
        if group["band_limit_hz"].isna().any():
            raise InvariantViolation(
                f"INV-17: condition '{cond}' has rows with no recorded cutoff. A "
                "missing cutoff on a non-exempt condition means the filter was "
                "never applied."
            )
        if (group["band_filter"].astype(str).str.len() == 0).any():
            raise InvariantViolation(
                f"INV-17: condition '{cond}' has rows with no filter spec recorded."
            )
        oob = group["band_oob_energy"].dropna().astype(float)
        if len(oob) and oob.max() > BAND_LIMIT_MAX_STOPBAND_ENERGY:
            raise InvariantViolation(
                f"INV-17: condition '{cond}' carries up to {oob.max():.2e} of its "
                f"energy above {LADDER_FMAX} Hz, over the "
                f"{BAND_LIMIT_MAX_STOPBAND_ENERGY:.0e} stopband allowance."
            )

    # One filter spec across every non-exempt condition, not just one cutoff.
    non_exempt = df[~df["band_exempt"].astype(bool)]
    specs = set(non_exempt["band_filter"].astype(str))
    if len(specs) > 1:
        raise InvariantViolation(
            f"INV-17: conditions were band-limited by different filters {sorted(specs)}. "
            "A filter's own rolloff is a signature; two filters means two signatures."
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
        "band_limit_hz",
        "band_exempt",
    ]
    return df.groupby(["tier", "condition"])[cols].first().reset_index()


def band_report(df: pd.DataFrame) -> pd.DataFrame:
    """INV-17 reporting helper: the delivered band, per condition.

    Out-of-band energy is rendered through :func:`data.invariants.format_oob`,
    which refuses to quote a number below MEASUREMENT_FLOOR. A figure there is
    float32 representation noise, and printing it would imply the filter was
    characterised to a precision it was not.
    """
    from .invariants import format_oob

    rows = []
    for (tier, cond), g in df.groupby(["tier", "condition"]):
        oob = g["band_oob_energy"].dropna().astype(float)
        rows.append(
            {
                "tier": tier,
                "condition": cond,
                "band_exempt": bool(g["band_exempt"].iloc[0]),
                "band_limit_hz": g["band_limit_hz"].iloc[0],
                "band_filter": g["band_filter"].iloc[0],
                "oob_max": format_oob(float(oob.max())) if len(oob) else "n/a (exempt)",
            }
        )
    return pd.DataFrame(rows)


def primary_ladder_frame(df: pd.DataFrame, *, why: str) -> pd.DataFrame:
    """INV-17. Drop band-exempt conditions before a ladder-wide comparison.

    Use this wherever conditions are pooled into one series. Exempt conditions
    are generated and measured like any other, but they differ from the ladder
    in bandwidth by design, so averaging or correlating across them measures the
    exemption rather than the vocoders.
    """
    if "condition" not in df.columns:
        raise InvariantViolation(f"INV-17: cannot filter conditions for {why}.")
    keep = ~df["condition"].map(is_band_exempt).astype(bool)
    return df[keep]


def dump_provenance(path: str | Path, payload: dict) -> Path:
    """Write the run-level provenance sidecar (git commit, env, seeds, versions)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path
