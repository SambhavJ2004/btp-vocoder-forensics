"""The vocoder ladder and its lookup table.

LADDER order is the study's prior on quality, worst to best. It is a hypothesis
about the X-axis, not a result: the ordering that appears in the main plot is
whatever UTMOS/PESQ/MCD actually measure. Do not let this list stand in for a
measurement.

Eight conditions are generated but only six are rungs.

``LADDER`` (6)              the primary ladder, what the headline correlation
                            runs over.
``CONTROL_CONDITIONS`` (2)  ``melgan_fullband`` and
                            ``bigvgan_v2_22khz_fullband``. Exempt from INV-17,
                            excluded from the primary correlation by
                            :func:`detectors.protocols.spearman_headline`, which
                            refuses to compute rho over a frame containing them.
``UNAVAILABLE`` (1)         ``specdiff_gan``: declared, audited, and not
                            generable because its weights were never released.
                            Kept in ``SPECS`` so the substitution is visible in
                            the audit trail rather than being an unexplained
                            gap. Never in ``ALL_CONDITIONS``.
"""

from __future__ import annotations

from collections.abc import Callable

from data.invariants import (
    BAND_EXEMPT_CONDITIONS,
    PRIMARY_LADDER,
    UNAVAILABLE_CONDITIONS,
    InvariantViolation,
    is_band_exempt,
)

from .base import Vocoder, VocoderSpec
from .bigvgan import BASE_SPEC as BIGVGAN_BASE_SPEC
from .bigvgan import FULLBAND_SPEC as BIGVGAN_FULLBAND_SPEC
from .bigvgan import LARGE_SPEC as BIGVGAN_LARGE_SPEC
from .bigvgan import BigVGAN
from .griffin_lim import SPEC as GRIFFIN_LIM_SPEC
from .griffin_lim import GriffinLim
from .hifigan import SPEC as HIFIGAN_SPEC
from .hifigan import HiFiGAN
from .melgan import SPEC as MELGAN_SPEC
from .melgan import MelGAN
from .melgan_fullband import SPEC as MELGAN_FULLBAND_SPEC
from .melgan_fullband import MelGANFullband
from .specdiff_gan import SPEC as SPECDIFF_SPEC
from .specdiff_gan import SpecDiffGAN
from .vocos import SPEC as VOCOS_SPEC
from .vocos import Vocos

# Expected quality order, worst -> best. Mirrors data.invariants.PRIMARY_LADDER,
# which is the definition; this is the ordered view of it.
LADDER: tuple[str, ...] = PRIMARY_LADDER

# Paired controls: real conditions, generated and measured like any other, but
# never averaged into the headline number.
CONTROL_CONDITIONS: tuple[str, ...] = (
    "melgan_fullband",
    "bigvgan_v2_22khz_fullband",
)

# Declared but not generable: weights are not publicly available. Kept in SPECS
# so the audit trail survives the substitution (INV-08) and so the replacement
# can be compared against what it replaced. Never in LADDER or ALL_CONDITIONS.
UNAVAILABLE: tuple[str, ...] = tuple(sorted(UNAVAILABLE_CONDITIONS))

# Everything Phase A generates.
ALL_CONDITIONS: tuple[str, ...] = LADDER + CONTROL_CONDITIONS

# Conditions shipped in the throwaway v0 dataset (week two) so that Persons B
# and C can build their pipelines while the full set is still generating.
V0_CONDITIONS: tuple[str, ...] = ("griffin_lim", "hifigan_v1")

SPECS: dict[str, VocoderSpec] = {
    "griffin_lim": GRIFFIN_LIM_SPEC,
    "melgan": MELGAN_SPEC,
    "hifigan_v1": HIFIGAN_SPEC,
    "vocos": VOCOS_SPEC,
    "bigvgan_base": BIGVGAN_BASE_SPEC,
    "bigvgan_112m": BIGVGAN_LARGE_SPEC,
    "melgan_fullband": MELGAN_FULLBAND_SPEC,
    "bigvgan_v2_22khz_fullband": BIGVGAN_FULLBAND_SPEC,
    # Declared but unavailable; excluded from ALL_CONDITIONS.
    "specdiff_gan": SPECDIFF_SPEC,
}

_BUILDERS: dict[str, Callable[[], Vocoder]] = {
    "griffin_lim": GriffinLim,
    "melgan": MelGAN,
    "hifigan_v1": HiFiGAN,
    "vocos": Vocos,
    "bigvgan_base": lambda: BigVGAN(BIGVGAN_BASE_SPEC),
    "bigvgan_112m": lambda: BigVGAN(BIGVGAN_LARGE_SPEC),
    "melgan_fullband": MelGANFullband,
    "bigvgan_v2_22khz_fullband": lambda: BigVGAN(BIGVGAN_FULLBAND_SPEC),
    "specdiff_gan": SpecDiffGAN,
}

# Keep the three declarations of "what is exempt" from drifting apart. The spec
# flag, the invariants set and the control list must agree, or a condition could
# be filtered by one code path and not another.
_declared_exempt = {
    k for k, v in SPECS.items() if not v.primary_ladder and k not in UNAVAILABLE_CONDITIONS
}
if _declared_exempt != set(BAND_EXEMPT_CONDITIONS) or _declared_exempt != set(
    CONTROL_CONDITIONS
):
    raise InvariantViolation(
        "INV-17: exemption declarations disagree. VocoderSpec.primary_ladder says "
        f"{sorted(_declared_exempt)}, invariants.BAND_EXEMPT_CONDITIONS says "
        f"{sorted(BAND_EXEMPT_CONDITIONS)}, registry.CONTROL_CONDITIONS says "
        f"{sorted(CONTROL_CONDITIONS)}."
    )
if set(LADDER) & set(BAND_EXEMPT_CONDITIONS):
    raise InvariantViolation(
        "INV-17: an exempt condition is in the primary ladder. Exempt conditions "
        "keep their full native band, so including one would put a bandwidth cliff "
        "back into the headline correlation."
    )


def get_vocoder(key: str) -> Vocoder:
    if key not in _BUILDERS:
        raise KeyError(f"unknown vocoder '{key}'. Known: {sorted(_BUILDERS)}")
    return _BUILDERS[key]()


def get_spec(key: str) -> VocoderSpec:
    if key not in SPECS:
        raise KeyError(f"unknown vocoder '{key}'. Known: {sorted(SPECS)}")
    return SPECS[key]


def primary_ladder_only(conditions) -> list[str]:
    """Drop exempt conditions from an arbitrary iterable, preserving order."""
    return [c for c in conditions if not is_band_exempt(c)]


def checkpoint_audit(include_unavailable: bool = True) -> list[dict[str, object]]:
    """Which conditions are actually runnable yet.

    A checkpoint counts as pinned when it carries either a HuggingFace revision
    SHA (content-addressed over the whole repo, verifiable without downloading
    anything) or a weight-file sha256 recorded at download time. Either
    satisfies INV-08. Conditions not hosted on HuggingFace can only use the
    second, which is why `vocoders.checkpoints` exists.

    The `pin` column says which mechanism is in force, so "pinned" is never a
    bare boolean hiding two different guarantees.
    """
    keys = list(ALL_CONDITIONS) + (list(UNAVAILABLE) if include_unavailable else [])
    rows = []
    for key in keys:
        spec = SPECS[key]
        unavailable = key in UNAVAILABLE_CONDITIONS
        needs_ckpt = spec.family != "signal_processing"

        if spec.checkpoint_revision:
            pin = "hf-revision"
        elif spec.checkpoint_sha256:
            pin = "sha256"
        elif not needs_ckpt:
            pin = "n/a"
        else:
            pin = "UNPINNED"

        mel_verified = not spec.mel.source.startswith("TODO")
        ckpt_known = bool(spec.checkpoint) and not spec.checkpoint.startswith(
            ("TODO", "BLOCKED")
        )
        rows.append(
            {
                "condition": key,
                "family": spec.family,
                "role": "unavailable"
                if unavailable
                else ("ladder" if spec.primary_ladder else "control"),
                "band": "exempt" if is_band_exempt(key) else "ladder",
                "checkpoint": spec.checkpoint or "(none required)",
                "pin": pin,
                "mel_verified": mel_verified,
                "blocked": unavailable
                or (needs_ckpt and not (ckpt_known and pin != "UNPINNED" and mel_verified)),
            }
        )
    return rows
