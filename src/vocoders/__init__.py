"""Vocoder adapters and the quality ladder. Person A.

Importing this package is cheap: no adapter loads torch or a checkpoint until
its ``load()`` is called, so a manifest audit works in an environment with no
vocoder dependencies installed at all (INV-13).
"""

from .base import ResynthesisResult, Vocoder, VocoderSpec
from .registry import LADDER, SPECS, V0_CONDITIONS, checkpoint_audit, get_spec, get_vocoder

__all__ = [
    "LADDER",
    "SPECS",
    "V0_CONDITIONS",
    "ResynthesisResult",
    "Vocoder",
    "VocoderSpec",
    "checkpoint_audit",
    "get_spec",
    "get_vocoder",
]
