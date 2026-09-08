"""INV-08 — checkpoint verification at download time.

Two pinning mechanisms, because the ladder has two kinds of checkpoint:

  HuggingFace-hosted (`bigvgan_*`, `vocos`)
      Pinned by repo revision SHA. Content-addressed over the whole repo,
      verifiable before downloading anything, and passed to `from_pretrained`
      so the load itself is pinned rather than merely documented.

  Everywhere else (`hifigan_v1` via Google Drive, `melgan` via torch.hub)
      No revision exists to pin. The only durable identifier is the digest of
      the weight file itself, which has to be computed after downloading and
      compared against a recorded expectation.

The second case is what this module implements. Record the digest in the
condition's `VocoderSpec.checkpoint_sha256` the first time a weight file is
fetched, then every subsequent fetch is verified against it.

A mismatch is not a warning. A checkpoint that changed under a fixed name is an
unlogged change to a condition, and it looks exactly like a result (INV-08).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from data.invariants import InvariantViolation

_CHUNK = 1 << 20  # 1 MiB


def sha256_file(path: str | Path) -> str:
    """Streaming SHA-256 of a file. Checkpoints are too big to slurp."""
    path = Path(path)
    if not path.is_file():
        raise InvariantViolation(f"INV-08: no checkpoint file at {path}")
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint(path: str | Path, expected_sha256: str, *, condition: str) -> str:
    """Assert a downloaded weight file matches its recorded digest.

    Call this immediately after download, before the weights are loaded and
    before any audio is generated from them. Returns the digest so a caller
    recording a digest for the first time can capture it.
    """
    if not expected_sha256:
        raise InvariantViolation(
            f"INV-08: '{condition}' has no recorded checkpoint_sha256 to verify "
            "against. Compute it with sha256_file() and record it in the "
            "condition's VocoderSpec before generating any audio -- an unpinned "
            "checkpoint means the condition is not reproducible."
        )

    actual = sha256_file(path)
    if actual != expected_sha256.lower().strip():
        raise InvariantViolation(
            f"INV-08: checkpoint digest mismatch for '{condition}'.\n"
            f"  expected {expected_sha256}\n"
            f"  actual   {actual}\n"
            f"  file     {path}\n"
            "The weights behind this name are not the weights the recorded "
            "results came from. Do NOT proceed: regenerate the condition, or "
            "restore the pinned checkpoint. A swapped checkpoint is an unlogged "
            "change to a condition and looks exactly like a result."
        )
    return actual


def fetch_and_verify(path: str | Path, spec, *, allow_first_use: bool = False) -> str:
    """Verify a downloaded checkpoint against its spec.

    ``allow_first_use`` is the one-time escape for recording a digest that does
    not exist yet: it computes and returns the digest instead of raising. It
    prints the value to paste into the spec, and is deliberately awkward -- it
    must never be the default path, or the pin means nothing.
    """
    if not spec.checkpoint_sha256:
        if not allow_first_use:
            raise InvariantViolation(
                f"INV-08: '{spec.key}' has no recorded checkpoint_sha256. Run once "
                "with allow_first_use=True to compute it, paste the value into the "
                "spec, and commit that before generating audio."
            )
        digest = sha256_file(path)
        print(
            f"[INV-08] first use of '{spec.key}': record this in its VocoderSpec\n"
            f'         checkpoint_sha256="{digest}",'
        )
        return digest
    return verify_checkpoint(path, spec.checkpoint_sha256, condition=spec.key)
