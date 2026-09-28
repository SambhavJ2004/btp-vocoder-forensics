"""Running 2019-2020 vocoder code on 2026 libraries, without pretending it is
unchanged.

HiFi-GAN (2020) and MelGAN (2019) both call two APIs that have since become
incompatible, and both die at import-or-first-use on Kaggle's current image:

  * ``librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax)`` -- librosa made this
    keyword-only, so the positional call raises ``TypeError: mel() takes 0
    positional arguments but 5 were given``;
  * ``torch.stft(...)`` without ``return_complex``, whose result they unpack as a
    real tensor with a trailing (real, imag) axis. torch 2.x returns a complex
    tensor and refuses the old form.

**Why a shim rather than a fork, a pin, or a reimplementation.** Pinning 2019
torch and librosa inside a Kaggle image is not feasible and would change every
other condition's environment (INV-13). Reimplementing the front-end would make
the condition "our version of HiFi-GAN", which is precisely what INV-02 says not
to do. Forking upstream hides the difference. A shim, recorded, is the only one
of the four that leaves the difference visible.

**What makes it honest.**

  * It is applied only AFTER the file's upstream sha256 has been recorded, and
    both digests -- before and after -- travel with the run.
  * Every rule is an exact string substitution whose replacement is
    *semantically identical* on the library versions involved: the same mel
    filterbank, the same magnitude. No behaviour is being corrected, only the
    call syntax that expresses it.
  * A rule that does not match **raises**. If upstream's text is not what this
    module expects, the right answer is to look, not to guess -- a fuzzy patch
    that silently no-ops would leave the condition running unshimmed code and
    failing later, or worse, running subtly different code.
  * ``describe()`` renders what was done, for the manifest's provenance.

This is an INV-02 matter and is recorded as one: the delivered condition runs
upstream's algorithm through two rewritten call sites, not upstream's bytes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from data.invariants import InvariantViolation


@dataclass(frozen=True)
class Rule:
    """One exact substitution, with the reason it preserves behaviour."""

    old: str
    new: str
    why: str


# librosa >= 0.10: `mel()` is keyword-only. Same filterbank, same argument order.
_LIBROSA_MEL = Rule(
    old="librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)",
    new=(
        "librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, "
        "fmin=fmin, fmax=fmax)"
    ),
    why="librosa.filters.mel became keyword-only; the positional order was "
        "(sr, n_fft, n_mels, fmin, fmax), so the mapping is exact.",
)

_LIBROSA_MEL_MELGAN = Rule(
    old="librosa_mel_fn(\n            sampling_rate, n_fft, n_mel_channels, mel_fmin, mel_fmax\n        )",
    new="librosa_mel_fn(\n            sr=sampling_rate, n_fft=n_fft, n_mels=n_mel_channels, "
        "fmin=mel_fmin, fmax=mel_fmax\n        )",
    why="Same change as HiFi-GAN's, against MelGAN's own parameter names.",
)

# torch >= 2.0: stft returns a complex tensor. |z| == sqrt(re^2 + im^2), and the
# +1e-9 upstream adds inside the sqrt is a guard against a zero gradient, which
# inference does not need -- but it is kept, outside the abs, so the numerical
# result is the same to floating point rather than merely close.
_TORCH_STFT_HIFIGAN = Rule(
    old="""    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True)

    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))""",
    new="""    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True,
                      return_complex=True)

    spec = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + (1e-9))""",
    why="torch.stft returns complex; sqrt(re^2+im^2+eps) is the same expression "
        "upstream wrote, with the axis unpack replaced by .real/.imag.",
)

_TORCH_STFT_MELGAN = Rule(
    old="""        fft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
        )
        real_part, imag_part = fft.unbind(-1)""",
    new="""        fft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
            return_complex=True,
        )
        real_part, imag_part = fft.real, fft.imag""",
    why="torch.stft returns complex; .real/.imag are the two components the old "
        "trailing-axis unbind produced.",
)

# file name as fetched -> the rules that file needs
SHIMS: dict[str, tuple[Rule, ...]] = {
    "meldataset.py": (_LIBROSA_MEL, _TORCH_STFT_HIFIGAN),
    "modules.py": (_LIBROSA_MEL_MELGAN, _TORCH_STFT_MELGAN),
}


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def apply_shims(path: Path, *, condition: str) -> dict:
    """Rewrite one upstream file in place. Returns what was done, for the record.

    Idempotent: a file already shimmed is left alone and reported as such, so a
    cached snapshot does not fail on a second load.
    """
    path = Path(path)
    rules = SHIMS.get(path.name)
    if not rules:
        return {"file": path.name, "rules": [], "shimmed": False}

    text = path.read_text(encoding="utf-8")
    before = _sha256(text)
    applied, already = [], []
    for rule in rules:
        if rule.old in text:
            text = text.replace(rule.old, rule.new)
            applied.append(rule.why)
        elif rule.new in text:
            already.append(rule.why)
        else:
            raise InvariantViolation(
                f"INV-02: {condition}: the compatibility shim for {path.name} expects "
                f"upstream text that is not there, and will not guess at a substitute. "
                f"Expected:\n{rule.old}\n"
                "Upstream's file has changed, or the wrong commit is pinned: read it "
                "and update vocoders.compat rather than loosening the match."
            )
    if applied:
        path.write_text(text, encoding="utf-8")
    return {
        "file": path.name,
        "upstream_sha256": before,
        "shimmed_sha256": _sha256(text),
        "rules": applied or already,
        "shimmed": bool(applied or already),
        "already_shimmed": bool(already and not applied),
    }


def describe(records: list[dict]) -> str:
    """One line per shimmed file, for a log or a provenance sidecar."""
    lines = []
    for record in records:
        if not record.get("shimmed"):
            continue
        lines.append(
            f"{record['file']}: {record['upstream_sha256'][:12]} -> "
            f"{record['shimmed_sha256'][:12]} ({len(record['rules'])} rule(s))"
        )
    return "\n".join(lines) or "no compatibility shims applied"
