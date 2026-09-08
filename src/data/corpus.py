"""Source corpora and the frozen utterance / split selection.

Person A owns this. The utterance list and the split assignment are computed
once, from a fixed seed, and committed. They are then shared byte-for-byte by
every condition and both detection protocols (INV-08, INV-09).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .invariants import GLOBAL_SEED, SPLIT_FRACTIONS, SPLIT_NAMES, InvariantViolation


@dataclass(frozen=True)
class Utterance:
    utt_id: str
    path: Path
    speaker: str
    text: str | None = None


class Corpus:
    """Base class for a source corpus of real recordings."""

    name: str = "base"
    native_sample_rate: int = 0
    # INV-09. Declared here so a corpus announces its own shape, and
    # cross-checked against the utterances it actually yields -- a declaration
    # that can lie is worse than none.
    is_multi_speaker: bool = False

    def utterances(self) -> list[Utterance]:
        raise NotImplementedError


class LJSpeech(Corpus):
    """LJSpeech-1.1. Single speaker, 22.05 kHz, ~13100 utterances.

    Phase A uses a ~3000-utterance subset (see the plan). Single-speaker means
    speaker identity cannot confound conditions, which is why it is the primary
    corpus; VCTK is the speaker-diversity extension if time allows.
    """

    name = "ljspeech"
    native_sample_rate = 22_050
    is_multi_speaker = False  # one reader; speaker cannot confound conditions

    def __init__(self, root: str | Path, subset_size: int | None = 3000):
        self.root = Path(root)
        self.subset_size = subset_size

    def utterances(self) -> list[Utterance]:
        meta = self.root / "metadata.csv"
        wavs = self.root / "wavs"
        if not meta.exists():
            raise FileNotFoundError(f"LJSpeech metadata.csv not found under {self.root}")
        items: list[Utterance] = []
        for line in meta.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            parts = line.split("|")
            utt_id = parts[0]
            text = parts[2] if len(parts) > 2 else None
            items.append(
                Utterance(utt_id=utt_id, path=wavs / f"{utt_id}.wav", speaker="LJ", text=text)
            )
        items.sort(key=lambda u: u.utt_id)  # deterministic before any sampling
        if self.subset_size is not None and self.subset_size < len(items):
            rng = np.random.default_rng(GLOBAL_SEED)
            idx = np.sort(rng.choice(len(items), size=self.subset_size, replace=False))
            items = [items[i] for i in idx]
        return items


class VCTK(Corpus):
    """VCTK 0.92. Multi-speaker, 48 kHz.

    Speaker-diversity extension. When VCTK is used, splits are speaker-disjoint
    as well as utterance-disjoint (INV-09) so that matched-condition detectors
    cannot key on speaker identity.
    """

    name = "vctk"
    native_sample_rate = 48_000
    is_multi_speaker = True  # forces an explicit speaker_disjoint decision

    def __init__(self, root: str | Path, subset_size: int | None = None):
        self.root = Path(root)
        self.subset_size = subset_size

    def utterances(self) -> list[Utterance]:
        raise NotImplementedError(
            "VCTK loader: enumerate wav48_silence_trimmed/<spk>/*.flac, keep mic1 only "
            "(mic2 has a different transfer function -- a per-file confound), and "
            "return sorted Utterance rows."
        )


def assign_splits(
    utterances: list[Utterance],
    *,
    speaker_disjoint: bool | None = None,
    seed: int = GLOBAL_SEED,
) -> dict[str, str]:
    """INV-09. Deterministic utt_id -> split map, computed once and committed.

    ``speaker_disjoint`` partitions by speaker first; for a single-speaker
    corpus it is a no-op and utterance-level partitioning is sufficient.

    **It has no default for a multi-speaker corpus.** Passing ``None`` there
    raises. The old default of ``False`` meant that moving to VCTK silently
    produced speaker-overlapping splits: the same voice in train and eval, so a
    detector could key on speaker identity and score well having learned nothing
    about vocoders. That failure is invisible -- the pipeline runs, the manifest
    validates, the EER just means something else. Fail closed and make the
    caller decide.

    The gate keys on the speakers actually present, not on the corpus's
    declaration, because the observation cannot be stale.
    """
    if abs(sum(SPLIT_FRACTIONS.values()) - 1.0) > 1e-9:
        raise InvariantViolation("INV-09: split fractions must sum to 1.0")

    observed = {u.speaker for u in utterances}
    if len(observed) > 1 and speaker_disjoint is None:
        raise InvariantViolation(
            f"INV-09: {len(observed)} speakers present and no speaker_disjoint "
            "decision was made. A multi-speaker corpus split by utterance alone "
            "puts the same voice in train and eval, so a detector can score well "
            "on speaker identity while learning nothing about vocoder artifacts. "
            "Pass speaker_disjoint=True (recommended) or speaker_disjoint=False "
            "explicitly, and say which in the thesis."
        )
    if speaker_disjoint is None:
        speaker_disjoint = False  # single speaker: the two are identical

    rng = np.random.default_rng(seed)
    if speaker_disjoint:
        keys = sorted({u.speaker for u in utterances})
    else:
        keys = sorted({u.utt_id for u in utterances})

    order = rng.permutation(len(keys))
    shuffled = [keys[i] for i in order]
    n = len(shuffled)
    bounds, acc = {}, 0
    for name in SPLIT_NAMES:
        take = int(round(SPLIT_FRACTIONS[name] * n))
        bounds[name] = shuffled[acc : acc + take]
        acc += take
    bounds[SPLIT_NAMES[-1]].extend(shuffled[acc:])  # remainder to the last split

    key_to_split = {k: name for name, ks in bounds.items() for k in ks}
    if speaker_disjoint:
        return {u.utt_id: key_to_split[u.speaker] for u in utterances}
    return {u.utt_id: key_to_split[u.utt_id] for u in utterances}
