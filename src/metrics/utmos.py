"""UTMOS — learned MOS predictor. Primary quality axis.

Primary because human listening tests are out of scope for a BTP budget, and
because it is non-intrusive: it scores the condition on its own terms rather
than against the reference, which keeps the X-axis conceptually independent of
the pairing that Phase A provides.

**Runs on the DERIVED zero-shot tier, not the archive.** UTMOS22 is a 16 kHz
model; scoring it at the archive rate would mean either resampling inside the
metric (a second filtering pass, forbidden by INV-01) or feeding the model
out-of-distribution input. It is one of the three external constraints that
force the derived tier to exist, alongside PESQ-WB and the pretrained ASVspoof
detectors.

Caveats to state in the thesis rather than discover in the viva:
  - As the primary X-axis metric, UTMOS is blind to everything above 8 kHz --
    the band the mechanism argument turns on. The headline correlation is
    therefore between a band-limited quality estimate and a full-band
    detectability estimate. That is defensible (it is how a listener-facing
    metric behaves) but it must be said, not assumed.
  - UTMOS is trained on VCC/BVCC listening data and is itself a neural model.
    A vocoder unlike anything in that training set can be scored unreliably.
  - It is a *predictor* of MOS, not MOS. Report it as such.
  - It is level-sensitive, so the loudness invariant (INV-04) is load-bearing
    here as well as on the detection side. Note the derived tier is deliberately
    not re-normalised, so its measured loudness sits a fraction of a LU below
    the archive target; that offset is identical across conditions by
    construction and so cannot bias the comparison.
"""

from __future__ import annotations

import numpy as np

from data.invariants import ZEROSHOT_SR, InvariantViolation

UTMOS_HUB = ("tarepan/SpeechMOS:v1.2.0", "utmos22_strong")


class UTMOSScorer:
    """Lazy wrapper so importing this module never pulls torch."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._model = None

    def load(self) -> None:
        import torch

        repo, name = UTMOS_HUB
        self._model = torch.hub.load(repo, name, trust_repo=True).to(self.device).eval()

    def score(self, wav: np.ndarray, sr: int = ZEROSHOT_SR) -> float:
        if sr != ZEROSHOT_SR:
            raise InvariantViolation(
                f"INV-01: UTMOS22 is a {ZEROSHOT_SR} Hz model; got {sr}. Score it on "
                "the derived zero-shot tier -- resampling here would be a second "
                "filtering pass on the signal."
            )
        if self._model is None:
            self.load()
        import torch

        with torch.no_grad():
            x = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0).to(self.device)
            return float(self._model(x, sr).squeeze().cpu())
