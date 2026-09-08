# CLAUDE.md — working rules for this repository

**Project:** *Does better-sounding mean harder-to-detect?* Measuring whether
perceptual quality and forensic detectability are orthogonal properties of a
neural vocoder. B.Tech project, NSUT, team of 3. Full design in
[btp-project-idea.md](btp-project-idea.md).

The entire study rests on one claim: **vocoder identity is the only free
variable.** Six vocoders regenerate the same recordings, and any measured
difference is therefore attributable to the vocoder. Every invariant below
exists to protect that claim. They are not preferences, and they are not
tunable defaults — **violating any one of them silently invalidates the study.**

Silently is the operative word. A leaked confound does not crash anything. It
produces a clean, plausible, publishable-looking plot of the wrong thing.

---

## The invariants

INV-01 to INV-05 are the confound controls from the plan's Phase A table.
INV-06 to INV-17 are their operational consequences — decisions the table
implies but does not spell out. Both kinds are binding.

Machine-readable copy: [`src/data/invariants.py`](src/data/invariants.py).
Code imports its constants from there so that no pipeline can quietly disagree
with this document.

Each invariant ends with an **Enforced by** clause naming the code that holds it
up. That clause is prose and can drift from the code while every test still
passes — which has happened. [`docs/enforcement_audit.md`](docs/enforcement_audit.md)
records the last audit of those claims against the code, with open divergences.

---

### INV-01 — Two rates, two single-step derivations. Archive high, derive low.

**Rule.** The **archive** is the primary artifact: **22.05 kHz mono**, forced by
LJSpeech's native rate. Vocoders resample **once**, from their own native output
rate to the archive rate.

The **16 kHz zero-shot set is a separate derived artifact**, produced by one
further downsample **from the archive** — never from source, and never fed back
into anything.

Two derivations, each a single filtering step. **Never chain.** No record may
carry more than one resample step, and the manifest rejects one that does.

```
source / vocoder native ──resample_once──▶ ARCHIVE 22.05 kHz   (primary)
                                                 │
                                                 └──downsample_once──▶ ZEROSHOT 16 kHz   (derived)
```

**Why.** The mechanism the study exists to measure is high-frequency vocoder
artifacts. At 16 kHz, Nyquist is 8 kHz, so delivering everything at 16 kHz would
low-pass away the exact evidence the band-limit ablation is built to find — the
study would have destroyed its own dependent variable before measuring it. At
22.05 kHz, Nyquist is 11.025 kHz and the ~8 kHz region stays real, measurable
content.

But the zero-shot protocol genuinely needs 16 kHz: pretrained ASVspoof detectors
expect it, and so do PESQ-WB (defined only at 16 kHz) and UTMOS22 (a 16 kHz
model). Hence two tiers rather than one compromise rate, which would serve
neither.

The direction is not symmetric and not negotiable. Deriving low from high
discards information the archive still holds. Deriving high from low would be
inventing it. And deriving the 16 kHz set from *source* rather than from the
archive would create two independent resampling paths whose anti-aliasing
imprints differ, so the two tiers would no longer be the same audio — a
per-condition label, which is the original confound wearing a new hat.

Chaining is forbidden for the reason it always was: a resampler is a filter, and
its rolloff is a signature. One pass is a recorded fact; two passes stack a
second filter's imprint that no manifest column describes.

**The one condition that takes a resample is measured, not assumed.** Vocos is
24 kHz native, so it is the only ladder condition whose output passes through a
resample (24000 → 22050) on the way to the archive. By the argument above that
resampler's rolloff is a per-condition signature — and the rolloff sits at
~10-11 kHz, above `LADDER_FMAX`, so INV-17 should erase it. That is an argument,
and the same shape of argument was wrong for Butterworth (see INV-17), so it was
tested.

Method: identical source audio down two paths —
`22050 → 24000 → 22050 → INV-17 → loudness → PCM_16` against
`22050 → INV-17 → loudness → PCM_16` — over 60 utterances, then three levels of
evidence.

| evidence | result |
|---|---|
| out-of-band energy, round-tripped | 5.61e-09 mean (max 6.27e-09) |
| out-of-band energy, direct | 5.61e-09 mean (max 6.27e-09) |
| `PCM16_OOB_FLOOR` | 2.0e-08 — both paths below it |
| A-vs-B relative RMS difference | 1.01e-05, i.e. **−99.9 dB** |
| largest per-band level difference | **0.0003 dB** (7-8 kHz band) |
| PCM_16 samples that differ at all | **379 / 1,766,178 = 0.021%**, never by more than 1 LSB |
| sanity probes (duration, silence, RMS, bandwidth) | **0.500 EER, all four** |
| logistic regression, 32 log-band energies | accuracy 0.508, **AUROC 0.502** |
| logistic regression, 513-bin log spectrum | accuracy 0.508, **AUROC 0.501** |

A spectral residual below the floor would not have been sufficient on its own —
that was the Butterworth mistake, where an identical filter attenuated without
equalising. What settles it is that a linear classifier given the full spectrum
cannot beat chance, and that the two paths produce **byte-identical PCM_16
output for 99.98% of samples**. The round trip is erased before quantisation and
quantisation finishes the job.

So Vocos's resample is exempted from the "rolloff is a signature" concern on
evidence. Two caveats on that evidence: it was measured on synthetic
speech-like audio (harmonic stacks with broadband noise), not on LJSpeech, and
it holds *because* `LADDER_FMAX` sits well below the resampler transition band.
**Re-run it if `LADDER_FMAX` ever rises above ~10 kHz, or if a condition is
added whose native rate is closer to 22.05 kHz** — a 22.05 → 22.05 "resample" is
a no-op, but 32 kHz → 22.05 kHz would put the transition somewhere new.
Reproduced by `TestVocosResampleSignature`.

**Constants.** `ARCHIVE_SR = 22_050`, `ZEROSHOT_SR = 16_000`,
`MAX_RESAMPLE_STEPS = 1`, `RESAMPLE_METHOD = "soxr_hq"` — the method is frozen
too, for the same reason the rates are. There is deliberately **no single
delivery-rate constant**: code names the tier it means.

**Which tier does what.**

| Consumer | Tier | Because |
|---|---|---|
| Band-limit ablation | archive | needs content above 8 kHz to exist |
| Band-wise LSD, MCD, F0 | archive | they measure the high band |
| Matched protocol | archive | upper bound on separability, uncapped |
| Bandwidth sanity probe | archive | derived tier shares one downsampler, always looks clean |
| Mismatched protocol | zeroshot | pretrained LA weights expect 16 kHz |
| PESQ-WB | zeroshot | defined only at 16 kHz |
| UTMOS | zeroshot | 16 kHz model |

Consequence to state in the thesis, not discover in the viva: the **matched EER
is measured on more signal than the mismatched EER**. The gap between the two
protocols therefore has two causes — the deployed detector's blind spot *and*
bandwidth — and cannot be attributed wholly to generalisation failure. Likewise
UTMOS and PESQ are blind above 8 kHz, so the headline correlation relates a
band-limited quality estimate to a full-band detectability estimate.

**Loudness interaction.** The derived tier is **not re-normalised**.
Downsampling discards energy above 8 kHz, which lowers measured loudness
slightly. Re-normalising would apply a gain *proportional to each file's
high-band energy* — encoding the artifact under study directly into the gain,
which is the worst available outcome. The drift is recorded in the manifest and
bounded by `DERIVED_LOUDNESS_DRIFT_LIMIT_LU = 2.0`; a larger drift means an
unintended second gain stage.

**Enforced by.** `data.preprocess.resample_once` / `derive_zeroshot`,
`data.invariants.check_waveform` (on every write, with `expected_sr`),
`data.audio_io.write_audio` (`sample_rate` is required — no default),
`data.manifest.validate_manifest` (tier/rate agreement, `resample_steps <= 1`,
derived rows must come from `ARCHIVE_SR`), `data.manifest.require_primary`
(refuses derived rows where the high band is being measured),
`detectors.protocols.DetectionResult` (carries `tier`; refuses a band limit at
or above tier Nyquist), `experiments.sanity_checks.bandwidth_probe`.

**Note.** Source audio *is* resampled to the vocoder's training rate before
analysis. That is not a second resample of a delivered file — it is a separate
signal path, on the input side, and it must be, because feeding a vocoder audio
at the wrong rate takes it out of distribution (see INV-02).

---

### INV-02 — Mel configuration: cannot be equalised; document per condition

**Rule.** Each vocoder keeps its own native mel front-end (`n_fft`, hop, window,
`n_mels`, `fmin`, `fmax`, log convention). Copy the numbers from the upstream
release, never guess them, and record them per condition in the manifest. Use
the vocoder's own extractor where it ships one, in preference to
`data.mel.compute_mel`.

**Why.** This is the one confound the design cannot remove, and pretending
otherwise would be worse than admitting it. Forcing a shared mel puts each
vocoder outside its training distribution — the measurement would then be of a
deliberately broken vocoder, not of the real one. So the honest control is
documentation: any EER difference between conditions is attributable to
"vocoder **and its own front-end** as a unit", and the thesis says exactly that
rather than implying a cleaner isolation than exists.

**Consequence for writing.** This belongs in the limitations section as a
stated scope boundary, not buried. `data.manifest.mel_config_table` generates
the per-condition table for it.

**Scope, after the Step-1 audit.** Every config in this ladder is numerically
identical except `fmax`, and INV-17 equalises the delivered band. What INV-02
still covers is implementation difference between extractors, not parameter
difference. See the INV-17 conflict note below and
[`docs/mel_configs.md`](docs/mel_configs.md).

**Enforced by.** `data.mel.MelConfig.source` — a config whose `source` still
starts with `TODO` is unverified, and `vocoders.registry.checkpoint_audit`
reports that condition as blocked.

**Currently dormant**: every condition's `source` is filled in, so the branch is
never taken. It is the kind of check that rots unnoticed once nothing trips it;
`TestLadder` exercises it against a synthetic spec rather than relying on a real
condition to stay broken.

---

### INV-03 — Silence: identical trimming, from the real reference

**Rule.** Trim boundaries are computed **once, on the real reference**, and
applied **by sample index** to every condition of that utterance. Never run a
trimmer independently per condition.

**Why.** Prior work on ASVspoof 2019 LA showed detectors partially keying on
leading and trailing silence duration rather than on spoofing artifacts — a
detector can score well while learning nothing about generation at all. Vocoders
differ in padding behaviour, so running a silence trimmer per condition would
produce systematically different silence durations *between conditions* and
hand the detector exactly that cue. The trim would have introduced the confound
it exists to remove.

Deriving the span from the reference and reusing it is what makes "identical
trimming" literally true rather than approximately true.

**Constants.** `TRIM_TOP_DB = 30.0`, `TRIM_FRAME_LENGTH = 512`,
`TRIM_HOP_LENGTH = 128`, `TRIM_PAD_MS = 20.0`.

**Enforced by.** `data.preprocess.derive_trim_span` / `apply_trim_span`,
`data.manifest.validate_manifest` (rejects per-condition trim spans),
`experiments.sanity_checks.silence_probe`.

---

### INV-04 — Loudness: one normalisation target, no per-file exceptions

**Rule.** Every condition is normalised to `-27.0 LUFS` (ITU-R BS.1770).
When a file would exceed the `-1.0 dBFS` peak ceiling, **drop that utterance
from every condition** — never attenuate it in one.

**Why.** Level differences between conditions are trivially learnable; a
detector can reach a low EER on RMS alone, having learned nothing forensic. The
X-axis needs it too: UTMOS and PESQ are both level-sensitive, so an
un-normalised set would confound *both* axes at once and the correlation between
them would be partly an artifact of gain.

The no-exceptions clause is the part that gets rationalised away under time
pressure. Quietly attenuating one clipping file makes that file's loudness
different from every other file in its condition — which is precisely the
confound the invariant exists to prevent. Dropping the utterance everywhere
keeps both the loudness guarantee and the pairing guarantee (INV-06) intact.
A conservative target is chosen so this rarely triggers.

**Enforced by.** `data.preprocess.normalise_loudness` + `would_clip`,
`data.invariants.check_waveform` (verifies against `LOUDNESS_TOLERANCE_LU` on
**archive** writes — the check is conditional on a `measured_lufs` being passed,
and `_emit_pair` deliberately omits it for the derived tier, which is not
re-normalised and legitimately drifts; that tier is bounded instead by
`check_derived_loudness`), `experiments.sanity_checks.loudness_probe`.

---

### INV-05 — Encoding: one container, one bit depth, everywhere

**Rule.** 16-bit PCM WAV for every condition, including real. No MP3, no FLAC,
no Opus — not as a final format and not as an intermediate. All audio I/O goes
through `data.audio_io`.

**Why.** Codecs and bit depths leave their own traces: quantisation noise
floors, dither, and in lossy formats the codec's own spectral fingerprint,
which is a far stronger signal than any vocoder artifact. A single condition
that passed through a lossy step at any point is separable on the codec alone.
A bit-depth difference is subtler and works the same way. Intermediates count —
compression is not undone by decompressing.

**Enforced by.** `data.audio_io.write_audio` refuses any non-`.wav` path. The
subtype is not *refused* but made unreachable: `subtype=PCM_16` is fixed inside
the call and the function exposes no parameter for it, so no caller can supply
another. `data.manifest.validate_manifest` re-checks the recorded subtype at the
table level, which is what catches a file written by some other route.

---

### INV-06 — Pairing: every condition holds exactly the real utterance set

**Rule.** Identical utterance IDs across every condition. Identical sample
lengths within an utterance. A drop is a drop from *all* conditions.

**Why.** Two things depend on it. Intrusive metrics (PESQ, MCD, F0 error) need
a genuine sample-aligned reference, which is the specific advantage resynthesis
buys over TTS. And the controlled-experiment claim needs the sets to be equal —
if condition A has 3000 utterances and condition B has 2950 because 50 failed to
synthesise, the missing 50 are probably the hard ones, and the comparison is
between different content as well as different vocoders.

**Enforced by.** `data.invariants.check_pairing`,
`data.preprocess.align_length` (errors beyond 50 ms rather than silently
padding), `detectors.protocols.score_condition`,
`experiments.sanity_checks.duration_probe`.

---

### INV-07 — Pipeline order is fixed

**Rule.** `resample_once → align_length → apply_ref_trim → normalise_loudness →
encode`. Same order, every condition, no exceptions.

**Why.** These steps do not commute. Measuring loudness before trimming
measures a different span of signal, so the same target lands at a different
actual level. Trimming after normalisation changes what the level was computed
over. If two conditions were processed in different orders — easy to do when
one is regenerated later — they end up with a systematic level or duration
offset that no individual step is responsible for.

**Scope.** Every condition now runs the same `PIPELINE_ORDER`, with no
exceptions at all. The band-exempt variant existed because INV-17 filtered at
generation and controls skipped that step; INV-17 no longer filters at
generation, so the exception is gone and INV-07's original "no exceptions"
wording is literally true again.

**Enforced by.** `data.preprocess.finalise_trimmed` is the **single
implementation** of the post-trim ordering (band-limit → loudness). It is
reached through two thin entry points that differ only in what they do *before*
the trim, which is the only thing that genuinely differs between conditions:

| entry point | used by | pre-trim work |
|---|---|---|
| `finalise_reference` | the real condition | derives the span, then trims by it |
| `process_condition_output` | every vocoder condition | resample → align_length → trim by the given span |

`PIPELINE_ORDER` records the order. **The equivalence is asserted by test, not
by this prose**: `TestINV07SingleOrdering` drives both entry points with
identical input and span and requires byte-identical output, and drives Phase A
end to end with an identity vocoder and requires the real and vocoder archive
files to be byte-identical.

That test is what makes INV-07 true. Before it existed, `build_real_condition`
open-coded the post-trim steps and the two implementations agreed only by
inspection — the invariant's own failure mode living inside its enforcement.

---

### INV-08 — Determinism: fixed seeds, pinned checkpoints, recorded provenance

**Rule.** `GLOBAL_SEED = 20260907` for subset selection and splits. Pin every
checkpoint's SHA-256 in its `VocoderSpec`. Fix sampler step counts and seeds for
any stochastic vocoder. Every manifest gets a provenance sidecar with the git
commit.

**Why.** A swapped checkpoint is an unlogged change to a condition, and it looks
exactly like a result. Six months of work across three people and three Kaggle
accounts will otherwise produce audio nobody can attribute to a specific model
version. This is also the thesis's reproducibility section — it is cheaper to
record it continuously than to reconstruct it in April.

**Two pinning mechanisms, because the ladder has two kinds of checkpoint.**

*HuggingFace-hosted* (`vocos`, `bigvgan_*`) pin by **repo revision SHA**:
content-addressed over the whole repo, verifiable before downloading anything,
and passed to `from_pretrained` so the load itself is pinned rather than merely
documented. Omitting the revision silently tracks the branch head.

*Everything else* (`hifigan_v1` via Google Drive, `melgan` via torch.hub) has no
revision to pin, so the only durable identifier is the **digest of the weight
file**, which must be computed after download and compared against a recorded
expectation. `vocoders.checkpoints.verify_checkpoint` does this and raises on
mismatch; `fetch_and_verify(..., allow_first_use=True)` computes a digest the
first time and prints it to paste into the spec. That escape is deliberately
awkward — if it becomes the default path the pin means nothing.

Either mechanism satisfies INV-08. The audit's `pin` column reports which one is
in force, so "pinned" is never a bare boolean hiding two different guarantees.

**Enforced by.** `vocoders.registry.checkpoint_audit` (no pin ⇒ blocked),
`data.manifest.dump_provenance`, `data.corpus.assign_splits` (seed).

`vocoders.checkpoints.verify_checkpoint` (digest mismatch ⇒ raise) exists and is
correct, but **activates with `load()`**: every adapter's `load()` is currently
`NotImplementedError`, so nothing verifies a digest on any real path yet. It is
a tool waiting for its caller, not an active guard.

The rule's *"fix sampler step counts and seeds for any stochastic vocoder"* has
**no mechanism at all** and is aspirational — it depends on whoever implements a
diffusion-style adapter honouring it.

---

### INV-09 — Splits are frozen and shared

**Rule.** Train/dev/eval assignment is a property of the **utterance**, computed
once from `GLOBAL_SEED`, committed, and shared byte-for-byte across every
condition, every detector and both protocols. Speaker-disjoint as well when the
corpus is multi-speaker.

**Why.** Content overlap between train and eval inflates matched-protocol
separability, and that inflation would be read as artifact information — a
direct error in the Y-axis. Re-splitting per condition is worse: each condition
then gets a differently-easy eval set, so the six EERs are no longer on a common
scale and the main plot compares points that were never comparable.

**Enforced by.** `data.corpus.assign_splits`,
`data.manifest.validate_manifest` (one split per utterance across conditions).

---

### INV-10 — No post-processing on vocoder output

**Rule.** Adapters return raw generator output at the native rate. No denoising,
EQ, peak limiting, dithering, DC removal, or silence padding. Normalisation
happens once, for every condition identically, in `data.preprocess`.

**Why.** Post-processing is a filter applied to one condition and not others,
which is the definition of a confound here. It also cuts against the research
question: a denoiser could plausibly remove the very high-band artifact the study
is trying to measure, so the result would be about the denoiser. If a vocoder's
upstream inference script includes a post-processing step by default, disable it
and note it in the spec.

**Scope.** This governs the **adapter** — anything in `vocoders/*` that could
differ per condition. Dataset-wide pipeline steps in `data.preprocess` that are
applied identically to every condition, real included (resampling, trimming,
loudness normalisation, and the INV-17 band limit), are not post-processing in
the sense INV-10 forbids. The test is "does any condition get something the
others do not?", not "is it a filter?". See the INV-17 conflict note.

**Enforced by.** Convention and review; `vocoders.base.Vocoder.synthesize`
documents it as part of the contract.

---

### INV-11 — Detectors see identical inputs

**Rule.** Same crop **duration** (4.0375 s, the ASVspoof convention), same
repeat-padding policy, same batching, across every condition and every protocol.

The convention is usually quoted as *64600 samples*, and that is correct only at
16 kHz. What carries across tiers is the duration: `crop_samples_for_rate(sr)`
returns **64600 at 16 kHz** and **89027 at 22.05 kHz**. Reusing the bare integer
at the archive rate would silently shorten the crop to 2.93 s, so the matched
protocol would see less signal than the mismatched one — the opposite of the
intent, and invisible.

**Why.** Zero-padding short utterances would make padded-silence duration a
function of utterance length — reintroducing INV-03's confound at the data
loader instead of the dataset, where none of the audio-level checks would catch
it. Repeat-padding avoids that. Differences in crop length between conditions
would likewise change how much signal each condition's detector sees.

**Enforced by.** `detectors.base.crop_samples_for_rate` (rate → length) and
`detectors.base.fixed_length_crop` (crop/pad), reached through
`detectors.base.Detector.crop_for`, which is the sanctioned entry point every
adapter's `score`/`fit` calls instead of cropping for itself.

**Not yet exercised**: every adapter's `score()` is `NotImplementedError`, so
nothing calls `crop_for` on a real path today. The policy is one function rather
than a convention, which is what makes it enforceable once adapters land — but
until then this is a contract, not an active guard.

---

### INV-12 — The sanity gate runs before any result is believed

**Rule.** `btpvf sanity --manifest ...` runs on every dataset version before
Person B or C consume it. Probes on duration, silence and RMS run on **both**
tiers; the bandwidth probe runs on the **archive tier only**, because on the
derived tier every condition has passed through the same downsampler and the
probe would report a reassuring 0.5 while saying nothing. All probes that run
must sit near chance (EER ≥ `SANITY_EER_FLOOR = 0.40`). A leak **blocks**
publication of that condition.

**Why.** This is the plan's own mitigation for "confound leakage invalidating
results", made executable. Each probe trains on a feature that carries no
vocoder artifact whatsoever; if such a feature separates real from vocoded, the
separation is a confound and any EER measured on that condition is measuring the
wrong thing. Cheap, CPU-only, and the difference between a defensible result and
a retracted one.

**Never** fix a leak by lowering the floor. Fix the pipeline and regenerate.
The bandwidth probe is the one legitimate grey area — a genuine high-band
vocoder artifact also moves effective bandwidth — so read a hit there as
"inspect the spectra", not automatically as "confound".

**Enforced by.** `experiments.sanity_checks.run_all`; the CLI exits non-zero on
a leak.

---

### INV-13 — One vocoder per environment

**Rule.** Never install two vocoders into the same environment. One isolated
Kaggle session per condition, output persisted to a Kaggle Dataset, manifests
merged afterwards. Importing this package must never require any vocoder's
dependencies.

**Why.** Their dependency trees genuinely conflict, and the failure mode is not
an install error — it is pip silently resolving to a different `torch` or
`librosa` version for one condition than another. Library version differences
change STFT edge behaviour and resampling filters, which lands straight on
INV-01 and INV-02. Isolation makes each condition's environment a recorded fact.

**Enforced by.** Lazy imports throughout; adapters load nothing until `load()`.
`TestINV13LazyImports` asserts that importing `vocoders`, `detectors`, `data`
and `metrics` pulls none of `torch`, `torchaudio`, `pesq` or `pyworld` — the
claim was true but unchecked, so a stray top-level `import torch` would have
passed CI and surfaced only as a dependency conflict on Kaggle.

---

### INV-14 — Matched and mismatched are never conflated

**Rule.** Every EER carries its protocol label. Never average, pool, or plot
them together without the distinction being visible. A detector that has seen
any project audio produces a **matched** result, whatever the intent.

**Why.** They answer different questions. *Mismatched* (an ASVspoof-2019-LA
detector, zero-shot, no adaptation) measures generalisation failure of deployed
systems — practical relevance. *Matched* (trained on real-vs-vocoder-X) measures
how much artifact information the signal contains at all, independent of one
detector's blind spots — an upper bound on separability, scientific relevance.
Conflating them is a recurring weakness in this literature, and separating them
cleanly is a stated contribution of this work. Losing the distinction inside our
own results would be an unusually embarrassing way to lose the contribution.

The gap between the two is itself a finding: it is how much of a deployed
detector's failure is blind spot rather than missing signal.

**Enforced by.** `detectors.protocols.DetectionResult.__post_init__` (a
mismatched result with in-project training conditions raises; a matched result
without them raises), `assert_protocols_not_pooled`.

---

### INV-15 — Credentials never enter the repository

**Rule.** No `kaggle.json`, `.env`, tokens, or keys — committed, pasted into a
notebook cell, or hard-coded. Credentials come from environment variables;
`.env.example` documents the names and holds no values.

**Why.** Three people, three Kaggle accounts, and notebooks that get shared and
forked. A leaked key is trivially recoverable from git history even after the
file is deleted, and it is not the team's data alone that is at risk.

**Enforced by.** [`.gitignore`](.gitignore). If a credential is ever committed,
rotate the key first, then clean the history — in that order.

---

### INV-16 — Every vocoder must be sample-aligned with its input

**Rule.** When a vocoder is first onboarded, cross-correlate real against
resynthesized over **20 files** (`ALIGNMENT_PROBE_FILES`). The peak lag must be
**exactly 0 samples**. Anything else blocks the condition and fails the manifest
gate. Fix the adapter — do **not** compensate by shifting the audio.

**Why.** Reference-derived trim spans (INV-03) apply the real reference's sample
indices to every condition. That is only meaningful if the vocoder's output is
sample-aligned with its input. A fixed algorithmic delay — a padding convention,
a centre/non-centre STFT mismatch, a filterbank group delay — shifts the whole
output by a constant.

The damage is specific, silent, and points the wrong way. A shifted condition
would have the reference trim span land on the wrong span of speech; would score
badly on MCD, PESQ and F0 error for a reason unrelated to reconstruction
quality, dragging the X-axis down; and would present a strong, trivially
learnable cross-correlation cue to the detector, dragging the Y-axis up.

**Both axes move together, in the direction that manufactures the correlation
the study is testing for.** A pure time shift would be read as a vocoder
"artifact" and would look exactly like the headline result. Of all the ways this
study can go wrong, this is the one that produces the most convincing wrong
answer.

A compensating shift is rejected because it is a per-condition operation applied
to one condition and not others — precisely what INV-10 exists to prevent.

**Constants.** `ALIGNMENT_PROBE_FILES = 20`, `ALIGNMENT_REQUIRED_LAG = 0`,
`ALIGNMENT_MAX_LAG_SAMPLES = 2048` (a search window, *not* a tolerance).

The modal lag across the probed files is used rather than the mean: a genuine
fixed delay shows the same integer on nearly every file, while an occasional
outlier is a file whose correlation locked onto a pitch period. A mean would
blur a real 1-sample delay into something that looks like rounding.

**Enforced by.** `data.alignment.check_vocoder_alignment` + `assert_aligned`
(run inside `experiments.phase_a_resynthesis.build_vocoder_condition`, before a
single file of the condition is written), `data.manifest.validate_manifest` via
the `alignment_checked` / `alignment_peak_lag` columns, and the
`<manifest>.alignment.json` sidecar.

---

### INV-17 — Full-band archive; the band limit is an ANALYSIS transform

**Rule.** The archive is **full-band**. Nothing is low-passed at generation.

Band-limiting to `LADDER_FMAX` is an **analysis-time transform**,
`data.preprocess.band_limit_comparison_set`, applied on request to an entire
comparison set — **including the real reference** — and never to one member of
it. `LADDER_FMAX` is *derived*: `min(audited fmax)` over the constraining
primary conditions, read from [`docs/mel_configs.md`](docs/mel_configs.md) via
`invariants.AUDITED_MEL_FMAX`, never written as a literal. It currently
evaluates to **8000.0 Hz**.

**Why (the premise this invariant was built on was false).** The original INV-17
low-passed at generation because an `fmax = 8000` mel front-end was assumed to
produce no output above 8 kHz. It does not. **`fmax` constrains the ANALYSIS the
vocoder consumes, not the synthesis it performs**: a time-domain upsampling
vocoder emits content across the full band regardless of what the mel carried.

Measured on 20 LJSpeech files, fraction of energy above 8 kHz:

| condition | >8 kHz fraction | |
|---|---|---|
| real | **0.01803** | |
| BigVGAN | **0.01480** | tracks real per-file |
| Griffin-Lim | **0.00000** | exactly zero |

Griffin-Lim is the exception, and it is the exception that produced the error.
It inverts the mel to a linear spectrogram and runs ISTFT, so it genuinely
cannot exceed `fmax`. **The cliff was measured on Griffin-Lim and generalised by
config inspection to conditions where it does not hold.**

What the old rule cost: everything above `LADDER_FMAX` in a neural vocoder's
output is **hallucinated** — the mel carried no information there, so the model
invented it from its prior. That is the most forensically interesting content on
the ladder, and generation-time filtering destroyed precisely it.

**This mirrors INV-01.** Archive high, derive low. The reasoning is identical
and was already written down one invariant earlier: information discarded at
generation cannot be recovered, while any narrower view can always be derived
from a wider archive. A band-limited comparison set is derivable from a
full-band archive; a full-band archive is not recoverable from band-limited
files. INV-01 applies it to sample rate, INV-17 applies it to bandwidth, and
INV-17 got it backwards until this was measured.

**What did not change.** The filter itself (`band_limit_to_ladder`, zero-phase
Chebyshev II order 12 / 100 dB), the measured floors, and the gate. They moved
from generation to analysis; they were not weakened. `check_band_limit` now
gates the *output of the analysis transform* rather than the archive — the
archive is full-band and would fail that check on every file, which is the point.

**The real reference is band-limited too, on the same terms.** That has not
changed either, only when it happens. `band_limit_comparison_set` takes the
whole set at once and refuses a set with no `real` member, because band-limiting
only the vocoded conditions leaves real holding a high band they lack — the same
separable cliff with its sign flipped.

**The argument still depends on PCM_16.** Detector inputs are read back from
written 16-bit files (`detectors.protocols.score_condition` → `read_processed`),
which is what destroys the filter's own float32 residual. Unchanged by this
rewrite, and still a contract rather than a coincidence: do not hand a detector
in-memory Phase A audio.

**Constants.** `LADDER_FMAX` (derived, 8000.0), `MEASUREMENT_FLOOR = 1e-15`,
`PCM16_OOB_FLOOR = 2e-8`, `BAND_LIMIT_FLOOR_MARGIN = 50`,
`BAND_LIMIT_FILTER_ORDER = 12`, `BAND_LIMIT_STOPBAND_DB = 100.0`,
`BAND_LIMIT_FILTER_SPEC`, `BAND_LIMIT_MAX_STOPBAND_ENERGY = 1e-6`.

Zero-phase (`sosfiltfilt`) for two reasons: a causal filter's group delay is
itself a phase artifact sitting in the band under study, and it would shift the
condition out of sample alignment with its reference, tripping INV-16.

**Chebyshev Type II, not Butterworth.** An identical filter *attenuates* but
does not *equalise*, so whatever it leaves behind is still a difference between
two signals that entered with different high-band content. Measured on a
harmonic-rich 22.05 kHz signal, as a fraction of total energy above the band:

| design | residual | verdict |
|---|---|---|
| unfiltered | 1.95e-03 | |
| Butterworth order 8 | 2.55e-05 | 4000x the quantisation floor — still separable |
| Butterworth order 24 | 2.15e-06 | 350x — still separable |
| **Chebyshev II order 12, 100 dB** | **1.78e-16** | below the floor |

The target is not "small" but **below the noise floor of the delivered format**.

**Measuring the stopband needs a Blackman window.** Not Blackman-Harris:
Blackman-Harris minimises the *peak* sidelobe (−92 dB) but has flat asymptotic
rolloff, while Blackman's peak is worse (−58 dB) and it rolls off at
−18 dB/octave, which is what matters at this frequency distance. Measured floors
on an off-bin tone: rectangular 3.0e-06, Blackman-Harris 4.2e-14, Nuttall
1.5e-12, **Blackman 5.9e-25**.

**Three floors sit under any reported figure.** Calibration
(`TestOutOfBandCalibration`) confirms linear recovery to within 0.1% from −40 dB
to −200 dB, so the measurement is never the limit:

| floor | value | what it is |
|---|---|---|
| measurement | ~1e-22 | Blackman + float64 rFFT |
| **float32 storage** | **1.8e-16** | `band_limit_to_ladder` returns float32; `MEASUREMENT_FLOOR` is set from this |
| PCM_16 delivery | ~2e-8 | the physically meaningful one; `PCM16_OOB_FLOOR` |

`format_oob()` renders anything at or below `MEASUREMENT_FLOOR` as "below
measurement floor" rather than quoting a number.

**Correlation exclusions.** `invariants.CORRELATION_EXCLUDED` maps each excluded
condition to *why*, and `spearman_headline` refuses a frame containing one,
quoting the reason. Excluded conditions are generated, measured and **reported**
like any other; what they are not is comparable on the headline axis.

| condition | reason |
|---|---|
| `griffin_lim` | **floor reference.** Structurally cannot emit above its mel fmax (ISTFT of an inverted mel), measured 0.00000. Its detectability is a bandwidth artifact, not a reconstruction artifact, and at the low-quality end of the ladder it would anchor a strong positive Spearman for a reason unrelated to the hypothesis. |
| `bigvgan_v2_22khz_fullband` | **paired control.** Differs from `bigvgan_112m` in training mel fmax alone, which is the variable it exists to isolate. |

**`melgan_fullband` was removed.** It was the same checkpoint as `melgan`
differing only in whether the generation-time filter ran. With a full-band
archive the two produce byte-identical audio, so it became a duplicate
condition. The contrast it gave is now an analysis choice on `melgan` itself.

**Enforced by.** `data.preprocess.band_limit_comparison_set` (the only
sanctioned way to band-limit; refuses a set without `real`),
`data.preprocess.band_limit_to_ladder` (the filter),
`data.invariants.check_band_limit` (gates the transform's output),
`data.manifest._validate_band` (**asserts the archive is full-band** — the
inverse of what it used to assert), `detectors.protocols.spearman_headline`
(refuses a `CORRELATION_EXCLUDED` condition, with the reason),
`data.manifest.primary_ladder_frame`, `vocoders.registry` import-time
cross-check.

**Manifest columns.** `archive_band_hz`, `archive_band` (always `full_band`),
`high_band_fraction` (measured per file — evidence, not a gate).

---

#### INV-17 vs the existing invariants — two real conflicts

These are stated rather than reconciled away.

**INV-10 (no post-processing) — RESOLVED, largely for the same reason.** INV-10
forbids "denoising, EQ, peak limiting" on vocoder output, and a low-pass filter
is EQ, so the generation-time INV-17 did exactly what INV-10 prohibited. With
the band limit moved to analysis time, no filter touches the archive at all and
the contradiction is gone. The scoping below is kept because it still applies to
the pipeline steps that *do* run at generation (resample, trim, loudness).

The rationales do not conflict — INV-10 exists because *"post-processing is a
filter applied to one condition and not others"*, and INV-17 is applied to all
of them, real included. So the resolution is one of **scope, and INV-10's text
is hereby scoped**: INV-10 governs the **adapter**, i.e. anything inside
`vocoders/*` that could differ per condition. INV-17 is a **dataset-wide
pipeline step** in `data.preprocess`, alongside resampling, trimming and
loudness normalisation, none of which INV-10 was ever read as forbidding.

The test is not "is it a filter?" but "does any condition get something the
others do not?". Adapters still return raw generator output.

**INV-07 (fixed pipeline order) — RESOLVED, the conflict is gone.** It used to
be real: band-exempt conditions skipped the generation-time `band_limit` step, so
they ran a different pipeline than INV-07's "no exceptions" allowed.

Moving the band limit to analysis time removed the exception rather than scoping
around it. Every condition now runs the identical `PIPELINE_ORDER`. This is worth
noting as a general shape: the INV-07 conflict was a *symptom* of INV-17 doing
its work in the wrong place, and it disappeared when that was fixed rather than
needing its own resolution.

**INV-02 (mel config) — no contradiction, but its scope shrinks.** INV-02 governs
the **input** to the vocoder; INV-17 governs the **output** of the pipeline. No
model is taken out of distribution: each vocoder still analyses with its own mel.

The `fmax` difference between configs no longer reaches the delivered archive
either — but not because it is filtered away. It never constrained the synthesis
in the first place, except for Griffin-Lim. What `fmax` *does* determine is the
frequency above which a condition's output is invented rather than reconstructed,
which is now measured directly by `metrics.spectral.high_band_distance` rather
than assumed away.

Vocos also re-widens INV-02: 24 kHz, 100 mel bands, `fmax` 12000, unlike the
five conditions that share 1024/256/1024 at 80 bands.

---

## Repository layout

```
src/vocoders/     Vocoder adapters + the quality ladder      Person A
src/data/         Corpus, confound enforcement, manifest     Person A
src/metrics/      UTMOS, PESQ, MCD, F0, band-wise LSD        Person B (X-axis)
src/detectors/    AASIST/RawNet2, both protocols, Phase C    Person C (Y-axis)
src/experiments/  Phase drivers, sanity gate, main plot      shared
```

`src/data/invariants.py` is the root of the dependency graph. Everything imports
from it; it imports nothing from the project.

---

## Working rules

**Confound controls are not configuration.** `experiments.config` holds paths
and compute knobs only. Nothing that affects the scientific comparison goes in a
YAML file that a tired person edits at 2am to make a run go through.

**Changing an invariant means regenerating everything.** Including the real
condition. A dataset with two INV values in it is not a dataset. If a value must
change, change it once, before generation, and say so in the thesis.

**`InvariantViolation` is never caught to keep a pipeline running.** It means
the affected condition must be regenerated, not patched.

**Metrics never resample, retrim or renormalise.** They consume what Phase A
produced. A metric that fixes up its input is hiding a Phase A bug.

**The v0 dataset is deliberately throwaway.** 200 utterances, Griffin-Lim and
HiFi-GAN, shipped in week two so B and C can build in parallel. It obeys every
invariant anyway — its purpose is to debug pipelines, which it cannot do if it
is not shaped like the real thing. The full dataset replaces it by changing a
path.

**Notebooks are drivers, nothing more.** Kaggle notebooks import this package
and call into it. Logic that lives only in a notebook is not in the
reproducibility record.

**Never pool an exempt condition into a ladder-wide number.**
`bigvgan_v2_22khz_fullband` is generated and measured like any other condition,
but it differs from the ladder in bandwidth by design. `spearman_headline`
refuses it outright; use `paired_bandwidth_contrast` for the comparison it
exists to support.

**Never compare EERs across tiers.** A 16 kHz zero-shot EER and a 22.05 kHz
matched EER were measured on different audio. `spearman_headline` refuses a
mixed-tier series and `protocol_comparison` keeps the tier in its index, but the
prose has to respect it too.

**Report n and spread, never a bare mean.** Six conditions is a small n. With
`n = 6` the Spearman p-value is close to uninformative, so the argument rests on
whether the per-condition EER confidence intervals overlap. Bootstrap CIs are
computed for this reason; do not drop them from the plot to make it tidier.

**Both outcomes are results.** A strong positive correlation quantifies the
tradeoff. A weak one says quality and detectability are orthogonal — a more
interesting result, and the one with a real security implication. There is no
version of this experiment that produces nothing, so there is never a reason to
nudge a pipeline toward a preferred answer.

---

## Current blockers

- **RESOLVED — SpecDiff-GAN replaced by Vocos.** Its repo ships `configs/` but
  no pretrained weights; inference expects a user-supplied `--checkpoint_file`.
  Substituted with `charactr/vocos-mel-24khz`, revision-pinned. `specdiff_gan`
  remains declared in `SPECS` and audited in
  [`docs/mel_configs.md`](docs/mel_configs.md) so the substitution is visible in
  the audit trail; it is excluded from `ALL_CONDITIONS` and can never be
  generated. `LADDER_FMAX` is unchanged at 8000 — Vocos's `fmax` of 12000 was
  never the minimum.
- **Mel configs unverified.** Every `MelConfig.source` still marked `TODO` must
  be filled in from the upstream release before that condition is generated.
  Run `btpvf audit` for the current state.
- **Checkpoint hashes unpinned.** No `VocoderSpec.checkpoint_sha256` is set yet
  (INV-08).
- **RESOLVED (superseded by INV-17) — mel `fmax` caps the vocoder band.**
  Fixed in the data rather than worked around in the analysis: every primary
  condition including real is now low-passed to `LADDER_FMAX = 8000`, and
  `bigvgan_v2_22khz_fullband` was added as an exempt paired control so bandwidth
  can still be measured against architecture. The audit that settled it is
  [`docs/mel_configs.md`](docs/mel_configs.md). Original text follows for the
  record.

  <details><summary>Original blocker</summary>

  **mel `fmax` caps the vocoder band below archive Nyquist.**
  Surfaced by the move to 22.05 kHz, and it needs a decision before the matched
  protocol is run.

  Three conditions use a mel front-end with `fmax = 8000`: `griffin_lim`,
  `hifigan_v1`, `specdiff_gan`. Their output therefore contains **exactly zero**
  energy between 8 kHz and archive Nyquist. Real audio at 22.05 kHz does carry
  that band (measured on the synthetic smoke corpus: 0.36% of total energy,
  against 0.0000% for Griffin-Lim). BigVGAN's front-end runs to `fmax = 12000`
  and does produce it; MelGAN's released config is unbounded and needs
  verifying.

  This is a hard bandwidth cliff, not a subtle artifact. A detector separates
  real from those three conditions perfectly on high-band energy alone, so their
  matched EER goes to ~0 and the Y-axis loses all resolution across the lower
  half of the ladder. Worse, the conditions it spares are the *high-quality*
  ones, so the confound points in exactly the direction of the hypothesis and
  would manufacture a textbook "quality correlates with detectability" result.

  INV-02 forbids the obvious fix: `fmax` belongs to the vocoder, and raising it
  puts the model out of distribution. The bandwidth sanity probe (INV-12)
  correctly fires `leaked = True` on the archive tier for these conditions —
  that is the gate working, not a false positive.

  Options, for the supervisor:
    1. Make the **8 kHz point of the band-limited sweep the primary comparison**.
       It is the like-for-like operating point across the whole ladder, the
       machinery already exists, and the full-band archive numbers then quantify
       the bandwidth cliff as a separate, genuine finding. *Recommended.*
    2. Report the cliff as the headline mechanism result and accept a saturated
       Y-axis over three conditions.
    3. Restrict the ladder to vocoders whose front-end spans the full band,
       which costs most of the quality range the study depends on.

  Until this is settled, do not read a low matched EER on those three conditions
  as an artifact finding.

  </details>

- **RESOLVED — INV-17 rebuilt on a measured premise.** The original invariant
  band-limited at generation because an `fmax = 8000` front-end was assumed to
  emit nothing above 8 kHz. Measured on 20 LJSpeech files that is false for every
  condition except Griffin-Lim (real 0.01803, BigVGAN 0.01480, Griffin-Lim
  0.00000). The archive is now full-band and band-limiting is an analysis-time
  transform. `griffin_lim` became a floor reference, excluded from the headline
  correlation; `melgan_fullband` was deleted as a duplicate. `LADDER_FMAX` is
  unchanged at 8000. Evidence in [`docs/mel_configs.md`](docs/mel_configs.md).

- **Regenerate every condition.** The archive changed meaning: files produced
  under the old INV-17 are band-limited at 8 kHz and cannot be reused. Nothing
  generated before this change is valid.

- **MelGAN is the band outlier, not the laggard.** The audit found
  `descriptinc/melgan-neurips` uses `mel_fmax=None` → 11025 Hz, so MelGAN is the
  *widest*-band primary condition. If it is ever re-sourced from a
  ParallelWaveGAN or ESPnet LJSpeech recipe (`fmax = 7600`), `LADDER_FMAX` drops
  to 7600 and every condition must be regenerated. Re-run the audit before
  swapping the source.
- **FILED, NOT STARTED — generate the constants that appear in invariant prose.**
  Every invariant quotes numbers that also live in code: the crop duration and
  its per-rate sample counts, `LADDER_FMAX`, the LUFS target, the Chebyshev
  filter spec, the measurement and delivery floors. Those should be **generated
  into CLAUDE.md from the code that defines them**, inside marked blocks, with a
  test that regenerates and diffs — failing when prose and code disagree.

  The rationale is a measured failure rate, not a preference. Three instances so
  far, all the same shape — one behaviour described in two prose locations:

  1. `np.blackman` in code, "Blackman-Harris" in a docstring **and** in this
     file. Caught only by a calibration task that went looking.
  2. **D-3**: the crop became duration-based; `aasist.py` was updated, INV-11's
     Rule was not, and it went on prescribing the 64600-sample crop that the
     change had fixed.
  3. The archive crop constant was written as `89033` in a code comment, this
     file, and `docs/enforcement_audit.md`. The real value is **89027**.

  The third is the one that settles it: it was introduced **while fixing the
  second**, in the same session, by someone who had just written the audit
  explaining the pattern. Care does not scale here. Prose cannot be tested;
  generated blocks can.

  Scope when picked up: mark the blocks, add `experiments/gen_constants.py`, add
  a test asserting regeneration is a no-op. Do not hand-edit inside a marked
  block afterwards — that is the failure mode returning by another route.

- **`hifigan_v1` and `melgan` remain unpinned (INV-08).**
  Neither checkpoint is on HuggingFace, so no revision SHA exists. The
  verification mechanism is implemented
  (`vocoders.checkpoints.fetch_and_verify`); what is missing is the digests
  themselves, which can only be computed by whoever first downloads the weights.
  Run once with `allow_first_use=True`, paste the printed value into the spec,
  and commit it. `btpvf audit` reports all three blocked until then.
  (`melgan_fullband` was removed, so the two-digests-must-match note no longer
  applies.)

- **Alignment probes not yet run.** No vocoder has an INV-16 record, so no
  vocoder condition can pass the manifest gate. The probe runs automatically on
  first onboarding; a non-zero lag is a blocker, not a tolerance to widen.
- **Two-tier storage cost.** Every utterance now exists twice. At ~3000
  utterances x 6 conditions x 2 tiers the archive is roughly 2.4x the previous
  footprint, against a 20 GB per-account Kaggle limit. Publish the tiers as
  separate Kaggle Datasets so a session can mount only the one it needs.
