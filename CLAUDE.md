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
INV-06 to INV-18 are their operational consequences — decisions the table
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

**Superseded in scope by INV-17's move to a full-band archive, and then by the
committed measurement.** The finding above was made when INV-17 band-limited at
generation, so both paths passed through the 8 kHz filter before being compared.
It shows the round trip is erased *below* `LADDER_FMAX` — it never showed the
round trip is invisible in a **full-band** archive, and it is not:

| committed evidence (60 LJSpeech files, PCM_16) | value |
|---|---|
| [10900, 11025) Hz energy, `vocos` | **8.63e-07** |
| same band, `bigvgan` (no resample) | 7.30e-09 — ~118x less |
| AUROC `vocos` vs `bigvgan` on that band | **0.9997** |
| AUROC `vocos` vs `real_roundtrip` on that band | **0.507** — indistinguishable |

Vocos is indistinguishable from *resampled real audio* in that band and trivially
separable from a vocoder that was never resampled. What distinguishes it there
is the resampler. So it is excluded from the high-band analysis and kept in the
band-limited primary analysis (INV-18). Source:
[`docs/measurements.md`](docs/measurements.md).

The original finding used synthetic audio. Whether the pinned round trip is
**at chance below** `LADDER_FMAX` on the real corpus is now committed: 0.5000 ±
0.0000 below, 0.9935 above (`data/measurements/probes.json`, 200 LJSpeech files,
grouped CV), and 0.5000 below, 0.9989 above with the pipeline's loudness
normalisation (see INV-18). Reproduced by `TestVocosResampleSignature` and
`TestINV18ResampleProbe`.

**Vocos's resampler is pinned, and differs from `RESAMPLE_METHOD`.** Both its
input-side (22050 → 24000) and output-side (24000 → 22050) resamples use
`vocoders.vocos.VOCOS_RESAMPLER`: torchaudio `sinc_interp_kaiser`,
`lowpass_filter_width=64`, `rolloff=0.9999`, `beta=14.769656459379492`. The
torchaudio **default** costs 18–20% of the energy above `LADDER_FMAX` (0.8198 over
60 files in `band_measurements.csv`, 0.8005 over 200 in `probes.json`) and is
weakly separable **below** it too (AUROC 0.5291 ± 0.0024). The kaiser pin keeps
0.9997 and reads 0.5000 below (`probes.json`). So the pin is load-bearing for the
primary analysis, not only for the high band. The earlier 60-file kaiser figure,
0.9998, is in no artifact and is superseded.

This is a real tension with *"the method is frozen too"* below, stated rather
than hidden: `RESAMPLE_METHOD = soxr_hq` still governs every other resample —
archive → zero-shot for all conditions, and native → archive for the 22.05 kHz
conditions, where it is a no-op. Vocos is the only condition that takes a
native → archive step at all, so "frozen across conditions" is vacuous for that
step, and a condition-specific pin is the honest representation. The pin fails
closed: without torchaudio `resample_with_spec` raises rather than substituting
another resampler, because another resampler leaves another rolloff.

Found while wiring this: the input-side resample previously called
`librosa.resample` with **no `res_type` at all** — neither `RESAMPLE_METHOD` nor
any pin.

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

**Loudness interaction.** The derived tier is **not re-normalised**. Since the
archive gain is referenced to the analysis band (INV-04), the derived tier is
already band-matched across conditions: it inherits a gain computed on the band
it is about to keep. Downsampling discards energy above 8 kHz, which lowers
measured loudness slightly — by each file's high-band share, which is the same
quantity `lufs_out` now records. Re-normalising would apply a gain *proportional to each file's
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

**This invariant defines silence RELATIVELY; the probe that checks it defines
silence ABSOLUTELY. Making the two agree is not optional.** `TRIM_TOP_DB` is a
level *below the file's loudest frame*, so it follows the recording. INV-12's
silence probe compares `|sample|` to a level in dBFS, which does not. They agree
only on a corpus whose noise floor happens to sit where that level was put, and
for three weeks it was put at −50.0 dBFS with no recorded derivation — about
3 dB **inside** LJSpeech's own floor of −53.5 dBFS. The probe was then measuring
how often the signal's noise crossed a level near its own noise: it read 15.5 ms
of the 40 ms this invariant guarantees. That was the defect, not the particular
number.

The fix is a threshold **derived from the corpus** (INV-12) plus a check tying it
back to here: this invariant leaves exactly `2 × TRIM_PAD_MS` of leading plus
trailing silence in every written file, identically for every condition, so at a
threshold that is reading silence the real reference must read that much. It is
the one quantity that is fixed before any condition exists, which is what makes
it usable as a calibration check rather than a second assumption.

**Enforced by.** `data.preprocess.derive_trim_span` / `apply_trim_span`,
`data.manifest.validate_manifest` (rejects per-condition trim spans),
`experiments.sanity_checks.silence_probe` (at the derived threshold),
`experiments.sanity_checks.silence_reference` (raises when the threshold does
not read `2 × TRIM_PAD_MS` on real).

---

### INV-04 — Loudness: one normalisation target, no per-file exceptions

**Rule.** Every condition is normalised to `-27.0 LUFS` (ITU-R BS.1770),
**measured on the signal band-limited to `LADDER_FMAX`** and applied to the
full-band audio. When a file would exceed the `-1.0 dBFS` peak ceiling, **drop
that utterance from every condition** — never attenuate it in one.

**Why.** Level differences between conditions are trivially learnable; a
detector can reach a low EER on RMS alone, having learned nothing forensic. The
X-axis needs it too: UTMOS and PESQ are both level-sensitive, so an
un-normalised set would confound *both* axes at once and the correlation between
them would be partly an artifact of gain.

**The target is defined on the analysis band, and that is a change in what the
number means.** It used to be the loudness of the whole signal. It is now the
loudness of the band the primary analysis actually sees. A file's full-band
level is therefore no longer `-27.0 LUFS`: it sits above it by however much that
file's high band contributes, which differs per condition and is recorded rather
than equalised.

The reason is measured, not theoretical. Full-band normalisation makes the gain
depend on high-band energy, so a condition with less of it is gained more, and
that extra gain lands on **every** bin — including the ones below `LADDER_FMAX`
that the primary analysis compares. The first full v0 run measured exactly that,
on 200 LJSpeech utterances:

| condition | gain delta vs real | in-band offset | × probe resolution | high-band fraction delta |
|---|---|---|---|---|
| `vocos` | 0.184 dB | −0.0022 dB | 0.6 | +0.0010 |
| `bigvgan_112m` | 0.231 dB | **0.0506 dB** | 14.4 | −0.0064 |
| `griffin_lim` | 0.763 dB | **0.3018 dB** | 86.2 | −0.0324 |

The in-band offset tracked the high-band deficit across all three: Vocos has
none and reads clean, Griffin-Lim has the largest and reads worst. A level offset
that large is trivially learnable, and it points the same way as the hypothesis —
the lowest-quality condition gets the biggest offset — so it would have
manufactured part of the headline correlation.

Referencing the gain to the band removes it at the source: every condition's
in-band loudness is now the same number by construction, so the primary analysis
compares like with like. The high-band analysis gains something too: it now
measures high-band energy against a low-band-matched level, which is the more
meaningful reference. What used to be a level offset is now visible where it
belongs, in `lufs_out` and in `high_band_fraction`.

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

**The gain is recorded, not just its result.** Every archive manifest row
records five values, so the change of reference is auditable per file:

- `lufs_in`, the full-band loudness before the gain — evidence, not the reference;
- `lufs_in_band`, the analysis-band loudness before the gain — **what the gain is
  referenced to**;
- `loudness_gain_db`, the gain the written file carries, measured from the arrays;
- `loudness_target_lufs`;
- `lufs_out_band`, the analysis-band loudness after the gain — it lands on target.

`target − lufs_in` is the gain the old full-band rule would have applied, so the
difference between the two rules stays computable from any manifest. Derived rows
record none of them, because they are never normalised.
`data.preprocess.normalise_loudness_recorded` is the single implementation, and
`normalise_loudness` is it with the record dropped.
`data.manifest._validate_loudness_record` gates on the record being present on
archive rows, on the gain equalling `target − lufs_in_band` within
`LOUDNESS_GAIN_RECORD_TOLERANCE_DB`, on `lufs_out_band` landing on target within
`LOUDNESS_TOLERANCE_LU`, and on derived rows carrying none of it. A manifest whose
gain is referenced to the full band is refused. `data.manifest.loudness_gain_report`
(`btpvf loudness`) compares the **measured** `lufs_out_band` across conditions, and
`TestINV04LoudnessRecord` pins the Griffin-Lim-shaped case — no high band at all,
which read 86× the probe resolution — at or below that resolution.

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
implementation** of the post-trim ordering (measure high band → measure band
loudness → normalise loudness; the two measurements are recorded, never applied). It is
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

Pinning is now enforced **on a real path** for three conditions, and
`vocoders.checkpoints.verify_checkpoint` (digest mismatch ⇒ raise) is called on
one of them:

- **`bigvgan_112m`.** `load()` downloads code, `config.json` and weights from the
  one pinned revision (`vocoders.bigvgan.fetch_pinned_snapshot`). It verifies the
  weight digest *before* anything from the snapshot is imported, and cross-checks
  the config against the audit (`check_hparams`).
- **`vocos`.** `from_pretrained` is called with the pinned revision, and the loaded
  front-end is checked against the audit (`check_feature_extractor`).
- **`griffin_lim`.** It has no checkpoint. Its phase init is seeded with
  `GLOBAL_SEED` per utterance, so it is the first condition with a mechanism for
  "fix seeds".

`hifigan_v1`, `melgan`, `bigvgan_base` and the full-band control still have no
working `load()`. Each adapter declares `required_pins`, which Phase A's preflight
checks before generating anything. So `bigvgan_base` and the full-band control,
pinned by revision only, are refused before generation rather than failing after
the real condition has been written. `checkpoint_audit` reports the same thing:
its `missing_pins` column names the absent pin and `blocked` is true, so
`btpvf audit` never lists a condition as runnable that the CLI would refuse
(`TestAuditAgreesWithTheCLI`).

The rule's *"fix sampler step counts"* is still aspirational for any future
diffusion-style adapter.

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

**Gated runs probe every utterance, not the eval split.** `btpvf resynth` runs
`run_all(split=None)`. The probes train nothing: each scores one scalar feature
directly, so there is no held-out set to protect. At a few hundred utterances the
eval split alone is small enough to cross the floor by chance. At 200 utterances
the eval split holds 40 per condition, and the EER's spread at true chance is
roughly 0.065. That is about a 6% chance per probe of falling below 0.40, so
across the 21 probes of a three-condition run a false leak is more likely than
not. Over all 200 utterances the spread is roughly 0.03. `btpvf sanity` still
defaults to `eval`.

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

**The silence probe's threshold is derived from the corpus, never written
down.** INV-03 defines silence relatively and this probe compares `|sample|` to
an absolute level, so the level has to be placed against the corpus's own floor
or the two definitions describe different things. Per dataset and per tier, from
the **real reference alone**:

```
floor     = SILENCE_FLOOR_PEAK_PERCENTILE of |sample| inside the quietest
            SILENCE_FLOOR_FRAME_PERCENTILE of real's RMS frames
threshold = floor + SILENCE_FLOOR_MARGIN_DB
```

with `SILENCE_FLOOR_FRAME_PERCENTILE = 10.0`, `SILENCE_FLOOR_PEAK_PERCENTILE =
99.9`, `SILENCE_FLOOR_MARGIN_DB = 8.0`, `SILENCE_PAD_AGREEMENT = 0.25`.

- **Peak against peak.** The probe thresholds `|sample|`, so the floor is a
  sample-peak statistic too. Expressed as an RMS it would leave the margin
  absorbing a crest factor of `sqrt(2 ln N)` — +10.4 dB over one 20 ms pad,
  +14.5 dB over a corpus — and the same constant would then mean something else
  at another sample rate or pad length. That is INV-11's "64600 samples" mistake
  in a new place.
- **The quietest decile, not `reference_silent_mask`.** INV-03's relative
  criterion admits quiet speech: on LJSpeech its median sits at −53.5 dBFS,
  essentially at its own cutoff. Calibrating on the loudest samples of a "quiet"
  band puts the threshold near the top of it.
- **Digital zeros are excluded.** A run of PCM_16 zeros is the absence of a
  recorded floor, not a floor; frames below one LSB carry no information about
  one and would drag the threshold to the quantiser.
- **It fails closed.** `2 × TRIM_PAD_MS` is the silence INV-03 leaves, so real
  must read that at the derived threshold, within `SILENCE_PAD_AGREEMENT`.
  Outside it the probe **raises** rather than reporting: too low and it counts
  the corpus's noise as signal, too high and it eats speech onsets, and either
  way the EER is not a silence measurement. The rule therefore also has a range
  requirement — a corpus whose floor is not far enough below speech is refused
  rather than measured, which `TestDerivedSilenceThreshold` pins.
- **It is recorded, not implied.** Every row of the sanity CSV carries
  `silence_threshold_dbfs`, `silence_floor_dbfs` and `silence_frame_rms_cut_dbfs`
  for its tier, and the provenance sidecar carries the whole calibration under
  `silence_threshold`. No EER in this study can be read without the threshold it
  was measured at.

**One column is recorded and deliberately ungated: `trim_pad_above_real_db`.**
The derived threshold sits above the trim pad's noise by design, so the silence
probe can no longer see a level difference *inside* the pad, and `rms_level`
cannot either — whole-file RMS is dominated by speech, and the pad is 40 ms.
The first v0 run measured +0.887 dB for `griffin_lim` against −0.086 for
`bigvgan_112m` and −0.803 for `vocos`, so the effect is real and condition-specific.
It is reported on every silence row and gates nothing, because a gate needs a
floor and there is no basis in this repository for choosing one — see
[`docs/measurements.md`](docs/measurements.md) for what choosing one would take.

**Diagnosing a hit is separate from deciding about it.**
`experiments.sanity_checks.silence_diagnostic` (`btpvf silence-diag`) reports, per
condition, the level inside the regions the REAL reference calls silent, the level
in the trim pad the probe actually reads, the two quiet-frame statistics the
threshold is derived from, and the probe's own feature swept across thresholds —
with the derived threshold swept alongside the fixed points and flagged, so what
the gate will read is visible next to what it would read elsewhere. `real` is a
row in that table, so the pipeline's own floor is visible next to each
condition's. It derives its reference **non-strictly**: a dataset whose threshold
disagrees with its pad is exactly what the diagnostic is for, so it reports what
the probe raises on. It changes no verdict and no floor.

**Enforced by.** `experiments.sanity_checks.run_all`; the CLI exits non-zero on
a leak. `experiments.sanity_checks.silence_reference` derives the threshold and
raises on pad disagreement; `_leading_trailing_silence` has **no default
threshold**, so no caller can obtain one without choosing it — a default is how
the old value went unexamined.

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

**One stated exception: the v0 subset.** `griffin_lim`, `bigvgan_112m` and
`vocos` may share one environment (`vocoders.registry.SHARED_ENV_REPORTED`),
because BigVGAN and Vocos were *reported* by the team to coexist in testing. The
repository did not measure that, and it is recorded as reported, not as a finding.

- **Refused otherwise.** `experiments.phase_a_resynthesis.preflight` refuses any
  other combination of neural conditions in one process.
- **Checkable afterwards.** Every run writes the package versions and GPU it used
  into the provenance sidecar (`environment`, `environment_shared_by`).
- **Stated in the notebook.** `notebooks/kaggle_v0.ipynb` says so in its opening
  cell rather than assuming it.

Adapters also declare the `runtime_modules` they need, and preflight checks those
are importable before any audio is generated.

**Upstream code runs only if it provably came from the pinned snapshot.**
BigVGAN's own code imports itself by generic top-level names (`utils`, `env`), so
`vocoders.bigvgan.import_upstream` refuses a same-named module already imported
from elsewhere, and refuses an import that resolved anywhere but the snapshot.
Both sides of that comparison are **resolved paths**: the module's file, and the
snapshot's own entry for that name. The HuggingFace cache keeps content in
`blobs/` and makes `snapshots/<revision>/` a tree of symlinks into it.

The first version resolved the module's file but compared it against the snapshot
*directory*, so every correctly loaded module failed. That is what stopped the
first Kaggle run at BigVGAN. The fixed comparison is an equality on what Python
actually loaded, which is at least as strict as the old prefix test.

- **On Linux (WSL)**, a stdlib-only script runs the old and fixed guards against a
  real symlinked cache. It reproduces the Kaggle failure with the old one; with the
  fixed one, both legitimate loads pass and the impostor and same-named cases
  still raise.
- **In the suite**, `TestUpstreamImportGuardOnTheHFCache` builds real file
  symlinks and skips where the OS refuses them (Windows without Developer Mode).
  `TestUpstreamImportGuardThroughADirectoryLink` uses a junction and runs
  everywhere; the old guard fails it.

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

**The one credential the pipeline reads: `HF_TOKEN`, optional.**
`vocoders.checkpoints.hf_token()` reads it from the environment. BigVGAN passes it
explicitly to every `hf_hub_download`. Vocos's `from_pretrained` has no token
parameter, and gets it through huggingface_hub's own reading of `HF_TOKEN`.
Unset, downloads are anonymous and may be rate limited; set, nothing about *what*
is downloaded changes, because every download is pinned by revision.

It is never printed, logged or recorded: `TestHFToken` runs a whole gated run with
a sentinel token and asserts it appears in no output, sidecar or error. The
notebook ships the Kaggle Secrets lines commented out.

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

**What did not change.** The filter family (`band_limit_to_ladder`, zero-phase
Chebyshev II, 100 dB per pass), the measured floors, and the gate. They moved
from generation to analysis; they were not weakened. The filter's **order and
passband edge** changed later, for a separate reason: see *The transition band*
below. `check_band_limit` now
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
`BAND_LIMIT_TRANSITION_HZ = 100.0`, `BAND_LIMIT_PASSBAND_LOSS_DB = 0.1`,
`BAND_LIMIT_STOPBAND_DB = 100.0` (per pass), `BAND_LIMIT_FILTER_ORDER` (derived,
53), `BAND_LIMIT_FILTER_SPEC`, `BAND_LIMIT_MAX_STOPBAND_ENERGY = 1e-6`.

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
The current order-53 design (below) lands at the same float32 floor, 1.8e-16
(float64: 4.8e-21).

**The transition band — the two analyses meet at `LADDER_FMAX`.** The filter used
to be specified by an order (12) and a stopband edge (`LADDER_FMAX`) alone.
`scipy.signal.cheby2` takes `Wn` as the frequency where rejection *first reaches*
`rs`, so the passband edge was never chosen. It fell 1.6 kHz below the band: the
primary analysis was effectively [0, 6.4 kHz], and 6.4–8 kHz, through the
fricative region, was in **neither** analysis. It went unnoticed because the only
passband figure ever quoted, "0.004 dB below 7 kHz", was energy-weighted. Speech
concentrates its energy low, so that number is small whatever the response does
near the edge.

The filter is now specified by **both edges** and the order is derived:
passband to `LADDER_FMAX − BAND_LIMIT_TRANSITION_HZ` at ≤ 0.1 dB total, stopband
from `LADDER_FMAX` at 100 dB per pass. Zero-phase response, measured directly
(pure tones, impulse through the implementation, and analytic, agreeing to 0.1 Hz):

| design | order | −1 dB | −3 dB | −20 dB | −100 dB | full stopband |
|---|---|---|---|---|---|---|
| old | 12 | 6419 Hz | 6555 Hz | 6876 Hz | 7616 Hz | 8000 Hz |
| **current** | **53** | **7915.5 Hz** | **7922.9 Hz** | **7940.3 Hz** | **7979.1 Hz** | **7998.8 Hz** |

**The transition can narrow but must not straddle `LADDER_FMAX`.** Moving it
above the band would leak attenuated [8000, 8000+δ) content into the primary
analysis, where real has energy and Griffin-Lim has none. That is the bandwidth
cliff INV-18 relies on the band limit to remove. So **[7900, 8000) Hz is still
in neither analysis at full weight.** That is 1.25% of the band, inside the
falling slope of the top 80-band mel filter.

**What it costs,** measured: order 12 → 53 (6 → 27 SOS sections, max pole radius
0.9948); impulse response 99.99%-energy span 2.8 → 24 ms; ~3.8× the compute
(~72 ms per 10 s of audio). The longer impulse response is linear and identical
for every member of a comparison set, so it smears but does not separate.
Rejected alternatives: elliptic (order 17, 0.1 dB passband ripple, 41 ms); Kaiser
FIR (cheaper, but its own residual is above `MEASUREMENT_FLOOR`); a narrower
7950 Hz edge (order 75, 37 ms). Full candidate table:
[`docs/measurements.md`](docs/measurements.md).

**No regeneration needed.** The band limit is an analysis transform, so changing
it re-runs analyses and leaves the archive alone. That is INV-17's own argument
paying out: a generation-time filter would have meant regenerating every
condition.

**The ablation sweep uses the same design.** `detectors.bandlimit.lowpass` was a
zero-phase Butterworth order 8, and it failed the sweep both ways it can. Its
transition *straddled* every cutoff (−6 dB at the cutoff, −20 dB 130–480 Hz above
it). It also attenuated without equalising, leaving 7e-06 to 3e-04 of total energy
above the cutoff. A retrained detector could still learn the band a cutoff was
meant to remove, and the collapse point would read too high. It now calls
`band_limit_sos` at each cutoff. Measured directly at every cutoff from 1 to
10 kHz: −1 dB at cutoff − 84..86 Hz, −3 dB at cutoff − 77..79 Hz, −20 dB at
cutoff − 59..62 Hz, ≤ −200 dB at and above the cutoff, and a residual at the
float32 floor. It costs order 31–60 against 8, a 22–44 ms impulse response against
1.5–5.8 ms, and 3–7× the filtering compute, all negligible next to retraining a
detector per cutoff. At `LADDER_FMAX` the sweep point is byte-identical to the
primary analysis. Per-cutoff table: [`docs/measurements.md`](docs/measurements.md).

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

**Exclusions are per analysis — see INV-18.** This section used to hold a single
`CORRELATION_EXCLUDED` set that kept `griffin_lim` out of the headline
correlation. That was wrong in the same way the original INV-17 was: the
disqualifying artifact is **band-localised**, above `LADDER_FMAX`, so a blanket
exclusion discarded usable data. INV-18 replaces it.

**Griffin-Lim is not exactly zero.** The 20-file table above reads 0.00000 to
five decimals. The committed 60-file measurement gives median **5.1e-9**, max
**5.5e-8** — the PCM_16 quantisation floor on no signal content, and 4 of 60 files
exceed `PCM16_OOB_FLOOR` itself. The conclusion survives: real's *minimum* is
6.4e-4, four orders of magnitude above Griffin-Lim's maximum.

**`melgan_fullband` was removed.** It was the same checkpoint as `melgan`
differing only in whether the generation-time filter ran. With a full-band
archive the two produce byte-identical audio, so it became a duplicate
condition. The contrast it gave is now an analysis choice on `melgan` itself.

**Enforced by.** `data.preprocess.band_limit_comparison_set` (the only
sanctioned way to band-limit; refuses a set without `real`),
`data.preprocess.band_limit_to_ladder` (the filter),
`data.preprocess.band_limit_sos` (both edges; fails closed if scipy's order
disagrees with `invariants.band_limit_order`),
`data.invariants.check_band_limit` (gates the transform's output),
`TestINV17BandLimitResponse` (pins −1 dB and −20 dB, measured by impulse and by
tone; asserts nothing at or above `LADDER_FMAX` survives; mutation-checks that the
order-12 design fails), `detectors.bandlimit.lowpass` (the sweep, same design at
every cutoff) with `TestBandlimit` (transition below every cutoff, residual at the
storage floor, derived order per cutoff, identical to the analysis band limit at
`LADDER_FMAX`, mutation-checks that Butterworth-8 straddles every cutoff),
`data.manifest._validate_band` (**asserts the archive is full-band** — the
inverse of what it used to assert). Membership of the two analyses is INV-18.

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

### INV-18 — Analysis membership is per band, and every exclusion cites a measurement

**Rule.** An analysis band may only include conditions whose processing history
introduces **no artifact within that band**, and **every exclusion must name a
measurement rather than an argument**.

Two analyses run over the ladder, with different membership:

| analysis | band | excluded |
|---|---|---|
| **primary** (`PRIMARY_ANALYSIS`) — the headline correlation | band-limited to `LADDER_FMAX` | **nothing** |
| **high-band** (`HIGH_BAND_ANALYSIS`) | [`LADDER_FMAX`, Nyquist) on the full-band archive | `griffin_lim`, `vocos` |

`griffin_lim` is **in** the primary correlation. That is intended, not an
oversight: its disqualifying artifact lives above `LADDER_FMAX`, which the primary
analysis removes.

**Why.** Two confounds found in this study are **band-localised**:

- **Griffin-Lim's mel-inversion cliff.** It inverts the mel to a linear
  spectrogram and runs ISTFT, so it cannot emit above `fmax`. The artifact exists
  only above `LADDER_FMAX`.
- **Vocos's resampler.** Its transition band leaves an imprint in
  [10900, 11025) Hz that makes it indistinguishable from resampled real audio
  and separable from an unresampled vocoder at AUROC 0.9997. Those figures
  describe a Vocos resampled with the torchaudio default, before the pin;
  the pinned kaiser round trip is still separable above the band at 0.9935,
  and at 0.9989 with the pipeline's loudness normalisation (`probes.json`). The
  exclusion cites all three.

A **blanket exclusion loses usable data**: removing Griffin-Lim from every
analysis throws away its below-band reconstruction behaviour, which is exactly
the low-quality anchor the ladder needs. A **blanket inclusion manufactures
results**: keeping it in the high-band analysis would score a structural absence
as a reconstruction artifact. The unit of membership has to be (condition,
band), not condition.

Both confounds were **found by measuring rather than by reasoning from
configuration**, and the reasoning that preceded them was wrong both times:
config inspection said `fmax = 8000` meant no output above 8 kHz (false for every
neural vocoder, INV-17), and argument said INV-17's band limit erased the Vocos
resample (true below the band, false in a full-band archive, INV-01). Hence the
second half of the rule. An exclusion justified by argument is a claim nobody has
checked; one that names a row can be recomputed.

**Exclusions and membership are different things.** An INV-18 exclusion removes
a ladder **rung** from one analysis and must cite evidence. A paired control
(`bigvgan_v2_22khz_fullband`) is not a rung at all — it is refused by
**membership**, needs no measurement, and never appears in
`ANALYSIS_EXCLUSIONS`. Conflating the two would either force a fake measurement
onto a design fact or let a real confound hide behind a design label.

**The primary analysis must actually be band-limited.** `spearman_headline`
refuses any detection row whose audio can carry content above `LADDER_FMAX` —
archive rows without `band_limit_hz`, or a tier whose Nyquist exceeds the band.
This is load-bearing rather than tidy: Griffin-Lim is only safe in the headline
*because* the headline is band-limited. Admit full-band rows and its cliff comes
straight back. Band-limiting removes the cliff's *content*, though, not its effect on the
full-band loudness gain; see the open item on per-condition gain deltas.

**Vocos below the band — established by committed measurement.** Vocos's
inclusion in the primary analysis rests on its resampler leaving no artifact below
`LADDER_FMAX`. `data/measurements/probes.json` measures exactly the pinned
resampler (parameters checked equal to `VOCOS_RESAMPLER`) on 200 LJSpeech files,
`GroupKFold(5)` by utterance, PCM_16, both without and with the pipeline's
loudness normalisation (resample → normalise → quantise):

| round trip vs real | high-band ratio | gain delta (normalised) | AUROC below, raw → normalised | above (control), raw → normalised |
|---|---|---|---|---|
| **kaiser (pinned)** | 0.9997 | 4.62e-05 dB | **0.5000 → 0.5000** | **0.9935 → 0.9989** |
| soxr | 0.9922 | 0.001534 dB | 0.5000 → 0.5000 | 1.0000 → 1.0000 |
| torchaudio default | 0.8005 | 0.02986 dB | **0.5291 → 0.5348** | 1.0000 → 1.0000 |

**0.5000 ± 0.0000 is degenerate, not noisy chance, and not a bug.** It means the
classifier gave both versions of every utterance the same score. lbfgs starts at
zero coefficients, and when the mean paired feature difference is under its
tolerance it stops at iteration 0. That makes the result a *resolution* (about
4e-4 in log-feature units), not a proof of identity. Two controls rule out a
pipeline fault: the same probe separates the same pairs **above** the band (0.9935),
and in the **same** band it reads a non-degenerate 0.5291 for the lossier default.
`data.measurements.probe_reading` refuses to hand out a below-band reading whose
above-band control is under `PROBE_CONTROL_MIN_AUROC`. Full reasoning, with the
mechanism reproduced: [`docs/measurements.md`](docs/measurements.md).

**The loudness step — measured, then removed at the source.** The `_norm` probes
measured the **old** rule, where the gain was referenced to the full band: a round
trip that removes high-band energy was gained more, and that shifted every
below-band bin. INV-04 now references the gain to the analysis band, so a pair
differing only above `LADDER_FMAX` receives the *same* gain and there is no shift
to detect. For such a pair the **raw** probes are now the pipeline-matching ones,
and they read 0.5000 below the band for the pin.

The bracket below is kept because it is what justified the change, and because it
still bounds any residual coupling. Measured on real audio, the coupling showed at
a 0.02986 dB gain delta (0.5291 → 0.5348) and not at 0.001534 dB or 4.62e-05 dB.

- **It is a bracket.** Onset lies in (0.001534, 0.02986] dB. "~0.0015 dB" is the
  highest measured null, not a located threshold.
- **It is this probe's resolution.** A gain delta moves every log feature equally,
  and lbfgs never leaves zero below ~0.0035 dB (natural log) or ~0.008 dB (log10).
  That predicts this bracket exactly. The shift does not vanish below it; it drops
  below what this linear probe resolves.
- **The pin's margin.** The pinned resampler sits **33× below the highest null**
  and 646× below the lowest detection. `LOUDNESS_COUPLING_MIN_MARGIN` requires 10×.
  33× does not reach 100×.

This also explains the local synthetic probe's 0.54–0.57 and the earlier
"configuration-sensitive, 0.50–0.69": synthetic audio loses far more high-band
energy to `soxr_hq`. The coupling is not specific to resamplers. See the open item
on per-condition gain deltas below.

**Normalisation and the high-band evidence.** Under the pipeline's normalisation,
above-band separability is not weaker: kaiser 0.9935 → 0.9989 (fold std 0.0056 →
0.0011), once vs twice 0.9928 → 0.9941. The Vocos exclusion and the uniform
round-trip rejection are at least as well supported under the real pipeline as the
un-normalised probe indicated. Each rise is within about one un-normalised fold
std and per-fold values are not recorded. So "not weaker" is established, while
"normalisation sharpens it" is not. Griffin-Lim's exclusion is unaffected.

The earlier "reported 0.502" is in no artifact and has no counterpart here. If it
was the default resampler, the committed 0.5291 disagrees with it.

**Rejected: the uniform round trip.** Considered and rejected — recorded so it is
not proposed again. The idea was to pass *every* condition through the same
22050 → 24000 → 22050 round trip, so Vocos would no longer be the only condition
with a resampler imprint. It fails because **resampling is not idempotent**: one
kaiser round trip versus two is separable at **AUROC 0.9928 above the band**
(0.9918 full; 0.9941 above with loudness normalisation; `probes.json`). Adding a round trip to the other conditions gives
every condition an imprint and Vocos a *different* one; it does not make them
path-identical. Below the band once and twice are indistinguishable (0.5000),
which is consistent: no round trip was needed there. The figure previously
recorded, 0.9903 from 60 files, is in no artifact. `REJECTED_ALTERNATIVES` now
cites the committed values, and `in_committed_artifacts` is derived from that
citation rather than set by hand.

**Cross-validation for any paired probe must be grouped by utterance.** Shuffled
folds split an utterance's two versions across train and test; because
utterances differ far more than the two paths do, the classifier is
systematically inverted — 0.13 against the grouped 0.57 on identical data.
`TestINV18ResampleProbe.test_paired_probe_must_group_by_utterance` pins this.

**Constants.** `PRIMARY_ANALYSIS`, `HIGH_BAND_ANALYSIS`, `ANALYSIS_EXCLUSIONS`,
`REJECTED_ALTERNATIVES`, `vocoders.vocos.VOCOS_RESAMPLER`,
`PROBE_CONTROL_MIN_AUROC = 0.95`, `PROBE_CHANCE_MAX_DEVIATION = 0.01`,
`LOUDNESS_COUPLING_OPERATING_PROBE = "real_vs_rt_kaiser_norm"`,
`LOUDNESS_COUPLING_MIN_MARGIN = 10.0`.

**Enforced by.** `data.invariants.ANALYSIS_EXCLUSIONS` (each `Exclusion` carries
its `MeasurementRef`s; Vocos's also cites the pinned resampler's above-band
0.9935, and 0.9989 under loudness normalisation), `data.measurements.verify` (recomputes every CSV statistic, and checks
every `probes.json` citation against the file),
`data.measurements.probe_reading` (a below-band reading never exists without a
separable above-band control), `vocoders.vocos.VOCOS_RESAMPLER.justification`
(cites kaiser ratio 0.9997, kaiser below 0.5000 raw and normalised, kaiser gain
delta 4.62e-05 dB, default below 0.5291), `data.measurements.loudness_coupling`
(derives the null/detection bracket and the pin's margin from every normalised
probe in `probes.json`; raises if not monotone in gain delta),
`detectors.protocols.spearman_headline`
(refuses excluded conditions with the evidence, refuses controls by membership,
refuses full-band rows), `detectors.protocols.refuse_excluded`,
`metrics.spectral.high_band_distance` (refuses excluded conditions; `condition`
is required so the refusal cannot be skipped), `data.manifest.analysis_frame`,
`metrics.runner.evaluate_archive` (records exclusions rather than scoring them).
Tests: `TestINV18PerAnalysisMembership` (every citation recomputed; refusals; the
high-band evidence clears the control threshold under loudness normalisation),
`TestINV18ResampleProbe` (**pinned kaiser at chance below the band, from
`probes.json`**, formerly a strict xfail; default not at chance; local
concentration above; grouped CV), `TestINV18ProbeControls` (every probe's
above-band control separable; a reading without one refused; the exact-0.5
mechanism reproduced), `TestINV18LoudnessCoupling` (**from `probes.json`, not a
local synthetic run**: the pinned resampler at chance below the band after
normalisation; soxr not detected; default detected as positive control; the
bracket; the pin's margin ≥ 10× below the highest null, tied to the pinned
parameters; the bracket reproduced from the solver's resolution),
`TestINV18RejectedAlternatives`,
`TestResamplerPin`, `TestMeasurementArtifacts` (`probes.json` measured exactly the
pin).

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

**Manifest paths are relative to the manifest.** `path` records where the audio
is relative to the manifest's own directory, and `data.manifest.read_manifest`
resolves it against that directory — the one resolution every consumer goes
through. A dataset gets renamed, published and mounted somewhere else: a
teammate's `/kaggle/input/<slug>/` is not the `/kaggle/working/` a run wrote
from. `write_manifest` refuses a path it cannot make relative, and the absolute
directory a run generated in is recorded once, in the provenance sidecar
(`generated_in`), where going stale is the point. This was a live bug: the first
v0 run recorded absolute `<out>.staging` paths, `run_gated` renamed staging to
`<out>.failed`, and every consumer that opens audio broke on a directory that no
longer existed — while `btpvf loudness`, which reads columns only, carried on.

**The v0 dataset is deliberately throwaway.** 200 utterances of `griffin_lim`,
`bigvgan_112m` and `vocos` (HiFi-GAN was the original plan, and is blocked on
weight sourcing), generated by `notebooks/kaggle_v0.ipynb` through the gated
`btpvf resynth`, so B and C can build in parallel. It obeys every
invariant anyway — its purpose is to debug pipelines, which it cannot do if it
is not shaped like the real thing. The full dataset replaces it by changing a
path.

**Notebooks are drivers, nothing more.** Kaggle notebooks import this package
and call into it. Logic that lives only in a notebook is not in the
reproducibility record.

**Membership is per analysis, and exclusions cite rows.** Filter with
`data.manifest.analysis_frame(df, analysis)` before pooling anything. Do not add
a condition to `ANALYSIS_EXCLUSIONS` without a `MeasurementRef` that resolves
against `data/measurements/` — an argument is not an exclusion (INV-18). The
paired control is refused by membership; use `paired_bandwidth_contrast` for what
it measures.

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
- **Checkpoint pins, partly resolved (INV-08).**
  - `bigvgan_112m` carries a revision and a weight digest (from
    `data/measurements/meta.json`), both enforced at `load()`.
  - `vocos` carries a revision, passed to `from_pretrained`.
  - `bigvgan_base` and the full-band control carry revisions only. Their adapter
    requires a digest, so preflight refuses them.
  - `hifigan_v1` and `melgan` remain unpinned (below).
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

- **RESOLVED — reported resampler measurements are now committed** in
  `data/measurements/probes.json` (200 files), **with different values** from the
  60-file reports, recorded rather than reconciled: kaiser 0.9998 / 0.9958 →
  ratio 0.9997, above 0.9935, full 0.9924. soxr_vhq 0.9936 → `rt_soxr` 0.9922,
  quality level unrecorded. One-vs-two 0.9903 → above 0.9928, full 0.9918. The
  band-limited 0.502 has no counterpart; the committed default below-band figure
  is 0.5291. See [`docs/measurements.md`](docs/measurements.md).
- **RESOLVED — the band limit left 6.4–8 kHz in neither analysis.** The order-12
  design put its −1 dB point at 6419 Hz. It is now specified by both edges and is
  −1 dB at 7915.5 Hz, −20 dB at 7940.3 Hz, at order 53. [7900, 8000) Hz remains
  the unavoidable transition (INV-17). Analysis-time only; no regeneration.
- **RESOLVED — the ablation sweep's Butterworth-8 straddled every cutoff.**
  −6 dB at the cutoff and 7e-06 to 3e-04 of energy surviving above it. Ported to
  the two-edge design (INV-17): −1 dB at cutoff − 85 Hz, ≤ −200 dB at and above
  the cutoff, order 31–60. Analysis-time only; no regeneration.
- **RESOLVED — the loudness coupling was real, and the target moved to the
  analysis band.** The first full v0 run measured it on 200 LJSpeech utterances:
  in-band offsets of −0.0022 dB (vocos), 0.0506 dB (bigvgan_112m) and 0.3018 dB
  (griffin_lim), tracking each condition's high-band deficit, against a probe
  resolution of 0.0035 dB. The estimates that opened this item (0.03–0.04 and
  0.15–0.20 dB) were the right size. INV-04 now references the gain to the
  band-limited signal, which removes the offset by construction; both loudness
  measurements stay on every row so the old rule remains computable. The archive
  must be regenerated — every file's gain changes.

  Closed by measurement, not by the argument: `band_offset_db` now reads ~1e-9 dB
  for all three conditions (~1e-6 × the probe resolution). Unanticipated, and
  recorded rather than acted on: the `rms_level` probe moved **closer** to chance
  under the new rule, not further — `griffin_lim` 0.630 → 0.480 archive and
  0.680 → 0.555 zeroshot, `bigvgan_112m` 0.565 → 0.530, `vocos` 0.495 → 0.500.
  RMS is unweighted and the high band is ~2% of total energy, so RMS is nearly an
  in-band measure; LUFS is K-weighted and gives that 2% far more say, so matching
  full-band LUFS was pushing the in-band level around to compensate. Both
  measurements of the same effect. [`docs/measurements.md`](docs/measurements.md).
- **OPEN, YOUR DECISION — griffin_lim trips the silence probe, and the probe is
  misspecified.** The first full v0 run: EER **0.350** archive, **0.355**
  zeroshot, against a 0.40 floor. Trim spans are reference-derived, so this
  cannot be a difference in silence LENGTH; it is what sits inside those regions.
  `btpvf silence-diag` has now been run on the run's own audio, and it **refutes**
  the reconstruction-noise hypothesis the synthetic diagnosis supported.

  - **No elevated floor.** Level inside the regions real calls silent, median
    dBFS: real −53.556, `bigvgan_112m` −53.524 (+0.033), `griffin_lim` −53.448
    (**+0.108**), `vocos` −53.628 (−0.072). The synthetic corpus predicted
    **+13.2 dB** for `griffin_lim` and overstated it by roughly a hundredfold.
    It was built with a broadband floor injected into the `griffin_lim` member,
    so the measurement recovered the injection; real Griffin-Lim inverts a mel
    that already carries LJSpeech's room tone.
  - **One real effect survives**, in the trim pad rather than in silence: the pad
    runs **+0.887 dB** above real for `griffin_lim`, against −0.086 for
    `bigvgan_112m` and −0.803 for `vocos`. About 1 dB in a 20 ms region, and a
    level difference rather than a duration one.
  - **The probe's threshold is −50 dBFS and the corpus floor is −53.5 dBFS**, so
    the threshold sits ~3 dB INSIDE the floor. INV-03 leaves exactly
    `2 × TRIM_PAD_MS` = 40 ms of leading plus trailing silence; at −50 dBFS the
    probe reads 15.5 ms of that pad, at −40 dBFS it reads 40.8 ms (the pad and
    nothing else), at −30 dBFS 88.9 ms (into speech). Separation exists only at
    or below −50 dBFS, where the feature is reading noise. That is a
    specification error whichever way the gate then reads.
  - **−50 dBFS has no recorded derivation.** It entered as a bare default
    argument on `experiments.sanity_checks._leading_trailing_silence` in the
    first commit and was never edited. It is in no constants block, no invariant,
    and no measurement compares it to a corpus noise floor. INV-03 defines
    silence relatively (`TRIM_TOP_DB` below the loudest frame); the probe that
    checks INV-03 defines it absolutely, and the two were never checked against
    each other.

  **The threshold is now derived (INV-12); the verdict is still yours.**
  `SILENCE_FLOOR_FRAME_PERCENTILE`, `SILENCE_FLOOR_PEAK_PERCENTILE`,
  `SILENCE_FLOOR_MARGIN_DB` and `SILENCE_PAD_AGREEMENT` live in
  `data/invariants.py`, the threshold is derived per dataset and tier from real's
  own floor, and the pad-agreement check raises rather than reporting a number
  nobody can interpret. `griffin_lim` is **not** marked `expected_by_design` and
  `SANITY_EER_FLOOR` is unchanged.

  What the next run settles: the derived threshold is predicted to land near
  −39 dBFS on LJSpeech — inside the −43..−37 dBFS band the sweep admits — but the
  quiet-frame statistics it is built from were not in the v0 output, so that is a
  prediction and not a measurement. `btpvf silence-diag` now prints
  `quiet_frame_rms_dbfs` and `quiet_peak_dbfs`, so the derivation is reproducible
  from its output. If the threshold does not read `2 × TRIM_PAD_MS`, the run
  stops and says by how much; that is the designed outcome, not a regression.

  Note what the change does and does not decide. Across the whole admissible band
  the silence EER is ≥ 0.43, so the gate passes there — a consequence of fixing
  the specification, not its justification, and the calibration uses real alone
  plus `TRIM_PAD_MS`, both fixed before any condition existed.
  Numbers: [`docs/measurements.md`](docs/measurements.md).
- **Regenerate every condition.** The archive changed meaning twice: files
  produced under the old INV-17 are band-limited at 8 kHz, and files produced
  before INV-04's move to the analysis band carry a full-band-referenced gain.
  Nothing generated before these changes is valid, including the first v0 run.

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

- **OPEN — first real Phase A run: partly executed.** The first Kaggle run of
  `notebooks/kaggle_v0.ipynb` established:
  - **Real condition.** 200/200 utterances through every gate, in 36 s.
  - **The weight digest.** `bigvgan_generator.pt` at the pinned revision matches
    `meta.json`'s sha256, which settles which file that digest was of.
  - **`--keep-failed`.** It left staging aside as `resynth_v0.failed`.

  The second run generated **all four conditions at 200 utterances**, so the
  INV-13 import-guard fix holds on the real cache and both neural adapters produce
  audio from real weights. It then failed the INV-12 gate on griffin_lim's silence
  probe (its own open item above), which is the gate working rather than a defect
  in it.

  That run also exposed the manifest-path defect: its recorded paths pointed into
  the `.staging` directory that `--keep-failed` had already renamed, so every
  consumer that opens audio failed on it. Paths are now relative to the manifest,
  and an older manifest's absolute paths are recovered from the layout its own
  columns record, so `resynth_v0.failed` is readable where it was left.

  Still **not** established:
  - **A complete pass.** No run has yet cleared every gate.
  - **Alignment.** The INV-16 records from that run are in
    `resynth_v0.failed/manifest.alignment.json`; they are not in this repository.
  - **The bandwidth probe on bigvgan_112m.** It did not fire in that run; whether
    it holds at 3000 utterances is untested.

  The next run also regenerates under band-referenced loudness (INV-04), so its
  audio differs from the v0 run's by each file's gain. Remove `resynth_v0.failed`
  before re-running; preflight refuses while it exists. Closes when the notebook
  completes on Kaggle with every gate passed.
- **Two-tier storage cost.** Every utterance now exists twice. At ~3000
  utterances x 6 conditions x 2 tiers the archive is roughly 2.4x the previous
  footprint, against a 20 GB per-account Kaggle limit. Publish the tiers as
  separate Kaggle Datasets so a session can mount only the one it needs.


## Git
Never run git commit, git push, git rebase, git checkout --, or any
command that modifies history or the working tree's committed state.
Stage with git add and stop. The human commits.