# VocoTrace Phase 4D: robustness redesign and source-diverse dataset plan

**Decision: prioritize C, acquiring and validating multi-speaker paired data.**
Implement its narrowly scoped ingestion/provenance prerequisites before generating
that data; then run one fixed clean-versus-augmentation comparison. Do not start
another legacy-data classifier search. The present evidence supports **B + C in
the root-cause classification**: training-distribution mismatch and dependence
on fragile acoustic cues. No audited preprocessing/decision correctness defect
explains the Phase 4C failures. There are prospective ingestion/enforcement gaps
to close, rather than a demonstrated one-line fix for the failed verifier.

2026-11-07, Asia/Calcutta. Work began on 2026-11-06; the audit directory retains
that start date. Starting manually created HEAD:
`5bf1ff88f996f641069e8bb003f1ace994cee71e`.
**All results and plans are development-only, non-citable.** No new detector,
normalizer, threshold, representation or model bundle is fitted or modified.
No original dataset, frozen eval audio/features/predictions or remote Kaggle
resource is used or changed. The proposed dataset counts are targets, not
newly acquired/generated observations.

## 1. Audit scope and evidence

Read AGENTS.md, the binding eighteen invariant rules in CLAUDE.md, the Phase
4A/4B/4C reports, measurement/mel/enforcement documentation and relevant corpus,
manifest, audio I/O, preprocessing, alignment, generation, feature, verifier,
calibration and partial-generator code. Historical reports and immutable model
artifacts remain historical evidence; a later commit does not make those runs
citable record-mode experiments.

The audit is saved in `runs/vocotrace-phase4d-audit-20261106/`:

- `analyze_invariance.py`, `plan_locked.json`, `invariance_validation.json`;
- `paired_score_sensitivity.csv`, `sensitivity_summary.json`, `feature_bands.csv`;
- `training_audio_processing.csv`, `input_resampling_contract.json`;
- preservation receipts and Ruff/pytest logs.

Use the unchanged Phase 4B v2 bundles and Phase 4C's saved 336 feature rows for
the same **12 training sources / 48 original recordings**, four sources from
each original outer-held fold. Each source is absent from its scoring model's
fit and calibration groups. The measurements cover **1,344 paired case/cell/
recording comparisons**: four contexts x seven cells x 48 recordings. They
are twelve source groups, not 1,344 independent examples. Prior score-method
selection used training OOF results, so this remains an exploratory diagnostic,
not an unbiased nested evaluation of that method selection.

Forty-eight original training WAVs are newly decoded to measure RMS/LUFS and
verify single-file inference and identity PCM16 encoding. No dev audio or
feature array is used. The source manifest is read for structural identities,
then filtered to training sources before waveform access. No frozen eval
waveform, feature array or individual prediction row is opened. Opaque historical
artifact hashing and structural/stat preservation do not constitute new eval
measurements.

## 2. Root causes: tested explanations

### Amplitude law and feature normalization

The existing descriptor contains 32 means and 32 temporal standard deviations
of log10 band power, using the same 8-kHz analysis band, PCM16 delivery,
89,027-sample first crop/repeat wrapper, periodic Hann 1,024-point frames and
256-sample hop. It does not subtract a recording's broadband level.

For a positive gain `g = 10^(dB/20)`, linear STFT/filtering gives
`P_g = g^2 P`. Where the log floor is inactive:

```text
mean(log10(P_g)) = mean(log10(P)) + dB/10
std(log10(P_g))  = std(log10(P))
```

Thus -6 dB changes every powered mean feature by -0.6, and -12 dB changes it
by -1.2. This is an expected mathematical response, not a missing factor of
two in the implementation. Two independent broadband synthetic checks confirm
the mean/std law to 2e-13 without fitting a model.

The saved scaler is a **feature-wise training-population** transform,
`z_j=(x_j-mu_j)/sigma_j`. Its fixed means/scales do not remove a new recording's
gain shift. For the binary linear head, the ideal gain-induced margin change is:

```text
delta_real_margin = (dB/10) * sum_j<32(binary_coefficient_j / binary_scale_j)
```

The measured delta from actual PCM16 features is independently reproduced by
`sum_j(delta_feature_j * coefficient_j / scale_j)` to 1e-12 for all 1,344
comparisons. The ideal gain formula additionally assumes no floor/quantization
effect; the audit records its residual instead of pretending it is exact on
encoded speech.

| Closed3 gain cell, 48 recordings | Ideal margin shift, median | Actual margin shift, median | Actual-minus-ideal, median [min, max] |
|---|---:|---:|---|
| -6 dB | +0.931267 | +0.941093 | +0.003776 [-0.300005, +0.247389] |
| -12 dB | +1.862533 | +1.913899 | +0.001211 [-0.403557, +0.537349] |

A positive delta is **more real-like**, which explains loss of synthetic-route
coverage under attenuation. It is not evidence that quieter speech is genuine.
The source head also changes: median largest absolute source-logit change is
1.2451 at -6 dB and 2.5130 at -12 dB. A fixed scaler is working as designed on
inputs outside its narrow level distribution.

As a feature-only diagnostic, subtracting the across-band mean from the 32
log-power means would cancel the ideal common gain offset. Its actual median
shape-change MAE is 0.003697 at -6 dB and 0.013743 at -12 dB, compared with
64-feature raw MAE 0.300523 and 0.601347. Standard-deviation MAE remains
0.004654 and 0.018270. Floors/quantization prevent perfect invariance. **No
centered features are fed to a historical model** and no improved attribution
result is claimed. Relative-power features are a future versioned hypothesis,
not an authorized change to the existing descriptor or a cure for filtering.

### RMS, loudness and encoding

All 48 measured original training inputs satisfy the current -27 LUFS
**analysis-band** target with maximum absolute deviation 0.000016022 LU; all are inside the
unchanged 0.5-LU tolerance. Full-band measurements agree with manifest
`lufs_out` to about 0.000014 LU. These local measurements do not reconstruct
the complete legacy generation/checkpoint history or re-admit the dataset.

| Actual condition | Median RMS dBFS | Median measured analysis-band LUFS |
|---|---:|---:|
| Genuine reference | -27.2212 | -26.999999 |
| BigVGAN-112m | -27.1105 | -27.000001 |
| Griffin-Lim | -27.2538 | -27.000002 |
| Vocos | -27.2232 | -27.000002 |

RMS and LUFS measure different properties; equal LUFS need not mean identical
RMS or equal spectral shape. A post-hoc per-class RMS correction would change
the established processing and could erase or introduce cues. None is applied.
The observed attenuated inputs intentionally depart from the canonical training
level; they are named analysis stress conditions, not damaged archive files
to repair. Re-normalizing them would remove the test being performed.

Identity PCM16 re-encoding gives **48/48 decoded-sample equality and 48/48 exact
feature equality**. This only establishes stability of PCM16 -> identical
PCM16 at the same rate. It does not cover quantization after attenuation,
different bit depths, lossy codecs or rate conversion. No codec is added to
the sanctioned pipeline. Gain followed by quantization is not generally
equivalent to gain applied to already computed spectral statistics.

### Training versus inference

`extract_comparison_features` and `extract_file_features` use the same filter,
PCM16 round trip, crop/pad and feature calculation. The singleton inference
path uses the real-key API slot for an unknown waveform; the independent
filter never uses a ground-truth reference or condition label. All **48/48
single-file feature vectors match their saved grouped training-path vectors
exactly**, maximum absolute difference zero.

There is therefore no reproduced grouped-versus-singleton processing bug in
this population. The important difference is the **input distribution**:
training audio was previously trimmed/level-normalized by Phase A, while the
inference entry checks rate/channels/encoding/duration but neither establishes
that processing history nor checks comprehensive domain support. It deliberately
does not silently resample, retrim or normalize user inputs. Header acceptance
alone is inadequate evidence that an input belongs to the calibration domain.
An application would need a separately versioned/admitted input protocol and
domain checks; copying Phase A normalization into inference without evaluation
would not establish robustness.

### Spectral dependence and confidence behavior

The 6-kHz stress condition applies the same sanctioned lower-band filter to
every condition, followed by the unchanged 8-kHz inference filter and PCM16
delivery. It does not expose the known **above-8-kHz** Griffin-Lim cliff or
Vocos resampler signature. Removing common below-8-kHz evidence is nevertheless
a substantial distribution shift for these absolute spectral features.

For genuine closed3 inputs, mean binary-margin movement is **-7.698436**.
Exact mean feature-contribution decomposition gives:

| Feature band | Contribution to genuine binary-margin change |
|---|---:|
| Below 4 kHz | +0.003711 |
| 4--6 kHz | +0.044946 |
| 6--8 kHz | -7.747094 |

The contributions sum to the observed movement; the strongly negative last
band dominates opposing small terms. This establishes this model's sensitivity
to the removed band, not a universal mechanism for all vocoders or proof that
every remaining cue is causal. Phase 4C's genuine median largest absolute
standardized feature rose from 2.2347 to 11.3435 under this filter.

Closed3 negative energy rises by median **+4.700950**, nominally *more*
known-like, while all 48 recordings become BigVGAN predictions. MSP rises by
median +0.252714. Meanwhile the unused distance score decreases by median
-1,284.5413, reflecting loss of proximity to clean training references.
The frozen selected energy gate is not a calibrated domain-support test.
Switching to distance now would be a new policy, not a correction to historical
predictions. Distance also does not automatically guarantee useful robustness:
it can reject everything after a benign channel shift.

### Thresholds and resampling

Binary routing is `real_margin <= cutoff`; source support is
`confidence >= cutoff`. Ties, source-class ordering, two-class logit mapping and
the returned states agree with the tested Phase 4B implementation. A real-like
branch is not unknown-source recognition or an authenticity certificate.

The predeclared local diagnostic shifts only the binary cutoff by +/-0.1 raw
margin units; it is **not** threshold selection or a probability interval.
On the closed3 clean 48 recordings, either direction changes one final source
decision. On the 6-kHz inputs, either direction changes **zero** final source
decisions: the failure is far larger than an infinitesimal boundary/tie issue.
The saved retain80-versus-retain95 confidence settings change 1/48 clean
closed3 decisions and zero 6-kHz decisions. Full counts for every context/cell
are recorded; no favorable perturbation is adopted.

The existing one-step 22,050 -> 16,000-Hz resampling preflight on these same
48 training inputs has valid PCM16 format/lengths, duration error at most
0.000055131 s and canonical derived-loudness-gate maximum drift 1.706413 LU.
Phase 4C verified matching bytes from `derive_zeroshot`. **All 48 are correctly
refused by the frozen 22,050-Hz model feature contract**. No resampled source
scores or feature-vector equivalence are invented. Rate-aware physical-band,
time-window, FFT/power-normalization and tier tests are required before scoring
16-kHz inputs. Upsampling them back or relabeling headers would bypass, not
validate, that requirement. A resampler changes signal/encoding behavior even
when delivered duration is preserved.

### Root-cause classification

| Explanation | Evidence-based finding |
|---|---|
| A: current correctness defect causes the 4C failures | Not reproduced. Exact grouped/singleton features, identity PCM16, gain law, linear score decomposition and decision arithmetic agree. This is scoped evidence, not proof that no unrelated bug exists. |
| B: training-distribution mismatch | Supported. Clean normalization/crop statistics do not cover changed gain, bandwidth or speech support. Inference header checks do not establish domain membership. |
| C: fragile acoustic cues | Supported. Uniform power offsets move logits; 6--8-kHz feature changes dominate the 6-kHz gate failure; less available speech changes pooled statistics. |
| D: combination | **B + C** best explains the observed verifier failures. Prospective ingestion gaps below are separate issues and must not be retroactively blamed for dormant paths in v0. |

## 3. Invariance measurements and decision flips

All entries below compare the **same 48 training recordings** to their clean
counterparts with the same frozen outer-held model/cutoffs. MAE averages 64
features per recording, then the table reports the median across recordings.
Score deltas are uncalibrated model units, not probabilities.

| Closed3 cell | Median feature MAE | Median real-margin delta | Median negative-energy delta | Argmax candidate flips / 48 | Final source-or-reject flips / 48 | Final state flips / 48 |
|---|---:|---:|---:|---:|---:|---:|
| Gain -6 dB | 0.300523 | +0.941093 | +0.296771 | 14 | 6 | 6 |
| Gain -12 dB | 0.601347 | +1.913899 | +0.924278 | 19 | 10 | 11 |
| Below 6 kHz | 0.658443 | -8.122688 | +4.700950 | 38 | 47 | 38 |
| Below 4 kHz | 1.343825 | +3.111479 | +1.818418 | 26 | 10 | 11 |
| First 1 s | 0.378707 | +1.515902 | +0.681441 | 16 | 16 | 16 |
| First 2 s | 0.153892 | -0.427451 | +0.284324 | 8 | 12 | 13 |
| Identity PCM16 encoding | 0 | 0 | 0 | 0 | 0 | 0 |
| Derived 16 kHz | Unsupported feature contract | Not scored | Not scored | Not scored | Not scored | Not scored |

`Final source-or-reject` compares class index or no source label. `Final state`
distinguishes real-like rejection from uncertainty rejection as well. Therefore
state changes can occur while both endpoints have no source label. Candidate
flips include candidates that the system never accepts. This prevents a
reject-all output from appearing to preserve source evidence.

For all held-generator contexts, final source-or-reject flip counts are:

| Cell | Without BigVGAN | Without Griffin-Lim | Without Vocos |
|---|---:|---:|---:|
| Gain -6 dB | 4/48 | 0/48 | 7/48 |
| Gain -12 dB | 7/48 | 0/48 | 7/48 |
| Below 6 kHz | 20/48 | 46/48 | 7/48 |
| Below 4 kHz | 12/48 | 4/48 | 7/48 |
| First 1 s | 18/48 | 8/48 | 7/48 |
| First 2 s | 7/48 | 11/48 | 3/48 |

Zero flips without Griffin-Lim under attenuation are **not** successful
invariance: the clean primary policy already returns zero labels on these
48 recordings, while its source argmax still changes on 10/48 and 11/48 inputs
at -6/-12 dB. Full per-condition counts/distributions and individual paired
changes remain in the CSV/JSON. No training/dev/eval reassignment, threshold
optimization, new fit or empirical gain/filter search occurs.

## 4. Source-diverse data: candidates and rights

Public source documentation and licenses were checked on 2026-11-06/07.
This is a suitability/rights audit, **not legal clearance or acquisition**.
Read the exact release notices again when acquiring it and retain copies/digests.
No audio archive, external dataset or checkpoint is downloaded in this phase.

### Candidate comparison

| Candidate | Verified public facts | Suitability and unresolved requirements |
|---|---|---|
| **VCTK 0.92** | Edinburgh describes 110 English speakers, common studio recording setup, two microphones and 48-kHz/16-bit distributed recordings. Its README/license explicitly identify CC BY 4.0. | Preferred **conditional** primary corpus. One fixed microphone, original lossless release, speaker/source identity checks, transcript overlap control and a multi-speaker ingestion preflight are still required. Do not use a lossy mirror. [Official release](https://datashare.ed.ac.uk/handle/10283/3443), [official README](https://datashare.ed.ac.uk/server/api/core/bitstreams/bb7edd96-5d96-4c0e-8989-1e45597e7b72/content). |
| **Original LibriTTS** | OpenSLR lists about 585 hours, 24 kHz, multiple speakers and CC BY 4.0; it explicitly states that the source materials were LibriVox MP3 and Gutenberg text. | Strong speaker breadth, but existing compression history is material to this forensic task. Not a lossless pristine-real replacement under the present no-lossy-intermediate design. Treat as a separately reviewed channel/history cohort, not silently admitted primary data. Vocos paper training uses LibriTTS training partitions, so upstream overlap is another concern. [Official source](https://www.openslr.org/60/), [Vocos paper, section 4.1](https://arxiv.org/html/2306.00814). |
| **LibriTTS-R** | OpenSLR identifies CC BY 4.0 and restoration with the Miipher model. | Do not call restored audio pristine genuine recordings. It could become a named processed-real control, with rights/history documented, but is not the primary negative class. [Official source](https://www.openslr.org/141). |
| **Hi-Fi TTS** | OpenSLR identifies CC BY 4.0, 10 speakers, 44.1 kHz, roughly 291.6 hours, and LibriVox/Gutenberg origins. | Smaller speaker breadth; original codec/restoration history and upstream model overlap require verification. Not presumed lossless just because delivered audio is high rate. Possible separately admitted supplementary cohort. [Official source](https://www.openslr.org/109/). |
| **LibriSpeech** | OpenSLR identifies 16-kHz speech and CC BY 4.0. | Unsuitable as a source for this full-band 22.05-kHz archive: upsampling would not restore missing bandwidth. A separately versioned 16-kHz research task would be needed. [Official source](https://www.openslr.org/12/). |

VCTK microphone choice needs explicit resolution: the unimplemented repository
stub says mic1, while TensorFlow's catalog notes very-low-frequency noise for
mic1 and technical issues for two mic2 speakers. This plan proposes **mic2 only**
for a new cohort, subject to measured inventory/quality eligibility, and records
the departure from the dormant stub. No fallback to mic1 for individual files
is permitted. PyTorch's loader documents missing transcript/audio cases; those
are starting checks, not assumed actual exclusion counts for our unacquired
subset. [TensorFlow catalog](https://www.tensorflow.org/datasets/catalog/vctk),
[PyTorch loader](https://raw.githubusercontent.com/pytorch/audio/main/src/torchaudio/datasets/vctk.py).

One additional channel-study candidate, **DAPS**, has an author Zenodo record
describing twenty speakers and studio/device variants. The rendered official
record did not expose usable license text, the author-site fetch failed, and
metadata/export endpoints were inaccessible in this review. Therefore its
release rights and exact raw/processed variants are **not verified here**;
do not admit it or copy a mirror's license assertion into our metadata.
[Author deposit](https://zenodo.org/records/4660670).

### What is verified versus what remains open

CC BY 4.0 grants sharing/adaptation subject to attribution and other terms.
Preserve creator/copyright/license/disclaimer notices, link the source/license,
identify changes and avoid implying endorsement or adding downstream restrictions
inconsistent with the grant. Source licensing does not supply all privacy,
publicity, personality or other rights. Review the relevant audio, text and
third-party notices for the actual release. [License deed](https://creativecommons.org/licenses/by/4.0/),
[legal code, sections 2--3](https://creativecommons.org/licenses/by/4.0/legalcode.en).

Development-only/non-citable describes this project's **research evidence
status**. It does not add a legal prohibition on uses permitted by an upstream
license or authorize relicensing the original recordings as our own work.

The VCTK release license is verified; whether the exact selected material and
intended release satisfy every relevant right/notice requirement remains a
release-review responsibility. Retain the corpus attribution to Yamagishi,
Veaux and MacDonald/University of Edinburgh, version 0.92 and DOI
10.7488/ds/2645. Preserve separate notices for any included transcripts rather
than assuming every upstream newspaper/book passage is freely reusable for
every purpose. Lack of a redistribution plan is not solved by private visibility.

BigVGAN's official code license and relevant model card identify MIT; Vocos's
official code and mel-model card likewise identify MIT. These establish the
stated software/model licensing context, **not a blanket license over input
recordings or all derived output**. Freeze the actual checkpoint revision,
weights hash, card/license version, generation code, front end and source rights.
Keep software notices with distributed code; retain source-audio attribution
and transformation lineage with derivative audio. No CC0 or all-output-MIT
claim is inferred. [BigVGAN license](https://raw.githubusercontent.com/NVIDIA/BigVGAN/main/LICENSE),
[BigVGAN model card](https://huggingface.co/nvidia/bigvgan_v2_22khz_80band_256x/raw/main/README.md),
[Vocos license](https://raw.githubusercontent.com/gemelo-ai/vocos/main/LICENSE),
[Vocos model card](https://huggingface.co/charactr/vocos-mel-24khz/raw/main/README.md).

The exact pinned revisions' rights still need matching to the releases actually
used; inspecting current main-branch pages is not that pin verification.
Griffin-Lim implementation/library notices must also be retained as applicable.
Before any separate Kaggle authorization, review the platform terms and
metadata license against this rights ledger. Do not copy btpvf-partial-v1's
legacy eligibility assumptions, choose a new public slug automatically or
assume an existing owner confirmation authorizes this new publication.

### Proposed bounded cohort and provenance contract

Target **50 eligible VCTK speakers x 10 eligible utterances = 500 source groups**,
each paired with real + the same three admitted resynthesis pipelines. That
would be 2,000 archive WAVs, plus 2,000 separately derived 16-kHz WAVs if all
current gates pass. **None exists yet.** Target counts may be missed; publish
actual eligible/excluded counts with measurements, never fabricate coverage.

1. Review the official release, rights and source-format/codec history before
   acquisition. Preserve archive/license checksums, acquisition URL/version/date,
   speaker IDs, native rate/encoding, microphone and prior processing history.
   Only future explicitly authorized acquisition may download the audio archive.
2. Enumerate the actual mic2 inventory with stable namespaced corpus/speaker/
   source IDs. Verify physical files, mono native-rate headers, transcripts,
   finite decoding, source hash, clipping/quality flags and recording origins.
   A metadata assertion alone does not prove authentic, unprocessed human audio.
3. Remove shared elicitation/rainbow prompts and normalized-transcript duplicates
   that would cross speakers; record exact matching evidence. Check near-duplicate
   recordings, source/document/session identity and any identified cross-corpus
   speaker aliases. All versions/microphones/channels of a source remain one
   group. Do not use source ID, text, speaker, filename or generator label as an
   inference feature.
4. Require at least ten admissible candidate utterances per speaker. Select
   eligible speakers by ascending SHA-256 of
   `20260907:vocotrace-ms-v1:speaker:<namespaced_speaker_id>` and each speaker's
   utterances by `20260907:vocotrace-ms-v1:utterance:<namespaced_source_id>`;
   break any hash ties by the canonical ID. Do not use detector success.
   Target 4.1--8 s after the fixed real-reference trimming
   policy for this initial cohort; record actual duration exclusions. These
   bounds are a new-cohort sampling rule, not an alteration of old archives or
   inference crop duration. Report resulting content/accent/duration selection
   bias and source coverage.
5. Freeze the speaker map once using existing `assign_splits(...,
   speaker_disjoint=True, seed=20260907)` and unchanged 70/10/20 fractions. For
   exactly fifty speakers this gives **35 train / 5 dev / 10 eval speakers**,
   or 350/50/100 source groups if ten per speaker survive. Namespaced source IDs
   and hashes inherit that map for every vocoder, tier, crop, augmentation and
   future partial control. Never redistribute old 140/20/40 or 123/17/34 groups.
   Post-generation exclusions reduce actual counts; they do not trigger a new
   split or replacement of final-evaluation speakers.
6. Generate through the existing Phase A ordering, native vocoder front ends,
   pinned checkpoints/resamplers, real-derived trim spans, one archive loudness
   normalization and PCM16 writer. All delivered archives remain 22,050-Hz mono
   full-band; derive 16 kHz once from those archives without renormalization.
   No adapter denoising, per-condition repair or alignment compensation.
7. Run current manifest/pairing/length/peak/loudness/alignment and sanity gates.
   At least twenty valid paired files are required for each alignment probe;
   require the existing zero-lag contract, not just matching file lengths.
   Record measured failures and stop rather than lowering a gate. Evaluate
   newly introduced resampling paths with the established processing-control
   methodology; do not assume old LJSpeech controls transfer to native 48 kHz.
   Do not insert chained-resampling controls as production archive records.
8. Save the complete inventory, parent/output hashes, code/environment/checkpoint
   identities, manifest/source/split digests, source and generator provenance,
   gate records and rights ledger. All new development outputs remain non-citable;
   only a later authorized valid record run tied to a user-created commit could
   support final research claims.

VCTK's acquired native FLAC files are lossless source containers, not sanctioned
processed artifacts. The proposed ingress is direct native decode through
`data.audio_io`, then the existing single output derivation; all materialized
processing intermediates/deliverables remain PCM16 WAV. **Resolve this acquisition-
versus-intermediate boundary against the strict INV-05 wording before admission**;
do not silently grant an exception, rewrite CLAUDE.md, create unsanctioned
48-kHz intermediates or treat a lossy mirror as equivalent. If the existing
rules disallow the proposed ingress, the cohort remains blocked until a
separate explicit policy decision or an admissible original source is available.

The legacy one-sided confound-gate orientation issue remains separate. Preserve
its existing 0.40 floor/rule; record both directional diagnostics and require
separate review of unresolved reversed separability. A structural/pin pass or
a one-sided historical verdict does not establish empirical confound freedom.

Upstream generator-training overlap is another separate axis. The Vocos paper
identifies LibriTTS training data; BigVGAN cards describe model-specific training
corpora/compilations, whose complete membership is not established here. New
speaker-disjoint **detector** splits do not prove those speakers were unseen by
each pretrained vocoder. Record known overlap and unknown membership, and reserve
a separately verified corpus/speaker acquisition for stronger generalization
claims. Do not relabel an uncertain upstream history as verified independence.

## 5. Existing infrastructure and required implementation

| Component | Reusable behavior | Work before new generation |
|---|---|---|
| `data.corpus` | Utterance/speaker abstraction; seeded speaker-disjoint assignment | `VCTK.utterances()` is still `NotImplementedError`; implement validated fixed-microphone enumeration, source IDs and eligibility receipts. |
| `phase_a_resynthesis` | Reference processing, paired native front ends, alignment, staging, gates and manifests | Generic helper accepts a Corpus, but preflight/gated CLI assumes LJSpeech `metadata.csv`/`wavs`; add corpus-specific preflight/speaker-disjoint plumbing without bypassing gates. |
| Source resampling | Real native -> archive uses explicit `RESAMPLE_METHOD`; Vocos input/output use its explicit pinned spec | Non-Vocos input path at line 307 calls `librosa.resample` without `res_type`. Current default **is** `soxr_hq`, matching the constant; this is an explicit-pinning/enforcement gap, not a reproduced current numerical method mismatch. Pin it and test a non-22.05-kHz fixture before expansion. |
| `audio_io`, preprocessing, manifests, alignment | Sanctioned PCM16 writing, reference trim, loudness/peak checks, grouping, recorded derivations | Preserve native decode provenance and all input/output resample specs; test 48-kHz source ingress and new source paths. No implicit channel mixing or shifted repair. |
| Source verification | Numeric frozen heads, train-only scalers, class maps and raw evidence | New experiment/protocol artifacts must have distinct identities; old bundles remain untouched. No advertised rate/gain/channel support beyond evidence. |
| Partial generator | Paired sample replacement, transitions, exact reconstruction, null and disjoint real-to-real controls, split/speaker checks | Keep whole-audio attribution primary. Apply to an explicitly eligible new-cohort stress subset after whole-audio validation; report donor exclusions. Do not train on boundary labels in the first attribution comparison. |
| Kaggle harness | Existing sole runner and provenance workflow | Future authorized vocoder generation uses it with one adapter environment per session. No new runner or remote execution is needed now. |

The non-Vocos input branch is dormant for the old 22.05-kHz BigVGAN/Griffin-Lim
source path. It cannot explain the old 4C amplitude/filter failures. Runtime
signature plus AST evidence is saved in `input_resampling_contract.json`.
No production source correction is made in this design phase.

## 6. Revised training and abstention protocol

Treat known source attribution, genuine false source matching and withheld
generator behavior as separate tasks. A classifier's real-like output is not
a genuine-speech claim. Source labels refer to a pinned reference pipeline and
front end, not an entire architecture, author, product or every synthetic voice.

Use equal speaker/source contribution and explicitly balanced **binary loss
mass**: 1/2 genuine, 1/2 synthetic, with synthetic mass divided equally among
the permitted known pipelines. This is not a claim that four-condition paired
files have equal binary class counts: they have one genuine and K synthetic
recordings per source. Keep the attribution loss balanced across known sources.
Do not mix a withheld generator into fit/scaler/reference/calibration/selection.
Every augmentation descendant divides its parent's weight; added views must
not multiply a speaker's or source's contribution.

Apply the same declared transform distribution to complete paired comparison
groups, including genuine. Gain, stopband, source-coordinate duration and any
future rate condition cannot be class-correlated. Preserve the original
archives; record transform order, gains, actual support, PCM16 identity and
failures. Fit normalization/scalers, class support references and any eventual
learned representation on **model-fit** speakers only. Calibration and assessment
have different speakers; all descendants and overlapping excerpts stay grouped.

The initial test keeps the existing 64-feature/linear model family to isolate
the effect of robustness training. No pretrained neural representation or
hyperparameter search is bundled into that comparison. A relative-level
descriptor or reproduced learned fingerprint reference requires a subsequent
separately specified experiment if these features remain inadequate.

Use a fixed class-support score (negative predicted-class Mahalanobis distance,
same regularized covariance family) for the new protocol, alongside a separate
genuine-false-match routing gate. Energy/MSP remain raw descriptive diagnostics,
not unknown probabilities. The new score choice is predeclared from these
training-only failure mechanisms and must not be selected by the new withheld
generator outcomes. Reproduce the frozen Phase 4B selected methods as references;
do not overwrite them or present the new choice as a historical correction.

Calibrate only on permitted genuine/known calibration speakers and the fixed
stress matrix. For the binary real-margin gate, form one **minimum genuine
margin per calibration speaker** over all that speaker's utterances/views,
then apply the existing whole-tie `genuine_gate(..., 0.05)` construction. This
is a conservative new-protocol objective controlling observed any-view routing
on calibration speakers. With seven calibration speakers it permits **zero**
such failures; it is not a population 5% risk guarantee. If this yields zero
useful known coverage, report failure rather than relax it.

For source support, use the existing retain95 construction on permitted known
calibration views, with equal speaker/source/view mass. With exactly ten
utterances per speaker and seven views this is ordinary equal-row weighting.
If exclusions change that balance, stop this preregistered comparison and issue
a revised prospective plan rather than silently alter weights/quantiles.
Confidence retention alone does not guarantee joint coverage or correctness.
Lock all cutoffs and models before assessment. Also report the old per-row
calibration reference descriptively if required, without selecting between
calibration rules after outcomes.

Every returned label needs both routing and source support; retain **no
supported attribution** / **inconclusive** outcomes and explicit rejection
reasons. Record false-known labels on genuine and withheld inputs. A processed
genuine input with a strong wrong source match is a safety failure even when
known-only F1 remains high. A reject-all system is not successful robust evidence.

## 7. Predetermined first experiment after data admission

**Status: proposal only; no model in this section is trained.**

### Data, roles and exact arms

- Prerequisite: the bounded new paired corpus above passes rights, ingestion,
  provenance and unchanged scientific gates. Require the planned 35 training
  speakers with ten eligible sources each. If unmet, report actual coverage
  and review a new plan before fitting; do not borrow dev/eval speakers.
- Use only those **35 training speakers / 350 sources** for the first comparison.
  Keep both original frozen eval and the proposed new ten-speaker eval isolated.
  Do not score the exposed old dev to choose parameters. The proposed five-speaker
  new dev remains unused until an independently reviewed lock/one-pass benchmark.
- Hash-sort the 35 training speaker IDs using namespace
  `20260907:vocotrace4e:outer:<speaker>` and distribute round-robin to five folds,
  seven speakers each. For outer fold f: assess f, calibrate `(f+1) mod 5`, fit the
  other three. This gives **21 fit / 7 calibration / 7 assessed speakers**, or
  210/70/70 sources if ten per speaker remain. Freeze the list before fitting.
- **F0:** unchanged Phase 4B final bundles and their selected score/cutoffs, scored
  on the same new training assessment population. No retraining/recalibration.
- **F1:** new clean-only fit, existing 64-statistic representation and independent
  binary/attribution StandardScaler + L2 logistic heads; C=1, same solver/tolerance/
  iterations/seed as Phase 4B. New protocol's fixed distance/support calibration.
- **F2:** identical to F1 except model fitting uses the seven fixed symmetric
  views below, with parent-balanced contributions. F1 and F2 use the **same**
  calibration-speaker/view protocol, assessment views, weighting rule, support
  score, cutoff rule and source maps. This isolates robustness training more
  meaningfully than changing representation, architecture and cutoffs together.

For F1/F2 run closed3 and all three leave-one-generator-out contexts, excluding
that context's withheld class from both heads, scalers, support statistics and
calibration. Five folds x four contexts x two arms x two heads = **80 planned
CPU logistic fits**, plus zero fits for F0. No C grid, temperature search,
repeated loss-weight sweep, generator-specific band selection or refit after
calibration. Keep all numeric bundles, folds and cutoffs separately versioned.

### Fixed augmentation/assessment matrix

Use clean, -6 dB, -12 dB, common 6-kHz stopband, common 4-kHz stopband, first
1 s and first 2 s. Each view is applied directly to the same original parent,
then the existing mandatory 8-kHz inference/PCM16/crop path. No combinations,
stochastic search, per-class processing, clipping repair, loudest-crop selection
or gain normalization is added. Repeat-padding remains 4.0375 s. F2 uses all
seven equal-weight views; F1 uses clean fit views only. Both are assessed on
all seven, and all calibration speakers receive the same fixed matrix.

These are **seen transform families** in a robustness-training comparison.
Passing them would establish scoped robustness on those conditions, not unseen
codecs/channels. Exact matched transformation coverage is the first question;
a separately preregistered new-transform test follows only if it passes.
Single-step 16-kHz resampling remains a format/loudness preflight, **excluded
from classifier scoring** until a separate rate-aware feature/model protocol is
implemented and independently checked. Do not use an old 22.05-kHz model on
resampled audio by silently converting it back.

Withheld-generator evaluation initially means the three **within-roster**
leave-one-out trials. A genuinely external fourth pipeline is a later separately
admitted target requiring its own checkpoint/code/license/front-end/alignment
evidence and paired generation on the same sources. Its admission must not be
conditioned on easy rejection. No additional independent family or unknown
identity capability is claimed from these three trials.

### Metrics and prospective continuation criteria

Report per cell/context/pipeline and worst cell: oracle known-source macro-F1/
confusion, known accepted accuracy and coverage, correct-known fraction over
all known inputs, genuine false source attribution, withheld wrong-label and
rejection rates **split by real-like versus uncertainty reason**, raw ranking,
all-input source-label precision/coverage with explicit mixture, and paired
candidate/gate/decision flips. Include unconditional argmax, binary-only and
reject-all references. Report worst-view-per-source and any-view-per-speaker
genuine false matches, not just per-recording means.

Use paired speaker-bootstrap resampling with all a speaker's source groups and
views retained together; at least 2,000 fixed-seed draws, undefined ratios
counted, conditional no-refit intervals explicitly labeled. Thirty-five
assessment speakers and seven calibration speakers per fold remain small;
there is no population low-error guarantee. Withheld contexts overlap in data
and are not independent replications of all possible generators.

Predeclared **engineering continuation criteria**, not replacements for any
scientific invariant or publication gate:

1. On F2, worst-cell genuine false source attribution <=5% of source recordings,
   with any-view speaker/source failure counts also disclosed. No hiding a
   failed pipeline/cell behind pooled averages or a degenerate bootstrap interval.
2. Every scored cell has >=20% known coverage and >=90% accepted-known accuracy;
   separately, all-input label precision >=90% for the declared genuine/known/
   withheld mixture. Undefined accuracy/precision and reject-all fail continuation.
3. Clean known-source macro-F1 loses no more than 0.02 versus **F1 on the same
   groups**, while minimum transformed-cell macro-F1 is >=0.70. Report all
   pipelines, including the B/V confusion that G can otherwise conceal.
4. Minimum-cell correct-known fraction improves by >=0.10 over matched F1;
   show paired speaker uncertainty and require its lower 95% conditional bound
   above zero for that prespecified worst-cell difference. If not supported,
   conclude insufficient evidence rather than choose a different cell/metric.
5. In each withheld context, wrong known-source labels <=5% per cell and useful
   known coverage retained. Explicitly distinguish rejection mechanism. Even
   passing this within-roster objective does not establish unknown-generator
   identification or justify calling the real-like branch genuine.

All five must hold to continue toward a broader evidence backend. These targets
are design choices motivated by intended utility/risk, not existing achieved
measurements or assurances about achievable statistical power. If they fail,
freeze the negative result and review representation/data limitations. Do not
run a sequence of new augmentation/threshold searches on these assessed folds
or the already exposed old dev. New proposals need a prospectively defined
assessment resource and must disclose all previous exposure.

For the minimum-cell comparison in criterion 4, compute each arm's minimum over
the **same six changed cells** within every bootstrap draw, then subtract F1
from F2; this evaluates the declared worst-case statistic, not a post-hoc chosen
favorable cell. Clean is reported separately. The recipe, interval rule and
failure handling must be locked with the future runnable experiment.

### Explicit Phase 4B reference

Preserve its exposed twenty-source dev macro-F1 **0.8108747**, 18/80 returned
labels with 17 correct, 0/20 genuine false labels, and each withheld dev trial's
19/20 real-like rejections / 0 uncertain / 1 wrong source label. Those are
historical reference results, **not a new tuning set or matched benchmark** for
a multi-speaker model. The current 4C twelve-training-source cells likewise
remain a fixed diagnostic reference: 10 clean closed3 labels, zero at -12 dB,
all 48 labeled BigVGAN after 6-kHz filtering including twelve genuine inputs.

F0 is additionally evaluated on the new **training-only assessment** population
after admission, so future tables can compare all arms on the same inputs.
The causal augmentation comparison is F2 versus F1; F0 versus either new arm
also changes training corpus/population/calibration and cannot isolate one cause.
Do not demand that a different-corpus score beat 0.8109 and call that a robust
improvement, or repeatedly inspect dev to force such a headline.

## 8. Compute, engineering and VocoPatch role

The initial data-generation target is at most 500 x 4 x 8 s = **16,000 seconds
of archive audio**, before exclusions. At 22,050-Hz mono PCM16 that is at most
705.6 MB decimal of waveform payload; optional 16-kHz derivatives add at most
512.0 MB. Headers, manifests and provenance add overhead; these are design
upper bounds, not generated sizes. Allow separate staging/failure space and
the official source download, whose release page currently lists 10.94 GB.
An archive-oriented official release may require the full download to obtain a
small subset; no unsupported partial downloader or lossy streaming mirror is
assumed. [Official VCTK files](https://datashare.ed.ac.uk/handle/10283/3443).

Only 350 x 4 x 7 = **9,800 feature vectors** are needed for all training views
at the target size; a float64 64-column matrix is about 5.02 MB, plus row metadata.
Use cached immutable features with source/transform digests, bounded per-source
audio memory and the existing CPU environment. F1 clean closed3 fits 840 binary
rows / 630 attribution rows per fold; F2 fits 5,880 / 4,410 rows. The eighty
planned linear fits require no GPU. No wall-time or memory performance result
is invented; profile the admitted pilot before scheduling the full run.

Vocoder generation needs separately approved compute and isolated adapter
environments through the existing Kaggle harness. First time a fixed twenty-
utterance training preflight per adapter after checkpoint verification; use its
measured real-time factor to estimate remaining generation time. GPU permission,
checkpoint/dataset acquisition and remote execution are not granted by this
design report. Existing pins and front ends remain fixed; no neural model is
trained here or promised as a cheap upgrade.

The next implementation deliverable should be **a CPU-tested multi-speaker
ingestion/preflight and provenance package**, including the explicit resampler
pin, fixed microphone, namespaced grouping and speaker-disjoint CLI/harness
support. Test native-rate source fixtures, missing/duplicate/source-overlap
cases, exact split inheritance, method pinning, gates, no eval decoder access,
and deterministic receipts before requesting acquisition/generation authorization.
This is the concrete work supporting choice C, not a new frontend.

Once validated whole-audio evidence exists, VocoPatch supplies a **secondary
controlled stress set**: mixtures, original real-only null controls and disjoint
real-to-real edited controls with sample labels and transition/ignore regions.
Keep all descendants with their source/speaker; report donor-eligibility bias
and the fraction of synthetic evidence actually inside the fixed input crop.
Do not force a pure-generator label onto a real/mixed recording, use edit truth
to choose inference crops, count windows as independent examples or claim that
splice controls eliminate every boundary confound. Localization training and
frontend work remain outside the first attribution experiment.

## 9. Verification, changed files and remaining limits

New no-fit diagnostics pass: two independent gain-law checks, 1,344 exact linear
score decompositions, 48 singleton/grouped feature replays, 48 identity PCM16
sample/feature re-encodes, source-disjoint model roles and input/historical
integrity checks. All 48 measured training originals satisfy the loudness
target; no input is repaired or excluded to obtain that result. Existing
resampling evidence is reused with its explicit unsupported-scoring limitation.

Focused existing tests: **55 passed**, 64.95 s, covering the Phase 4B verifier
and Phase 4C replay/stress driver. Source/test Ruff passes; repository Ruff
retains the eleven existing generated-notebook errors. Full pytest: **962 passed,
4 failed, 7 skipped, 1 xfailed**. The failures retain their established identities:

- `test_kaggle_harness.py::TestGeneratedNotebook::test_the_committed_notebook_is_exactly_what_the_harness_generates`
- `test_phase_a_run.py::TestKaggleNotebook::test_the_generated_notebook_is_the_committed_one`
- `test_phase_a_run.py::TestUpstreamImportGuardThroughADirectoryLink::test_namespace_package_through_the_link_passes`
- `test_phase_a_run.py::TestUpstreamImportGuardThroughADirectoryLink::test_namespace_package_from_elsewhere_fails`

The first two are notebook CRLF/LF byte comparisons and the other two are Windows
directory-junction `WinError 5` failures. Existing link-privilege skips and the
unimplemented-ablation expected failure remain. No new failure appears. This is
implementation verification, not a new remote experiment or population-validity
claim.

Independent replay compares every one of the 1,344 new score-change/flip records
against the unchanged Phase 4C prediction records, including raw score deltas,
candidate changes, routing/support gates, source/reject states and accepted
labels lost/gained. All agree. Preservation checks pass for **4,621 prior file
paths across 19 roots**, with unchanged paths/sizes/mtimes and protected non-WAV
digests, including both Phase 4B runs and the Phase 4C output. All previously
tracked file hashes also agree. Original/partial datasets, historical pilots,
old reports and model bundles are unchanged. Eval waveform preservation is
stat/inventory evidence only; eval bytes/features are not refreshed or scored.

The only tracked-tree addition is **this report**. Diagnostic scripts,
measurement CSV/JSON and logs are in the new ignored audit directory; no
production source, pre-existing test, CLAUDE.md, invariant, old report, checkpoint
pin, dataset, historical run or model bundle is modified. The user's existing
`.gitignore` edit is preserved. All changes remain unstaged; Git writes are manual.

Limitations remain substantial: one measured speaker/corpus, twelve selected
training source groups, old training-OOF exposure, only three included pipelines,
unresolved legacy source/code/checkpoint eligibility, fixed short pooled features,
incomplete rate/codec/channel/unknown coverage and no achieved improvement.
Public license declarations do not establish every source right or scientific
admissibility. The proposed fifty-speaker cohort is still a small, conditional
English-read-speech pilot and does not prove upstream-training independence.
No deployment or publication readiness follows from this audit.

## 10. Reproduction

From the repository PowerShell environment:

```powershell
& '.venv/Scripts/python.exe' -B -X utf8 `
  runs/vocotrace-phase4d-audit-20261106/analyze_invariance.py

& '.venv/Scripts/python.exe' -B -m pytest -o addopts='' -q -ra `
  tests/test_vocotrace_abstention.py tests/test_source_verification.py `
  --basetemp runs/vocotrace-phase4d-audit-20261106/pytest-focused-1

& '.venv/Scripts/python.exe' -B -m pytest -o addopts='' -q -ra `
  --basetemp runs/vocotrace-phase4d-audit-20261106/pytest-full-1

& '.venv/Scripts/python.exe' -B -m ruff check src tests
& '.venv/Scripts/python.exe' -B -m ruff check .
& '.venv/Scripts/python.exe' -B -X utf8 `
  runs/vocotrace-phase4d-audit-20261106/verify_audit.py
git --no-optional-locks rev-parse HEAD
git --no-optional-locks status --short
git --no-optional-locks diff --cached --name-only
git --no-optional-locks diff --check
```

These are actual invocation names; use a **new** basetemp for another test run.
The no-fit analysis refuses to overwrite its existing sensitivity summary. For
replay in a new audit, copy that diagnostic script to a fresh audit directory;
its parent determines its output root. Input paths identify unchanged earlier
runs. The plan, full paired records, validation, before/after receipts and
source/working-tree identities distinguish newly measured training-only facts
from historical values and future proposals. No command above trains a new
research detector, publishes data or changes Git state.
