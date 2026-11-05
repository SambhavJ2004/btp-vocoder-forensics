# VocoPatch Phase 3D: acoustic signal and dataset failure investigation

2026-11-05. **Development-only, non-citable.** Starting manually created HEAD:
`79a4229f441befee85203f1b7cd87b440e1d90ac`. Phase 3A/B/C implementations and
artifacts are frozen inputs. The pre-existing `.gitignore` modification is
preserved. No new empirical neural training, model download, installation,
dataset edit, eval use, Kaggle operation or Git write is authorized here.

## Scope and fixed methods

The investigation separates physical paired differences, standalone classifier
ranking, nuisance associations and dataset validity. A measurable paired
difference does not establish a usable detector: these intrusive measurements
have the real reference, while a deployed detector does not. Conversely, weak
performance of these fixed representations does not establish acoustic
indistinguishability for every possible detector.

The complete plan is saved before new measurements in
`runs/vocopatch-phase3d-audit-20261105/predeclared.json`. The production driver is
`python -m experiments.phase3d`; pure numerical methods live in
`experiments.acoustic_diagnostics`. All new predictive/acoustic analyses use
frozen **training groups only**. The inherited input checker integrity-hashes
train/dev audio and parses shared metadata containers, but dev waveforms are
not decoded or used for any new choices. Eval WAV bytes and annotation records
are not consumed. No held-out split is relabeled as pristine; prior exposure
disclosed in earlier reports remains.

### Structural and acoustic alignment audits

Reuse existing manifest/tier/header/group/label validation, recorded digests and
the independent scalar splice/PCM16 reconstruction oracle for each training
output. This verifies actual donor sample mapping, untouched real samples,
internal crossfades, pure synthetic/real cores, transition masks, complete
duration and decoded null identity. Read originals and outputs through the
sanctioned mono 22,050-Hz PCM16 path. Hash every consumed original source before
and after analysis. Current common filtering applies the unchanged below-8-kHz
comparison-set transform to real, all vocoders and both controls together,
including its PCM16 round trip and band gate.

Separately rerun the existing INV-16 probe on all eligible training pairs per
vocoder. The 50-ms energy-envelope modal lag must be exactly zero; the existing
conditional waveform gate and all constants remain unchanged. Individual lags,
modal share, correlations, gate applicability and failures are retained. No
shift, alignment repair, time warping or replacement of recorded metadata is
performed. A nonzero per-file lag is distinct from failure of the existing
condition-level modal gate; a modal pass does not certify perfect phonetic
alignment for every utterance. Any gate failure blocks all new nuisance fits.
Descriptive failure evidence can still be recorded; it is not research admission.

### Activity, sampling and frozen score analysis

The original real reference defines an RMS activity proxy: maximum full-file
512-sample/128-hop RMS minus 30 dB, floored at two PCM16 steps. Three fixed bins
are below this threshold, threshold to ten times threshold (roughly -30 to
-10 dB relative to reference maximum), and at least ten times threshold. This
is not a phonetic VAD or a voiced/unvoiced ground truth.

Primary sample activity bins use the real reference at the **same positions**
for every condition, projected from complete 64-ms/16-ms RMS windows using
midpoint cells. Secondary bins use each actual condition's 64-ms RMS under the
same real-reference threshold. Label counts, raw margin distributions, FPR and
recall are retained by activity, condition and original source. The sampling
audit compares all available pure windows with the actual recorded CNN's
hash-seeded cap-24-per-record/class sampler; its 128-ms windows use their actual
RMS and support. This is a within-sampler comparison, not a claim that 64-ms and
128-ms activity distributions are interchangeable.

Use four explicitly separated frozen score sets: Phase 3B's A64 train OOF
predictions (the unchanged Phase 3A specification), Phase 3C train OOF scores,
the original Phase 3A final-model train scores, and scores of the frozen Phase
3C final model on its existing train cache. Final-model train scores are
**in-sample diagnostics**, not held-out performance. Both OOF sets retain the
same source folds. No neural optimizer is called by this investigation.

Scores are projected to samples independently of truth. All model comparisons
use the same conservative 5,249-sample effective support guard from Phase 3C,
including transitions, record edges, local context, smoothing and projection.
This numerical IIR-support convention is not a finite-support proof. Frozen
models retain their original train-OOF 5% thresholds; source-specific thresholds
are not fitted. Report source score distributions, FPR/recall, source variance,
correlations with RMS/activity and the ten largest FP/TP contributors alongside
their sample exposure. Associations do not identify a causal nuisance.

For an explicitly **ORACLE source-shift diagnostic**, subtract each source's
known null-control valid-real mean margin from all its scores and recompute
ranking. The known control and source identity make this unavailable to a
single unknown recording at inference. It is neither a new deployed detector
nor a reported threshold-performance improvement.

Hard negatives are matched-time real target cores, edited-real cores, full
null-control real, disjoint real donor/away cores, and valid null real at least
one second from both raw edit boundaries. Core means use the stricter guard and
its closed-boundary convention: `[S+F+5249+1, E-F-5249)`, 11,111 samples;
missing far-real support is reported rather than manufactured. Report pooled
core-mean AUROC and within-source paired win probability/contrast separately:
pooled ranking includes between-source differences. Speech-active same-position
paired acoustic comparisons below use the real-derived frame mask for both
members, avoiding class-specific diagnostic positions.

### Paired acoustics and full-band membership

Use the generator's exact pure-core interval with its 3,308-sample effective
filter guard on both sides: 14,994 samples for the fixed recipe. Compare real
and each original vocoder at identical coordinates without lag compensation.
Non-padded complete 64-ms periodic-Hann frames have a 16-ms hop and 2,048-point
one-sided PSD. A real-reference active-frame mask selects the same frames in
both conditions. PSD has a 1e-12 floor; LSD is RMS `10*log10(PSD_syn/PSD_real)`.
These original full-source waveforms have no splice at the selected coordinates;
their acoustic interval is distinct from the 11,111-sample valid-mask interval
used to average predictions on edited recordings. The inherited acoustic guard
keeps comparability with Phase 3C, while prediction averages obey the stricter
closed label-mask bounds.
Also report log-shape distance after removing each frame's constant log offset,
band energy differences, RMS ratio, zero-lag waveform correlation and relative
waveform error. Exact active-frame sample agreement is a diagnostic of copies,
not proof of upstream model identity.

The spectral envelope is a real-cepstral lifter of log10 PSD retaining q=0..22
and its symmetric negative-quefrency coefficients (about 1 ms). Periodicity is
the maximum overlap-normalized autocorrelation in 50–500-Hz lag ranges on
mean-removed Hann frames. Together with below-band spectral flatness it supplies
a limited harmonic/noise **proxy**. It is not validated HNR, F0, phonetic voicing
or perceptual quality. Silent/undefined measurements remain missing, with counts.

The primary comparison uses the validated common below-8-kHz PCM16 analysis.
A separately labeled archive-tier full-band diagnostic uses unchanged source
WAVs. **INV-18 membership remains binding:** BigVGAN may receive a full-band
distance; Griffin-Lim and Vocos retain their measured high-band exclusions and
receive no full/high-band reconstruction distance. Their raw bandwidth/energy
profiles are processing-confound diagnostics only. High-frequency silence or
resampling edges must not be interpreted as generalizable synthesis evidence.
No full-band detector is fitted and no high-band exclusion is waived.

### Four limited nuisance ablations

Exactly four fixed C=1 L2 logistic recipes reuse the frozen A64 acoustic feature
geometry and three train-source folds. This is a diagnostic factorial comparison,
not architecture or hyperparameter search:

1. **balanced:** at most 72 pure windows per source and binary class, sampled
   independently of source performance.
2. **activity_matched:** within each source and each of the three own-RMS bins,
   select equal real/synthetic counts, capped at 24 each per bin.
3. **rms_normalized:** use exactly the balanced indices, with each 64-ms waveform
   frame scaled in RAM to RMS 0.03 before the unchanged feature extractor.
4. **activity_rms:** use the activity-matched indices and the same RMS control.

Normalization is waveform-only and shared by classes. Frames at/below two LSB
RMS stay unchanged and are counted, rather than amplifying numerical silence.
Record gains and peaks; do not clip or rewrite WAVs. Batching independent
normalized frames through the sanctioned feature extractor is regression-tested
against individual frame calls. Fixed 4.0375-second context/padding conventions
are preserved; every supervised local frame contains original samples only.

Each fold fits its scaler only on sampled fit-source windows and weights BCE
logistic loss equally by source and class, `N/(2*G*n_source_class)`. Class/activity
matching can change the number of fitted windows; these are not equal-budget
architecture comparisons. Activity/bin/source metadata selects training samples
only and never enters the 31-dimensional classifier. Held-out predictions cover
all windows, with no oracle activity mask. Report the fixed 0.5 point and the
train-OOF 5% sample-FPR cutoff separately constrained on each control category.
No dev/eval threshold or winner is selected. OOF threshold estimates are
development selections, not an unbiased final performance estimate.

### Source-level uncertainty and dataset selection

Aggregate first to one statistic per original utterance, preserving its paired
conditions. Use 2,000 fixed-seed percentile bootstrap resamples of source
statistics for 95% intervals on source means. No windows, files or overlapping
crops are independent resampling units. Intervals are conditional on the frozen
predictions/selected cutoff and observed source subset; they omit training,
hyperparameter, speaker and corpus uncertainty. Undefined ratios/measurements
are reported missing. Fewer than ten supporting sources gives no interval.
These exploratory intervals do not replace the unresolved research-level paired
uncertainty work or establish scientific admission.

The construction analysis compares the original 140 **train** real utterances:
123 admitted and 17 excluded, retaining the exact existing exclusion reasons.
Only excluded real-training WAVs are additionally decoded; no excluded or
held-out synthetic condition is fabricated. Duration/RMS/activity differences
describe selection, not randomized causal effects. The original frozen
140/20/40 assignments and eligible subset remain unchanged.

## Findings and research decision

**Recommendation B: redesign the training comparison around matched-content
hard negatives before increasing model capacity.** This specifically addresses
target/background content mismatch; it is not a claim that an RMS/activity-only
correction solved the task. Current evidence shows weak matched-content
discrimination in the tested representations, especially for neural vocoders,
and a much larger contrast against unmatched real background. It does not
establish that synthesis leaves no acoustic signal. No additional neural model
is trained or downloaded in Phase 3D.

The evidence does not support a gross label/splice/encoding error or a simple
constant per-utterance score shift as the primary explanation. Activity affects
scores, but the limited matched-activity/gain ablations do not improve the
reference. Paired acoustics differ, while an unknown recording lacks the
matched-time real reference that makes those differences observable. The
remaining ambiguity is useful standalone signal versus representation and
content-conditioned training. The next experiment should remove the documented
content mismatch from training comparisons, with control FPR central to its
decision. This identifies the best-supported next control, not a proven single
causal explanation for every error.

### Source and construction validity

All 615 training outputs reproduce their declared PCM16 splice exactly, including
unchanged outside samples, donor cores, fade indexing and null identity. All 509
consumed original training-file hashes match: 492 admitted real/vocoder files
and 17 additional excluded-real files. Frozen source grouping, archive tier,
22,050-Hz mono PCM16 encoding, sample lengths and labels agree. Current common
filtering follows the existing sanctioned transform and gates. This does not
authenticate historical upstream checkpoints/generation bytes or certify all
historical processing; those provenance limitations remain.

| Current INV-16 training probe | Envelope mode (samples) | Exact-zero share | Mean envelope correlation | Mean absolute waveform correlation | Conditional waveform gate |
|---|---:|---:|---:|---:|---|
| BigVGAN | 0 | 64/123, 52.03% | 0.98927 | 0.65555 | Not applied: modal share 37.40% |
| Griffin-Lim | 0 | 38/123, 30.89% | 0.98018 | 0.18645 | Not applied |
| Vocos | 0 | 23/123, 18.70% | 0.99499 | 0.21435 | Not applied |

All three existing condition gates pass unchanged. Median / p95 / maximum
absolute envelope lags are 0 / 7.8 / 22 samples for BigVGAN, 3 / 14 / 36 for
Griffin-Lim and 7 / 26.8 / 82 for Vocos. The corresponding waveform modal lags
are +1, -93 and -55 samples; their modal shares do not satisfy the existing
conditional gate (and the latter two lack the required mean phase correlation).
Do not reinterpret a modal pass as exact per-file alignment, or these scattered
phase-sensitive lags as a validated fixed delay. No correction is applied.

The admitted/excluded train cohorts have median duration **7.143 / 2.185 s**,
p05–p95 **3.830–9.540 / 1.344–3.426 s**. Median below-band RMS is
0.04339 / 0.04434 and median active fraction 0.8269 / 0.8865. The 17 exclusions
remain 11 donor-unavailable, 3 too-short and 3 without an active target. Thus
construction strongly selects longer utterances; it does not simply select
sources with the highest activity fraction. Neither this selection nor the
fixed one-second/fade recipe represents arbitrary partial-audio attacks.

### Acoustic differences by vocoder

These are one paired-core statistic per source, using the same real-active
frames. Parentheses give 95% conditional source-bootstrap intervals for the
**mean**; they are not uncertainty bounds on individual windows or final research
generalization.

| Below-8-kHz quantity, source mean | BigVGAN | Griffin-Lim | Vocos |
|---|---:|---:|---:|
| Log-spectral distance, dB | 5.8748 (5.8068–5.9472) | 6.2670 (6.1888–6.3514) | 6.2892 (6.2287–6.3522) |
| Cepstral-envelope distance, dB | 0.9167 (0.8872–0.9517) | 1.4295 (1.3827–1.4843) | 1.0014 (0.9720–1.0383) |
| RMS difference, dB | +0.1310 (+0.1007–+0.1623) | +0.1043 (+0.0809–+0.1270) | +0.0097 (-0.0133–+0.0328) |
| Periodicity-proxy difference | +0.00056 (-0.00080–+0.00186) | -0.03942 (-0.04218–-0.03682) | -0.00548 (-0.00721–-0.00367) |

Median zero-lag waveform correlations are 0.7132 / 0.0061 / -0.0008; relative
waveform RMS errors are 0.7649 / 1.4190 / 1.4159. These are not near-identical
waveforms, especially when phase is reconstructed. Yet broad envelopes and RMS
remain close, and periodicity changes are small for BigVGAN/Vocos. Phase-related
intrusive errors do not establish standalone detectability, perceptual quality,
or a quality-versus-detectability correlation. Complete bandwise energy and
across-utterance distributions are retained in the acoustic tables.

Exploratory archive full-band mean LSD is **6.7947 dB for BigVGAN**. Griffin-Lim
and Vocos full/high-band distances remain **excluded**, with the existing
measurement-backed reasons. Median high-band energy fractions of matched real /
BigVGAN / Griffin-Lim / Vocos cores are **0.005083 / 0.006098 / 1.90e-8 /
0.005492**. Griffin-Lim is at the PCM16 noise-floor scale; common-filtered
fractions are roughly 1.35e-8–1.96e-8 for all conditions. These are descriptive
PSD profiles, not new forensic high-band scores. No new matched resampler control
or historical resampler authentication is performed, so specific attribution of
the current Vocos profile to an upstream implementation remains unresolved.

### Activity and training exposure

Both frozen OOF systems use exactly the same valid-sample activity partition:

| Reference activity | Valid samples | Synthetic samples | Sources with real / synthetic support |
|---|---:|---:|---|
| Low energy | 11,101,925 | 207,840 | 122 / 63 |
| Active quiet | 43,753,755 | 2,691,690 | 123 / 123 |
| Active loud | 21,790,670 | 1,200,429 | 123 / 109 |

At the frozen 5% cutoffs, source-mean real FPR percentages (95% conditional CI)
are A64 **0.0508 (0.0150–0.0955)** low, **3.7409 (3.1905–4.3336)** quiet,
**8.7122 (7.5208–10.0821)** loud; CNN **0.4910 (0.2552–0.7693)**,
**5.6369 (4.7878–6.4913)**, **5.4222 (4.3792–6.5759)**. Real speech activity,
not just near-silence, is a major location of false alarms.

Source-mean synthetic recall percentages are A64 **0.0 (0–0)** low,
**5.3565 (3.8276–7.1245)** quiet, **11.0918 (8.2497–14.4674)** loud; CNN
**1.7188 (0–4.1013)**, **5.2724 (3.6733–7.0179)**,
**4.4520 (2.1009–7.4387)**. Missing-bin support is not assigned a false zero.
These source means differ from sample-pooled rates because source durations
and activity support differ.

Available A64 pure windows: 215,303 real and 14,187 synthetic. Low-energy windows
are 31,788 real (14.76%) and 738 synthetic (5.20%). The CNN's 128-ms available
windows have 20,386/209,639 low-energy real (9.72%) and 147/12,711 synthetic
(1.16%). Its actual cap sampler selects 1,393/14,760 (9.44%) and 99/8,856
(1.12%), respectively: the cap does not substantially change these proportions.
Positive/background energy composition differs, but the sampler is not shown
to manufacture an additional large energy bias. The two frame lengths must
not be compared as if they used the same activity measurement.

A separate coordinate-count audit calls the existing tested sampler on position
tokens, with no audio decoding or fitting. A64 has **4,729 / 215,303** real
negative windows at eligible same-time synthetic-target coordinates (2.196%).
The CNN selects **327 / 14,760** such negatives (2.215%, from 4,237 available
matched positions), alongside 8,856 selected synthetic examples. These count
eligible positions, not guaranteed one-to-one selected pairs. Thus even though
the cap preserves broad energy proportions, matched-content negatives receive
little exposure. `paired_negative_exposure.json` records all source counts,
cache identities and the audit script digest. This directly motivates the paired
training control; it does not prove that reweighting will recover accuracy.

### Source variability and negative controls

On the common sample mask, raw A64 OOF AUROC is **0.589817** and the ORACLE
null-source-centered value is **0.589916**. CNN gives **0.564611 → 0.560897**.
Subtracting a known per-utterance negative mean does not rescue ranking.
Between-source negative-mean variance versus mean within-source real-margin
variance is **0.01270 vs 0.24244** for A64 and **0.02384 vs 0.30164** for CNN.
These are descriptive logit variances, not an estimated random-effects ICC.

Negative source mean scores correlate with real activity fraction (Spearman
rho **0.548 / 0.573**) much more than with real RMS (**0.110 / 0.108**).
Source FPR versus activity is only **0.056 / 0.253**, versus RMS **0.041 /
-0.018**. Correlations do not identify causes; fold-specific means remain in
the saved tables because OOF heads can also have different score scales.

The ten largest FP sources contribute **22.44% / 25.34%** of A64/CNN FPs,
with **11.01% / 10.17%** of real sample exposure. The ten largest TP sources
contribute **34.30% / 47.39%** of TPs. Performance varies, but pooled false
alarms are not solely one exceptional utterance. Source F1 p05/median/p95 is
**0 / 0.03349 / 0.29907** for A64 and **0 / 0 / 0.20045** for CNN. Source-mean
F1 is **0.07948 (0.06183–0.09836)** and **0.04528 (0.03290–0.05838)**.

Matched-content real is the hardest negative. Full real backgrounds and far
real can make discrimination look easier because speech content/activity also
change. Pooled core-mean AUROC and paired within-source wins must be separated:
nearly consistent small paired score increments can coexist with weak pooled
ranking against a large natural-speech baseline. Full per-vocoder negative
tables are saved, including all 123 sources with far-real support. Neither null
identity nor real-to-real controls eliminate every splice confound.

| OOF system / vocoder | Matched real AUROC | Edited-real core | Away real core | Far real mean | Full null mean | Matched within-source win probability (95% CI) |
|---|---:|---:|---:|---:|---:|---|
| A64 / BigVGAN | 0.538965 | 0.597197 | 0.598718 | 0.710820 | 0.734946 | 0.6911 (0.6016–0.7724) |
| A64 / Griffin-Lim | 0.667526 | 0.724040 | 0.723709 | 0.824245 | 0.847049 | 0.9756 (0.9431–1.0000) |
| A64 / Vocos | 0.516227 | 0.575980 | 0.576310 | 0.677110 | 0.699055 | 0.6098 (0.5203–0.6992) |
| CNN / BigVGAN | 0.503933 | 0.515566 | 0.515434 | 0.710886 | 0.745786 | 0.5366 (0.4472–0.6260) |
| CNN / Griffin-Lim | 0.511402 | 0.524291 | 0.523366 | 0.715183 | 0.750413 | 0.6585 (0.5772–0.7398) |
| CNN / Vocos | 0.503536 | 0.514707 | 0.515698 | 0.711415 | 0.747835 | 0.5691 (0.4878–0.6585) |

These are ORACLE diagnostic comparisons, not localization accuracy. All use
123 paired sources. Target, edited and away cores have equal duration; far/full
means average different amounts of audio. Their different content, activity,
selection and averaging variance prevent a clean causal interpretation of the
AUROC difference alone. The paired win requires a known real counterpart and
known target, neither available to deployed inference.

The **mean margin** decomposition is more direct. A64 synthetic-versus-other-real
contrasts in mixed recordings are 0.1423/0.2322/0.1249 for BigVGAN/Griffin-Lim/
Vocos, but matched-time contrasts are only 0.02759/0.11749/0.01019. CNN values
are **0.1625/0.1667/0.1635 versus 0.001387/0.005581/0.002396**. CNN matched
contrast source-mean 95% intervals are [0.000069,0.002730],
[0.003109,0.007967] and [0.000730,0.004135]. Most of its target/background
mean contrast is already present in real speech at the target. This describes
a comparison mismatch; it is not a percentage attribution of errors to a
proven causal confound. Coarse RMS-bin matching leaves phonetic content unmatched.

The final CNN's in-sample source F1 at the OOF cutoff is only **0.01817**,
with lower score variance than its OOF mixture. The top ten sources contain
91.41% of its TPs. This is a score-transport/stopping/calibration warning, not
held-out performance. Final-model train diagnostics are kept separate from
OOF results and are not used to tune a replacement threshold.

### Limited nuisance results

All rows below use train-group OOF scores. Fixed 0.5 results remain in the
artifacts; at that point all variants still have roughly 43–47% control FPR.
At the train-OOF-selected 5% point:

| Recipe | Fitted windows | Window AUROC | EER | Source F1 (95% conditional CI) | Sample IoU | Synthetic recall |
|---|---:|---:|---:|---|---:|---:|
| Balanced | 17,712 | 0.589615 | 0.440746 | 0.07848 (0.06136–0.09653) | 0.04097 | 7.6045% |
| Activity matched | 12,128 | 0.572432 | 0.447342 | 0.06820 (0.05361–0.08482) | 0.03713 | 6.9622% |
| RMS normalized | 17,712 | 0.579608 | 0.444421 | 0.06900 (0.05264–0.08659) | 0.03712 | 6.9540% |
| Activity + RMS | 12,128 | 0.565674 | 0.452245 | 0.06629 (0.05233–0.08163) | 0.03581 | 6.7297% |

Both control categories meet the declared OOF 5% sample cap in every recipe
(about 4.78–5.00%). None improves the frozen A64 source F1 0.07948 on the same
mask. Intervals overlap; this is no proof that all nuisance confounds are absent.
It is evidence against these simple, fixed matching/normalization corrections
being sufficient. No best ablation is deployed and no dev/eval selection occurs.

### Options and next experiment

| Option | Supporting evidence / expected benefit | Remaining risks / decision |
|---|---|---|
| A — one frozen pretrained representation | Paired acoustic changes and small matched-content score increments exist; the tiny CNN largely loses even Griffin-Lim separation. A richer waveform/time-context representation could test representation limits. | Worth a later bounded test if the paired-content control fails. More capacity can retain the same content shortcut; pretrained encoders may also discard forensic detail. Reproduce its reference benchmark and verify model/data provenance first. |
| **B — paired-content hard-negative protocol** | Mean target/background score contrast is much larger than the same-time vocoder increment, especially for the CNN. Coarse RMS/activity matching does not match phonetic content. Source offsets do not explain the gap. Pairing real and synthetic training windows at identical times removes this comparison mismatch. | **Recommended next.** No recovery is guaranteed: the representation may still be inadequate or useful signal intrinsically weak. Preserve ordinary backgrounds, both controls, strict masks and source-group validation, and measure whole-recording false alarms. |
| C — narrow to attribution/another task | Vocoder-specific acoustic differences make attribution a possible separate study. | This phase provides no new train-group attribution benchmark; intrusive differences or excluded high-band processing cues are insufficient evidence for switching the objective. |

Proposed single next experiment: keep the A64 representation and fixed C=1 head,
and compare the frozen reference with a predeclared same-time paired sampler.
For each synthetic training window, include its original-real null window at
the identical coordinates; retain fixed quotas of ordinary background and
real-to-real negatives. Keep the existing archive/below-band processing,
source folds, strict support masks, source/class weighting and a fixed budget.
Use grouped train-only validation and control-aware threshold/calibration
selection. Report source-paired uncertainty, matched-content discrimination,
whole-recording low-FPR recall and recording-level FA. This tests training
content mismatch directly; it does not introduce difference-to-reference input
features. **Inference remains waveform-only** with no real counterpart, source
ID or oracle timestamps. If that control remains weak, a separately bounded
frozen-representation probe is justified. Resolve authenticity/alignment/admission
limitations before research interpretation. Eval stays isolated; no new
architecture sweep, runtime or model download is authorized by this report.

## Reproduction

Run from repository PowerShell. The paths below are the actual run names;
choose fresh output/basetemp names for repeats. Existing runs are refused.

```powershell
& '.venv/Scripts/python.exe' -B -X utf8 -m experiments.phase3d --dataset-dir runs/btpvf-partial-v1-20261104 --source-manifest runs/onboarding/v0-kaggle-input/manifest.csv --baseline-dir runs/vocopatch-phase3a-baseline-20261104 --ablation-dir runs/vocopatch-phase3b-ablation-v2-20261104 --cnn-dir runs/vocopatch-phase3c-cnn-20261105 --out-dir runs/vocopatch-phase3d-diagnostics-v2-20261105
& '.venv/Scripts/python.exe' -B -m pytest -o addopts='' -q -ra tests/test_phase3d_diagnostics.py --basetemp runs/vocopatch-phase3d-audit-20261105/pytest-focused-3
& '.venv/Scripts/python.exe' -B -m pytest -o addopts='' -q -ra --basetemp runs/vocopatch-phase3d-audit-20261105/pytest-full-2
& '.venv/Scripts/python.exe' -B -m ruff check src tests
& '.venv/Scripts/python.exe' -B -m ruff check .
& '.venv/Scripts/python.exe' -B -X utf8 runs/vocopatch-phase3d-audit-20261105/verify.py
& '.venv/Scripts/python.exe' -B -X utf8 runs/vocopatch-phase3d-audit-20261105/paired_exposure.py
```

Outputs retain physical/score/activity/source tables, sampling counts, alignment
and scalar reconstruction records, input/code/environment digests, numerical
fold models and OOF outputs where admitted, group intervals, configuration and
complete SHA-256 inventory. Failure preserves only its fresh staging directory
and receipt. Old datasets, models and run artifacts are never overwritten.

## Correctness correction, integrity and checks

One defect was found in the **new diagnostic**, not in existing audio, generator
labels or saved predictions. The initial hard-core mean began at `S+F+5249`.
The existing closed transition dilation marks that first sample ignored;
independent reproduction gives labels `[-1,1]` at the old first two positions.
The corrected first index is `S+F+5249+1`, with an explicit pure-label assertion
for every condition. Core length is 11,111 samples. A regression independently
checks both mask boundaries and every retained label. No guard or gate is relaxed.

The initial completed `runs/vocopatch-phase3d-diagnostics-20261105/` is preserved
as superseded development evidence. The source snapshot and reproduction receipt
are in `core_boundary_fix.json` / `phase3d_before_core_boundary_fix.py` under
the audit directory. **Use the corrected `...-diagnostics-v2-20261105/` run.**
It repeats the complete cold analysis rather than modifying the first run.
Its physical tables and sampling are byte-identical, all twelve numeric linear
model files are identical, and localization/activity counts are unchanged.
Only the diagnostic core averaging and its dependent summaries change.

The independent final verification passes:

- Complete corrected inventory and recorded source digests match.
- Every one of the 1,476 synthetic diagnostic core means agrees with the mean
  of its independently valid synthetic samples; no ignored sample enters it.
- Separate projection/count calculations reproduce frozen A64/CNN OOF and
  A64 final-train results. Weighted sklearn AUROC/EER, oracle-centered ranking
  and source-bootstrap intervals agree independently.
- Original source splits and saved fold memberships remain fixed. No dev
  waveform is decoded, no eval WAV bytes or labels are used, and no neural
  optimizer is called by the new experiment.
- All **4,008 protected prior-phase file paths, sizes and mtimes** match the
  before-state snapshot, and complete non-receipt prior model/run artifacts
  retain their digests. All 509 consumed original-training source hashes match.
  The inherited checker confirms current train/dev WAV hashes; eval and demo
  WAV bytes are intentionally not rehashed. All 870 generated WAV paths,
  sizes and mtimes remain unchanged, with byte verification explicitly scoped
  to consumed audio. The first Phase 3D run remains sealed and unchanged too.

Focused pytest: **16 passed**, 86.07 s. Full suite after the correction:
**884 passed, 4 failed, 7 skipped, 1 xfailed**, 1,246.33 s. The failures are the
same two generated-notebook CRLF/LF comparisons and two Windows junction
`WinError 5` cases. Link-privilege skips and the existing unimplemented-ablation
expected failure remain. Source/test Ruff passes; repository-wide Ruff retains
11 generated-notebook findings. No existing test is changed or weakened.

The corrected CPU driver took **1,405.83 s**, with one BLAS thread and tests
sharing the machine. The superseded first pass took 1,376.43 s; both costs are
reported rather than hidden. Each analysis fits exactly twelve lightweight
logistic heads, with no new empirical neural run. Existing optimizer regression
tests use synthetic fixtures as part of pytest. The corrected output contains
**53 files / 45,260,889 bytes**, with no generated WAV dataset or model download.
These wall times include integrity/scoring/serialization and are not isolated
throughput benchmarks.

Changed repository files:

- `src/experiments/acoustic_diagnostics.py`: tested physical/statistical helpers.
- `src/experiments/phase3d.py`: fresh-output training-only audit/analysis driver.
- `tests/test_phase3d_diagnostics.py`: CPU numerical, mask, grouping, deterministic
  sampling, source-immutability, decoder-trap and gate-blocking coverage.
- This report and the README pointer.

Supplementary receipts, logs and independent audit/count scripts live only in
the new ignored audit directory. HEAD remains
`79a4229f441befee85203f1b7cd87b440e1d90ac`, the index is empty, and all changes
are unstaged. The pre-existing `.gitignore` bytes, `CLAUDE.md`, scientific gates,
source datasets, earlier pilots and frozen model artifacts are unchanged.
Nothing is uploaded. Single speaker, legacy provenance, prior held-out exposure,
fixed edit/fade construction, limited representations and conditional uncertainty
remain scientific limitations. This report supports the next controlled
development experiment, not publication or deployment.
