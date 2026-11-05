# VocoPatch Phase 3C: paired-signal and oracle-core diagnostics

2026-11-05. **Exploratory development only, non-citable.** Starting manually
created HEAD: `fcd8f5cb747533471dca9e824d4a12e4ed0339a5`. The dated output paths
retain their original `20261104` identifiers. Code digests identify the actual
working-tree implementation. No invariant, source dataset, split, historical
experiment or scientific admission gate is changed. No eval audio is opened.

**Outcome:** the controlled oracle narrowly justified testing one small learned
model, but the CNN did not improve the frozen baseline on train-group OOF or
the exposed dev benchmark. At the train-selected 5% point its dev synthetic
sample recall is **0.994%**, source-macro F1 **0.01247** and null/edited-real
sample FPR **0.797% / 0.706%**. Retain Phase 3A as the reference; this CNN is an
experimental, weak localizer, not a deployable detection result.

## Fixed design and interpretation

`python -m experiments.paired_signal` adds a separate training-only diagnostic
driver. It reuses the Phase 3A identity receipt and strict Phase 3B input
validation, frozen source grouping, sanctioned audio I/O, identical whole-file
below-8-kHz filtering/PCM16 round trip, 64-ms/16-ms acoustic extractor, corrected
directional EER and deterministic three-fold source assignments. Existing code
and saved models are read-only. Dev WAVs are integrity-hashed by the inherited
input checker, but are not decoded or used to fit/select these diagnostics.
Eval WAV bytes and eval annotation records are not consumed; shared metadata
containers are parsed before train/dev filtering as in Phase 3B.

For every training source, compare six roles: original real at the edit target,
original real at the disjoint donor position, the actual real-to-real edited
target, and the three original full-vocoder outputs at the identical target
coordinates. The original null is checked for exact waveform identity and is
represented once by the real target; duplicating it would reweight identical
negatives. The real-to-real role contains only real speech but relocates content.

For raw edit `[S,E)` with internal fades `F`, the diagnostic target is
`[S+F+3308, E-F-3308)`. The matched away interval starts at the real-to-real
donor start plus the same offset from `S` and has identical length. It is
disjoint from the raw target, not an independently sampled distant utterance.
These coordinates are checked against actual generator pure-core annotations.
For the fixed 1-second/220-sample-fade recipe the diagnostic core is 14,994
samples (0.6800 seconds). Ground-truth edit intervals, oracle intervals and
predicted intervals are separate concepts; this experiment produces no
localization intervals.

The 3,308-sample guard is the inherited numerical effective IIR guard, not exact
finite filter support. Actual partial-vs-full-donor residuals are measured
separately for all vocoders and the relocated real donor. Full-vocoder cores
allow investigation of resynthesis independently of most splice influence;
their equivalence to a filtered edited core is measured rather than presumed.

Per-role tables retain physical RMS, sample peak, 512-sample/128-hop RMS activity
distributions, 24 mean PSD bands, eight 1-kHz PSD summaries, in-band energy and
31-feature mean/std distributions. Paired differences are within source and
time for vocoders; away/control comparisons also change speech content.
Near-edge tables describe both sides of each raw onset/offset in 0–50,
50–150, 150–250 and 250–400-ms bins. These tables are acoustic residuals,
not detector error rates or evidence of boundary-confound immunity.

## Oracle classification and nuisance controls

Each core becomes one 62-dimensional observation: mean and population standard
deviation of the existing 31 features over its 64-ms frames. All roles share
source-group-disjoint train folds. Each fold fits its scaler and C=1,
class-balanced L2 logistic classifier using only that fold's fit groups.
Higher margins mean synthetic. The corrected shared EER receives the opposite
score sign under its higher-is-real convention. Sigmoid scores are uncalibrated.

Four fixed variants use identical eligible groups:

| Variant | RMS gain control | Speech-activity matching |
|---|---|---|
| raw | None | All core frames |
| rms | RMS 0.03 over selected frame support | All core frames |
| active | None | Shared active frames |
| rms_active | RMS 0.03 over selected frame support | Shared active frames |

Activity is an RMS proxy, not a phonetic VAD: the threshold is 30 dB below the
maximum 512-sample frame RMS of the full filtered real source, with a two-LSB
floor. Target frame positions are the intersection of activity in real and all
three vocoders; away/control positions intersect their two activity masks.
Masks are fixed before gain control. Require at least eight common frames for
each pair set, otherwise exclude the entire source from all four oracle
variants with measured evidence. RMS normalization uses the union of selected
64-ms frame supports, applies a single floating-point gain in RAM, records
gains/peaks, and does not clip, rewrite WAVs or change archive loudness.

This selection uses known intervals and other conditions' activity; it is
explicitly ORACLE and potentially selects easier/more active cores. The model
receives only acoustic vectors, not source IDs, names, timing or activity-count
metadata. Original target and away/control have different speech content;
matched-time real-versus-vocoder AUROC is therefore reported separately as well
as pooling all three negative types. Away and edited real cores are correlated
within a source, not independent replicates.

Before real measurements, `predeclared.json` fixed the conditional learned-model
decision: the **rms_active train OOF** pooled AUROC must be at least 0.60, with
synthetic-core recall at least 0.10 at a threshold whose OOF FPR is at most 0.05
separately for real-target, real-away and edited-real cores. Ties move together;
the most restrictive negative-kind cutoff is used. This is a development
decision about proceeding, **not a new scientific gate**. Its in-sample OOF
threshold is a train development selection, not an unbiased final performance
estimate. Core FPR cannot be substituted for full-recording sample FPR.

The conditional candidate was one compact 24-log-mel, 128-ms/hop-16-ms,
eight-channel 3x3 CNN with frequency-pair pooling, Adam 0.001, batch 256,
L2 0.0001, dropout 0.10, maximum 20 epochs and patience four. Train-only inner
group validation would choose epochs inside each outer fit fold, followed by
outer refits and a final median-epoch fit. No architecture search, installation
or model download was planned. The result section records whether this
condition was met; a failed fixed diagnostic does not prove that every possible
learned representation lacks signal.

## Observed training-only paired signal

All **123 training source groups** passed the fixed oracle activity/RMS rules;
there were zero new oracle exclusions. Three folds each hold out 41 sources.
There are 738 core observations per variant (369 synthetic, 369 real), not
738 independent utterances. The 492 original training source files match their
catalog hashes. All 492 measured edited-vs-original donor cores are decoded
PCM16-identical after the common analysis/guard, including the relocated real
core; maximum difference is zero in these measured cores. This empirical fact
does not give the IIR finite support or certify other fade/filter recipes.

| Train-group OOF oracle | Pooled AUROC | Directional EER | Recall at each-negative 5% cap | Source-macro core F1 |
|---|---:|---:|---:|---:|
| raw | 0.623930 | 0.401084 | 0.081301 | 0.087418 |
| rms | 0.615734 | 0.401084 | 0.067751 | 0.069532 |
| active | 0.623482 | 0.411924 | 0.116531 | 0.135734 |
| rms_active | 0.618290 | 0.417344 | 0.105691 | 0.124623 |

The last row meets the predeclared proceed criterion **narrowly**, allowing one
compact CNN experiment. Its uncalibrated cutoff is 0.8558567534507251. Real-target
core FPR is 6/123 (4.878%); real-away and edited-real FPR are each 4/123 (3.252%).
Recall is 39/369. Do not interpret these core rates as localization rates.

| rms_active oracle vocoder | AUROC vs all three real roles | AUROC vs same-time real only | Recall at pooled cutoff |
|---|---:|---:|---:|
| BigVGAN | 0.581444 | 0.536453 | 0.065041 |
| Griffin-Lim | 0.709564 | 0.675326 | 0.203252 |
| Vocos | 0.563862 | 0.518210 | 0.048780 |

Thus there is modest pooled signal, dominated by Griffin-Lim; same-time neural
vocoder separation is weak. Adding away-time negatives improves ranking, so
content/target selection contributes. Source-specific six-core AUROC spans
0–1, with p05 0.2222, median 0.8889, p95 1 and mean 0.6865. These tiny per-source
rankings are coarse and differ from pooled AUROC because pooled comparisons also
cross sources and fold score scales. They are not confidence bounds.

Median paired vocoder-vs-real RMS differences are +0.1403 dB BigVGAN, +0.1152 dB
Griffin-Lim and +0.0253 dB Vocos; away/control real is +0.5559 dB versus target
real. Corresponding median in-band energy differences are +0.1409/+0.1196/
+0.0332 dB and +0.4728 dB. Median activity-fraction difference is zero for every
role, but the complete source distributions are retained. RMS control slightly
lowers AUROC; activity matching improves low-FPR recall more than ranking.
These small ablations cannot uniquely attribute the residual to gain/activity.
Spectral shape, content, alignment and vocoder/front-end processing remain
possible causes.

The paired RMS p05–p95 range is -0.118 to +0.382 dB for BigVGAN, -0.104 to
+0.303 dB for Griffin-Lim and -0.228 to +0.186 dB for Vocos. Away/control real
spans -4.310 to +5.015 dB, with activity-fraction changes -0.1395 to +0.1658;
vocoder activity differences have much narrower source distributions. This
supports treating the away-time comparison as a content-changing control,
not an acoustically interchangeable copy of target-time real.

Selected median paired 1-kHz PSD-summary differences, in dB (all eight bands
and 31-feature source distributions are saved in the diagnostic/audit tables):

| Condition vs target real | 0–1 kHz | 3–4 kHz | 6–7 kHz | 7–8 kHz |
|---|---:|---:|---:|---:|
| BigVGAN | +0.226 | +0.116 | -0.451 | -0.261 |
| Griffin-Lim | +0.204 | -0.579 | -0.498 | -0.519 |
| Vocos | +0.028 | -0.029 | +0.126 | +0.049 |
| Away / relocated real | +0.076 | -1.129 | -0.194 | +0.290 |

These are means of three adjacent fixed linear-PSD bands followed by a paired
dB ratio, not new full-band analyses or causal feature importances. Sub-band
differences remain after suppressing known above-8-kHz shortcuts.

The edited-real and away-real features are identical in measured guarded cores,
as expected from their exact waveform match. Both are useful explicit negatives
but do not provide independent acoustic replications. Their pooled cutoff FPR is
lower than same-time real, rather than revealing a special edited-core cue here.
This says nothing about seams excluded from these cores; the boundary tables and
full-recording control errors are separate required diagnostics.

## Compact learned-model specification

The fixed 128-ms input contains seven 32-ms periodic-Hann PSD subframes with a
16-ms hop and a final anchored subframe, 1,024-point FFT, the existing 24 HTK
triangles below 8 kHz, and log10 floor 1e-12. Store patches as float32; train in
float64. A valid 3x3 convolution has eight channels, ReLU, mean pooling of
adjacent frequency pairs, flattening of 11x5x8 activations, training-only 10%
dropout and one linear logit. Total trainable parameters: **521**, plus 24
training-fitted per-band means and scales. Gradient checks cover every trainable
parameter, including fixed-mask dropout and L2 terms.

Use all 123 frozen training groups, not an oracle-selected subset. Only windows
entirely within one class after the inherited 3,308-sample filter guard can fit
the model. Up to 24 windows per recording and binary label are selected by a
fixed source/example-specific seed. Null and real-to-real recordings and real
backgrounds in every mixed recording supply negatives. This cap samples the
whole edited-real recording; it does not oversample its transplanted core.
Every source/class receives equal total BCE weight: `N/(2*G*n_source_class)`.
Train-only sampled patches fit per-band mean/std across observations and time;
no per-record gain/activity correction or oracle cropping is fed to the CNN.

Reuse the three outer folds. Inside each 82-source fit partition, a fixed hash
order holds out `ceil(0.2*82)=17` sources and fits 65. Adam trains for at most
20 epochs, batch 256, learning rate 0.001, L2 0.0001 on kernel/head weights;
stop after four epochs without inner weighted BCE improvement of 0.0001.
Discard inner weights and refit all 82 sources from scratch for the selected
epoch count. Outer held-out sources never fit normalization, weights or stopping.
The final 123-source model uses the integer median of the three epoch counts.
There is one architecture/optimizer recipe and no dev-dependent refit.

Thresholds at 1%, 5% and 10% are selected from outer OOF predictions to constrain
sample-weighted FPR separately in both actual control recordings. Also retain
the predeclared 0.5 reference; sigmoid scores are uncalibrated under a balanced
loss. Freeze the final weights, epoch count and thresholds in
`selection_locked.json` before the first dev decode, then benchmark the exposed
17-group dev set. Fixed centered smoothing is three scores; no gap merging.
Inference uses a numeric bundle, waveform, work directory and operating point,
without labels, source IDs, timing, activity masks or ground-truth intervals.

The primary sample comparison retains the previous **4,544-sample origin-label
guard** so its denominator matches Phase 3A/B. It does not claim to isolate the
larger CNN's entire effective support near a boundary. A second, conservative
**5,249-sample** mask accounts for filter guard + half 128-ms window + smoothing
hop + half projection cell. Secondary metrics keep the original selected
thresholds and do not optimize them again. Both masks remain effective numerical
IIR assumptions, not finite-support proofs. Window AUROC uses pure local-window
labels; common-mask sample ranking provides a comparable denominator when
window lengths differ.

## Learned results and frozen-baseline comparison

Successful run: `runs/vocopatch-phase3c-cnn-20261105/`. Inner validations selected
2/11/3 epochs (6/15/7 epochs actually executed including patience); outer refits
used 2/11/3 and the final model used their median, **3 epochs**. This is seven
fits and 47 executed fit-epochs, with source groups held out at both levels.
The final sampler contains **23,616 windows**: 14,760 real and 8,856 synthetic.
Full train scoring covers 268,210 windows (209,639 real / 12,711 synthetic /
45,860 ignored); dev covers 39,065 (31,020 / 1,755 / 6,290). The network sees
615 train recordings and benchmarks 85 dev recordings, not independent file
replicates. There are 123 and 17 independent source groups, respectively.

| Ranking | Frozen A64 | CNN128 |
|---|---:|---:|
| Train-group OOF window AUROC | 0.588927 | 0.564953 |
| Train-group OOF window directional EER | 0.440811 | 0.460787 |
| Train OOF source-macro sample F1, 5% point | 0.081536 | 0.049462 |
| Train OOF synthetic sample recall, 5% point | 0.077524 | 0.050190 |
| Exposed dev window AUROC | 0.588464 | 0.572465 |
| Exposed dev window directional EER | 0.434268 | 0.436467 |
| Dev raw common-mask sample AUROC | 0.589322 | 0.571279 |
| Dev raw common-mask sample EER | 0.435319 | 0.438147 |
| Dev smoothed common-mask sample AUROC | 0.595722 | 0.579714 |

Window metrics have different pure-window support because 64 and 128 ms differ.
The common-mask sample metrics use the same **11,659,610 valid dev samples**,
including 638,571 synthetic samples. OOF cutoff selection guarantees its training
sample-tail constraints, not unseen-source FPR or recording specificity.

All following results are the **already exposed dev benchmark**, using choices
fixed inside train. `1/5/10%` denotes the train-OOF target for each control,
not a promised dev FPR. Rates in the three rate columns are percentages.

| System / point | Cutoff | Source F1 | Global IoU | Synthetic recall % | Null FPR % | R2R FPR % | Null / R2R recordings with FA |
|---|---:|---:|---:|---:|---:|---:|---|
| A64 / 0.5 | 0.500000000 | 0.141026 | 0.067935 | 58.4165 | 44.4370 | 44.4637 | 17 / 17 |
| A64 / 1% | 0.701689315 | 0.001095 | 0.000514 | 0.0553 | 0.3992 | 0.4749 | 9 / 9 |
| A64 / 5% | 0.641217757 | 0.042459 | 0.020571 | 2.9851 | 2.5554 | 2.6765 | 17 / 16 |
| A64 / 10% | 0.609198185 | 0.085340 | 0.040243 | 8.1955 | 6.0892 | 6.0699 | 17 / 17 |
| CNN / 0.5 | 0.500000000 | 0.134188 | 0.066313 | 47.0516 | 36.1774 | 35.2152 | 17 / 17 |
| CNN / 1% | 0.653932075 | 0.000000 | 0.000000 | 0.0000 | 0.0397 | 0.0397 | 1 / 1 |
| CNN / 5% | 0.610151113 | 0.012467 | 0.008780 | 0.9941 | 0.7965 | 0.7058 | 9 / 9 |
| CNN / 10% | 0.584271172 | 0.038409 | 0.023488 | 3.1444 | 2.1050 | 1.8006 | 15 / 14 |

Each control category contains 17 recordings. Lower CNN false alarms accompany
lower recall; these are not equal-empirical-FPR comparisons and no dev cutoff is
retuned to make them match. At its 5% train point CNN dev micro precision is
0.06995, micro F1 0.01741 (distinct from source F1), TP 6,348, FP 84,406,
FN 632,223, TN 10,936,633. On train OOF its separate null/R2R sample FPRs are
4.8898% / 4.9981%, yet 119/123 and 117/123 control recordings have false alarms.

Dev per-vocoder CNN results at the same pooled 5% cutoff:

| Vocoder | Window AUROC | EER | Source sample F1 | Global sample IoU | Sample recall % |
|---|---:|---:|---:|---:|---:|
| BigVGAN | 0.575451 | 0.428643 | 0.016742 | 0.010767 | 1.1599 |
| Griffin-Lim | 0.571359 | 0.439316 | 0.012096 | 0.007688 | 0.8283 |
| Vocos | 0.574516 | 0.435897 | 0.014468 | 0.009227 | 0.9941 |

The model does not recover even the stronger Griffin-Lim oracle separation in
full recordings. In the valid region within 250 ms of raw boundaries, CNN recall
is 2.9412% and real FPR 0.3636%; farther away they are 0.7299% and 0.7735%.
For A64 they are 2.2347% / 5.0052% near, 3.0869% / 2.5683% away. Near metrics
refer only to samples surviving the guard, excluding transitions and much of the
near region. They neither measure fade localization nor eliminate seam cues.

At the CNN 5% point, null/R2R controls contain **16 / 14 valid false-alarm
intervals**, totaling **0.84236 / 0.74639 seconds** of false-positive real samples.
A64 has 65 / 64 and 2.70254 / 2.83061 seconds. Across all 85 CNN recordings
there are 85 raw predicted intervals and 72 valid-real false-positive runs,
3.82794 seconds total FP duration; median FP run is 47.98 ms and p95 128.07 ms.
At 0.5 there are 1,628 valid-real FP runs, median 64.04 ms / p95 314.58 ms.
Raw predicted intervals can include ignored samples; valid-real runs are split
by ignored or positive samples and are a different quantity. Ground truth only
scores predictions; it does not create or trim inference intervals.

The larger **5,249-sample** support mask preserves the conclusion without
reselecting cutoffs: CNN 5% dev source F1 0.009394, IoU 0.006544 and recall
0.7470%, versus A64 0.042983 / 0.020697 / 3.0858%. Raw common-mask sample AUROC
is 0.570236 vs 0.592097. The primary result is not rescued by stricter support.

## Cost, verification and checks

Recorded local CPU wall time, with BLAS limited to one thread and tests sharing
the machine: paired diagnostics **251.75 s**, including twelve small oracle
logistic fits; CNN driver **641.99 s**, including train feature preparation
313.47 s, dev preparation 29.27 s, three inner/refit/prediction fold blocks
16.95/37.97/19.51 s and final fit 4.24 s. The remainder includes scoring,
diagnostics, serialization and final identity checks. These driver times exclude
their initial identity admission and separate verification/tests; they are not
isolated throughput benchmarks. No GPU, installation or model download occurred.
Label-free single-file inference took 0.512 s on 9.273 s of audio within the
driver; it is offline, with zero-phase filtering and centered lookahead.

Diagnostic output: 167 files / 10,999,268 bytes. CNN output: 64 files /
85,780,219 bytes; its standalone three-file inference bundle is 10,825 bytes.
There are no generated WAV datasets in either new experiment output.

Independent receipts in the audit directory establish:

- Every diagnostic/CNN inventory digest and recorded current source digest
  matches; all 268,210 training window labels agree with a separate scalar
  interval/guard and full-window purity calculation.
- The twelve oracle refits and their OOF scores reproduce exactly; AUROC/EER
  agree with independent sklearn ROC calculations. All CNN outer/inner sources
  are disjoint, match the previous frozen folds, and fit normalization only on
  their training membership. Loaded fold models reproduce all OOF scores and
  the final model reproduces every cached dev score exactly.
- The final 3-epoch optimizer refit reproduces all six numeric arrays exactly
  (4.06 s) when the original extraction memory layout is restored from NPZ.
  NPZ reload is C-contiguous; extraction preserves transposed time-major inner
  strides. The audit restores `x.transpose(0,2,1).copy().transpose(0,2,1)` before
  refitting. Without that restoration, std reductions differ by up to 4.04e-13
  and trained kernel entries by 4.61e-15. The first check failure is retained;
  no saved model, cache, tolerance gate or dataset was changed to pass it.
- Standalone CLI inference receives only bundle/WAV/workdir. Its intervals are
  identical to saved predictions; unrestricted CLI BLAS scores differ by at most
  5.55e-17 from the one-thread driver. Exact recorded scoring uses the recorded
  one-thread environment, not a cross-platform floating-point promise.
- All **3,777 protected file inventories, sizes and mtimes** match the before
  snapshot: v0, both pilots, expanded data, prepared upload, Phase 3A, Phase 3B
  and the earlier correctness audit. Complete non-receipt artifacts of prior
  model/audit runs retain their SHA-256 values. All 700 consumed train/dev WAV
  hashes and 492 original training source hashes match; eval's 170 WAVs and
  16 demo WAVs are intentionally not rehashed/opened. Thus all 870 generated
  paths/sizes/mtimes are unchanged, with the byte-hash check scoped explicitly
  to consumed audio. No claim of a fresh eval-byte audit is made.

Focused checks: **29 passed** in 82.08 s. Full pytest: **868 passed, 4 failed,
7 skipped, 1 xfailed**, 700.54 s. Failures remain the same two generated-notebook
CRLF/LF comparisons and two Windows junction `WinError 5` cases. Existing
link-privilege skips and the unimplemented-ablation expected failure remain.
Source/test Ruff passes; whole-repository Ruff retains 11 generated-notebook
findings. No existing test or scientific gate is weakened. Development fixes
included creating the inference work directory and adding the conservative
support report; initial test/import/fixture failures remain in audit logs.

HEAD is unchanged and the index has no staged paths. Changes are new
`src/experiments/paired_signal.py`, `src/detectors/temporal_cnn.py`,
`src/experiments/learned_temporal.py`, `tests/test_paired_signal.py`,
`tests/test_temporal_cnn.py`, `tests/test_learned_temporal.py`, this report and
the README pointer. The pre-existing `.gitignore` edit is byte-unchanged.
All changes remain unstaged. No Kaggle, eval scoring or Git write occurred.

Next stage: retain the frozen reference and do not promote or enlarge this CNN
based on dev outcomes. Predeclare one train-group-controlled investigation of
activity/background sampling, between-source score shifts and low-FPR validation
or temporal aggregation. The oracle's longer truth-selected cores do not imply
that a short-window CNN can localize them. Resolve provenance/admission and
independent-cohort limitations before a citable or deployment claim.

## Reproduction and artifact layout

From repository PowerShell, using fresh output and pytest basetemp names:

```powershell
& '.venv/Scripts/python.exe' -B -X utf8 -m experiments.paired_signal --dataset-dir runs/btpvf-partial-v1-20261104 --source-manifest runs/onboarding/v0-kaggle-input/manifest.csv --baseline-dir runs/vocopatch-phase3a-baseline-20261104 --out-dir runs/vocopatch-phase3c-diagnostics-20261104
& '.venv/Scripts/python.exe' -B -X utf8 -m experiments.learned_temporal run --dataset-dir runs/btpvf-partial-v1-20261104 --source-manifest runs/onboarding/v0-kaggle-input/manifest.csv --baseline-dir runs/vocopatch-phase3a-baseline-20261104 --diagnostics-dir runs/vocopatch-phase3c-diagnostics-20261104 --out-dir runs/vocopatch-phase3c-cnn-20261105
& '.venv/Scripts/python.exe' -B -m pytest -o addopts='' -q -ra tests/test_paired_signal.py tests/test_temporal_cnn.py tests/test_learned_temporal.py --basetemp runs/vocopatch-phase3c-audit-20261104/pytest-phase3c-3
& '.venv/Scripts/python.exe' -B -m pytest -o addopts='' -q -ra --basetemp runs/vocopatch-phase3c-audit-20261104/pytest-full-2
& '.venv/Scripts/python.exe' -B -m ruff check src tests
& '.venv/Scripts/python.exe' -B -m ruff check .
```

The driver refuses existing targets/staging directories and outputs overlapping
inputs. A failure preserves its new staging directory with a failure receipt;
it never overwrites an earlier experiment. Success seals every output SHA-256.
The run includes configuration/folds, source/code/environment provenance,
physical diagnostic tables, measured exclusions, per-source activity/gain
evidence, twelve numeric fold bundles, all oracle features/OOF scores, per-kind,
per-vocoder and per-source metrics, the conditional decision and validation.
NPZs contain numeric/string arrays, never pickled estimators. The oracle bundles
are diagnostic classifiers; the separate CNN bundle supports label-free temporal
inference and remains a weak development model.
The executed CNN run adds train/dev spectrogram caches, outer fold bundles and
loss histories, train OOF/locked dev scores and intervals, 1/5/10% and 0.5
operating-point metrics, conservative support-mask metrics, the locked final
numeric bundle and a label-free single-file agreement receipt. Inference can
run independently of dataset/label files with
`python -m experiments.learned_temporal predict --bundle ... --wav ... --work-dir ...`.

The concrete CLI check used:

```powershell
& '.venv/Scripts/python.exe' -B -X utf8 -m experiments.learned_temporal predict --bundle runs/vocopatch-phase3c-cnn-20261105/bundle --wav runs/btpvf-partial-v1-20261104/audio/0c817611b39d7de466544d41.wav --work-dir runs/vocopatch-phase3c-audit-20261104/inference-cli-work
```

The audit directory contains the before-state protection receipt, predeclaration,
logs and a separate read-only verification script. Tests use synthetic fixtures
and check physical units, gain invariance, shared activity masks, score
orientation/ties, per-kind FPR constraints, disjoint folds, deterministic
repetition, immutable inputs, fresh outputs and a dev/eval decoder trap.
The CNN tests add all-parameter finite differences, deterministic Adam fitting,
source/class weighting, train-only normalization, masks/padding, safe numeric
bundle loading and label-free inference. The six-source integration fixture's
actual oracle fails the 5% cap; the test verifies that refusal before stubbing
only the conditional prerequisite in RAM to test full orchestration. No actual
experiment uses that stub. The real run requires the sealed measured criterion.

## Scientific scope

This remains one legacy single-speaker corpus, three known vocoder/front-end
systems, fixed-duration edits and one fade recipe. Eligibility already favors
sources with a valid disjoint active donor. Source/generation/alignment
provenance and appropriate partial-audio signal-gate admission remain unresolved;
common filtering and structural integrity do not establish those facts.
INVs 17/18 and prior EER-orientation/paired-uncertainty limitations are unchanged.

Within-source comparisons reduce content confounding but do not remove phase,
alignment, spectral-shape, loudness or front-end effects. Activity and gain
ablation differences are descriptive, not causal identification. Similar null
and real-to-real behavior does not eliminate all splice shortcuts. A group
macro distribution is not a validated confidence interval. The exposed 17-group
dev cohort and previously reviewed legacy eval cohort are not pristine final
research holdouts; neither is used for this conditional decision. Independent
provenance and an untouched cohort remain necessary before research claims.
No new perceptual quality score is measured here, so these known-vocoder
differences do not establish a quality-versus-detectability relationship.
The CNN also changes local context, training sampling and loss weighting versus
the frozen linear system; this is a comparison of two systems, not an isolated
causal test of representation alone. There is one fixed seed/recipe and no
validated source-cluster uncertainty interval. OOF thresholds do not guarantee
the same FPR after the median-epoch full refit or on new sources. Early stopping
minimizes inner balanced BCE rather than directly optimizing low-FPR recall.
Bitwise reproduction checks are specific to the recorded local implementation,
libraries and CPU numerical environment.
