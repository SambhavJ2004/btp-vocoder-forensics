# VocoPatch Phase 3E: paired-content negative weighting

Date: 2026-11-05. **Development-only, non-citable.** The 17-source dev set is a
previously exposed development benchmark. Eval is unused. This is a controlled
CPU logistic experiment, not a new neural model, final research record or
deployment qualification.

**Decision: C — stop optimizing this localization approach and reconsider the
detection problem.** Train-only selection retains A, the original protocol.
Increasing paired real-negative mass to 25% or 50% fails to improve primary OOF
recall. Small dev gains at the locked cutoffs coincide with higher sample FPR,
remain uncertain across sources, and largely disappear at equivalent empirical
FPR. This is a negative result for this controlled feature/model/weighting test;
it does not establish that synthetic speech contains no usable forensic signal.

## Question and fixed design

Does increasing the weight of same-source, same-time real speech improve the
existing temporal detector's discrimination at controlled false-positive rates?
Phase 3D motivates this test but does not establish that reweighting will help.

The plan was saved in
`runs/vocopatch-phase3e-audit-20261105/predeclared.json` before empirical fitting.
The only candidates are A (original protocol), B (25% paired-negative mass) and
C (50%). All use the original 31 below-8-kHz spectral features, 64-ms windows
(1,411 samples), 16-ms hops (353 samples), C=1 logistic regression, three-sample
centered smoothing and no gap merging. No new frequency representation, gain
normalization or architecture is introduced. The common zero-phase filter and
PCM16 comparison path are unchanged.

The primary sample mask retains the original A64 guard, 4,544 samples:
`3308 + ceil(1411/2) + 353 + ceil(353/2)`.
The 3,308-sample feature-label guard and requirement for a completely pure
feature window remain unchanged. A secondary 5,249-sample scoring guard permits
comparison with the wider-support Phase 3C/3D analyses; thresholds are held fixed.
The numerical IIR support guard is not a claim of exactly finite filter support.

## Pair construction and retained controls

Only the 123 admitted frozen training sources contribute supervised pairs.
For every source, the original real PCM waveform is checked against the existing
null control, before and after the sanctioned comparison processing. Its complete
feature matrix and window starts must exactly equal the sealed A64 null cache.
Real-reference RMS and the recorded activity threshold must also reproduce.
There is no new offset estimation, shifting, trimming or alignment repair.

A coordinate is eligible when it is a pure synthetic window in each of the three
mixed conditions and is speech-active in all three synthetic signals and their
real counterpart. Activity uses the already recorded Phase 3D real-reference
threshold: maximum 512/128 RMS minus 30 dB, floored at two PCM16 steps. Selection
uses each 64-ms window's RMS. At least eight shared active coordinates per source
are required; failure aborts rather than dropping a source or lowering the rule.
The resulting 123 groups all qualify: **4,460 distinct real windows**, with
**13,380 synthetic/real pair links**, 30–39 shared coordinates per source.

Pairs have identical `[start_sample, end_sample)` coordinates, source-real
identity, preprocessing and window geometry. Modal-zero alignment gates inherited
from the verified, unchanged Phase 3D source audit do not establish perfect
per-file/per-frame alignment. Ground truth and activity select supervised rows
and weights only. No coordinate, filename, group identity or vocoder annotation
becomes an inference feature.

The paired real feature is already present in the null-control training rows;
it is reweighted once, not copied three times for its three pair links. Every
original pure positive and negative window remains. This includes inactive
positive speech, unpaired null windows, real-to-real edits and ordinary real
background in mixed recordings. Ambiguous/transition windows remain ignored.

## Exact weights and normalization

In each fit partition let N be the number of original pure windows and N_c the
count for class c. A retains its original balanced class weight
`w_c = N / (2 N_c)`. For source g with n_g negative windows its original negative
mass is `M_g = w_0 n_g`. With m_g distinct eligible matched real windows, B/C use

```
paired negative weight   = M_g * alpha / m_g
other negative weight   = M_g * (1-alpha) / (n_g-m_g)
positive window weight  = w_1 (unchanged)
alpha                   = 0.25 or 0.50
```

Each source-by-class total and both global class masses are preserved; total
weight is N. This controls source contribution relative to A, rather than
silently replacing its duration-dependent source weighting with equal-source
weighting. Every original pure window retains positive finite weight. B/C use
explicit sample weights and no second class-weight factor. C, solver, tolerance
and maximum iterations remain fixed. Effective sample size is reported because
concentrating negative mass reduces it, even though row counts do not change.

The full fit has 270,670 original windows: 41,180 ignored, **215,303 pure real**
and **14,187 pure synthetic**. Each condition contributes 45,898 pure windows;
each mixed condition contains 4,729 positives. All candidates fit the same
229,490 rows, with class masses 114,745 each (floating-point roundoff aside).
The original per-window weights are 0.5329465915 real and 8.088038345 synthetic.
There are 738 low-activity positives, all retained. The 13,380 eligible synthetic
pair links cover 94.31% of positive rows; none of the other 807 positives is lost.

| Negative stratum | Distinct rows | A weight mass | B weight mass | C weight mass |
|---|---:|---:|---:|---:|
| Matched original real | 4,460 | 2,376.942 | 28,686.250 | 57,372.500 |
| Other null real | 41,438 | 22,084.241 | 16,912.353 | 11,274.902 |
| Real-to-real control | 45,898 | 24,461.183 | 18,739.297 | 12,492.865 |
| Ordinary real in mixed recordings | 123,507 | 65,822.635 | 50,407.100 | 33,604.734 |

Matched real rises from 2.0715% to 25% or 50% of negative mass. Full-fit effective
sample sizes are 53,239.86 / 45,081.84 / 29,859.12 for A/B/C; these are weight
concentration diagnostics, not independent-utterance counts. Negative low-activity
mass changes from 16,941.31 to 12,960.15 / 8,640.10, while every positive activity
bin's mass is unchanged. Full condition/activity/source counts and weights are
saved in `training_distribution.json` and each fold's `fit.json` / weight NPZ.

The intervention necessarily shifts negative condition and activity composition
as it emphasizes active null-control coordinates. Source/class totals are
controlled, but this is not a factorial experiment isolating paired content
from activity, control composition or weight concentration. No such causal
claim is made.

Within each fit partition, A fits the original unweighted StandardScaler on
that partition's original pure windows. B/C reuse its **exact numeric mean and
scale**. No held-out source contributes to that scaler. The replayed A heads
must exactly match all four arrays of the historical three fold heads and the
original Phase 3A final head before proceeding.

## Selection, lock and evaluation

The original seeded three-fold source assignment is reused: 82 fit and 41 held
training groups per fold. Each source's five recordings stay together. Train
OOF thresholds are computed separately for each candidate at 1%, 5% and 10%
control-sample FPR; **both** null and real-to-real controls must satisfy the cap.
Threshold ties move together. A fixed 0.5 reference is also retained.

Selection maximizes synthetic sample true positives (equivalently micro recall,
because the positive denominator is fixed) at the 5% OOF point. Ties use higher
source-macro F1, lower maximum control FPR, then lower paired-negative mass.
All three final models, operating points, preferred configuration and numeric
bundle inventories are written to `selection_locked.json` **before dev features
are loaded**. Dev scoring cannot change them. Unit coverage traps a dev cache
load before that lock and checks its complete bundle inventories.

Twelve lightweight logistic fits are performed: nine outer-fold heads and three
final heads. No neural optimizer is invoked by this experiment. Inference uses
the existing waveform-only numeric context bundle. A single-input replay checks
that margins equal cached-feature prediction without an original real waveform,
source manifest, labels or score centering.

Window AUROC/EER use the unchanged window-label protocol. Sample F1, IoU, recall
and control FPR use the common primary mask, and are also saved with the larger
secondary mask. Recording false alarms mean at least one predicted event on a
control; a low sample FPR does not imply a low recording false-alarm rate.
Per-vocoder, near/away boundary, group variability and diagnostic predictions
are saved at every operating point. Source-macro F1 includes undefined cases
as zero using the existing policy.

The fixed OOF cutoffs are the actual benchmark operating points. In addition,
an **evaluation-only empirical ROC statistic** reports maximum synthetic recall
under the same 1/5/10% caps on each scored set. It uses final smoothed sample-cell
scores and whole ties; no cutoff is exported or used to select a model. Dev ROC
statistics describe ranking under equivalent empirical FPR conditions; they
are optimistic descriptive metrics, not newly calibrated dev thresholds or
guarantees about deployment. Raw rank correlation against A and same-time
synthetic-minus-real OOF margin differences help separate discrimination from
a change of calibration.

Paired 2,000-resample source-group intervals use fixed predictions and cutoffs.
They do not refit the model, redo threshold selection or account for choosing
among three candidates. OOF control calibration and model selection share OOF
data; these estimates are development selection evidence, not unbiased final
generalization estimates. Relative gains are reported alongside absolute changes.

The predeclared research-guidance heuristic calls an OOF improvement meaningful
only if recall rises by at least two percentage points, its paired conditional
95% interval is strictly above zero and group F1 does not fall. It recommends C
if that fails; B if it passes but the locked candidate fails dev control transport
or loses recall against A; otherwise A. This heuristic changes no scientific
admission gate and cannot reselect or recalibrate a model on dev.

## Measured results

The primary point is each candidate's **own train-OOF-selected 5% control-FPR
cutoff**, applied unchanged to dev. Percent signs denote rates; F1 and IoU are
unit-scale values. All 615 train recordings (123 sources) and 85 dev recordings
(17 sources) are covered; their five variants are not independent utterances.

| Split | Model | Locked cutoff | Window AUROC | Window EER | Group F1 | Global sample IoU | Synthetic recall |
|---|---|---:|---:|---:|---:|---:|---:|
| Train OOF | A original | 0.6412177570 | 0.588927 | 0.440811 | 0.081536 | 0.043489 | 7.7524% |
| Train OOF | B paired25 | 0.6137356763 | 0.591023 | 0.438289 | 0.081357 | 0.042517 | 7.5928% |
| Train OOF | C paired50 | 0.5901679693 | 0.590983 | 0.435972 | 0.076414 | 0.039038 | 7.0112% |
| Exposed dev | A original | 0.6412177570 | 0.588464 | 0.434268 | 0.042459 | 0.020571 | 2.9851% |
| Exposed dev | B paired25 | 0.6137356763 | 0.589734 | 0.429301 | 0.047904 | 0.023460 | 3.4676% |
| Exposed dev | C paired50 | 0.5901679693 | 0.588400 | 0.433671 | 0.045668 | 0.022533 | 3.4526% |

| Split / model | Null sample FPR | Real-to-real sample FPR | Null recording FA | Real-to-real recording FA |
|---|---:|---:|---:|---:|
| OOF A | 4.9992% | 4.8077% | 122/123 (99.19%) | 118/123 (95.93%) |
| OOF B | 4.9987% | 4.8381% | 123/123 (100%) | 119/123 (96.75%) |
| OOF C | 4.9992% | 4.9283% | 122/123 (99.19%) | 119/123 (96.75%) |
| Dev A | 2.5554% | 2.6765% | 17/17 (100%) | 16/17 (94.12%) |
| Dev B | 2.7759% | 2.8061% | 17/17 (100%) | 16/17 (94.12%) |
| Dev C | 3.1203% | 3.1244% | 16/17 (94.12%) | 15/17 (88.24%) |

Recording false-alarm rates remain unacceptable for an audio-localization
application. The small count reduction for C on dev does not overcome weak
recall or justify selecting C after seeing this benchmark.

### Absolute and relative changes

Compared with A at each locked 5% point:

| Split / candidate | Recall change, percentage points | Relative recall change | Conditional paired recall-change 95% interval, pp | Group F1 change | Relative F1 change |
|---|---:|---:|---:|---:|---:|
| OOF B | -0.1596 | -2.06% | [-0.6014, +0.2600] | -0.000179 | -0.22% |
| OOF C | -0.7412 | -9.56% | [-1.8294, +0.2224] | -0.005122 | -6.28% |
| Dev B | +0.4825 | +16.16% | [-0.2914, +1.6034] | +0.005445 | +12.82% |
| Dev C | +0.4674 | +15.66% | [-0.9446, +2.1258] | +0.003209 | +7.56% |

All these recall-change intervals include zero. Conditional group-F1 change
intervals are OOF B [-0.004085, 0.003882], OOF C [-0.014408, 0.003684],
dev B [-0.003738, 0.018301], and dev C [-0.013591, 0.021003]. Large relative
dev percentages arise from a baseline recall below 3%; the absolute gains stay
below half a percentage point. Dev does not reverse the OOF selection.

### Equivalent empirical FPR and operating tradeoff

These are the evaluation-only ROC recalls under the same **separate** null and
real-to-real FPR caps, with no new operating threshold exported:

| Scored set | Maximum FPR in each control | A recall | B recall | C recall |
|---|---:|---:|---:|---:|
| OOF | 1% | 1.7612% | 1.7971% | 1.6371% |
| OOF | 5% | 7.7524% | 7.5928% | 7.0112% |
| OOF | 10% | 14.4715% | 14.6992% | 14.6894% |
| Exposed dev | 1% | 0.5528% | 0.4422% | 0.3870% |
| Exposed dev | 5% | 6.8664% | 6.9217% | 6.5197% |
| Exposed dev | 10% | 13.2916% | 13.1641% | 13.8990% |

At a common 5% empirical dev cap, B gains just **0.0553 pp** of recall
(approximately 0.81% relative); C loses 0.3467 pp. This differs from the larger
apparent gains at the locked cutoffs, where the actual dev control FPRs differ.
At 1% empirical FPR both candidates are worse than A on dev. Small 10% gains
do not satisfy the predeclared 5% objective and involve a much looser FPR budget.

The actual locked OOF-1% dev recalls are 0.0553% / 0 / 0 (A/B/C), with maximum
control FPRs 0.4749% / 0.6263% / 0.6358%. At the locked OOF-10% cutoffs, dev
recalls are 8.1955% / 8.7577% / 9.4934%, with maximum control FPRs
6.0892% / 6.4970% / 7.0987%. Saved metrics include both separate control rates
and F1/IoU for every point. At a shared numerical threshold 0.5, dev recall falls
58.42% → 55.87% → 53.56% as control FPR also falls from approximately 44.5% to
42.5% and 40.4%; none is useful. Comparing that threshold alone would confuse
a score/calibration change with improved discrimination.

### Per-vocoder results at the locked 5% points

F1 is source-macro within the indicated mixed condition, not the all-five-condition
group F1 above. Recall is sample micro; ranking is window based.

| Split | Model | Vocoder | AUROC | EER | Group F1 | Recall |
|---|---|---|---:|---:|---:|---:|
| OOF | A | BigVGAN | 0.574939 | 0.452962 | 0.087838 | 6.9741% |
| OOF | B | BigVGAN | 0.573748 | 0.454104 | 0.084031 | 6.5617% |
| OOF | C | BigVGAN | 0.568832 | 0.452841 | 0.073835 | 5.7105% |
| OOF | A | Griffin-Lim | 0.634167 | 0.405888 | 0.116213 | 9.6079% |
| OOF | B | Griffin-Lim | 0.643891 | 0.398479 | 0.120459 | 9.8715% |
| OOF | C | Griffin-Lim | 0.654897 | 0.388715 | 0.115158 | 9.1703% |
| OOF | A | Vocos | 0.563161 | 0.462043 | 0.083976 | 6.6752% |
| OOF | B | Vocos | 0.560497 | 0.462043 | 0.082151 | 6.3451% |
| OOF | C | Vocos | 0.553366 | 0.467949 | 0.079895 | 6.1530% |
| Dev | A | BigVGAN | 0.580826 | 0.444408 | 0.043481 | 2.9851% |
| Dev | B | BigVGAN | 0.580450 | 0.440800 | 0.054738 | 3.8143% |
| Dev | C | BigVGAN | 0.576497 | 0.441784 | 0.056885 | 3.9801% |
| Dev | A | Griffin-Lim | 0.634970 | 0.395100 | 0.049303 | 3.3168% |
| Dev | B | Griffin-Lim | 0.643324 | 0.387340 | 0.061888 | 4.2667% |
| Dev | C | Griffin-Lim | 0.650650 | 0.391768 | 0.054038 | 3.7692% |
| Dev | A | Vocos | 0.556713 | 0.464012 | 0.042508 | 2.6534% |
| Dev | B | Vocos | 0.552152 | 0.466054 | 0.037538 | 2.3217% |
| Dev | C | Vocos | 0.543907 | 0.472286 | 0.040146 | 2.6083% |

The pooled ranking increase hides a tradeoff: Griffin-Lim AUROC rises, while
BigVGAN and Vocos AUROC decline. There is no consistent cross-vocoder improvement.
These are not perceptual-quality-versus-detectability research conclusions.

### Boundary support, groups and discrimination

Away from the existing ±0.25-s boundary diagnostic region, OOF recalls are
7.8231% / 7.5755% / 6.8216% and F1 values 0.080290 / 0.077716 / 0.069897.
Dev away-boundary recalls are 3.0869% / 3.6519% / 3.7786%, with F1
0.040487 / 0.046792 / 0.046425. With the wider 5,249-sample guard and unchanged
cutoffs, OOF recall is 7.8196% / 7.5806% / 6.8347%; dev is
3.0858% / 3.6464% / 3.7645%. Neither check changes the OOF selection or recovers
strong localization. Guarding boundaries does not eliminate every editing cue.

| Split / model | Source F1 median [5th, 95th percentile] | Source recall median [5th, 95th percentile] |
|---|---:|---:|
| OOF A | 0.04273 [0, 0.29268] | 3.7590% [0, 26.2322%] |
| OOF B | 0.04212 [0, 0.29302] | 4.6988% [0, 27.2529%] |
| OOF C | 0.04223 [0, 0.28503] | 3.7590% [0, 20.6746%] |
| Dev A | 0.01643 [0, 0.13391] | 1.8795% [0, 10.7132%] |
| Dev B | 0.01013 [0, 0.18948] | 0.9398% [0, 14.0963%] |
| Dev C | 0.01519 [0, 0.16627] | 1.3684% [0, 13.7204%] |

There are zero-recall groups in every setting. Dev B's improved mean accompanies
a lower median; the effect is not uniformly shared across sources.

OOF raw-margin Spearman correlation against A is 0.98858 for B and 0.91821 for C.
Mean within-source same-time synthetic-minus-real margins (BigVGAN / Griffin-Lim /
Vocos) are A: 0.02599 / 0.11759 / 0.00799; B: 0.02495 / 0.11499 / 0.00814;
C: 0.02381 / 0.11159 / 0.00830. Source-mean paired ordering win fractions are
A: 0.56747 / 0.75619 / 0.51908; B: 0.56896 / 0.76899 / 0.52536;
C: 0.57018 / 0.78308 / 0.52525. These are paired-window diagnostics, **not AUROC**
and not the Phase 3D core-mean oracle statistic; their sampling units differ.

There are small ranking changes, especially for Griffin-Lim, rather than a
purely monotone recalibration. However, the high correlation, lowered cutoffs,
weak matched-pair margins, absent primary OOF gain and almost unchanged
equivalent-FPR dev recall give no evidence of a useful general localization
improvement. The controlled experiment does not justify replacing A or starting
another neural architecture. Reconsider the target, available data and validation
protocol before further model optimization; no new task is started here.

## Reproduction and artifacts

Use the existing project environment and a fresh output directory. No dependency
installation, model download, original data regeneration or Kaggle job is needed.

```powershell
& '.venv/Scripts/python.exe' -B -X utf8 -m experiments.paired_training `
  --dataset-dir runs/btpvf-partial-v1-20261104 `
  --source-manifest runs/onboarding/v0-kaggle-input/manifest.csv `
  --baseline-dir runs/vocopatch-phase3a-baseline-20261104 `
  --ablation-dir runs/vocopatch-phase3b-ablation-v2-20261104 `
  --review-dir runs/vocopatch-phase3d-diagnostics-v2-20261105 `
  --out-dir runs/vocopatch-phase3e-paired-training-20261105-replay
```

The saved run is `runs/vocopatch-phase3e-paired-training-20261105/`. Fresh-output
and separate-input checks reject overwrite/resume; a failed run retains its
staging receipt. Artifacts include the plan, source/data/environment digests,
counterpart checks, exact pair coordinates, fold memberships, per-fold/full
weight vectors, distribution summaries, twelve numeric bundles, OOF/dev margins,
thresholds, metrics, diagnostics, lock, research decision, validation and complete
SHA-256 inventory. No new audio dataset is generated. Supplementary before/after
checks, logs and the independent replay script are in the new ignored audit
directory `runs/vocopatch-phase3e-audit-20261105/`.

The empirical driver completed in **584.07 seconds**, using one BLAS thread,
while tests shared the machine. The sealed output contains **184 files /
50,963,402 bytes** (approximately 48.60 MiB). This is end-to-end integrity,
feature verification, fitting, scoring and serialization time, not an isolated
training-throughput benchmark. Numeric prediction replay is checked for every
saved OOF/dev model; repeated weighted fitting is tested on CPU fixtures.
No second full B/C empirical refit or cross-platform determinism claim is made.

## Integrity, checks and review scope

The driver rechecks the inherited dataset identity and all 509 reviewed original
training-source digests after completion. Its successful receipt records 123
train groups, 17 dev benchmark groups, identical normalization across candidates,
exact A numeric arrays and inference without pair labels or source adjustments.
The existing strict train/dev admission checker verifies 700 consumed partial
WAV hashes plus metadata; it does not decode or rehash the 170 eval or 16 demo
WAVs. Shared metadata containers are parsed for integrity, but no eval annotation
records are used to fit, select, score or analyze a model.

The supplementary independent checker verifies the complete new inventory and
source-code digests, exact A historical OOF/dev prediction arrays, original A
head arrays, new-model prediction replay, all 13,380 pair coordinates and
activity checks, retained positive weights, each source/class weight budget,
source-disjoint frozen fold memberships, and the lock's unchanged bundles.
It compares all 4,114 protected prior file paths, sizes and mtimes across twelve
roots, and complete non-receipt hash inventories for seven prior model/run roots.
The original v0, expanded dataset, upload folder, both pilots and both Phase 3D
runs are protected. Eval/demo preservation evidence is deliberately limited to
inventory/stat comparison; no forbidden eval byte read is presented as a hash
check. The 870 generated WAV paths, sizes and mtimes stay unchanged; consumed
train/dev bytes are separately verified. No original or prior artifact is written.

Focused tests: **23 passed**, 22.62 seconds. New CPU coverage checks exact matched
coordinates and active intersections, deduplication, invalid source/coordinate/
split/activity evidence, all source/class budgets, retained controls and positive
weights, deterministic fitting, train-only normalization, whole-score ROC ties,
selection constraints, paired source uncertainty, lock-before-dev access,
label-free inference, fresh-output rejection and fixture input immutability.
The end-to-end fixture mocks only inherited review admission because six toy
sources cannot establish an INV-16 experiment; the empirical run uses strict
real-data admission and unchanged existing gate receipts.

Full pytest: **907 passed, 4 failed, 7 skipped, 1 xfailed**, 722.84 seconds.
The four existing failures are two generated-notebook CRLF/LF comparisons and
two Windows directory-junction `WinError 5` cases. Link-privilege skips and the
existing unimplemented quantization-ablation expected failure remain. There are
no new test failures. Source/test Ruff passes; repository-wide Ruff retains the
same **11 generated-notebook findings**. No pre-existing tests are weakened.
An earlier full-suite invocation ended without a final summary; the fresh
`pytest-full-2` run is the authoritative complete result.

Exact test invocations (these basetemps were fresh when used; choose new names
for future reruns):

```powershell
& '.venv/Scripts/python.exe' -B -m pytest -o addopts='' -q -ra `
  tests/test_paired_training.py `
  --basetemp runs/vocopatch-phase3e-audit-20261105/pytest-focused-3
& '.venv/Scripts/python.exe' -B -m pytest -o addopts='' -q -ra `
  --basetemp runs/vocopatch-phase3e-audit-20261105/pytest-full-2
& '.venv/Scripts/python.exe' -B -m ruff check src tests
& '.venv/Scripts/python.exe' -B -m ruff check .
& '.venv/Scripts/python.exe' -B -X utf8 `
  runs/vocopatch-phase3e-audit-20261105/independent_verify.py
```

Changed repository files are `src/experiments/paired_training.py`,
`tests/test_paired_training.py`, this report and its README pointer. The pre-existing
`.gitignore` modification is preserved byte-for-byte. HEAD remains
`bc2f470c8790de0dd3a2c534535a9f96a35e3041`; the index is empty and all new work is
unstaged. No dependencies, checkpoints, scientific gates, processing algorithms,
split assignments, `CLAUDE.md` rules or previous run artifacts are changed.
There is no Kaggle operation, final eval scoring or new empirical neural training.

## Scientific limits and next decision

Legacy source provenance remains unresolved, so these working-tree development
fits are not citable record-mode research evidence. A successful integrity
receipt does not grant scientific admission to the underlying legacy archive.
The source corpus is single-speaker; duration/disjoint-donor eligibility excludes
26 of the original 200 sources, and the present 123/17 cohorts inherit that bias.
No additional source is excluded in Phase 3E. Same-utterance grouped folds prevent
variant/window leakage but cannot establish generalization to unseen speakers,
corpora, recording channels, codecs, generators or editing methods.

Prior dev exposure has influenced the overall research direction even though
this run locks selection before dev access. Seventeen reused dev sources do not
provide a fresh final benchmark. The untouched 34 eval sources are not used to
settle these exploratory questions. Overlapping windows, identical background
across variants and paired source reuse make recordings/windows correlated;
the source group is the resampling/partition unit. Conditional intervals, the
fixed three-candidate search and a weight-concentration change do not isolate all
causes or quantify full model-selection uncertainty.

Fixed one-second splices, fixed fade construction, retained vocoder processing
differences, residual low-band cues and numerical filter tails remain potential
confounds. Genuine-splice controls and away-boundary masks investigate some
editing effects rather than eliminating all of them. The unchanged below-8-kHz
scope preserves the existing band-membership rules and makes no full-band or
quality-axis claim. Balanced-training sigmoids are not calibrated deployment
probabilities, and even the 5% sample cap leaves almost every control recording
with a false alarm.

The evidence supports **recommendation C for this localization approach**.
Paired-negative scarcity was a testable hypothesis; emphasizing those negatives
does not produce the required OOF gain, and exposes no newly correctable pairing,
normalization or implementation defect. Retain the original model as the frozen
reference. Further work should first reconsider the detection target and data/
evaluation design rather than widen this weighting search or deploy a localizer.
Code, tests and the negative-result report are ready for manual review/commit;
the models are not ready for an application or publication.
