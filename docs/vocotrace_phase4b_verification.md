# VocoTrace Phase 4B: verification benchmark and safe abstention

**Outcome:** the benchmark and guarded decision interface work reproducibly,
but these features do **not** establish reliable unknown-generator recognition.
Three-class dev source precision is 94.44% on only 18/80 returned labels. With
Griffin-Lim withheld, both returned dev labels are wrong. All three held-out
contexts reject 19/20 withheld dev recordings by returning no supported source;
none identifies them through the explicit unknown/inconclusive branch.

2026-11-05. Starting manually created HEAD:
`db056a79186cbfc5c1ae44707f5a3a15658ea5d4`.
**Development-only, non-citable.** This is a small whole-audio verification
benchmark, not general open-set recognition, an authenticity certificate or a
new neural architecture. Legacy generation provenance and scientific admission
limitations remain. No frozen eval audio/features are consumed or newly scored.

## Scoring correction

The zero-temperature defect was reproduced before changing the energy helper.
`detectors.attribution.energy_score` now requires a finite positive numeric scalar
temperature and finite logits with a nonempty class axis. Zero, negative, NaN,
infinite, boolean and nonscalar temperatures raise explicit errors. Values not
representable as positive float64 temperatures also fail explicitly.

The definition and orientation remain **E(z) = -T logsumexp(z/T)**, with lower
energy nominally known-like. The implementation centers before dividing, avoiding
overflow from huge logits or tiny T. A fallback handles an overflowing product
whose sum is nevertheless representable. Unrepresentable final float64 scores
raise rather than saturating. Tests compare the original formula on ordinary
inputs, equal logits, constant shifts, extreme values and product cancellation.
No historical score, prediction or model file is changed. The old pilot does
not use this helper.

Energy is not an unknown probability: `E(z+c*1)=E(z)-c`, while class decisions
and softmax probabilities are unchanged. The dormant open-set stub's reject-all
prose is corrected too: a constant score has AUROC 0.5 with both classes present.
No unsupported open-set driver is promoted to a functioning research result.

## Populations and historical baseline audit

Use the original paired **whole-utterance** archive, not partial recordings
relabeled as pure vocoder audio. Frozen groups remain **140 train / 20 dev /
40 eval**, one speaker, four conditions per source: real, BigVGAN-112m,
Griffin-Lim and Vocos. New features cover 560 training and 80 dev recordings.
The often-mentioned 17-group dev population belongs to the donor-eligible
partial-audio subset. All 20 original whole-audio dev sources are already exposed
and are used only after locking choices. No new split ratio or reassignment occurs.
The VocoPatch 870 recordings/174-source dataset and its 123/17/34 eligible
partitions remain unchanged and are not additional independent observations.

The existing manifest resolver/validator and pilot frame guards check complete
source groups, conditions, tiers, speaker/split consistency and duplicate rows.
The original manifest, frozen group-map and source-sidecar digests must match the
historical model's identities. All **640 train/dev audio hashes** must match the
historical feature-row records. The comparison-set extractor verifies equal
paired waveform lengths. Eval identities are structural metadata only; eval
waveforms are not hashed/opened. No alignment compensation, resampling, new
loudness normalization, gate relaxation or source-file repair is performed.

The feature path is unchanged: common below-8-kHz zero-phase filtering on real
and all three vocoders, sanctioned PCM16 write/reread and band verification;
89,027-sample first crop/repeat padding; 1,024-sample periodic Hann frames with
256 hop; 32 log-power-band means followed by 32 population standard deviations.
All inference features are acoustic. Names, labels, source IDs, split and
calibration metadata select/audit rows but do not enter the 64-value vector.
The model only sees the first approximately 4.0375 seconds of long recordings.

Historical pilot artifacts are checked as immutable opaque files. The existing
aggregate attribution matrix is `[[36,0,4],[0,40,0],[10,0,30]]`, ordered
BigVGAN/Griffin-Lim/Vocos; its macro-F1 recalculates to **0.8826733710**. Historical
binary AUROC **0.6791666667**, EER **0.3833333333**, matrix `[[80,40],[18,22]]`
and joint detected-and-attributed fraction **0.5833333333** remain stored
historical eval aggregates. No eval prediction row or feature array is used
to recompute their ranking or add new eval statistics.

Fresh cold training features reproduce both original train-only scaler means
and scales to 1e-12, on 560 binary and 420 synthetic attribution rows. After
the new choices are locked, the old frozen model's previously selected dev
benchmark is replayed from the new dev features. It is not a new selection
criterion. The old three-class model is never reused as a two-class held-out
vocoder model, warm start, scaler or source-support reference.

## Three separate tasks and fixed models

The predeclaration is
`runs/vocotrace-phase4b-audit-20261105/predeclared.json`; the executed plan is
saved with each run. It fixes four contexts:

| Context | Known synthetic classes | Withheld class |
|---|---|---|
| `closed3` | BigVGAN-112m, Griffin-Lim, Vocos | None |
| `without_bigvgan_112m` | Griffin-Lim, Vocos | BigVGAN-112m |
| `without_griffin_lim` | BigVGAN-112m, Vocos | Griffin-Lim |
| `without_vocos` | BigVGAN-112m, Griffin-Lim | Vocos |

For each context, report **A: closed-set attribution on true known inputs**,
**B: false source attribution on genuine inputs**, and **C: withheld-vocoder
source rejection** separately. Do not pool these into an overall classification
accuracy. A high conditional known-source F1 does not settle tasks B or C.

Every fit uses the existing StandardScaler + L2 logistic family, fixed C=1,
lbfgs, max_iter=2000, tol=1e-6, seed 20260907 and one BLAS thread. Binary
labels remain `0=synthetic, 1=real`; a higher binary margin means real. Binary
loss uses the original balanced-class rule without duplicating real rows.
Its scaler fits genuine plus permitted known synthetic rows. Attribution uses
only permitted known synthetic rows, equal counts per class and an independent
train-only scaler. No hyperparameter grid, new neural representation or new
probability calibrator is fitted.

For two classes, sklearn's single source log-odds margin m becomes logits
`[-m/2,+m/2]`, reproducing sigmoid(m), not sigmoid(2m). For three classes use
the original multinomial logits. This mapping and class order are saved.

## Train-only grouping, calibration and lock

Hash-sort source IDs by the recorded seed and `vocotrace4b:outer` namespace,
then assign round-robin to three training-only folds (47/47/46 sources).
Within each outer fit population, a separately hash-ordered every-third source
forms calibration; the rest fit the model. The resulting outer roles are
62 model-fit sources, 31/31/32 calibration sources and 47/47/46 outer-held
sources. Each source's conditions remain together. Saved `folds.json`, bundle
metadata and prediction row IDs identify every source/class membership.

Final models fit folds 0+1 (**94 sources**) and calibrate on fold 2 (**46**).
There is **no full-train refit after calibration**. Keeping a model/calibration
partition costs fit data but avoids transferring a fold cutoff blindly to a
refitted score scale. These C=1, 94-fit-source models are not a causal comparison
against the old dev-selected 140-fit-source pilot.

For a withheld class U, no U row enters that context's binary fit, attribution
fit, either scaler, class centroid/covariance, confidence cutoff, genuine gate
or score-method selection. Fixed acoustic extraction may populate a shared
cache for all contexts, but no statistics/parameters are shared between their
learned heads. Poisoning excluded U rows is regression-tested to leave fitted
state and thresholds unchanged. U remains known in the separately defined
three-class reference context; that reference's choices are not transferred
to U's held-out context.

The three predeclared confidence scores, all oriented **higher known-like**, are:

1. **Distance:** negative squared Mahalanobis distance to the predicted class's
   centroid in attribution-standardized feature space. Fit class centroids
   and pooled within-class covariance on known synthetic model-fit rows only;
   regularize as `0.9*S_within + 0.1*I`, with denominator `N_known-K`.
2. **MSP:** maximum softmax of source logits, explicitly uncalibrated.
3. **Negative energy:** `-E` at fixed T=1, explicitly uncalibrated. The public
   energy helper itself retains its opposite, lower-known-like orientation.

In a symmetric two-class head, MSP and negative energy are monotone functions
of |m| and therefore largely redundant ranking candidates, not independent
evidence of unknownness. Three-class behavior need not have this equivalence.

Calibrate a binary gate from genuine calibration margins: the largest finite
`<=` boundary below the first disallowed genuine score, with at most **5%**
genuine calibration acceptance into the synthetic route. Move whole ties.
Independently choose each confidence `>=` cutoff on **all known synthetic
calibration scores**, allowing at most 5% (`retain95`) or 20% (`retain80`)
confidence rejection. Do not condition this calibration on correct labels or
on having passed the binary gate. The joint system is not guaranteed 95%/80%
known coverage; the binary gate can reject many known synthetic inputs.

Select the score method per context using **known/genuine outer-OOF predictions
only**, at the primary retain95 point. Require observed OOF genuine false source
attribution <=5% and nonzero known coverage; maximize correctly attributed known
fraction, then accepted-known accuracy, lower genuine false attribution, then
fixed score order. If none qualifies, choose **reject-all source labels**.
Retain80 and reference policies are descriptive tradeoffs, not extra selection
candidates. The 5% target is an engineering operating objective, not a change
to the scientific confound gate or its 0.40 floor.

All four choices, final models and thresholds are saved and hashed in
`selection_locked.json` before producing **any withheld-class OOF scores**
or dev scores. Dev features are decoded/loaded only after that lock. The
withheld class never supplies a threshold example for its own context. These
OOF estimates still share data with score-method selection and are exploratory,
not unbiased nested estimates of that final method-selection procedure.

## Decision interface and metric denominators

`detectors.source_verification` exposes numeric bundles and waveform-only
`predict_file`. Its output states are:

- `known_source`: the binary synthetic route and selected confidence cutoff
  both pass; a label from that context's permitted known classes is returned.
- `no_supported_synthetic_source`: the real-margin gate does not pass. This
  is **not** an authenticity certificate and is not reliable identification
  of genuine speech in the presence of unknown generators.
- `unknown_or_inconclusive`: source support fails after the synthetic route,
  or a reject-all source policy is active. This does not name or establish
  the identity of an unseen generator.

Outputs explicitly set `genuine_claim_supported=false`,
`unknown_generator_identity_supported=false` and `probabilities_calibrated=false`.
No source labels, filenames or true edit locations are needed at inference.
The original `PilotBundle` and its historical predictions remain unchanged;
the new two/three-class numeric schema has its own strict loader and checksums.

Each context saves standalone known-source confusion/F1, known coverage,
accepted-known accuracy, correct-known fraction including all rejections,
known false rejection, genuine false attribution, withheld rejection, rejection
state breakdown, per-vocoder results, score AUROCs and source variability.
AUROC uses explicitly labeled two-population ranking; missing populations give
null rather than a fabricated score. Rejection success is not unknown identity
recognition: an unknown vocoder falling into the real-like branch still has
no source label, and that branch is reported separately.

Also report **all-input source-label coverage** (all source labels / all inputs)
and **all-input source-label precision** (correct known labels / all source
labels). These include wrong labels on genuine and withheld inputs, while the
known-only accepted accuracy does not. Their fixed mixture is three known plus
one genuine recording per source in closed3, or two known, one genuine and one
withheld recording in a held-out context. This is not deployment prevalence
and not a replacement for the three separate task results. Zero accepted labels
give undefined accuracy/precision, not 100%.

Reference policies include unconditional known argmax, the same binary gate
without confidence rejection, and rejecting all source labels. Show both how
often labels are produced and their correctness; reject-all's perfect withheld
source rejection cannot be presented as a useful verifier.

Uncertainty uses 2,000 fixed-seed resamples of source groups with all conditions
together. Intervals condition on fitted scores/cutoffs and do not repeat model
fitting or score-method selection. Undefined bootstrap ratios are counted.
Overlapping classes across the three withheld trials do not create independent
generator replications. Twenty dev sources, one speaker and three source
pipelines provide limited evidence about future error rates.

## Actual results

Use the corrected sealed run
`runs/vocotrace-phase4b-verification-v2-20261105/`.
OOF uses 140 source groups; the exposed dev benchmark uses 20. Each context
contains four input conditions per source: 560 OOF or 80 dev recordings.
Across contexts the CSVs have 2,240 OOF and 320 dev prediction entries, but
repeat the same 560/80 input recordings, not new independent observations.

The locked primary methods are **negative energy** for closed3, **MSP** without
BigVGAN, **MSP** without Griffin-Lim, and **distance** without Vocos. No case
needed the reject-all fallback. These choices remain unchanged after all
withheld/dev results below; neither a better-looking unselected score nor a
secondary operating point is promoted afterward.

### A. Standalone known-source attribution

These matrices score all true known synthetic recordings, before binary routing
or abstention. Rows=true, columns=predicted; B=BigVGAN-112m, G=Griffin-Lim,
V=Vocos. This is explicitly not end-to-end performance.

| Context | Class order | OOF known macro-F1 | OOF confusion matrix | Dev known macro-F1 | Dev confusion matrix |
|---|---|---:|---|---:|---|
| closed3 | B,G,V | 0.748436 | `[[92,1,47],[0,133,7],[47,4,89]]` | 0.810875 | `[[11,0,9],[0,20,0],[2,0,18]]` |
| Without B | G,V | 0.964284 | `[[136,4],[6,134]]` | 1.000000 | `[[20,0],[0,20]]` |
| Without G | B,V | 0.642839 | `[[91,49],[51,89]]` | 0.657670 | `[[9,11],[2,18]]` |
| Without V | B,G | 0.989284 | `[[137,3],[0,140]]` | 1.000000 | `[[20,0],[0,20]]` |

Griffin-Lim is easy to distinguish from these two neural sources, whereas
BigVGAN and Vocos remain substantially confused. Two-way perfect dev results
involving Griffin-Lim are not evidence of general source recognition.

The unchanged historical model reproduces its old, already selected dev
AUROC **0.7016666667**, binary EER **0.3000000000** and synthetic-only attribution
macro-F1 **0.8832603711**. The new fixed-C=1 models use different fit/calibration
budgets, so their lower three-way F1 is not a bug in the original model or an
isolated causal effect of adding rejection. The historical eval figures in the
earlier section were not rescored.

### Known-source coverage, correctness and false rejection

Primary retain95 operating point; accuracy conditions on a source label returned
for a **true known** input. Correct fraction divides by every true known input,
including rejections and wrong labels. The latter is the end-to-end known-source
success quantity, not standalone classifier F1.

| Split / context | Known labels / known inputs | Correct / known labels | Correct fraction of all known | Known false rejection | Selective known macro-F1 |
|---|---:|---:|---:|---:|---:|
| OOF closed3 | 117/420 (27.86%) | 101/117 (86.32%) | 24.05% | 72.14% | 0.324549 |
| OOF without B | 121/280 (43.21%) | 119/121 (98.35%) | 42.50% | 56.79% | 0.520644 |
| OOF without G | 26/280 (9.29%) | 16/26 (61.54%) | 5.71% | 90.71% | 0.102397 |
| OOF without V | 81/280 (28.93%) | 79/81 (97.53%) | 28.21% | 71.07% | 0.384068 |
| Dev closed3 | 18/60 (30.00%) | 17/18 (94.44%) | 28.33% | 70.00% | 0.357843 |
| Dev without B | 21/40 (52.50%) | 21/21 (100%) | 52.50% | 47.50% | 0.604119 |
| Dev without G | 1/40 (2.50%) | 0/1 (0%) | 0% | 97.50% | 0.000000 |
| Dev without V | 11/40 (27.50%) | 11/11 (100%) | 27.50% | 72.50% | 0.354839 |

Selective known F1 counts abstentions as false negatives. It therefore does
not inherit the high accepted-subset accuracy by dropping most hard examples.

### B. Genuine false attribution

| Context | OOF genuine source labels | Dev genuine source labels |
|---|---:|---:|
| closed3 | 6/140 (4.29%) | 0/20 |
| Without B | 4/140 (2.86%) | 0/20 |
| Without G | 6/140 (4.29%) | 0/20 |
| Without V | 5/140 (3.57%) | 0/20 |

All selected OOF points meet the predefined **observed** <=5% criterion. The
three-way MSP candidate does not: its 8/140 genuine false labels (5.71%) make
it ineligible despite matching energy's 101 correct-known labels. No gate or
budget was relaxed to admit it. Unconditional argmax assigns a source to every
genuine recording and has 100% genuine false attribution by construction.

Zero dev errors is not proof of a <=5% future rate. The same twenty genuine
sources recur across contexts; this is not eighty independent genuine tests.
All final genuine **gate-only** calibration rates are 2/46 (4.35%); confidence
can reject further. These empirical cutoffs are not risk certificates under
new speakers, channels, processing or unknown generators.

### C. Held-out-vocoder non-attribution

| Withheld class / split | No source label / withheld inputs | Via no-supported-source branch | Via unknown/inconclusive branch | Incorrect known-source labels |
|---|---:|---:|---:|---:|
| B / OOF | 139/140 (99.29%) | 138 | 1 | 1 |
| G / OOF | 124/140 (88.57%) | 124 | 0 | 16 |
| V / OOF | 129/140 (92.14%) | 129 | 0 | 11 |
| B / dev | 19/20 (95%) | 19 | 0 | 1 |
| G / dev | 19/20 (95%) | 19 | 0 | 1 |
| V / dev | 19/20 (95%) | 19 | 0 | 1 |

The dev 95% headline is chiefly a **real-like/binary-gate rejection**, not a
successful unknown-class classifier. The same gate rejects 41/60 known dev
inputs in closed3, 17/40 without B, 39/40 without G and 29/40 without V.
Unknown-source identification and reliable genuine authentication are therefore
**unsupported categories**, even though returning no source often avoids a
wrong attribution. The interface deliberately does not call these recordings
authentic or name their unseen generator.

Known-versus-withheld confidence AUROC, higher score nominally known-like:

| Withheld class | OOF distance | OOF MSP / negative energy | Dev distance | Dev MSP / negative energy |
|---|---:|---:|---:|---:|
| B | 0.522194 | 0.382755 | 0.477500 | 0.286250 |
| G | 0.576607 | 0.378342 | 0.618750 | 0.061250 |
| V | 0.516964 | 0.715230 | 0.555000 | 0.713750 |

Scores below 0.5 are not automatically reversed after seeing U. The extreme
0.06125 MSP result without G means the withheld Griffin-Lim inputs tend to
look *more* source-confident than the known inputs, directly contradicting a
general maximum-confidence unknownness assumption. Distance is not a generally
strong alternative here either. The two-class MSP/energy AUROCs coincide as
expected from their monotonic relationship.

Binary real-versus-known AUROC is OOF/dev: closed3 0.717993/0.740833;
without B 0.830255/0.855000; without G 0.622653/0.645000;
without V 0.769668/0.800000. The poor known-neural-versus-genuine separation
without Griffin-Lim limits useful source-label coverage.

### All-input source-label precision

The following counts include false source labels on **both genuine and withheld
inputs**, in the declared four-condition-per-source mixture. They are not a new
overall classification accuracy and must not be transferred to deployment priors.

| Split / context | All returned source labels / all inputs | Correct labels / all returned labels |
|---|---:|---:|
| OOF closed3 | 123/560 (21.96%) | 101/123 (82.11%) |
| OOF without B | 126/560 (22.50%) | 119/126 (94.44%) |
| OOF without G | 48/560 (8.57%) | 16/48 (33.33%) |
| OOF without V | 97/560 (17.32%) | 79/97 (81.44%) |
| Dev closed3 | 18/80 (22.50%) | 17/18 (94.44%) |
| Dev without B | 22/80 (27.50%) | 21/22 (95.45%) |
| Dev without G | 2/80 (2.50%) | 0/2 (0%) |
| Dev without V | 12/80 (15.00%) | 11/12 (91.67%) |

The without-G case is unusable despite 95% withheld non-attribution and zero
genuine false attribution on dev: neither of its two returned source labels is
correct. Reporting only the known-only F1 or withheld rejection would conceal
this failure. Its locked method is retained for honest reporting, not promoted
as an operationally reliable verifier.

### Per-vocoder behavior and tradeoffs

Three-class known-source results (140 OOF or 20 dev true inputs per vocoder):

| Split / class | Standalone precision | Standalone recall | Standalone F1 | Source-label coverage after abstention | Correct fraction after abstention |
|---|---:|---:|---:|---:|---:|
| OOF B | 0.661871 | 0.657143 | 0.659498 | 8.57% | 5.00% |
| OOF G | 0.963768 | 0.950000 | 0.956835 | 61.43% | 58.57% |
| OOF V | 0.622378 | 0.635714 | 0.628975 | 13.57% | 8.57% |
| Dev B | 0.846154 | 0.550000 | 0.666667 | 5.00% | 0% |
| Dev G | 1.000000 | 1.000000 | 1.000000 | 70.00% | 70.00% |
| Dev V | 0.666667 | 0.900000 | 0.765957 | 15.00% | 15.00% |

Fourteen of the seventeen correct closed3 dev source labels are Griffin-Lim.
The only accepted true BigVGAN recording is mislabeled. High aggregate accepted
precision does not establish useful coverage for BigVGAN or Vocos.

In held-out dev trials, the known source-label coverage/correct fraction pairs
are: without B, G 90%/90% and V 15%/15%; without G, B 5%/0% and V 0%/0%;
without V, B 0%/0% and G 55%/55%. Full per-class and per-source records for all
OOF/dev policies remain in the metric JSONs; no unsupported class is dropped.

The fixed stricter retain80 point gives the following tradeoffs for the selected
method. It was not used to choose the score or revise a cutoff after U/dev:

| Split / context | Known coverage | Accepted-known accuracy | Genuine false attribution | Withheld rejection |
|---|---:|---:|---:|---:|
| OOF closed3 | 24.29% | 91.18% | 2.86% | N/A |
| OOF without B | 37.14% | 100% | 2.14% | 99.29% |
| OOF without G | 8.57% | 66.67% | 2.86% | 89.29% |
| OOF without V | 24.64% | 98.55% | 1.43% | 95.00% |
| Dev closed3 | 28.33% | 94.12% | 0/20 | N/A |
| Dev without B | 37.50% | 100% | 0/20 | 95.00% |
| Dev without G | 2.50% | 0% | 0/20 | 95.00% |
| Dev without V | 22.50% | 100% | 0/20 | 100% |

Without confidence rejection, closed3 OOF returns labels for 28.33% of known
inputs at 85.71% accepted-known accuracy and 6.43% genuine false attribution.
Selected energy reduces genuine false attribution to 4.29%, at 27.86% known
coverage and 86.32% accepted-known accuracy. On dev, binary-only returns 19 known
labels with 17 correct; energy returns 18 with the same 17 correct. Much of the
coverage loss is the binary gate, not the confidence method. Without G, the
primary MSP policy equals the binary-only decisions on OOF and dev.

Reject-all has 100% withheld rejection and 0% known coverage, with undefined
accepted precision. Conversely, unconditional argmax has 100% known coverage
but 100% genuine false attribution and 0% withheld rejection. Both endpoints
are saved so a superficially impressive rejection/accuracy number is visible
in its proper coverage context.

### Source variability and uncertainty

Conditional dev source-bootstrap 95% intervals:

| Context | Known coverage | All-input label coverage | All-input label precision |
|---|---|---|---|
| closed3 | [20.0%, 41.67%] | [15.0%, 31.25%] | [85.0%, 100%] |
| Without B | [42.5%, 62.5%] | [21.25%, 33.78%] | [87.5%, 100%] |
| Without G | [0%, 7.5%] | [0%, 7.5%] | [0%, 0%] among defined replicates |
| Without V | [17.5%, 37.5%] | [8.75%, 21.25%] | [76.47%, 100%] |

Without G, 696/2,000 bootstrap replicates produce no source labels; accepted
accuracy/precision is undefined in those replicates. The remaining all-zero
interval is not evidence of certainty about future error. Genuine-dev intervals
are [0,0] because bootstrap cannot invent unobserved errors: even zero errors
among twenty independent genuine observations has a one-sided 95% binomial
upper bound of 13.91%. The single-speaker dependence further limits that
illustration's relevance to deployment. All three withheld-dev rejection
intervals are [85%,100%], conditional on these fitted systems and sources.

Per-source correct-known fraction has median 1/3 in closed3 (OOF and dev),
with 5th percentile zero and 95th percentile 2/3. Without G its dev value is
zero for every source. Without B and without V the dev medians are 1/2, with
zero lower-tail sources. Detailed source counts expose heterogeneity hidden
by aggregate precision. There is no speaker/corpus uncertainty estimate and
no new frozen-eval uncertainty calculation.

## Reproducibility, corrections and checks

The first cold run `runs/vocotrace-phase4b-verification-20261105/` is preserved
sealed. A code review identified that known-only accepted accuracy omitted wrong
source labels on genuine/withheld inputs; the reporting correction was recorded
in `metric_reporting_correction_plan.json` before empirical metric inspection.
The corrected v2 run adds all-input coverage/precision and their source intervals.
It changes no model, confidence candidate, calibration or selection rule.

The v2 replay explicitly reuses the first run's feature cache only after checking
its complete inventory, dataset identity, feature configuration and six unchanged
feature-processing source dependencies. Dev feature arrays are loaded after
the new lock. No original WAV or old run is overwritten. The independent audit
finds **all 48 bundle files byte-identical**, all numeric arrays exact, all eight
score/prediction CSVs byte-identical and every original metric field unchanged.
Only the additional reporting fields differ. This is deterministic replay,
not an expanded search or dev-driven optimization.

Each run performs **32 lightweight logistic fits** (16 binary/attribution pairs);
both passes together perform 64, plus unrelated synthetic test fixtures. The
cold driver took **299.19 s** and the cached replay **16.69 s**, with one BLAS
thread and tests sharing the machine. These include integrity/scoring/output
work and are not isolated throughput benchmarks. The corrected output contains
**142 files / 4,595,860 bytes**, no new WAV dataset, GPU work, installation or
model download.

Saved artifacts include the fixed specification, original frozen identities,
train/dev feature rows and arrays, exact source-fold memberships, fit/calibration
classes and groups for all sixteen numeric bundles, per-fold/final thresholds,
selection comparisons and lock, all scores/decisions, 72 metric files, historical
aggregate audit and dev replay, label-free inference check, validation and
complete SHA-256 inventory. Supplementary independent scripts, logs and receipts
are in `runs/vocotrace-phase4b-audit-20261105/`.

Cold reproduction from repository PowerShell, using a fresh output name:

```powershell
& '.venv/Scripts/python.exe' -B -X utf8 -m experiments.vocotrace_verification `
  --manifest runs/onboarding/v0-kaggle-input/manifest.csv `
  --baseline-dir runs/forensic-pilot-step3-20261012/v0-final `
  --out-dir runs/vocotrace-phase4b-cold-replay
```

The actual corrected replay command was:

```powershell
& '.venv/Scripts/python.exe' -B -X utf8 -m experiments.vocotrace_verification `
  --manifest runs/onboarding/v0-kaggle-input/manifest.csv `
  --baseline-dir runs/forensic-pilot-step3-20261012/v0-final `
  --feature-cache runs/vocotrace-phase4b-verification-20261105 `
  --out-dir runs/vocotrace-phase4b-verification-v2-20261105
```

The new optional cache is explicit; source/configuration mismatch, corruption,
unsafe links, row-order mismatch and reused output/staging paths are rejected.
Cold reproduction remains available. Inference needs only a trusted bundle,
supported WAV and existing work directory:

```python
from detectors.source_verification import predict_file

result = predict_file(
    "runs/vocotrace-phase4b-verification-v2-20261105/closed3/bundle",
    "supported_archive_input.wav",
    work_dir="runs",
)
```

The strict numeric loader uses no pickle and verifies exact file inventory,
hashes, class mapping, positive scales, finite parameters, positive-definite
support precision, environment and implementation identities. Hashes establish
integrity, not authenticity. An unselected fold bundle requires an explicitly
declared diagnostic score; the final bundle has a locked default.

Focused pytest: **32 passed**, 34.36 s after the reporting/cache regression
extension. Coverage includes invalid/extreme energy values and old semantics,
whole calibration ties, two-class logits, analytic support distances, train-only
normalization, disjoint fit/calibration/held sources, poisoned excluded classes,
genuine-cap selection and reject-all, independently derived confusion/F1 and
all-input denominators, bundle integrity, three decision states, full orchestration
with eval-decoder traps, lock-before-dev/cache loads, exact cached replay,
immutable inputs and corrupt-cache refusal. A new fixture initially referenced
the first staging path during cache replay; it was corrected to check each
actual run's lock. No existing test is changed or weakened.

Final full pytest: **939 passed, 4 failed, 7 skipped, 1 xfailed**, 500.21 s.
The same two generated-notebook CRLF/LF comparisons and two Windows junction
`WinError 5` failures remain. Source/test Ruff passes; repository-wide Ruff
retains 11 existing generated-notebook findings. The earlier full-suite log is
also retained; the final `pytest-full-2` run covers the corrected code.

```powershell
& '.venv/Scripts/python.exe' -B -m pytest -o addopts='' -q -ra `
  tests/test_source_verification.py `
  --basetemp runs/vocotrace-phase4b-audit-20261105/pytest-focused-3
& '.venv/Scripts/python.exe' -B -m pytest -o addopts='' -q -ra `
  --basetemp runs/vocotrace-phase4b-audit-20261105/pytest-full-2
& '.venv/Scripts/python.exe' -B -m ruff check src tests
& '.venv/Scripts/python.exe' -B -m ruff check .
& '.venv/Scripts/python.exe' -B -X utf8 `
  runs/vocotrace-phase4b-audit-20261105/independent_verify.py
```

Use new basetemp/output names on rerun. The independent checker reproduces all
72 metric files' point calculations from the CSVs, known-only method selection,
all 16 models' saved predictions, scaler statistics and calibration cutoffs,
all source partition separations, current source/output digests and replay
equivalence. All **4,321 protected prior file paths/sizes/mtimes** and protected
non-WAV artifact hashes match; prior Phase 4A receipt files are unchanged too.
Original v0, partial datasets/pilots/upload folder, historical models and previous
experiment outputs remain unchanged. Eval-WAV preservation is deliberately an
inventory/stat check, not a forbidden new waveform hash/decode audit.

One new audit snapshot overlapped the authorized energy-helper edit. Its initial
receipt is retained as `before_initial.json`; the original source digest was
recovered from matching Git HEAD and the verified Phase 4A receipt, with the
correction explicitly recorded. This bookkeeping correction changes no model,
source dataset or experimental result. The user's pre-existing `.gitignore`
bytes remain unchanged.

Changed repository files:

- `src/detectors/attribution.py`: validated stable energy and corrected stub prose.
- `src/detectors/source_verification.py`: modular evidence/abstention interface,
  strict versioned numeric bundles and single-file inference.
- `src/detectors/verification_metrics.py`: separately scoped metrics, coverage,
  precision, source variability and conditional uncertainty.
- `src/experiments/vocotrace_verification.py`: bounded train-group benchmark,
  calibration/selection lock, provenance and verified optional feature reuse.
- `tests/test_source_verification.py`: CPU regressions and synthetic integration.
- This report and the README pointer.

All changes remain unstaged, with the index empty. No Kaggle or Git write
operation occurs. `CLAUDE.md`, all eighteen rules, scientific gates, checkpoint
pins, generation code, archived audio and frozen partitions are unchanged.

## Scientific decision and next milestone

The hybrid direction is supported **as a controlled research workflow**, with
some closed-set source separation and a reproducible abstention benchmark.
It is **not** established as robust attribution on arbitrary uploads. Useful
coverage is heavily Griffin-Lim-dependent, BigVGAN/Vocos remain hard to separate,
and high withheld rejection mostly arises from a gate that also misses known
synthetic speech. The without-G context fails even at tiny accepted coverage.
Calibrated authenticity probabilities, reliable genuine classification and
general unknown-generator recognition remain unsupported.

The archive is single-speaker and legacy/non-citable, with unresolved generation
provenance and current signal-admission/orientation issues. Shared low-band
processing removes known direct high-band cues but does not identify a causal
vocoder fingerprint or eliminate historical in-band processing, frontend,
content, crop/padding and channel shortcuts. Only three particular source
pipelines are represented, not independently varied generator families or
checkpoints. This phase tests no new gain/codec/resampling/partial-mixture
robustness conditions. No quality-versus-detectability claim follows.

Keep the original model and this benchmark as frozen references. The next
bounded model milestone should reproduce **one published whole-audio fingerprint
baseline** under its documented protocol, then use this harness for preregistered
genuine/known/withheld and controlled-processing tests. First resolve provenance
and data eligibility, and obtain distinct calibration and final unknown-source
cohorts before any broader claim. Any new representation needs a new explicit
feature/bundle protocol; do not silently replace this baseline. VocoPatch remains
a useful future mixture/control stress resource, not a solved localizer.

Defer a user-facing application until there is useful class-balanced coverage
at an independently justified false-attribution risk, including genuinely unseen
generators and channels. The present label-free interface supports offline
research inspection; code is ready for manual review/commit, models are not
ready for forensic or authenticity decisions on arbitrary uploads.
