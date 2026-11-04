# VocoPatch Phase 3B: failure analysis and controlled context comparisons

2026-11-04. **Exploratory development only, non-citable.** Starting manually
created HEAD is `67c559062072e75b25960336be9fecff0a6ea25b`; Phase 3A was still
uncommitted working-tree code at this task's start. All actual code digests and
environment versions are saved separately. Neither phase is a valid research
record-mode run. The dataset, scientific gates and all earlier artifacts stay
unchanged. There is no final eval, Kaggle operation or neural training.

**Outcome: no alternative won the predefined train-group comparison.** A64 was
retained. Its 5%-target operating point reduces dev control FPR to **2.555% null /
2.677% real-to-real**, but sensitivity falls to **2.985%**, group-macro F1 to
**0.04246** and global IoU to **0.02057**. This is a specificity/recall tradeoff,
not improved score ranking or a useful practical localizer. The original 0.5
point remains 0.14103 group F1, 0.06794 IoU and about 44.4% control FPR.

## Independent correctness audit

The source-packaged command `python -m experiments.temporal_review` reproduces
the Phase 3A audit into a fresh directory. It does not overwrite an old run.
The successful real audit is `runs/vocopatch-phase3b-correctness-v2-20261104/`;
the additional initial audit, test logs and final verification live under
`runs/vocopatch-phase3b-audit-20261104/`.

No Phase 3A correctness defect was found. On **700 train/dev recordings**:

- Source identities, frozen splits, complete groups and sample annotations agree
  with the existing manifest and pinned package identity.
- The existing independent scalar PCM16 splice oracle reconstructs all 700
  waveforms exactly, including untouched samples, donor cores and fades. Sample
  indices and timestamps agree. This checks declared source/sample pairing; it
  does not newly certify legacy upstream acoustic alignment or provenance.
- Independently constructed guards, full-window labels, midpoint projection,
  three-score smoothing, predicted intervals and sample confusion agree with
  saved outputs. AUROC and interpolated directional EER match sklearn ROC points.
- All four original numeric model arrays (train-only scaler mean/scale and
  classifier coefficients/intercept) reproduce **exactly** after refitting the
  original 64/16 ms, C=1 configuration. All 700 saved margin vectors reproduce
  exactly. There is no dev normalization or refit.
- Higher margins mean synthetic. Shared EER receives negative margins under its
  higher-is-real convention. The 0.5 cutoff is applied to uncalibrated,
  class-balanced logistic outputs; it is not a population posterior threshold.

Recomputed dev raw window AUROC **0.588464**, EER **0.434268**, source-macro sample
F1 **0.141026** and sample IoU **0.067935** match Phase 3A. Sample confusion is
TP 373,031, FP 4,852,424, FN 265,540, TN 6,168,615. Train resubstitution
source-macro F1 remains 0.150597.

The unchanged common low-pass transform is applied to all five conditions,
including real, and round-tripped through sanctioned PCM16 WAV I/O before
features. Its IIR has infinite support. The fixed 3,308-sample effective guard
has measured impulse-tail bound 0.00193022 PCM16 steps, not exact finite support
or immunity to quantizer/edge effects. The source implementation, filter gate,
4.0375-second shared contexts and crop/repeat-padding helpers are unchanged.

## Observed Phase 3A failure pattern

At threshold 0.5, null and real-to-real valid-sample FPRs are 0.444370 and
0.444637, respectively; every dev control recording has a false alarm. Null
recording FPR ranges from 0.3347 to 0.5945; real-to-real from 0.2888 to 0.6115.
The issue is widespread across real recordings, not one exceptional source.

About **73.0% of all false-positive samples are more than one second from an
edit boundary** (3,542,295 / 4,852,424). Near-boundary FPR is 0.4913, versus
0.4393 farther than 250 ms. The scoring mask removes actual fades and their
adjacent support, so this diagnostic cannot measure correctness inside those
excluded transitions. Similar control averages do not eliminate splice confounds.
Repeated real backgrounds also make many errors identical across vocoders.

Approximate diagnostic median smoothed scores are 0.4876 for null real,
0.4890 for real-to-real real, and 0.5124/0.5284/0.5016 for the synthetic portions
of BigVGAN/Griffin-Lim/Vocos. The quantile summaries use a documented deterministic
per-record stride sample; full sample histograms and all metric counts are exact.
These distributions overlap heavily. The balanced fitting prior is one half,
although only 14,187/229,490 = 6.18% of fitted Phase 3A windows are synthetic.
No posterior calibration claim follows from sigmoid scores.

Across dev, raw predicted intervals have median duration 80.0 ms and 95th
percentile 320.2 ms. Valid-real false-positive runs have median 80.0 ms and 95th
percentile 288.3 ms. Raw intervals include ignored samples; valid-real runs split
at ignored/positive samples and are a different statistic from event matching.

The fixed diagnostic cutoff 0.75 nearly removes control FPs (null 0.0151%,
real-to-real 0.0303%) but yields **zero synthetic sample recall for every
vocoder**. Cutoffs 0.9, 0.95 and 0.99 yield no positive samples. These are
descriptive probes of the saved dev outputs, not thresholds selected for Phase
3B. Weak ranking and score overlap limit what threshold shifts can achieve.
In-band energy/content/processing shortcuts remain possible causal explanations;
the diagnostics do not establish any single acoustic cause.

## Fixed design and train-only grouped selection

The initial plan was saved before fitting in `predeclared.json`, and the driver
requires it to equal its fixed plan. No search was expanded after seeing results.

| Candidate | Local context | Hop | Acoustic vector |
|---|---:|---:|---|
| A64 | 64 ms | 16 ms | Original 31 spectral features |
| B256 | 256 ms | 16 ms | Same 31 features at longer FFT support |
| C512 | 512 ms | 16 ms | Same 31 features at longer FFT support |
| Dmel256 | 256 ms | 16 ms | 24 log-mel means and 24 standard deviations |

A/B/C reuse the original extractor, with a separate explicit config extension
for 512 ms; the original schema and original model loader remain unchanged.
D uses 32 ms periodic-Hann subframes, 8 ms subframe hops with a final anchored
frame, 24 HTK-spaced triangular filters below `LADDER_FMAX`, normalized discrete
PSD filter weights, log10 floor 1e-12, and population standard deviation across
subframes. It has 48 features and adds no metadata. It shares B's outer support
and label geometry. All candidates retain the same 4.0375-second contexts,
common low-pass/PCM16 transform, complete coverage and repeat-padding policy.

Only waveform samples enter features. IDs, vocoder names, labels, split names,
edit locations and masks never enter the predictor. This satisfies data-access
isolation; the inherited zero-phase filter and centered smoothing use lookahead.
**This is offline inference, not a streaming-causal algorithm.**

Window supervision still requires 100% one class after the fixed 150 ms filter
guard; mixed, ignored or padded windows stay -1. Longer windows lose more pure
training/metric windows. Sample metrics deliberately keep the **same 4,544-sample
Phase 3A scoring guard for every candidate and operating point**. It is a
sample-truth mask, not a claim that a 512 ms feature is pure near an edit.
Larger contexts do not receive easier masks that conceal their boundary errors.

Predictions use integer sample coordinates but retain the 16 ms hop/context
resolution; integer indexing is not a claim of sample-perfect estimated boundaries.

Candidate window AUROC/EER use their respective pure windows, hence different
support sets. A further common-mask **sample-weighted AUROC/EER** compares
ranking on identical valid samples: each projected center cell contributes
its actual real/synthetic counts as weights. This is algebraically equivalent
to expanding every valid sample, without treating samples as independent trials.
Both raw margins and smoothed probabilities are reported.

The 123 train sources form three disjoint 41-source folds. Sort source IDs by
SHA256 of `20260907:temporal3b:<group_id>` and assign round-robin. All five
recordings of a source remain together. Each of 12 fold models fits its scaler
and balanced C=1 logistic regression only on its 82 training groups. Preserve
lbfgs, max_iter=2000, tolerance=1e-6 and one-thread BLAS. Every train recording
gets exactly one out-of-fold (OOF) margin vector. No new C, smoothing or merging
search is performed: three-window centered probability mean and zero gap merge
are fixed. Window-equal base weighting is retained, so longer sources still
contribute more training loss.

For each candidate, select minimum deterministic >= thresholds on the OOF
control scores at predefined **1%, 5%, 10% sample FPR caps for each control
type separately**. Count exact valid samples in midpoint cells, move entire
score ties together and use nextafter at the excluded boundary. A cutoff one
float above 1 explicitly permits the all-negative solution for saturated ties.
There is no recall guarantee. Constraints are empirical pooled OOF caps, not
per-source or future-data guarantees. Refitting the full train model can shift
its score scale relative to fold models.

Select the candidate by OOF source-macro sample F1 at the 5% cap, then lower
maximum control FPR, then shorter context and candidate name. Save and hash
selection before any new dev feature extraction. Fit four full-train models,
then report all four fixed candidates on the previously exposed 17 dev groups
at 0.5 and the three OOF-selected cutoffs. Dev never changes this selection.
OOF operating points are themselves tuned on OOF controls and candidate F1,
so OOF selection scores are optimistic, not nested-validation estimates.

## Runtime results

Train-only selection chose **A64** at the predefined 5% control-FPR point. The following dev results did not alter selection. All rows use 123 train source groups / 615 recordings and 17 dev groups / 85 recordings.


### Train-group OOF comparison at the 5% point


| Candidate | Window AUROC | Window EER | Common-sample AUROC | Common-sample EER | Group F1 | Null FPR | R2R FPR |
|---|---|---|---|---|---|---|---|
| A64 | 0.5889 | 0.4408 | 0.5890 | 0.4412 | 0.0815 | 4.999% | 4.808% |
| B256 | 0.5988 | 0.4352 | 0.5980 | 0.4359 | 0.0797 | 4.999% | 4.648% |
| C512 | 0.5859 | 0.4292 | 0.5885 | 0.4328 | 0.0603 | 4.999% | 4.793% |
| Dmel256 | 0.5611 | 0.4623 | 0.5605 | 0.4634 | 0.0506 | 4.999% | 4.824% |


### Descriptive dev comparison at the 5% point


| Candidate | Window AUROC | Window EER | Common-sample AUROC | Common-sample EER | Group P | Group R | Group F1 | Global IoU |
|---|---|---|---|---|---|---|---|---|
| A64 | 0.5885 | 0.4343 | 0.5893 | 0.4353 | 0.0968 | 0.0299 | 0.0425 | 0.0206 |
| B256 | 0.6017 | 0.4378 | 0.5916 | 0.4418 | 0.2100 | 0.0211 | 0.0365 | 0.0152 |
| C512 | 0.6675 | 0.3383 | 0.6263 | 0.4016 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Dmel256 | 0.6081 | 0.4215 | 0.6070 | 0.4286 | 0.0494 | 0.0111 | 0.0146 | 0.0085 |


### Dev operating-point tradeoffs


| Candidate | OOF target / reference | Cutoff | Sample P | Sensitivity | Group F1 | IoU | Dev null FPR | Dev R2R FPR |
|---|---|---|---|---|---|---|---|---|
| A64 | fixed_05 | 0.5000 | 0.0714 | 0.5842 | 0.1410 | 0.0679 | 44.437% | 44.464% |
| A64 | fpr_01 | 0.7017 | 0.0073 | 0.0006 | 0.0011 | 0.0005 | 0.399% | 0.475% |
| A64 | fpr_05 | 0.6412 | 0.0621 | 0.0299 | 0.0425 | 0.0206 | 2.555% | 2.677% |
| A64 | fpr_10 | 0.6092 | 0.0733 | 0.0820 | 0.0853 | 0.0402 | 6.089% | 6.070% |
| B256 | fixed_05 | 0.5000 | 0.0694 | 0.5195 | 0.1440 | 0.0652 | 40.783% | 40.492% |
| B256 | fpr_01 | 0.7763 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.257% | 0.363% |
| B256 | fpr_05 | 0.6989 | 0.0520 | 0.0211 | 0.0365 | 0.0152 | 2.055% | 2.498% |
| B256 | fpr_10 | 0.6531 | 0.0788 | 0.0899 | 0.0898 | 0.0439 | 6.224% | 6.279% |
| C512 | fixed_05 | 0.5000 | 0.0799 | 0.6057 | 0.1719 | 0.0760 | 41.524% | 39.908% |
| C512 | fpr_01 | 0.8565 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.333% | 0.651% |
| C512 | fpr_05 | 0.7573 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2.865% | 3.386% |
| C512 | fpr_10 | 0.6947 | 0.0676 | 0.0962 | 0.0850 | 0.0413 | 7.546% | 7.752% |
| Dmel256 | fixed_05 | 0.5000 | 0.0722 | 0.5638 | 0.1465 | 0.0684 | 42.940% | 42.062% |
| Dmel256 | fpr_01 | 0.7978 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.272% | 0.272% |
| Dmel256 | fpr_05 | 0.7141 | 0.0345 | 0.0111 | 0.0146 | 0.0085 | 1.786% | 1.695% |
| Dmel256 | fpr_10 | 0.6668 | 0.0922 | 0.0802 | 0.0864 | 0.0448 | 4.907% | 4.362% |

Targets name OOF constraints; the last two columns are **actual dev FPRs**, which need not equal or meet those targets. Within each candidate, raw window/common-sample AUROC and EER do not change with the decision cutoff.

Cutoffs in tables are rounded for reading. Bundles retain complete float64
cutoffs, including nextafter tie boundaries. Use those numeric values for
reproduction rather than the rounded table entries.

At the selected A64 5% point, **17/17 null and 16/17 real-to-real dev recordings
still contain a false alarm**. At its 1% point, 9/17 of each control type contain
alarms, while sensitivity is only 0.0553%. Lower sample FPR is not a comparable
reduction in recording-level false alarms. None of the tested low-FPR operating
points provides useful localization recall on these dev data.

C512 has a descriptively higher dev pure-window AUROC (0.6675), but its
common-sample AUROC is 0.6263 and its 5% point has zero synthetic recall. Its
pure-window positive set is much smaller (537 versus A64's 1,959 on dev).
It did not win train-group validation, so it is not promoted using these
previously exposed dev results. At cutoff 0.5 it reaches group F1 0.1719 while
still making roughly 40% control-sample errors. That is not practical success
or evidence that the longer representation robustly generalizes better.


### Per-vocoder dev performance at the 5% point


| Candidate | Mixed condition | Window AUROC | Window EER | Source F1 | Sensitivity | Sample IoU |
|---|---|---|---|---|---|---|
| A64 | bigvgan_112m | 0.5808 | 0.4444 | 0.0435 | 0.0299 | 0.0237 |
| A64 | griffin_lim | 0.6350 | 0.3951 | 0.0493 | 0.0332 | 0.0263 |
| A64 | vocos | 0.5567 | 0.4640 | 0.0425 | 0.0265 | 0.0211 |
| B256 | bigvgan_112m | 0.6023 | 0.4356 | 0.0439 | 0.0249 | 0.0204 |
| B256 | griffin_lim | 0.6291 | 0.4178 | 0.0428 | 0.0252 | 0.0207 |
| B256 | vocos | 0.5784 | 0.4555 | 0.0233 | 0.0133 | 0.0109 |
| C512 | bigvgan_112m | 0.6640 | 0.3464 | 0.0000 | 0.0000 | 0.0000 |
| C512 | griffin_lim | 0.6795 | 0.3240 | 0.0000 | 0.0000 | 0.0000 |
| C512 | vocos | 0.6613 | 0.3408 | 0.0000 | 0.0000 | 0.0000 |
| Dmel256 | bigvgan_112m | 0.6079 | 0.4244 | 0.0141 | 0.0100 | 0.0084 |
| Dmel256 | griffin_lim | 0.6097 | 0.4175 | 0.0192 | 0.0133 | 0.0112 |
| Dmel256 | vocos | 0.6136 | 0.4200 | 0.0152 | 0.0100 | 0.0085 |

These rows use each mixed condition and its real background; controls are not duplicated per vocoder. Repeated backgrounds remain correlated.


### Dev boundaries and source variability at the 5% point


| Candidate | Near FPR | Away FPR | Near R | Away R | Source F1 min | Median | P95 | Max |
|---|---|---|---|---|---|---|---|---|
| A64 | 5.005% | 2.568% | 0.0223 | 0.0309 | 0.0000 | 0.0164 | 0.1339 | 0.3665 |
| B256 | 1.576% | 2.242% | 0.0206 | 0.0212 | 0.0000 | 0.0000 | 0.2224 | 0.2451 |
| C512 | 5.796% | 3.091% | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Dmel256 | 1.982% | 1.790% | 0.0294 | 0.0086 | 0.0000 | 0.0000 | 0.0705 | 0.1520 |

Near means within 250 ms of an edit boundary, after the fixed support mask. These strata have different positive prevalence; comparisons do not identify a causal boundary shortcut.


### Selected-model false-alarm intervals on dev


| Point | Null recording FA | R2R recording FA | Null events/min | R2R events/min | Valid-real FP runs | Median s | P95 s |
|---|---|---|---|---|---|---|---|
| fixed_05 | 1.0000 | 1.0000 | 249.6310 | 254.1697 | 2058 | 0.0800 | 0.2883 |
| fpr_01 | 0.5294 | 0.5294 | 7.3755 | 8.5101 | 67 | 0.0320 | 0.0480 |
| fpr_05 | 1.0000 | 0.9412 | 36.8773 | 36.3100 | 303 | 0.0320 | 0.0961 |
| fpr_10 | 1.0000 | 1.0000 | 76.5913 | 76.0240 | 626 | 0.0320 | 0.1139 |


### Actual pure-window membership


| Candidate | Split | Real | Synthetic | Ignore | Total |
|---|---|---|---|---|---|
| A64 | train | 215303 | 14187 | 41180 | 270670 |
| A64 | dev | 31796 | 1959 | 5650 | 39405 |
| B256 | train | 198533 | 9762 | 55000 | 263295 |
| B256 | dev | 29475 | 1350 | 7560 | 38385 |
| C512 | train | 177449 | 3861 | 72155 | 253465 |
| C512 | dev | 26483 | 537 | 10005 | 37025 |
| Dmel256 | train | 198533 | 9762 | 55000 | 263295 |
| Dmel256 | dev | 29475 | 1350 | 7560 | 38385 |


### Observed computation and storage


| Candidate | Dev features s | 3 fold fits s | Full fit s | Dev linear head s | Train matrix MiB | Dev matrix MiB |
|---|---|---|---|---|---|---|
| A64 | 5.6945 | 8.2441 | 4.1965 | 0.0184 | 64.0165 | 9.3197 |
| B256 | 26.1668 | 7.6077 | 3.9071 | 0.0929 | 62.2722 | 9.0785 |
| C512 | 54.7120 | 6.0103 | 3.2042 | 0.0167 | 59.9473 | 8.7568 |
| Dmel256 | 48.4131 | 12.2216 | 5.7005 | 0.0234 | 96.4215 | 14.0570 |

Shared fresh dev decode/filter/PCM16 processing: **18.333 s**. Train feature extraction was reused and has no per-candidate extraction timing in the completed receipt. All 16 fits use one-thread BLAS. Timings overlap other checks and are not isolated benchmarks.

The sealed run contains **171 files**, **343.081 MiB**, and no WAV files. Feature caches dominate storage; numeric model bundles are much smaller.

## Cache-reader defect caught during Phase 3B

The first new run completed its train feature pass, then was interrupted before
selection/dev extraction. The new cache reader accessed compressed NPZ members
inside its per-record loop. NumPy inflates a member on every access, so the
entire feature matrix was repeatedly decompressed. This was a Phase 3B I/O
performance defect, not a Phase 3A mapping, feature or model correctness defect.
Both new readers now materialize each compressed array once before slicing.
An access-count regression catches repeated decompression without a flaky time
limit. The interrupted `.staging` directory is retained and marked interrupted;
its incomplete run supplies no empirical model result.

The fresh v2 run reuses only its completed train feature caches. Reuse requires
matching data/config/environment/dependency identities and archived driver bytes;
the extraction/writer syntax trees must be unchanged. Cache hashes, exact frozen
row membership, dimensions, finiteness, window grids and reconstructed pure
labels are checked. New copies have identical complete cache hashes. No fitted
model, threshold or dev feature is reused. The synthetic replay test compares
full extraction against cache reuse and requires identical selection, numeric
bundles and OOF/dev score files. A fresh uncached run remains supported.

The interrupted run did not reach `costs.json`, so its per-candidate train
extraction timings are unavailable. Report fresh dev extraction timings and
matrix sizes; cache-read time is not extraction time. Concurrent tests also
make observed runtimes unsuitable as isolated speed benchmarks.

## Reproduction and output contract

Use repository PowerShell and the existing environment. These commands record
the actual paths used. Choose fresh output and pytest basetemp paths for reruns;
pytest may delete an existing basetemp. Neither command resumes or overwrites
an existing experiment directory. For a full extraction rerun, omit the two
cache arguments and use a fresh output path; this reproduces the same numeric
inputs rather than depending on the interrupted attempt.

```powershell
& '.venv/Scripts/python.exe' -B -m experiments.temporal_review --dataset-dir runs/btpvf-partial-v1-20261104 --source-manifest runs/onboarding/v0-kaggle-input/manifest.csv --baseline-dir runs/vocopatch-phase3a-baseline-20261104 --out-dir runs/vocopatch-phase3b-correctness-v2-20261104 --protected-dir runs/vocopatch-phase1-v2-20261103 --protected-dir runs/vocopatch-pilot-v0-20261027 --protected-dir runs/btpvf-partial-v1-kaggle-upload-20261104
& '.venv/Scripts/python.exe' -B -m experiments.temporal_ablation --dataset-dir runs/btpvf-partial-v1-20261104 --source-manifest runs/onboarding/v0-kaggle-input/manifest.csv --baseline-dir runs/vocopatch-phase3a-baseline-20261104 --audit-dir runs/vocopatch-phase3b-correctness-v2-20261104 --out-dir runs/vocopatch-phase3b-ablation-v2-20261104 --train-cache-run runs/vocopatch-phase3b-ablation-20261104.staging --cache-source-snapshot runs/vocopatch-phase3b-audit-20261104/aborted-driver.py
& '.venv/Scripts/python.exe' -B -m pytest -o addopts='' -q -ra tests/test_temporal_ablation.py --basetemp runs/vocopatch-phase3b-audit-20261104/pytest-focused-5
& '.venv/Scripts/python.exe' -B -m pytest -o addopts='' -q -ra --basetemp runs/vocopatch-phase3b-audit-20261104/pytest-full-2
& '.venv/Scripts/python.exe' -B -m ruff check src tests
& '.venv/Scripts/python.exe' -B -m ruff check .
```

The run stores fixed configuration, fold assignments, input/code/environment
provenance, the locked selection, all 16 numeric bundles, feature caches, OOF
and dev window scores/labels, and per-operating-point metrics and predicted
intervals. Original scores remain available. GT editing/core/buffer fields
remain distinct from model timestamps. Bundle loading uses numeric arrays and
`allow_pickle=False`, checksums, finite shapes/scales and implementation/runtime
compatibility. `predict_context_file` needs one WAV and an operating-point name,
with no reference, labels or vocoder metadata. Timings are observations; their
bytes are not claimed deterministic. Numeric state and scores are deterministic
under the recorded implementation/environment and single-thread policy.

Eval WAVs are neither hashed nor decoded in Phase 3B. Shared manifest/label
JSON containers are integrity checked and parsed, but filtering precedes any
use of eval annotation records. The loader binds the existing Phase 3A full
package identity, checks the complete path inventory and unsafe links, hashes
all current consumed train/dev WAVs and metadata, and discloses skipped WAVs.
This scopes new access; it does not change or weaken the full dataset validator.
The legacy eval partition was previously exposed and is not a pristine test.

## Tests, preservation and remaining limitations

Final focused tests: **50 passed**, 144.37 s. The additional cache regression
requires one decompression per array, and cached replay requires identical
selection, numeric bundles and OOF/dev prediction files. Other coverage includes
longer-window geometry, local waveform-only access, silence/empty/short inputs,
fixed config rejection, seeded group-disjoint folds, tied weighted cutoff
minimality, both control caps, saturated all-negative decisions, exact expanded
sample ROC agreement, source-only normalization, eval byte isolation, dev-lock
ordering, safe bundle loading and standalone inference.

Final full pytest: **839 passed, 4 failed, 7 skipped, 1 xfailed**, 741.73 s.
Failure IDs exactly match Phase 3A: two generated-notebook CRLF/LF comparisons
and two Windows junction WinError 5 failures. The existing unsafe-link privilege
skips and unimplemented ablation expected failure remain. Ruff **passes for
`src tests`**; whole-repository Ruff retains the same **11 generated-notebook
findings**. No tests or scientific gates were weakened to obtain these results.

`runs/vocopatch-phase3b-audit-20261104/verification.json` passes an independent
reconstruction of **all 2,800 candidate/recording predictions across the 700
train/dev recordings**, at all four operating points. It verifies the pure
window labels, midpoint cells, smoothing, sample confusions, source-macro F1,
saved intervals and exact OOF control-cap attainment. Source IDs/folds agree;
four standalone dev-file inferences agree with the saved predictions. All 16
bundles passed numeric roundtrip checks in the driver. The selected parameters
and complete 171-file run inventory remain sealed and unchanged.

Preservation checks retain every protected file's inventory, size and mtime in
v0 (1,606 files), initial pilot (77), Phase 1 pilot (91), expanded dataset (901),
prepared upload folder (906), and the Phase 3A model run (15). All consumed
original-source and generated-output hashes match, as does the complete Phase
3A model-run SHA inventory and every recorded Phase 3A/3B implementation digest.
The copied train feature caches have identical complete hashes to their inputs.
Eval WAV integrity is scoped to recorded inventory/size/mtime and previously
pinned metadata; **no fresh eval WAV byte hash is claimed**. The dataset's frozen
eligible partitions remain 123/17/34; train/dev model inputs remain 615/85.
No dataset, historical validation, earlier pilot or Phase 3A result is rewritten.
Git HEAD is unchanged, the index has no staged paths, and all changes remain
unstaged. The final verification receipt records complete working-tree status.

## Interpretation and next model decision

The main observed causes are weak ranking in this acoustic representation and
an unsuitable 0.5 decision point under class balancing and low synthetic
prevalence. Widespread null errors and substantial errors far from edits rule
out splice seams as the sole explanation. They do not establish absence of
seam shortcuts or identify a specific causal acoustic confound. There is no
evidence of a Phase 3A label/index/normalization/orientation bug.

The retained A64 model has **exactly the original fitted arrays and score
ranking**. Its lower-FPR result is entirely an operating-point change. The
longer/log-mel alternatives fail the predefined primary train-group comparison;
they are retained rather than tuned further. The small A64/B256 OOF F1 gap is
not a statistically established difference. Dev's better C512 ranking is a
descriptive, support-dependent observation, not a new model-selection license.

**A neural model is not yet established as the appropriate next remedy.** I
would first prioritize training-group-only speech-activity/gain diagnostics,
real hard-negative analysis and threshold stability. The fixed-statistics
log-mel result does not test an uncollapsed temporal spectrogram model, so a
small, predeclared neural capacity comparison could be a subsequent exploratory
task after those checks. It would need the same source folds, both controls,
false-positive targets and dev-exposure disclosure. There is no evidence here
supporting a large model, deployment, final evaluation or a promised neural
improvement. No neural model was trained in Phase 3B.

No research admission or provenance deficiency is resolved by a better local
metric. The corpus is single speaker and only three known vocoders. Disjoint
donor eligibility, fixed edit durations, content-relocating real controls,
in-band gain/content cues and prior dev exposure limit interpretation. Boundary
masks exclude hard transition evidence. Source-group variability is reported
without pretending windows are independent or inventing confidence intervals.
No paired-bootstrap/F0 methodology or INV-12 gate-orientation change is included.

Changed in Phase 3B: new `src/detectors/temporal_context.py`,
`src/detectors/temporal_operating.py`, `src/experiments/temporal_review.py`,
`src/experiments/temporal_ablation.py`, `tests/test_temporal_ablation.py`, this
report and its README pointer. Phase 3A source/tests/report and the earlier
protocol edits remain as they were. The pre-existing `.gitignore` edit is
preserved. All changes remain unstaged; every Git operation was read-only.
