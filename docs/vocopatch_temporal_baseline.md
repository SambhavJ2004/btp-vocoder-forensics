# VocoPatch Phase 3A: first temporal localization baseline

2026-11-04, Asia/Calcutta. **Exploratory development only, non-citable.**
Starting manually created HEAD: `67c559062072e75b25960336be9fecff0a6ea25b`.
The implementation and local experiment below use working-tree code whose
digests are recorded separately. No final eval, Kaggle execution/publication,
large neural model, dataset regeneration or Git write is part of this task.

**Outcome: the linear baseline is a poor practical localizer.** Dev source-group
macro sample F1 is **0.1410**, with approximately **44.4% valid-sample false
positives on both real-only control types**. Every dev control recording has a
false alarm. This is a reproducible baseline result, not application readiness,
evidence of confound-free forensics, or a citable research finding.

## Data, independence and leakage checks

Reuse `runs/btpvf-partial-v1-20261104/` and the original frozen-source manifest
`runs/onboarding/v0-kaggle-input/manifest.csv`. Existing strict package schema
and inventories are checked; original manifest identity, output digests,
source/donor grouping, split inheritance and train/dev label semantics must
agree. Require all three mixed classes, null and real-to-real controls per
source. No utterance, split or source pairing is repaired or reassigned.

| Consumed split | Source groups | Recordings | Raw PCM samples | Duration (s) |
|---|---:|---:|---:|---:|
| Train | 123 | 615 | 96,082,850 | 4,357.498866 |
| Dev | 17 | 85 | 13,989,130 | 634.427664 |

Each source supplies five highly correlated recordings. Overlapping windows,
repeated real backgrounds and crops are not independent source utterances.
Train has 123 source groups; model selection/diagnostics have 17 dev groups.
Train figures are resubstitution diagnostics. Dev figures are optimistic
selection-set results after the search below, not untouched validation.

The existing 34-source/170-recording eval partition is **not decoded,
featurized, scored, selected on or evaluated**. Package hashes and frozen ID
metadata include eval for integrity only. The loader filters to train/dev
before using interval labels, and extraction has no eval branch. A CPU test
fails on any attempted eval waveform decode. All source groups and original
140/20/40 source assignments remain frozen; 123/17/34 is the eligible subset.
The legacy eval exposure disclosed in Phase 1 remains; it is not a pristine
blind test. No final research holdout claim follows from this run.

## Common preprocessing and feature definition

Require archive-rate mono PCM_16 WAV and read through sanctioned audio I/O.
Process a complete five-example source group at a time; the real null occupies
the reference slot. Apply the unchanged verified
`band_limit_comparison_set` to every member with stopband edge `LADDER_FMAX`
(8 kHz). Materialize temporary PCM_16 analysis WAVs, reread them and check the
decoded analysis band. Apply no additional gain, normalization, trimming,
resampling or condition-specific processing. All archive inputs stay unchanged.
The same per-waveform path works for single-file inference without a real
reference or class label. Synthetic-file and real-file names give identical
inference for identical audio in tests.

Common context length remains **4.0375 s / 89,027 archive samples**, using the
existing crop/repeat-padding helpers. Local acoustic frames lie within those
contexts. Context overlap of local-window-length minus one preserves a complete
global sliding grid; recordings are not reduced to their first context. Full
windows use only original samples. Inputs shorter than one local frame are
repeat-padded for inference and excluded from supervised windows. Empty WAVs
are rejected explicitly. Silent/short feature arrays remain finite. This
development temporal API is separate from clip-level `Detector.score`; it
preserves common context/processing across every condition and changes no
INV-11 helper, research detector adapter or CLAUDE.md rule.

The requested local acoustic window/hop search is a new Phase 3A development
implementation of the earlier planned windowing workflow. It does not change
the generation pipeline or claim admission to a matched/mismatched thesis
protocol. As a model trained on project audio it is a **matched development**
experiment; its protocol identifier is
`vocopatch_temporal_logistic_archive8k_v1`.

Each local frame has **31 acoustic features**:

- 24 log10 mean PSD bands with equal-width edges across [0,8 kHz).
- Log10 integrated in-band power; normalized centroid, spread and 85% rolloff;
  spectral flatness; log-spectral slope; zero-crossing rate.

Use a periodic Hann window, next-power-of-two FFT and one-sided PSD
`|FFT|^2 / (sr * sum(Hann^2))`, doubling non-DC/non-Nyquist bins. Integrate
in-band power with bin width `sr/n_fft`; log floor is `1e-12`. No bin at or
above 8 kHz enters the vector. No IDs, labels, paths, splits, vocoder identity,
edit coordinates, masks or recording durations are appended as features.
Metadata is used only for supervision, traceability and stratified reporting.

## Labels, IIR support and score masks

Raw sample labels remain the generator's complete half-open intervals:
0 real, 1 pure synthetic core, -1 transition/ignore. Both control types contain
no synthetic labels; their matched transition buffers remain ignored. The
real-to-real control's intentional content relocation is not alignment repair.

The existing zero-phase IIR has infinite support. A fixed **150 ms / 3,308
sample** effective guard is calibrated from its centered 131,072-sample impulse
response, independently of corpus labels. The measured absolute tail sum is
`2.945281774151175e-08`; a bounded input difference of 2 yields a measured-tail
bound of **0.00193022 PCM16 steps**, below the fixed 0.01-step tolerance.
This numerical calibration is **not** a proof of finite support, exact filtered
sample origin or immunity to quantizer-boundary changes. End effects/infinite
tails and residual ambiguity remain limitations. No filter gate is relaxed.

For supervision, dilate raw ignore regions and every raw class transition by
this guard; also ignore recording-edge guards. This includes hard splices with
no raw fade. A full window is label 1 only if **100%** of its guarded samples
are 1; label 0 only if 100% are 0. Any ignore, mixed origin or repeat padding
makes its supervised/window-metric label -1. It still receives an inference
score. Both real-only controls contribute eligible negative windows.

Sample localization metrics use one **fixed 4,544-sample / 206.077 ms guard**
for the entire search. It includes filter guard, maximum half-frame support,
maximum smoother-neighbor displacement and maximum half-hop projection cell.
Thus a candidate cannot improve selection merely by scoring a smaller or
different denominator. Raw annotation files are never rewritten. The guards
remove some raw pure-core samples from scoring; scores/intervals are still
produced on complete recordings. No truth mask is an inference input.

## Predeclared search and selected model

| Setting | Frame samples | Hop samples | FFT | Best dev source-macro sample F1 | Control sample FPR |
|---|---:|---:|---:|---:|---:|
| 32 ms / 8 ms | 706 | 176 | 1,024 | 0.1370 | 0.5561 |
| 64 ms / 16 ms | 1,411 | 353 | 2,048 | **0.1410** | **0.4445** |

For each frame setting fit C in `{0.01,0.1,1}`: six regularized logistic
regressions in total. Use existing `fit_head`, StandardScaler fitted on pure
**train windows only**, lbfgs, max_iter 2,000, tolerance `1e-6`, seed 20260907,
and class weights `N/(2*N_class)`. Each labeled window has equal base weight;
longer utterances contribute more training windows. There is no fit/refit on
dev and no learned attribution head. Limit BLAS to one thread for fitting and
saved predictions. Group-equal training weighting remains future work.

For each fit, try thresholds `{0.2,0.35,0.5,0.65,0.8}`, centered smoothing
widths `{1,3}` and merge gaps `{0,32}` ms: **120 dev operating candidates**.
Select maximum source-group macro sample F1 pooled across each source's five
recordings; tie-break by lower control sample FPR, lower C, frame/hop,
smoothing/merge simplicity, then higher threshold. No candidate uses eval.
The small advantage of the selected frame is descriptive, not a significance
claim or a controlled causal effect of window size alone.

Selected: **64/16 ms, C=1, threshold=0.5, centered three-window mean, merge gap
0 ms**. Scaler uses 229,490 labeled train windows. Actual weights are
0.5329465915 for real and 8.088038345 for synthetic. Positive margins mean
synthetic. Sigmoid outputs are **uncalibrated class-balanced logistic outputs**,
not calibrated population probabilities. The shared EER routine receives
negative margins so its higher-is-real convention is preserved; no direction
optimization or EER-interpolation boundary chooses the operating threshold.

Window centers are projected to midpoint cells covering the original recording;
thresholded contiguous cells form intervals, then permitted short gaps merge.
Postprocessing receives scores and geometry only. Original margins/probabilities
and smoothed probabilities are all retained. Ground-truth editing/core/buffer
timestamps and model-predicted timestamps have separate JSON fields.

## Actual window counts

| Setting and split | Real (0) | Synthetic (1) | Ignore (-1) | All windows |
|---|---:|---:|---:|---:|
| 32/8 train | 437,968 | 29,952 | 76,430 | 544,350 |
| 32/8 dev | 64,573 | 4,137 | 10,565 | 79,275 |
| **64/16 train** | **215,303** | **14,187** | **41,180** | **270,670** |
| **64/16 dev** | **31,796** | **1,959** | **5,650** | **39,405** |

Ignored windows are not forced into a binary class and are not fitted or scored
as real/synthetic windows. Sample metrics have their separate fixed valid mask.

## Observed performance

| Metric | Train resubstitution | Dev, selected on dev |
|---|---:|---:|
| Raw window AUROC, pooled | 0.6116 | 0.5885 |
| Raw directional window EER | 0.4231 | 0.4343 |
| Window precision / recall / F1 | 0.0821 / 0.6292 / 0.1452 | 0.0756 / 0.5850 / 0.1339 |
| Source-macro window AUROC / EER | 0.6117 / 0.4195 | 0.5888 / 0.4309 |
| Sample precision / recall / F1 | 0.0778 / 0.6305 / 0.1385 | 0.0714 / 0.5842 / 0.1272 |
| **Source-group macro sample F1** | **0.1506** | **0.1410** |
| Global valid-sample temporal IoU | 0.0744 | 0.0679 |
| Mean best-reference valid-mask IoU, misses zero | 0.4181 | 0.3609 |

Dev valid sample confusion is TP 373,031, FP 4,852,424, FN 265,540,
TN 6,168,615. The 11,659,610 scored dev samples still represent **17 source
groups**, not millions of independent experiments. Group-level results are
saved as well as pooled window/sample counts. There are no window-wise
confidence intervals or claims of statistical significance.

### Real-only control false alarms

| Dev control | Window FPR | Valid-sample FPR | Recordings with false alarms | Events / valid minute |
|---|---:|---:|---:|---:|
| Same-source null | 0.4453 | 0.4444 | 17/17 | 249.63 |
| Real-to-real splice | 0.4459 | 0.4446 | 17/17 | 254.17 |

These rates fail practical specificity. High null false alarms show that the
problem is not confined to introduced splice seams. Similar pooled control
rates do not establish absence of editing confounds: most samples share real
backgrounds, and the relocated real control is not content matched. Train
control sample FPR is 0.4713 null and 0.4631 real-to-real.

### Per-vocoder dev diagnostics

Each row evaluates that vocoder's mixed recordings, including their real
backgrounds. It does not duplicate either shared control once per vocoder.

| Mixed condition | Window AUROC | Window EER | Source-macro sample F1 | Best-reference IoU mean |
|---|---:|---:|---:|---:|
| BigVGAN-112m | 0.5808 | 0.4444 | 0.2050 | 0.3231 |
| Griffin-Lim | 0.6350 | 0.3951 | 0.2441 | 0.4572 |
| Vocos | 0.5567 | 0.4640 | 0.1904 | 0.3025 |

Control-only AUROC/EER are null because they have no positives. These model
EERs are not INV-12 scalar-confound-probe gate verdicts. The three mixed
conditions have identical dev real-background FP counts (926,443 samples),
reflecting their shared real speech; those counts are not independent evidence.

### Localization and boundary diagnostics

Of 51 dev pure-core reference events, 11 match at valid-mask IoU >=0.50 and
2 at >=0.75. At 0.50, event precision/recall/F1 are
0.00497/0.21569/0.00972, with 2,202 false events and 40 misses. Mean best IoU
can look less poor because it chooses the best among many spurious segments;
report it alongside false alarms, global IoU and event counts.

Onset/offset absolute errors are **0.2958/0.3011 s**, conditional on the 11
matches at IoU 0.50. Misses remain reported separately; these are not errors
over all reference events. Bounds refer to the raw pure synthetic core,
not physical fade onset.

After support masking, samples within 250 ms of raw edit boundaries have FPR
0.4913 and recall 0.5449; farther samples have FPR 0.4393 and recall 0.5895.
Near F1 is 0.3797 versus 0.1174 away, but near positive prevalence is also
much larger and the actual fade/adjacent support is excluded. This difference
does not establish a causal splice-boundary shortcut or its elimination.

## Predicted versus true examples

The deterministic example set takes the first sorted source in each consumed
split and all five classes. Full examples are under
`runs/vocopatch-phase3a-audit-20261104/examples.json`. Below is dev source
**LJ005-0165**. The edit is [5.9737,6.9737) s; the mixed pure core is
[5.9837,6.9637) s. Controls have no synthetic interval. The displayed prediction
is the **longest predicted interval**, chosen without ground truth; all other
predictions remain saved. They are model outputs, not editor timestamps.

| Class | True pure synthetic core (s) | Longest model interval (s) | Total model intervals | Best valid-mask IoU |
|---|---|---|---:|---:|
| BigVGAN | 5.9837–6.9637 | 5.1149–5.4030 | 39 | 0.1410 |
| Griffin-Lim | 5.9837–6.9637 | 5.1149–5.4030 | 40 | 0.3665 |
| Vocos | 5.9837–6.9637 | 5.1149–5.4030 | 39 | 0.3101 |
| Null | none | 5.1149–5.4030 | 41 | not applicable |
| Real-to-real | none | 5.1149–5.4030 | 41 | not applicable |

## Artifacts, reproducibility and verification

Run: `runs/vocopatch-phase3a-baseline-20261104/`.

- `config.json`, `selection.json`: full predeclared search, realized rules,
  all 120 dev candidates, selected operating parameters, window counts and
  train-only class/scaler diagnostics.
- `bundle/`: JSON metadata and four numeric scaler/linear arrays in NPZ with
  `allow_pickle=False` loading, shape/finiteness checks, dependency/processing
  compatibility and checksums. No pickled sklearn estimator is deserialized.
- `train/`, `dev/`: numeric window margins/probabilities/smoothed probabilities,
  labels and record offsets/IDs; per-record GT versus predicted intervals;
  complete aggregate, source-group, class and control metrics.
- `provenance.json`, `validation.json`, `artifacts.sha256.json`: dataset/source
  identities, existing HEAD plus actual code digests, Python/platform/library
  versions, seed, one-thread BLAS policy, exact model roundtrip and file hashes.

The immutable model run has no eval prediction output. Source data have not
been modified. Separate audit outputs are under
`runs/vocopatch-phase3a-audit-20261104/`, so checking the run does not alter its
sealed inventory. An independent saved-score audit reconstructs all train/dev
window labels, smoothed probabilities, sample intervals and confusion counts;
it agrees with source-macro F1 and independently verifies AUROC/EER with sklearn
ROC points. Ten single-file signal inferences (one source per split, all five
classes) agree with cached training-driver predictions. CLI prediction is also
exercised on a dev file without truth input. Synthetic end-to-end repeated
runs have identical complete inventories, and the real saved model reproduces
all cached feature-matrix margins exactly after numeric bundle loading.

Final `runs/vocopatch-phase3a-audit-20261104/verification.json` passes a
before/after SHA-256 comparison of **3,581 protected files**: original v0
(1,606), initial pilot (77), Phase 1 pilot (91), expanded dataset (901) and
prepared upload directory (906). All **870 generated WAVs** are unchanged.
Frozen eligible groups remain 123/17/34 and recordings remain 615/85/170.
The run's complete 14-file inventory plus its inventory receipt (15 files,
7,058,614 bytes) verifies, and every recorded implementation digest still
matches the current source. Standalone CLI scores and intervals agree with
the corresponding saved dev recording. HEAD is unchanged and the Git index
has no staged paths. These checks establish local integrity, not historical
generation provenance or final research eligibility.

The commands below record the paths used for this run, from repository
PowerShell. For reruns, choose fresh output and pytest basetemp names. Training
refuses existing output paths; pytest can delete an existing basetemp, so do
not reuse the audit directories below.

```powershell
& '.venv/Scripts/python.exe' -B -m pytest -o addopts='' -q -ra tests/test_temporal_localization.py --basetemp runs/vocopatch-phase3a-audit-20261104/pytest-focused-1
& '.venv/Scripts/python.exe' -B -m experiments.cli temporal train --dataset-dir runs/btpvf-partial-v1-20261104 --source-manifest runs/onboarding/v0-kaggle-input/manifest.csv --out-dir runs/vocopatch-phase3a-baseline-20261104 --setting 32:8 --setting 64:16 --c 0.01 --c 0.1 --c 1 --threshold 0.2 --threshold 0.35 --threshold 0.5 --threshold 0.65 --threshold 0.8 --smoothing 1 --smoothing 3 --merge-ms 0 --merge-ms 32 --seed 20260907
& '.venv/Scripts/python.exe' -B -m experiments.cli temporal predict --bundle runs/vocopatch-phase3a-baseline-20261104/bundle --wav runs/btpvf-partial-v1-20261104/audio/0c817611b39d7de466544d41.wav --work-dir runs/vocopatch-phase3a-audit-20261104/inference-work
& '.venv/Scripts/python.exe' -B -m pytest -o addopts='' -q -ra --basetemp runs/vocopatch-phase3a-audit-20261104/pytest-full-1
& '.venv/Scripts/python.exe' -B -m ruff check src tests
& '.venv/Scripts/python.exe' -B -m ruff check .
```

Focused tests: **38 passed**, 51.11 s. Full pytest: **789 passed, 4 failed,
7 skipped, 1 xfailed**, 296.31 s. Four failures match Phase 2 exactly: two
generated-notebook CRLF/LF comparisons and two Windows junction WinError 5
failures. Existing physical-link privilege skips and ablation expected failure
remain. Source/test Ruff passes; whole-repository Ruff retains the same 11
generated-notebook findings. No gate/test is weakened and no new failure appears.

## Limits and next decisions

This pipeline is numerically reproducible but this trained model has poor
specificity and localization. Further changes need a separately declared
development experiment, for example an explicit control-FPR operating constraint
or gain/content-robust feature ablations. The current search is not expanded
after seeing its poor results, and eval remains unused.

Common low-pass processing suppresses known above-band cues but does not prove
absence of in-band gain/processing or acoustic-content cues. Absolute power is
a feature; source loudness and composite drift can matter. The largest selected
standardized coefficients include the two upper PSD bands, spectral slope and
in-band power; correlated coefficients are not causal feature importance.
RMS-active target selection versus broadly sampled real negatives is another
possible shortcut. Boundary/control diagnostics investigate rather than remove
all these concerns.

The corpus is single speaker, the disjoint-donor eligibility policy selects
longer/active utterances, real-to-real negatives change content, and the fixed
edit duration/fades limit external validity. The 17 dev groups were reused for
model/threshold selection, and there is no source-group confidence interval.
Historical source/generation/alignment provenance and applicable research signal
gates are not newly established. Passing package/analysis integrity checks is
not an INV-12 admission receipt. Paired uncertainty/F0 methodology, high-band
membership rules and the prior gate-orientation limitations remain separate.

Changed files: new `src/detectors/temporal_features.py`, `temporal_metrics.py`,
`temporal_model.py`, `src/experiments/temporal_localization.py`,
`tests/test_temporal_localization.py`, this report, CLI additions and pointers in
README/the earlier model protocol. No scientific invariant, Phase A, dataset,
frozen split, checkpoint pin or Git history/index is changed. All work remains
unstaged; the pre-existing `.gitignore` edit is preserved. No upload, final eval,
large neural training or frontend is performed.
