# Evaluation correctness: EER and confound orientation

Step 2, 2026-10-12. Starting HEAD:
`84d101f30ed00efd89f08d9eeb6d02c62e6cc5b2`. The existing untracked
`onboarding_changes.patch` was present at the start and is untouched. This task
corrects one shared metric, adds a separate diagnostic and documents downstream
effects. No model, generation, dataset, checkpoint or invariant changes.

## Directional EER and the defect

The project convention remains **higher score means more bona fide**. EER is a
fraction, not a percentage. Correctly oriented perfect separation gives 0;
perfectly reversed separation gives 1. Ordinary detector EER is not minimized
over score directions. A reversed score convention must be corrected explicitly
in the adapter, not concealed in metric evaluation.

Previously, `detectors.eer.compute_eer` stably sorted concatenated real-then-spoof
samples and advanced its error rates after each individual sample. In a tied
group this created class-dependent intermediate decisions that no score threshold
could make. It also chose the closest observed FRR/FAR pair and averaged it,
rather than finding the linearly interpolated equal-error crossing. Four real and
four spoof scores all equal to zero therefore returned 1 instead of 0.5.

The public signature remains `compute_eer(bonafide_scores, spoof_scores)` returning
`(eer, threshold)`. Each class must supply a nonempty, finite, real, one-dimensional
score vector. NaN, infinity, complex/non-numeric values, scalar inputs, ragged
arrays and multidimensional arrays raise `ValueError`; batch/class dimensions
are no longer silently flattened. Different class lengths are valid: error rates
are normalized separately by their class counts. An empty class is a missing
class, not an EER of zero or chance.

## Corrected ROC calculation and threshold meaning

A deterministic decision accepts real when `score > threshold`. With an ascending
sweep, the initial threshold is minus infinity (accept everything). At each
distinct score, reject the **entire equal-score group**. The error rates are

```
FRR = rejected real / number of real
FAR = remaining spoof / number of spoof
```

The endpoints are `(FRR,FAR)=(0,1)` and `(1,0)`. Between successive attainable
points, `d = FRR - FAR` increases monotonically. At an exact `d=0` operating point,
return its error rate. Otherwise, for the bracketing points L and R:

```
w   = -d_L / (d_R - d_L)
EER = FRR_L + w * (FRR_R - FRR_L)
```

FAR interpolated with the same weight equals this value. Score differences and
threshold midpoints are unnecessary, avoiding overflow at finite float64 extremes;
the initial sentinel replaces the arbitrary `min_score - 1e-6`. FAR counts remaining
spoofs directly to avoid cancellation changing an exact rational crossing into an
apparent adjacent segment.

Interpolation mixes legitimate threshold decisions. It does not invent a
deterministic threshold that separates identically scored samples by label. For
constant scores, the ROC jumps from `(0,1)` to `(1,0)`: their midpoint is EER 0.5,
but no single deterministic threshold attains both error rates at 0.5.

The returned threshold is the score boundary at the **upper threshold end** of
the crossing segment, or the exact crossing point's boundary. It identifies that
transition, rather than promising that applying this threshold reproduces an
interpolated EER. For example, all-zero scores return `(0.5, 0.0)`; the `>` decision
at 0 rejects every sample. Deployment thresholds still require explicit dev-set
calibration for the intended decision policy. Consumers currently record this
threshold; they do not implement a deployment decision from it.

## Regression evidence

`tests/test_evaluation_correctness.py` covers these hand-derived cases:

| Real scores | Spoof scores | Expected directional EER |
|---|---|---|
| four zeros | four zeros | 0.5 |
| two sevens | five sevens | 0.5 |
| `[0,1,2]` | `[0,1,2]` | 0.5 |
| repeated 5 | repeated -5 | 0 |
| repeated -5 | repeated 5 | 1 |
| `[0,1,1]` | `[0,0,1]` | 1/3, attained at threshold 0 |
| `[0,1,1,2]` | `[0,0,1]` | 3/10, interpolated |
| `[1,2,3]` | `[0,1]` | 1/5, interpolated |
| `[0,3,5]` | `[1,2]` | 1/3, interpolated, no ties |

For the 3/10 case, the bracketing points are `(1/4,1/3)` and `(3/4,0)`; their
intersection weight is 1/10. For the 1/5 case, `(0,1/2)` to `(1/3,0)` crosses at
weight 3/5. These expected values were derived independently of the implementation.

Additional cases check all class-wise score permutations of the mixed-tie example,
sign reversal, float64 extremes, input immutability, explicit invalid-input errors,
bootstrap validation/reproducibility and matched/mismatched protocol consumers.
Twelve deterministic tied/unequal-class examples compare against the installed
scikit-learn `roc_curve(drop_intermediate=False)`, followed by interpolation of
FAR at `FRR-FAR=0`. The trusted ROC implementation independently constructs the
attainable grouped operating points; it is not a copy of the production algorithm.

Bootstrap uses the same corrected metric on every draw. Only its score-vector
validation was aligned with EER; the existing independent class-wise resampling,
seed, percentile interval and public interface are retained. Constant identical
scores give a degenerate `(0.5,0.5)` interval; perfect/reversed separation gives
`(0,0)`/`(1,1)`. Paired/grouped uncertainty remains a separate methodological review.

## Confound-probe orientation

All four current probes pass a raw physical feature directly into directional
EER; none learns or calibrates a bona-fide score orientation.

| Probe | Raw feature and possible reversed direction |
|---|---|
| `duration` | Total seconds. A condition could be longer or shorter; proper paired validation separately requires identical lengths. |
| `silence_duration` | Leading plus trailing quiet seconds at the real-derived threshold. A condition can have more or less silence. |
| `rms_level` | Broadband RMS amplitude. A condition can be louder or quieter; band-referenced normalization does not fix full-band RMS ordering. |
| `bandwidth_rolloff` | Mean spectral rolloff at 0.995 energy, archive only. A condition can have higher or lower rolloff; known design exceptions remain explicitly separate. |

The code implicitly treats a larger feature as more real by calling directional
EER without a sign transform. There is no scientific basis for assuming that
ordering always holds for these physical features. Current `_verdict` tests only
`eer < SANITY_EER_FLOOR`, with `SANITY_EER_FLOOR=0.40` and the existing bandwidth
`expected_by_design` exceptions.

After fixing ties, real zeros versus condition ones correctly has **directional
EER 1**. `_verdict` still returns `leaked=false`, although the feature separates
perfectly in the reverse direction. This is independent of the tie defect:
constant identical distributions previously also returned 1 incorrectly, but now
return 0.5; genuinely reversed separation must continue to return 1.

`probe_orientation_diagnostic(real_features, condition_features)` is a new,
separate helper in `experiments.sanity_checks`. It computes both score directions
with the **same** corrected `compute_eer` and returns `eer_forward`, `eer_reverse`
and `eer_separability = min(eer_forward, eer_reverse)`. It emits no gate verdict,
uses no new floor, and is not called by `run_all`, `_verdict`, the probes or Phase A.
It measures monotonic scalar separability, not every possible nonlinear feature
relationship, and must not replace ordinary detector EER.

Tests reproduce reversed separation through all four actual feature probes using
synthetic fixtures, as well as directly through `_verdict`. They assert that the
new diagnostic is not called by those probes and that the current one-sided rule,
strict floor comparison and design exceptions remain unchanged.

## Downstream and historical limitations

| Consumer/evidence | Effect of the correction |
|---|---|
| `detectors.protocols.score_condition` | Corrected EER/threshold and CI; protocol/tier/training labels preserved. Both Phase B detection drivers use this path. |
| `bootstrap_eer_ci` | Every draw is now tie-aware/interpolated; saved CI values may change. Sampling design was not rewritten. |
| Four sanity probes, `silence_diagnostic` | Corrected directional values flow through existing calculations. `_verdict` and `leaked_at_floor` retain their original lower-tail rule. |
| `experiments.analysis`, Spearman/contrast reporting | Reads EER/CI artifacts; regenerate dependent summaries if underlying measurements change. No automatic rewriting of saved CSVs. |
| Source-attribution utilities | No direct EER consumer today; folds, family labels and energy score are unaffected. Open-set metric implementation remains a stub. |
| `data/measurements/probes.json`, band-energy records | AUROC/energy evidence does not call this shared EER and is not numerically changed by this patch. |

**Gate policy is unchanged; identical numerical outcomes are not promised.**
Fixing the shared EER necessarily changes the numbers received by its consumers.
A corrected value can cross the unchanged floor. No publication-gate orientation
rule, threshold, calibration, generation logic or exception was changed in this
task. No historical dataset gate was rerun or dataset eligibility newly certified.
Interpolation can also change unique-score estimates, so revalidation is not
limited to populations containing ties.

Revalidation scope:

- `data/measurements/silence_threshold_sweep.csv`: quantized silence durations can
  tie, and all EER/`leaked_at_floor` rows need raw-feature re-evaluation. The saved
  pad-agreement failure is independent of EER and is not overturned by this fix.
- EER tables in `docs/measurements.md`, especially the RMS comparison discussion
  using distance from 0.500 while verdicts use only the lower tail. The historical
  0.630/0.680 RMS records are warning signs for direction, **not corrected results**.
  Do not derive revised verdicts by subtracting an old EER from one.
- Saved legacy v0 and Griffin-Lim iteration sanity CSVs, including both tiers,
  and their gate receipts. The locally inspected v0 table has 21 rows and no
  recorded EER 1 rows; that does not prove its partial ties were handled correctly.
- The saved base run's bandwidth EER 0.395 is near the 0.40 floor and needs
  re-evaluation from its actual features; do not infer that it now passes or fails.
- Any external/saved detector EER, threshold, bootstrap interval and downstream
  correlation obtained with the old algorithm. No completed detector research
  run was established in onboarding; future results require a corrected metric
  identity and benchmark reproduction.

Saved historical files were only inspected and were not overwritten. No corrected
empirical values are claimed here. Aggregate EER and feature means are insufficient
to reconstruct a grouped ROC; raw per-file scores/features, the exact population,
calibration, pairing, code and artifact identities are required. Legacy/iteration
provenance limitations also remain, independently of evaluation correctness.

## Proposed gate correction: separate approval required

After review, compute both orientations for each eligible scalar probe and use
`eer_separability < SANITY_EER_FLOOR` for leakage, retaining forward/reverse values
and existing `expected_by_design` exceptions. Keep the floor exactly 0.40, the
strict comparison, every real-derived silence calibration check, tier restriction,
measured exclusion and population/split rule. This preserves existing lower-tail
rejections and adds the symmetric upper-tail rejections; it cannot weaken the
current separability acceptance criterion. It is a policy change and is **not
activated by this task**.

Before activation:

1. Add gate-path tests for both perfectly separated directions, identical tied
   distributions, partial ties, exact 0.40 boundaries and existing design exceptions.
2. Test `run_all`, CLI failure/publication behavior, `.failed` staging and receipts,
   calibration disagreement/refusal, both tiers and archive-only bandwidth. Keep
   grouping/pairing and the ungated trim-pad-level field unchanged.
3. Re-evaluate null sampling behavior across the approved population sizes under
   the two-direction diagnostic. If false-leak rates raise concerns, document
   them for review; do not tune the acceptance floor to admit a desired result.
4. Recompute the historical populations above from authenticated bytes/raw features
   into **new** versioned artifacts, preserving old records and documenting old/new
   metric and verdict semantics. Missing raw evidence leaves the result unverified.
5. Obtain explicit approval for the implementation and any separate authoritative
   documentation change. All 18 `CLAUDE.md` invariants remain binding.

## Verification and next prerequisites

Baseline full suite, fresh workspace basetemp: **403 passed, 4 failed, 4 skipped,
1 xfailed**, 430.87 seconds, exit 1. Failures match onboarding: two CRLF/LF notebook
byte comparisons and two sandbox-denied directory junctions. Four file-symlink
cases skip; the ablation PCM_16 contract remains the expected failure.

Final focused selection: **115 passed**, 81.50 seconds, exit 0: all **82 new
regression cases** plus the existing EER, protocol-label, silence-diagnostic,
derived-threshold and committed-calibration tests.

Final full suite: **485 passed, 4 failed, 4 skipped, 1 xfailed**, 393.74 seconds,
exit 1 (494 collected). Failure identities match the baseline exactly; no new
failure or additional skip appeared. The initial focused run caught a boundary
rounding issue at an exact 1/3 crossing; direct remaining-spoof counts corrected
it before the final passing focused/full runs. No test was weakened or skipped.

Ruff on `src tests`: **passed**. Repository-wide Ruff: **the same 11 generated
notebook errors** (E401: 2, I001: 4, UP031: 2, E402: 3), exit 1. No notebook or
lint configuration change. `git diff --check` passes. AST comparison against HEAD
confirms every pre-existing sanity function/class is unchanged; the only added
definition is `probe_orientation_diagnostic`.

Commands use the existing environment and repository configuration:

```powershell
& '.venv\Scripts\python.exe' -m ruff check src tests
& '.venv\Scripts\python.exe' -m ruff check .
& '.venv\Scripts\python.exe' -m pytest -o addopts='' -q -ra --basetemp runs/evaluation-correctness-20261012/pytest-final
```

Focused selection: `tests/test_evaluation_correctness.py`,
`test_invariants.py::TestEER`, `test_invariants.py::TestINV14Protocols`, and
`test_phase_a_run.py::{TestSilenceDiagnostic,TestDerivedSilenceThreshold,
TestSilenceCalibrationArtifacts}`, with a separate fresh `pytest-focused-final`
basetemp. Logs/temp fixtures and `verification.json` are in ignored
`runs/evaluation-correctness-20261012/`. Do not reuse these basetemp directories:
pytest can delete their contents; choose a new path when rerunning.

Files changed: `src/detectors/eer.py`, `src/experiments/sanity_checks.py`,
`tests/test_evaluation_correctness.py`, this document and `docs/measurements.md`.
All modifications are unstaged. Existing `AGENTS.md`, `CLAUDE.md`, generated
notebook, historical measurements, manifests, audio and checkpoint pins remain
unchanged, as does the user's pre-existing untracked patch.

Before binary training: separately review/activate the orientation gate, establish
dataset-admission provenance and revalidate signal gates, preserve split grouping,
then approve shared features/model/persistence/evaluation implementation. Pretrained
scientific comparisons additionally require pinned checkpoints and reproduced
reference benchmarks. F0 and paired bootstrap methodology issues remain separate
tasks. No model implementation, Kaggle execution or Git write was performed.
