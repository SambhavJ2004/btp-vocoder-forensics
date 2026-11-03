# VocoPatch Phase 1 scientific review and control improvements

2026-11-03. Starting HEAD: `bd43850198371bb3d87ff2fec683297ab69b70f7`.
The original generator, tests, guide and README/CLI edits were already unstaged
at the start of this review. The pre-existing `.gitignore` edit is untouched.
All artifacts remain **development-only and non-citable**. No publishing,
detector training, original-v0 modification, invariant change or Git write occurs.

## Independent audit before source edits

Read `AGENTS.md`, applicable `CLAUDE.md` rules, the original pilot guide, all
generator/validator/demo code and all 97 original test cases. Before editing,
capture the generator source, all **1,606 original dataset files**, and all
**77 initial-pilot files** by SHA-256. Reproduce the defects below on copies,
never on the original dataset or pilot. Evidence is under
`runs/vocopatch-review-20261103/`: `prechange_audit.json`,
`audit_before_changes.md`, `baseline_identities.json` and `before/`.

| Area | Audit finding and action |
|---|---|
| Real/vocoder pairing | Existing manifest/table gates, same utterance IDs, mono/rate/count checks and recorded alignment evidence are consistent. No silent alignment repair is present. Legacy alignment/generation provenance is still not authenticated. |
| Sample replacement | The existing `[start,end)` replacement and untouched outside samples are correct for sanctioned decoded PCM_16 input. No off-by-one defect was established. |
| Crossfade | Entry weights `(k+1)/(F+1)` and reversed exit weights stay inside the edit; the pure donor core is `[start+F,end-F)`. Strictly interior fade weights and ignored samples agree. |
| Labels | Original mixed core label 1, outside label 0, and fade label -1 are consistent. Null controls contain only real audio with a mirrored ignore policy, not synthetic labels. |
| Reconstruction | The original checker called the same `splice` and writer as generation and required temporary writes beside the package. Replace this shared implementation check with an independent scalar fade/PCM oracle; validation now writes no files. |
| Duration metadata | **Confirmed defect:** `abs(actual_duration - NaN) > tolerance` is false, so a non-finite recorded duration passed. Reject nonnumeric, non-finite and nonpositive duration records before checking tolerance. |
| Inventory | **Confirmed defect:** excluding by basename hid any nested file called `artifacts.sha256.json`. A copied package containing such an unlisted file still passed. Exclude only the root receipt, include nested names, and reject links/escaped paths. |
| Provenance claims | **Confirmed defect:** contradictory `processing` and `phase_a_changed` claims passed after updating an ordinary receipt hash. Enforce the actual processing, time-to-sample rounding, Phase A/gate status and non-admission claims with strict types. Self-hashes remain integrity checks, not authentication. |
| Split coverage | Global selection reused `GLOBAL_SEED`, the seed used to freeze split membership. The initial selected ranks are all train. Preserve this legacy recipe; add opt-in explicit counts within each existing split with separately namespaced RNGs. Do not relabel utterances. |
| Metadata leakage | No predictor consumes metadata here. Formats/rates are uniform and primary filenames are opaque hashes. Source/class/index/split/ignore-mask metadata are labels/provenance and must not become classifier inputs. |
| Regeneration | Fresh output/staging guards, preserved failure diagnostics and post-validation publication remain. Old source files and the first pilot are never overwritten. |

These findings do **not** establish that the initial pilot has incorrect samples
or labels. The confirmed defects concern rejection/record validation, and the
control/coverage changes extend its scientific usefulness. Existing four
platform/notebook test failures are tracked separately, without weakening tests.

## Null controls versus real-to-real controls

The existing **null control** calls the same splice/encode pipeline with real as
both reference and donor at identical indices. Difference-form interpolation
must be an exact decoded-sample identity. This checks processing symmetry,
indexing, encoding, accidental gain/filter/length changes, and label bookkeeping.
It cannot create a new acoustic seam, measure real-to-synthetic boundary
mismatch, or demonstrate that editing artifacts are absent.

The opt-in **real-to-real control** uses a different active excerpt from the
**same original real utterance**. It never uses a vocoder waveform and never
borrows another utterance, speaker or split. Its donor interval:

- Has the same sample length as the target and satisfies the same real-only RMS
  activity/margin policy.
- Is disjoint from the target; no circular wrap, resampling, stretching, gain,
  padding or interpolation of sample positions is introduced.
- Is selected deterministically in its own `real-donor:<utterance>` RNG namespace.
- Has a different decoded pure core from the target; a degenerate identical
  excerpt cannot silently become another null control.

Donor sample `donor_start+k` maps to target `start+k`. The existing complementary
crossfade is applied inside the target. The whole recording retains its original
length, and every sample outside the target stays exact. Donor/target bounds,
source hashes and activity decisions are recorded explicitly.

**Speech content is not matched.** This relocation can repeat/remove words or
phonemes and change local prosody, phase, level and acoustic continuity. It is
an intentional editing operation, not a repair of vocoder alignment or a claim
of unchanged linguistic content. The source speaker/recording/group is shared;
temporal correspondence inside this control is deliberately different.

Both null and real-to-real recordings have `has_synthetic=false`, no synthetic
source ID, and no label-1 intervals. Their cores and untouched samples are real
(label 0); fades retain the matched transition/ignore label -1. Real-to-real has
`same_source_null_edit=false`, its own `example_class`, explicit donor indices,
`content_policy=different_real_region_not_content_matched` and
`intentional_excerpt_relocation=true`.

Each utterance/variant has one shared null control and, when enabled, one shared
real-to-real control alongside its three mixed examples. Mixed examples reference
both control IDs. Thus the two kinds remain separate without duplicating real
controls once per vocoder. Every source and derived variant stays in its original
frozen group/split. New controls are shown with audio and a green real-origin
timeline in the offline preview; red still denotes only synthetic cores.

Neither control category eliminates all splice-related confounds. The real-only
relocated excerpt does not reproduce the exact generator-dependent phase,
bandwidth, level or residual mismatch at a real-to-vocoder seam. It is a useful
negative editing control, not proof of confound-free synthetic detection.

## Explicit frozen-split selection

Use repeatable `--split-count train=N`, `dev=N`, `eval=N` flags. Omitted splits
have request zero; duplicate/unknown/negative/noninteger requests are errors.
An explicitly supplied total must equal their sum. The bounded pilot limit of
12 utterances remains; no new default split ratios are introduced.

Within each requested original partition, sort the existing IDs and permute with
a stable hash-derived `vocopatch-split-selection|seed|split` RNG. Test eligibility
in that order. No split is calculated, moved or redistributed, and no shortfall
is backfilled from another partition. Selection records store full candidate
orders, requests, actual selected counts, exclusions and shortfalls per split.

If a split has too few eligible sources, valid generated files may still pass
sample integrity, but `split_coverage_complete=false` and explicit shortfalls
remain. Integrity status is not a claim that the requested coverage was met.
Enabling real-to-real controls requires an eligible distinct donor for each
selected variant. An unsuitable first variant excludes its whole source group;
an unavailable optional second variant is recorded separately. No condition-only
drop or threshold relaxation is used.

The user explicitly chose **4 train, 4 dev, 4 eval** for the regenerated pilot.
Those are sampling counts within existing partitions, not new split assignments
or a changed 140/20/40 source partition.

## Compatibility and validation

Legacy defaults remain schema `vocopatch_partial_v1`, with the original global
sampling order, IDs, source/target intervals, audio and label semantics. New
control and/or explicit-split modes use `vocopatch_partial_v2`. New configuration
fields are omitted when serializing a legacy recipe. The initial folder remains
immutable; changed code provenance is recorded honestly in reproductions.

The validator supports both schemas. It independently reconstructs entry/exit
weights using scalar calculations, copies the correctly indexed donor core,
rounds to float32, and checks the decoded PCM_16 floor grid. It never calls
`splice` or the audio writer, and creates no temporary WAVs. A regression test
injects a generator-only crossfade error and requires validation to reject it.
Signed/near-grid calibration tests check the PCM_16 oracle against sanctioned
I/O. The unchanged outside/core arrays and null identity receive exact checks.

Validation recomputes frozen source selection, donor mapping, labels, metadata,
source hashes, format/counts, class/split totals, preview copies and file inventory.
New real-only controls require real source identities and non-null changed cores;
donor/group/split corruption is rejected. The original scientific pipeline and
all 18 invariants remain unchanged; no historical gates are newly certified.

## Exact local commands

Run from repository PowerShell with the existing environment. Output names must
be fresh. The reviewed command explicitly enables the new controls and counts:

```powershell
& '.venv\Scripts\python.exe' -m experiments.cli partial build --manifest runs/onboarding/v0-kaggle-input/manifest.csv --out-dir runs/vocopatch-phase1-v2-20261103 --split-count train=4 --split-count dev=4 --split-count eval=4 --real-to-real-controls --seed 20260907 --segment-seconds 1.0 --crossfade-ms 10.0 --boundary-margin-seconds 0.25 --variants 1
& '.venv\Scripts\python.exe' -m experiments.cli partial validate --dataset-dir runs/vocopatch-phase1-v2-20261103 --source-manifest runs/onboarding/v0-kaggle-input/manifest.csv
```

The initial recipe can still be reproduced to a different directory:

```powershell
& '.venv\Scripts\python.exe' -m experiments.cli partial build --manifest runs/onboarding/v0-kaggle-input/manifest.csv --out-dir runs/vocopatch-review-20261103/legacy-reproduction --n-utterances 12 --seed 20260907 --segment-seconds 1.0 --crossfade-ms 10.0 --boundary-margin-seconds 0.25 --variants 1
```

For a future rerun, replace the already-used output names above with new ones.
There is no upload or detector-training command in this workflow.

## Verification and generated results

Focused verification: **152 passed** (97 existing plus 55 new tests), 131.28 s.
Original 97 alone: passed. New review cases cover real-only sample provenance,
control semantics, donor/target mapping, zero/crossfades, all variants/grouping,
split-count errors and shortages, repeatable/relocatable generation, unchanged
source files, all reproduced audit defects, independent reconstruction, and CLI.

Source/test Ruff passes; repository-wide Ruff retains the same **11 generated
notebook errors**. Full pytest: **694 passed, 4 failed, 4 skipped, 1 xfailed**,
414.79 s. Failure IDs match the previous pilot exactly: two notebook CRLF/LF byte
comparisons and two Windows directory-junction permission failures. File-symlink
privilege skips and the pre-existing ablation expected failure are unchanged.
No new failure/skip appeared; no scientific gate or test was weakened.

Actual reviewed output: `runs/vocopatch-phase1-v2-20261103/`.

| Frozen split | Utterances | BigVGAN-112m mixed | Griffin-Lim mixed | Vocos mixed | Null | Real-to-real | Files |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 4 | 4 | 4 | 4 | 4 | 4 | 20 |
| dev | 4 | 4 | 4 | 4 | 4 | 4 | 20 |
| eval | 4 | 4 | 4 | 4 | 4 | 4 | 20 |
| Total | 12 | 12 | 12 | 12 | 12 | 12 | 60 |

All 60 recordings are mono PCM_16 at 22,050 Hz, totaling **428.256916100 s**.
The requested 4/4/4 coverage was met with no shortfall. Eval candidate
`LJ005-0137`, rank 2 in its split's candidate order, was excluded for
`no_distinct_disjoint_real_donor_region`; the next eligible eval candidate was
selected. No threshold or split was changed to admit it. Default edit/fade
parameters remain 22,050 edit samples and 220 samples per fade; no second
variant was requested in this regenerated pilot.

Build-time and standalone final-folder validation both passed. An independent
sample audit of all 60 files confirmed exact outside samples and donor cores,
complete labels, unchanged frozen groups, exact null identity and exclusively
real source origins for all 12 real-to-real controls. Direct fade calculations
agreed within the PCM_16 quantization step (maximum 0.995476 steps, rounded up).
The preview SVG is valid XML and all preview labels/copies pass validation.

Legacy reproduction in `runs/vocopatch-review-20261103/legacy-reproduction/`
reproduced the original 48 WAVs and label/manifest files **byte-for-byte**.
Of the initial pilot's 77 files, 75 reproduced exactly; only the new code
provenance and its dependent inventory differ, as expected. All 77 files in the
original pilot folder remain unchanged. A full before/after inventory also
verified all **1,606 original v0 files unchanged**, not only selected WAVs.

The successful fixture runs and data checks are logged with these commands:

```powershell
& '.venv\Scripts\python.exe' -m pytest -o addopts='' -q -ra tests/test_partial_audio.py tests/test_partial_audio_review.py --basetemp runs/vocopatch-review-20261103/pytest-focused-final
& '.venv\Scripts\python.exe' -m pytest -o addopts='' -q -ra --basetemp runs/vocopatch-review-20261103/pytest-full-final
& '.venv\Scripts\python.exe' -m ruff check src tests
& '.venv\Scripts\python.exe' -m ruff check .
```

Logs and receipts are in ignored `runs/vocopatch-review-20261103/`. Fresh pytest
basetemp directories are used; do not reuse them because pytest may delete them.

## Remaining scientific limits and commit review

Legacy generation/alignment provenance is not authenticated by current hashes.
The corpus is single speaker and the sample is small. RMS activity is not a
phonetic/semantic annotation. Real-to-real controls are not content matched and
introduce their own acoustic/linguistic discontinuities. Full-band Griffin-Lim
and Vocos signatures, crossfade cues and loudness drift remain concerns; relevant
analysis-band exclusions and future label transformations still apply.

These are ground-truth editing artifacts, not learned localization predictions,
research-admitted measurements or proof of generalization. Dev/eval audio is
inspected here under the user's explicit sampling request; no detector or model
selection uses it. A future evaluation protocol must account for this development
inspection. Publishing and model training require separate work and authorization.

## Changed files and manual commit readiness

Phase 1 changes are in `src/data/partial_audio.py`,
`src/experiments/partial_audio.py`, `src/experiments/partial_audio_demo.py`,
`src/experiments/cli.py`, new `tests/test_partial_audio_review.py`, this new report,
the original pilot guide's review pointer, and README's review pointer. The
original 97-test file remains unchanged. The preceding pilot's edits were
already unstaged, so manual review should include that foundation as well.

The implementation is ready for **manual review and commit**, with the explicitly
documented pre-existing test/lint limitations. This is not scientific publication
or model-training readiness. No identified audit defect remains unresolved in
the tested paths. All file changes remain unstaged; the existing `.gitignore`
edit is untouched. `CLAUDE.md`, Phase A, all scientific gates/checkpoint pins and
the original dataset are unchanged. No Git writes or remote operations occurred.
