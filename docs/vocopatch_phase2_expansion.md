# VocoPatch Phase 2: complete-source development expansion

2026-11-04 follow-up: the independent review's recursive demo-file admission
gap is addressed in [the packaging integrity fix](vocopatch_packaging_integrity_fix.md).
That report and the [owner/licensing review](vocopatch_licensing_owner_review.md)
record stricter package checks and current publication blockers. The production
dataset and historical results below remain unchanged.

2026-11-04. Starting HEAD: `6628c7a3f6bb95eae2f7a177060e9bca2f827d98`.
**Development-only, non-citable.** Phase 1 was manually committed before this
work. The pre-existing `.gitignore` change is untouched. No models, installs,
remote operations, source-dataset writes or Git writes are part of Phase 2.

## Design and eligibility census

Read the repository rules, binding invariant rules, both Phase 1 reports and
the existing generator/validator/demo/tests. Reuse the reviewed sample editor,
independent PCM reconstruction oracle, source admission gates, labels, controls
and deterministic donor/target recipe. No Phase A processing, signal gate,
alignment, checkpoint pin or `CLAUDE.md` rule is changed.

The legacy default still caps pilots at 12 sources and serializes their original
configuration. A separate explicit `--all-utterances` mode scans every original
source group (bounded to 200 for this development release), requires both
controls and exactly one variant, and rejects quotas/subset requests. Derive
the source population from the original manifest, not a new split function.
The original 140 train / 20 dev / 40 eval assignments remain unchanged.
Within each existing partition reuse Phase 1's namespaced ordering; inspect all
candidates rather than stopping after a small quota. There is no replacement
across splits, alignment repair, parameter adaptation or selective vocoder drop.

| Frozen split | Original groups inspected | Eligible groups | Excluded |
|---|---:|---:|---:|
| train | 140 | 123 | 17 |
| dev | 20 | 17 | 3 |
| eval | 40 | 34 | 6 |
| Total | 200 | 174 | 26 |

| Exact whole-group exclusion reason | Train | Dev | Eval | Total |
|---|---:|---:|---:|---:|
| `no_distinct_disjoint_real_donor_region` | 11 | 3 | 5 | 19 |
| `too_short_for_interior_segment` | 3 | 0 | 1 | 4 |
| `no_active_interior_segment` | 3 | 0 | 0 | 3 |

The donor failure means no qualifying distinct disjoint donor **for the already
deterministically selected target**, not that every imaginable target would
fail. No target is retried or shortened to improve admission. Exclusions retain
split/rank/reason, observed source paths and hashes, and, when decoding passed,
validated source records, measured length/peak, selected target/activity and
disjoint active donor candidate count. Fixed source bytes/configuration allow
the validator to recompute every exclusion. No condition-only output survives
a group exclusion.

Excluded source IDs (their exact per-source reasons/measurements are in
`selection.json`, also repeated in `validation.json`):

- Train, donor unavailable: LJ047-0157, LJ015-0052, LJ022-0121, LJ016-0347,
  LJ048-0058, LJ030-0105, LJ047-0022, LJ039-0041, LJ007-0169, LJ039-0113,
  LJ028-0500.
- Train, too short: LJ020-0064, LJ041-0050, LJ024-0003.
- Train, no active target: LJ036-0002, LJ033-0017, LJ021-0054.
- Dev, donor unavailable: LJ046-0046, LJ017-0133, LJ012-0026.
- Eval, donor unavailable: LJ005-0137, LJ050-0053, LJ046-0188, LJ047-0056,
  LJ008-0069.
- Eval, too short: LJ037-0076.

`source_census_complete=true` means all 200 were examined. The existing
`split_coverage_complete` flag is false because exclusions leave fewer than the
requested original population. A passed integrity status neither conceals nor
waives that coverage shortfall. The source partition stays 140/20/40; the
eligible output subset is 123/17/34, with no new split ratio.

## Output semantics and local organization

The generation folder is `runs/btpvf-partial-v1-20261104/`. Its `audio/` contains
only primary manifest examples; `demo/` contains offline review copies, which
must not be globbed into training. Keep manifests, labels, selected-source
catalog, selection/exclusion records, provenance, validation and complete
SHA-256 inventory together. The annotation schema remains
`vocopatch_partial_v2`; dataset name `btpvf-partial-v1` is a separate identity.

Every eligible source produces one BigVGAN-112m mixed, Griffin-Lim mixed, Vocos
mixed, null and different-region real-to-real control. The default Phase 1
parameters remain seed 20260907, target 22,050 samples, 220-sample internal
fades, 5,512-sample edge margins, RMS frame 512/hop 128, 30 dB activity threshold,
minimum fraction 0.8. No global gain/filter/resample/trim is applied to edits.
Unchanged samples outside the edit and pure donor samples inside are checked
exactly after PCM_16 decoding. Fade samples are independently reconstructed.

Labels are 0 real, 1 pure synthetic and -1 ignored transition. Real-to-real
uses only a disjoint excerpt from the same real utterance and has no synthetic
labels; it intentionally changes content/continuity. Null audio remains a
decoded identity. Both controls retain matched ignored boundary buffers. These
controls investigate different issues; neither proves absence of all seams.

The upcoming model protocol is in
[vocopatch_training_protocol.md](vocopatch_training_protocol.md). It defines
train-only fitting/statistics, dev-only tuning, frozen eval isolation, sample
and event metrics, source-grouped evaluation and confound comparisons. Eval
review exposure from Phase 1 is disclosed. No learned model or eval-dependent
parameter optimization is performed in this expansion.

## Local Kaggle preparation

`partial prepare-upload` makes a separate fresh folder at
`runs/btpvf-partial-v1-kaggle-upload-20261104/`, after complete generation
validation. It copies all validated dataset bytes and adds this release's
[dataset card](vocopatch_partial_v1_dataset_card.md), protocol, preparation
receipt and private-upload review plan. The preparation receipt preserves
the original generation file inventory and its receipt hash. A new inventory
covers the copy plus added documents; original artifacts remain unchanged.
`partial validate-upload` reconstructs the copy and verifies original copied
bytes, documents, new inventory and the strictly private unauthorized plan.

No credentials, environment files, source WAV tree, tests, caches or unrelated
runs are copied. Offline demo WAVs are explicitly marked duplicate review
assets. Only manifest examples are training observations. The local preparation
module has no network, Kaggle import or upload command. It is not another runner.

Owner and source/derivative license remain unresolved, so provide only a
`dataset-metadata.template.json` with null owner identity and no license claim.
There is deliberately no executable `dataset-metadata.json`.
`publication_ready=false` and `upload_authorized=false`. The repository MIT
code license is not an audio redistribution license. Future separately
authorized publishing must use existing Kaggle harness/wrapper conventions,
review owner/collision/license/privacy, explicitly include all directories
(installed CLI defaults to `skip`), verify remote file inventory and retrieved
hashes, and create a new private dataset rather than versioning `btpvf-v0`.
The harness's existing code-dataset publisher is specialized to `btpvf-code`
and its code license; do not reuse its slug/license for this audio dataset.

## Exact reproduction commands

Run from repository PowerShell using the existing environment. These are the
initial output names; substitute **fresh names** for reruns. Never reuse pytest
basetemp directories, which pytest may remove. Generate only after CPU
validation passes. No command below contacts Kaggle or trains a model.

```powershell
& '.venv/Scripts/python.exe' -m pytest -o addopts='' -q -ra tests/test_partial_audio.py tests/test_partial_audio_review.py tests/test_partial_audio_expansion.py --basetemp runs/vocopatch-phase2-audit-20261104/pytest-focused-final
& '.venv/Scripts/python.exe' -m experiments.cli partial build --manifest runs/onboarding/v0-kaggle-input/manifest.csv --out-dir runs/btpvf-partial-v1-20261104 --all-utterances --real-to-real-controls --seed 20260907 --segment-seconds 1.0 --crossfade-ms 10.0 --boundary-margin-seconds 0.25 --variants 1
& '.venv/Scripts/python.exe' -m experiments.cli partial validate --dataset-dir runs/btpvf-partial-v1-20261104 --source-manifest runs/onboarding/v0-kaggle-input/manifest.csv
& '.venv/Scripts/python.exe' -m experiments.cli partial prepare-upload --dataset-dir runs/btpvf-partial-v1-20261104 --source-manifest runs/onboarding/v0-kaggle-input/manifest.csv --out-dir runs/btpvf-partial-v1-kaggle-upload-20261104 --dataset-card docs/vocopatch_partial_v1_dataset_card.md --protocol docs/vocopatch_training_protocol.md
& '.venv/Scripts/python.exe' -m experiments.cli partial validate-upload --dataset-dir runs/btpvf-partial-v1-kaggle-upload-20261104 --source-manifest runs/onboarding/v0-kaggle-input/manifest.csv
& '.venv/Scripts/python.exe' -m pytest -o addopts='' -q -ra --basetemp runs/vocopatch-phase2-audit-20261104/pytest-full-final
& '.venv/Scripts/python.exe' -m ruff check src tests
& '.venv/Scripts/python.exe' -m ruff check .
```

Code provenance records the existing manually created HEAD and hashes the
actual generator/validator/CLI bytes; HEAD alone does not imply these Phase 2
working-tree bytes have been committed. Generation determinism is defined for
the same source/configuration/code/runtime. A later manual commit changes
provenance HEAD while the audio/labels must still reproduce.

## Scientific limits

Legacy source provenance and historical alignment/gate claims remain unverified.
This does not fix gate orientation, revalidate historical EERs, implement paired
uncertainty, establish F0 validity or admit data to the thesis. The corpus is
single speaker and known-vocoder only. Eligibility changes the observed subset;
do not generalize without studying that selection bias. RMS activity is not
speech/phonetic annotation. Fixed one-second edits, matched target positions
and fade length may become easy cues. Real-to-real controls alter content and
do not match every synthetic seam. Composite loudness may drift. Full-band
Griffin-Lim/Vocos cues remain and INV-17/18 restrictions still apply. Transforms
with temporal support require remapped/widened ignore labels before metrics.
No detector performance, unseen-vocoder rejection, final holdout validity or
scientific upload/research readiness is claimed.

## Executed verification and results

Build-time reconstruction and complete inventory checks passed before the fresh
staging directory was renamed to the final generation folder.

| Frozen split | BigVGAN mixed | Griffin-Lim mixed | Vocos mixed | Null | Real-to-real | Total WAVs |
|---|---:|---:|---:|---:|---:|---:|
| train | 123 | 123 | 123 | 123 | 123 | 615 |
| dev | 17 | 17 | 17 | 17 | 17 | 85 |
| eval | 34 | 34 | 34 | 34 | 34 | 34 | 170 |
| Total | 174 | 174 | 174 | 174 | 174 | 870 |

There are 522 mixed recordings and 348 real-only controls. Primary duration is
**6,151.370748299 seconds** (1 h 42 min 31.371 s), counting each derivative once
and excluding preview copies. Primary WAV size is **271,313,730 bytes**;
the full immutable generation package contains **901 files, 282,720,385 bytes**.
Logical file sizes are reported; filesystem allocation overhead is not included.

The initial **178 focused tests passed before audio generation**. Final focused
verification after preparation-receipt review passed **182 tests**, including
all 152 existing partial-audio/review cases and 30 new expansion/preparation
cases (118.48 s). The new
cases exercise beyond-pilot population size, complete census/exclusions,
unchanged partitions, control/group semantics, deterministic source/output
bytes, Phase 1 recipe compatibility, duplicate source hashes crossing splits,
safe copying/output guards, absent executable owner/license metadata, copied
byte/inventory/document corruption and CLI preparation/validation. An initial
test-fixture path error was corrected before generation; no gate/test was
weakened. Source/test Ruff passes; repository Ruff retains the same **11**
generated-notebook findings (2 E401, 4 I001, 2 UP031, 3 E402).

Final full pytest: **724 passed, 4 failed, 4 skipped, 1 xfailed** in 307.63 s. The
four failures match Phase 1, with no new failure or skip:

- `tests/test_kaggle_harness.py::TestGeneratedNotebook::test_the_committed_notebook_is_exactly_what_the_harness_generates`
- `tests/test_phase_a_run.py::TestKaggleNotebook::test_the_generated_notebook_is_the_committed_one`
- `tests/test_phase_a_run.py::TestUpstreamImportGuardThroughADirectoryLink::test_namespace_package_through_the_link_passes`
- `tests/test_phase_a_run.py::TestUpstreamImportGuardThroughADirectoryLink::test_namespace_package_from_elsewhere_fails`

The first two compare CRLF notebook bytes against LF-generated bytes. The
second two hit Windows junction access denial (WinError 5). Four file-symlink
cases retain their WinError 1314 privilege skips; the existing unimplemented
ablation PCM_16-driver check remains an expected failure. No notebook or
platform test was edited, bypassed or weakened.

Standalone validation of the final generation folder passed. A separate sample
audit of all 870 outputs checked source hashes, exact untouched samples and
donor cores, independently weighted transitions, complete label coverage, real
origins for both negative control categories, and original grouping/splits.
No cross-split group or source-file-hash leakage was found. Fade error relative
to the unquantized independent calculation was at most 0.995476 PCM_16 steps
(rounded upward); the validator's independent quantized oracle agrees exactly.

All **1,606 original v0 files**, **77 initial-pilot files** and **91 Phase 1
files** match the complete pre-work hash snapshot. All 60 Phase 1 examples are
present in the expansion with **identical manifest rows and WAV bytes**. This
checks actual compatibility, in addition to synthetic fixture evidence.

The second complete generation at
`runs/vocopatch-phase2-audit-20261104/reproduction/` uses the same build command
and parameters with only `--out-dir` changed. `determinism.json` records the
complete file/hash comparison, including the root inventory receipt. The
original package remains immutable throughout preparation.

The final preparation review reproduced one defect in the **new Phase 2
preparation validator** on a copied synthetic fixture: an incorrect original
receipt digest and contradictory claimed operations passed after ordinary
inventory resealing. No generated WAV, source, pilot or real preparation
receipt was wrong. The fix explicitly validates preparation identity/scope,
binds the stored original receipt digest to its canonical complete file map
(supporting LF/CRLF relocation), and requires both documentation digests. Four
new resealed-corruption cases cover identity, operations, original receipt
digest and omitted documentation. Self-hashes remain integrity checks, not
authentication. `preparation_scope_audit.json` records rejection of the same
fixture after the fix. The earlier preparatory copy is preserved under
`runs/vocopatch-phase2-audit-20261104/pre-final-upload/`; it is not the final
upload-review folder and is not included in that folder. Final focused/full
tests and fresh preparation follow this correction.

The final upload-review folder contains **906 files, 282,837,884 bytes**,
including all 870 primary WAVs, duplicate offline preview assets, labels,
manifests, source evidence, exclusions, the dataset card/protocol, both
generation/preparation provenance records and a complete new checksum inventory.
Its fresh build-time upload validation passed. Upload readiness is **local
review only**: owner/license review, explicit authorization and remote
privacy/path/hash verification remain outstanding.

Evidence and logs are under `runs/vocopatch-phase2-audit-20261104/`:
`before.json`, `eligibility.json`, `independent_audit.json`, `determinism.json`,
`preparation_scope_audit.json`,
build/reproduction/standalone validation logs, focused/full pytest logs and
both Ruff logs. `verify_expansion.py` contains the independent sample audit.
`validate-upload-final.log` records standalone validation of the final upload
folder. `verification.json` consolidates final counts, sizes, source/pilot
protection after preparation, test/lint outcomes, code identities and final
inventory digests. These ignored receipts are development evidence, not
citable research results.

## Changed files and review status

Tracked edits: `README.md`, `docs/vocopatch_partial_pilot.md`,
`src/data/partial_audio.py`, `src/experiments/partial_audio.py` and
`src/experiments/cli.py`. Added files: `src/experiments/partial_audio_upload.py`,
`tests/test_partial_audio_expansion.py`, this report,
`docs/vocopatch_partial_v1_dataset_card.md` and
`docs/vocopatch_training_protocol.md`. The existing `.gitignore` edit predates
this work and is unchanged. Data/audit outputs remain under ignored `runs/`.

Every change remains unstaged. Manual code/data review is possible locally;
upload and training remain unauthorized. No source dataset, old pilot,
scientific invariant, checkpoint pin or Git object/index was changed. Successful
local validation is not license clearance, research admission, untouched eval
status or proof that a later Kaggle upload will work.
