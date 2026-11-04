# VocoPatch partial-audio development pilot

2026-11-04: [Phase 2 complete-source expansion](vocopatch_phase2_expansion.md)
adds explicit all-source scanning and local private-upload preparation. The
pilot recipes/artifacts and historical results below remain unchanged.

2026-11-03 review: [Phase 1 audit and control improvements](vocopatch_phase1_review.md)
adds opt-in real-to-real controls and explicit frozen-split counts. Legacy defaults,
the initial pilot and the historical results below are preserved. Validation now
uses an independent read-only reconstruction check and closes the documented
duration, inventory and provenance-validation gaps.

2026-10-27. Starting HEAD: `bd43850198371bb3d87ff2fec683297ab69b70f7`.
This implements a bounded, local editing/labeling pilot on the existing paired
`btpvf-v0`. **Every generated example and report is development-only and
non-citable.** No localization model, detector prediction, accuracy, frontend,
remote job or Kaggle upload is part of this task. The updated user instruction
explicitly defers publishing until a separate review and authorization.

## Scope and reused contracts

`data.partial_audio` contains the sample-domain editor, activity sampling and
interval labels. `experiments.partial_audio` performs preflight, generation and
reconstruction/integrity validation. `experiments.partial_audio_demo` writes an
offline review folder. `btpvf partial build|validate` exposes these functions.

The implementation reuses `read_manifest`, portable path resolution,
`validate_manifest`, `select_tier`, `require_primary`, `check_pairing`, sanctioned
`probe`/`read_processed`/`write_audio` and the existing streaming SHA-256 helper.
It requires no new dependency and does not import neural or classifier models.

The original Phase A code, all 18 invariants, scientific gates, calibration,
checkpoint pins, frozen splits and original v0 audio/metadata are unchanged.
These edited recordings have a separate manifest schema. They are not Phase A
outputs or research-admitted detector inputs. Their `tier` description is
`archive_derived_partial_edit`: a full-band edit at archive rate, **not** the
16 kHz zeroshot resampling path. Source catalog records retain `tier=archive`.
No new zeroshot files are generated.

## Source admission and sampling

Default source: `runs/onboarding/v0-kaggle-input/manifest.csv`, with 200 utterance
groups and conditions `real`, `bigvgan_112m`, `griffin_lim`, `vocos`. The existing
manifest contains both 22,050 Hz archive and 16,000 Hz zeroshot rows; only archive
rows are consumed. Existing table gates run unchanged before selection. Duplicate
rows/paths, conflicting speaker identities and unknown splits fail preflight.
Multi-speaker inputs must preserve existing speaker-disjoint splits.

Sort source IDs, permute them with NumPy `default_rng(20260907)`, and inspect in
that fixed order until at most 12 eligible groups have been selected. No split is
generated or reassigned. The complete candidate order, selected group identities,
candidate ranks and explicit exclusions are saved in `selection.json`. IDs beyond
the sampling limit are uninspected candidates, not rejected examples.

For every candidate, all four conditions must exist, resolve inside the original
dataset tree, and be readable mono RIFF/WAVE PCM_16 at 22,050 Hz. Header checks
precede decoding so the generic reader cannot silently downmix a stereo file.
Actual durations must agree with the source manifest within half a sample.
Decoded sample counts must be identical across the group. An identical real and
purported vocoder recording is unsuitable for this pilot. No time shift, padding,
truncation, gain, resampling or alignment repair is performed.

Missing/unsupported files, length/duration mismatch, insufficient interior room
or absent activity exclude the **whole utterance group**, with a reason. There
is no condition-specific partial inclusion. Existing `InvariantViolation` failures
are not converted to sampling exclusions: they stop the build. If nothing is
eligible, no generated audio or published dataset directory is created.

Header/ID/count agreement verifies structural pairing. The source's recorded
alignment checks/lag, checkpoint, code revision and seed are retained as recorded
evidence. No new empirical alignment benchmark or proof of historical generation
bytes is claimed. The legacy provenance limitation remains.

## Speech-active interior region

Default edit duration is 1.0 second = **22,050 samples**. Boundaries remain at
least 0.25 seconds from each recording edge, rounded to **5,512 samples**.
Sample conversions use nearest-integer rounding with ties to even; requested and
realized fade durations are both stored.

The activity proxy uses only the real reference:

1. Compute RMS on complete `TRIM_FRAME_LENGTH=512` sample frames at
   `TRIM_HOP_LENGTH=128`, without padding or changing the waveform.
2. A frame is active when RMS is at least
   `max(maximum_RMS * 10**(-TRIM_TOP_DB/20), 2/32768)`, with `TRIM_TOP_DB=30`.
   A reference whose maximum RMS is at/below two PCM_16 steps is unsuitable.
3. Enumerate candidate starts `margin + k*hop`, keeping intervals inside both
   margins. At least 80% of frame centers in an interval must be active, and
   its first and last included frames must both be active.
4. Select a candidate with an utterance/variant-specific RNG derived from the
   SHA-256 of `seed|utterance_id|variant`. Selection never depends on vocoder
   features or labels. All three vocoders and their shared null control use the
   same interval for that utterance/variant.

This is an RMS activity proxy, not semantic speech detection, word alignment or
a claim that every selected sample is voiced. Short pauses can occur inside the
interval. Save the measured threshold, frame/hop specification, eligible candidate
counts and exact chosen bounds so the decision can be reconstructed.

## Editing and crossfade definition

All sample ranges are **zero-based, half-open**. For edit `[s,e)` and fade length
`F`, the two fades lie **inside** the edit:

- Entry transition: `[s,s+F)`.
- Pure donor core: `[s+F,e-F)`.
- Exit transition: `[e-F,e)`.
- Everything outside `[s,e)` remains the decoded real samples exactly.

For an entry transition sample `k=0,...,F-1`, donor weight is `(k+1)/(F+1)`;
exit weights are the reversed sequence. The core's donor weight is 1, and
outside the edit it is 0. Every transition sample has strictly positive real
and donor contributions. Complementary linear weights keep the operation convex;
there is no equal-power gain bump. The implementation computes
`real + weight*(donor-real)` in float64 and stores float32 before sanctioned
PCM_16 encoding. This difference form makes same-source null edits exact
identities rather than introducing arithmetic rounding changes.

Default requested fade is **10 ms**, rounded to **220 samples per boundary**
(actual **9.977324263 ms**). The default pure donor core therefore contains
**21,610 samples**. Zero fade is supported and explicitly labeled as a hard
splice. An empty core, out-of-range boundary or nonfinite input is rejected.

Source WAVs are already processed archive recordings. Apply no further trim,
normalization, filtering, cropping, resampling or compression to any output.
Mixed recordings and controls use the identical editor and PCM_16 writer. Global
renormalization would change samples outside the labeled edit and encode a
different processing history, so it is not silently added. Composite loudness
may consequently drift; no Phase A loudness admission is claimed for edits.

One variant is the default. Optional variant 2 is attempted only after all four
variant-1 outputs pass sample validation. It must use a disjoint eligible
interval in a **separate recording**, still containing only one synthetic
segment. If no such region exists, retain variant 1 and record the variant-2
exclusion. A validation failure stops the build; it never triggers a silent retry
or a replacement region chosen to conceal a failed output.

## Controls and labels

There is **one real-only same-source null control per utterance/variant**.
It uses real as both reference and donor through the same splice/encode path.
All three mixed conditions reference that control's ID. This avoids three
duplicate control files merely because three synthetic conditions exist.
Its decoded samples must equal the original real waveform exactly.

Labels are complete run-length intervals covering every output sample:

| Label | Meaning |
|---|---|
| `0` | Real samples, or the all-real core of a null control |
| `1` | Pure paired-vocoder donor core |
| `-1` | Transition/ignore buffer |

Controls have `has_synthetic=false`, `same_source_null_edit=true` and no pure
synthetic range. Their **physical audio is real everywhere**. They receive the
same ignored editing-boundary buffers as the mixed examples, explicitly as a
matched ignore policy; `-1` does not imply synthetic content in those controls.
No control sample is labeled synthetic.

Null controls test the editing/encoding path. They do **not** reproduce the
real-to-synthetic mismatch or establish that crossfades eliminate splice cues.
Different-region real-to-real edits, fade/no-fade ablations and boundary-focused
tests remain future experiments. These controls cannot alone establish a model
learns vocoder artifacts rather than boundaries.

## Artifacts and provenance

Output WAV filenames are opaque deterministic recipe IDs in one `audio/` folder;
class, split and vocoder names are not embedded in their filenames. IDs bind the
schema, seed, utterance, variant, source audio hashes, edit bounds and fade length.
Filenames/metadata must never be supplied as model features.

| File | Content |
|---|---|
| `manifest.json`, `manifest.csv` | Output IDs/paths, group/speaker/original split, source condition and source identities, edit samples/seconds, crossfade/ignore/core ranges, configuration/seed, input/output hashes, changed-sample counts, passed validation and non-citable flags |
| `labels.json` | Complete half-open sample intervals and label meanings |
| `sources.json` | Selected original source paths relative to source manifest, hashes, measured format/counts and recorded upstream evidence |
| `selection.json` | Fixed candidate order, selected utterances, real-derived activity decisions and whole-group exclusions |
| `variant_exclusions.json` | Optional second-variant exclusions |
| `provenance.json` | Configuration, source manifest/sidecar hashes, recorded upstream fields, current code hashes/HEAD and runtime versions |
| `validation.json` | Recomputed counts, durations, split populations and local integrity status |
| `artifacts.sha256.json` | Complete package file/hash inventory, excluding its own self-hash |
| `README.md`, `demo/` | Human-readable explanation, representative audio, original sources, null controls, timestamps and timelines |

`source_dataset_name=btpvf-v0` is the declared source. Its local version tag is
the original manifest SHA-256, accompanied by every selected source WAV digest.
An upstream Kaggle version is not invented. Existing provenance fields such as
`git_commit` are preserved **as recorded**, and missing fields stay missing.
Recorded historical metadata is not proof that those bytes executed. Current
generator source hashes identify unstaged working-tree code independently of
the existing Git HEAD; no Git write is performed.

## Validation and failure handling

Before writing, preflight validates source groups and sampling. Each written
output is then decoded through sanctioned I/O and independently checked for:

- Mono PCM_16 RIFF/WAVE at 22,050 Hz and exact original sample count.
- Valid interior boundaries, exact sample-to-time conversion and complete labels.
- Exact real samples outside the edit and exact donor samples in its pure core.
- Exact reconstruction of every encoded transition sample from source audio,
  recorded bounds, fade weights and the same sanctioned PCM_16 encoding.
- At least one changed core sample for a synthetic example; zero changed samples
  over the entire null control.
- Correct group/split, source IDs, hashes and deterministic region selection.
- CSV/JSON/label agreement, original source/sidecar integrity and complete output
  inventories with no unmanifested audio.
- Preview audio/source copy hashes and SVG/HTML/timestamp agreement with the
  actual labeled sample positions.

The build uses a fresh `<out-dir>.staging` directory outside the source tree.
It publishes `<out-dir>` only after complete reconstruction/inventory validation.
Existing output/staging paths are refused. Failure preserves staging plus
`failure.json` for diagnosis, re-raises the error and leaves no successful final
folder. It does not continue past scientific invariant failures.

`partial validate` is read-only. It requires the original source manifest/tree,
recomputes selection and samples, and checks the saved validation summary and
all package hashes. A package and source tree can be relocated together while
preserving their bytes. Integrity hashes detect changed bytes; they are not
signatures, upstream authentication or proof of research eligibility.

## Reproducibility commands

From the repository in PowerShell, with the existing environment. Use a **new**
output name; generation never overwrites a prior run:

```powershell
& '.venv\Scripts\python.exe' -m experiments.cli partial build --manifest runs/onboarding/v0-kaggle-input/manifest.csv --out-dir runs/vocopatch-pilot-v0-reproduction-20261027 --n-utterances 12 --seed 20260907 --segment-seconds 1.0 --crossfade-ms 10.0 --boundary-margin-seconds 0.25 --variants 1
& '.venv\Scripts\python.exe' -m experiments.cli partial validate --dataset-dir runs/vocopatch-pilot-v0-reproduction-20261027 --source-manifest runs/onboarding/v0-kaggle-input/manifest.csv
```

The generated offline review entry is `demo/index.html`; it requires no Python,
network, model or server. `demo/timeline.svg` and `demo/timestamps.csv` also work
independently. Preview copies and originals in `demo/` are not extra training
examples. Keep every derivative/preview of a source utterance in its frozen split.

For identical source bytes, parameters, implementation, environment and HEAD,
tests require identical complete package inventories/hashes across output paths.
After a manual commit, provenance HEAD/code identity can legitimately differ;
audio and labels must still reproduce from the recorded generation recipe.

## Proposed Kaggle integration: review first

No authentication query, remote mutation, upload, dataset creation/versioning or
kernel execution is performed. There is no new runner or publisher in this patch.

After separate approval, use the existing `experiments.kaggle_harness` CLI wrapper
and current CLI-managed account. Prepare a separate upload directory/manifest for
`<authenticated-owner>/btpvf-partial-pilot-v0`, preserving the immutable local
validation receipt. Resolve source/derivative licensing before writing dataset
license metadata; do not invent a license or inherit permissions by assumption.
Include validated audio, manifests/labels, configuration, provenance/validation,
README and a bounded set of previews. Exclude test fixtures, audit logs, original
credential paths, unrelated source data and accidental duplicate training copies.

Use the installed CLI's metadata and directory-handling conventions (existing
harness documentation records that default directory mode can skip folders).
After read-only identity/collision checks, a future authorized operation may
create exactly one **new private** dataset, never `--public` and never a version
of original `btpvf-v0`. Verify owner, slug, version, privacy, complete file listing
and retrieved identities before reporting success. Preserve an exact sanitized
failure receipt and stop on an authentication/upload failure; do not retry a
failed or ambiguous create blindly. No Kaggle URL is claimed for this local pilot.

## Verification and local generation results

Focused synthetic-fixture verification: **97 passed**, 49.49 seconds. Source Ruff
passes; repository-wide Ruff retains the same **11 generated-notebook errors**.
Full suite: **639 passed, 4 failed, 4 skipped, 1 xfailed**, 271.00 seconds. The four
failure IDs match the previous pilot: two generated-notebook CRLF/LF byte
comparisons and two Windows directory-junction permission failures. The same
four file-symlink privilege skips and existing ablation expected failure remain.
No test was weakened, and no new failure or skip appeared.

The tests cover independent crossfade mathematics, label coverage/ignore buffers,
null-edit identity, input immutability, malformed configuration/arrays, activity
selection and seed determinism, zero fade, disjoint/gated second variants,
structural source errors, whole-group exclusions, unchanged source bytes,
reproducible/relocatable packages, corrupt samples/labels/splits/hashes/previews,
path containment, failure staging, no overwrites and read-only CLI validation.

Commands used fresh workspace basetemp directories in the existing environment:

```powershell
& '.venv\Scripts\python.exe' -m pytest -o addopts='' -q -ra tests/test_partial_audio.py --basetemp runs/vocopatch-audit-20261027/pytest-focused-final
& '.venv\Scripts\python.exe' -m pytest -o addopts='' -q -ra --basetemp runs/vocopatch-audit-20261027/pytest-full-final
& '.venv\Scripts\python.exe' -m ruff check src tests
& '.venv\Scripts\python.exe' -m ruff check .
```

Choose new basetemp names for reruns; pytest can remove an existing basetemp.
Audit logs/receipts are under ignored `runs/vocopatch-audit-20261027/`.

### Actual local pilot: development-only, non-citable

Output: `runs/vocopatch-pilot-v0-20261027/`. Generation ran **after** all 97
focused tests passed. Full reconstruction/inventory validation passed before the
staging directory was renamed to its final location.

| Item | Actual result |
|---|---|
| Original source utterances used | 12 |
| Selected original source WAV hashes verified | 48 |
| Mixed recordings | 36: 12 each BigVGAN-112m, Griffin-Lim and Vocos |
| Real-only same-source null controls | 12 |
| Primary dataset WAVs | 48 |
| Total primary duration | 323.412970522 s (5 min 23.413 s) |
| Mixed-recording duration | 242.559727891 s |
| Control-recording duration | 80.853242630 s |
| Pure synthetic core samples | 777,960 (35.281632653 s) |
| Encoding | Mono RIFF/WAVE PCM_16, 22,050 Hz |
| Original frozen splits represented | Train: 12 groups; dev/eval: 0 |
| Whole-group exclusions inspected | 1 |
| Optional second variants | Disabled; no second-variant exclusions |
| Local sample/label/hash validation | Passed |
| Remote upload | Not attempted; deferred by the user |

`LJ036-0002` at candidate rank 11 was excluded with
`no_active_interior_segment`. The next eligible candidate was used, with no
change to the activity/fade policy. The 12 selected IDs and exact per-file
regions are in `selection.json` and `manifest.json`. No source utterance was
reassigned to another split. The fixed default permutation produces a train-only
subset on this v0; this is a development artifact with **no held-out population**.
Do not invent dev/eval rows or report evaluation metrics from this selection.

The self-contained `demo/` contains five representative mixed clips across all
three vocoders and two utterances, two shared control copies, and seven original
source copies (14 WAV copies total). These copies are additional review assets,
not independent examples to add to the 48-file dataset. Open `demo/index.html`;
the exact timestamps and SVG are also saved separately. The SVG is valid XML
and is checked against the sample labels by the reconstruction validator.

Original manifest/provenance sidecars and all 48 selected source WAV hashes were
rechecked unchanged. Existing v0 is not modified or versioned. No new upstream
generation, signal-gate admission, localization model or predictions are claimed.

A separate sample audit, independent of the editor's reconstruction helper,
checked all 48 outputs and source hashes again. It confirmed exact untouched
outside samples, exact donor cores, null-control identity and complete labels.
Direct complementary-weight calculations agreed at the fades within PCM_16
quantization (maximum absolute difference 0.995476 PCM steps, rounded upward).
Standalone `partial validate` also passed on the final relocated-from-staging
folder. Audit receipts are `verification.json` and
`independent_sample_audit.json` in the audit directory.

## Scientific limitations

This is a small single-speaker, known-vocoder manipulation pilot on legacy v0,
not a complete localization dataset. RMS activity is not phonetic annotation.
Known full-band Griffin-Lim bandwidth and Vocos resampling signatures can remain
inside inserted regions; their research-analysis restrictions are unchanged.
Crossfades can leave phase/level/boundary cues, and whole-recording loudness may
drift. Same-source null controls do not eliminate these confounds. Source
alignment/generation provenance and historical signal gates are not newly proven.
Labels describe the saved PCM_16 samples: future filtering, resampling or cropping
must transform the labels/ignore policy as needed and revalidate temporal support.
No learned detector, localization accuracy, generalization result, confidence
interval, unseen-vocoder rejection or thesis-citable evidence is produced.
