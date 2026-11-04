# btpvf-partial-v1: local expanded development dataset

2026-11-04. **Development-only, non-citable. Private upload preparation only.**
No Kaggle dataset has been created or published. Historical source generation,
checkpoint execution and alignment provenance remain unresolved. Current source
hashes establish byte identity, not authentication or research eligibility.

This dataset reuses paired btpvf-v0 archive real/BigVGAN-112m/Griffin-Lim/Vocos
audio and the reviewed Phase 1 editor. Every eligible source contributes five
full-length recordings: three single-vocoder partial-synthetic examples, one
same-source/same-index null control and one different-region real-to-real
edited control. The latter uses only a disjoint active excerpt of the same real
utterance; it changes linguistic content/continuity without synthetic speech.
Null controls remain exact decoded identities. Neither control removes all
boundary, level, phase, bandwidth or generator-dependent confounds.

The original 200 source groups have 140/20/40 frozen train/dev/eval assignments.
Complete eligibility scanning preserves those assignments, never backfills
between splits, and excludes an entire group if any required input or real
donor fails. Actual eligible counts, all excluded IDs/ranks/reasons and class
counts are machine-readable in `selection.json` and `validation.json`.
Successful integrity validation does not imply zero exclusions: check both
`source_census_complete` and `split_coverage_complete`.

Actual locally validated expansion:

| Split | Sources | BigVGAN mixed | Griffin-Lim mixed | Vocos mixed | Null | Real-to-real | Recordings |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 123 | 123 | 123 | 123 | 123 | 123 | 615 |
| dev | 17 | 17 | 17 | 17 | 17 | 17 | 85 |
| eval | 34 | 34 | 34 | 34 | 34 | 34 | 170 |
| Total | 174 | 174 | 174 | 174 | 174 | 174 | 870 |

The 26 excluded sources comprise 19 without a distinct disjoint real donor
for the fixed selected target, four too short for the interior target and
three without an active target. Their exact IDs, split, ranks, reasons, source
hashes and available measurements are retained. No thresholds were relaxed.
Primary recordings total **6,151.370748299 seconds**, **271,313,730 bytes**.
The immutable generation folder has 901 files, **282,720,385 bytes** including
metadata and duplicate offline previews; additional preparation documentation
and receipts increase the separate upload copy's size. These are dataset
integrity counts, not research measurements or model performance.

The fixed Phase 1 recipe is seed 20260907, one 1.0 s target per utterance,
10 ms requested internal complementary linear fades, 0.25 s edge margins,
30 dB real-only RMS activity threshold with minimum active fraction 0.8,
512-sample RMS frames at hop 128, and one distinct disjoint real donor.
Time rounding is nearest integer with ties to even: target 22,050 samples,
fades 220 each, margins 5,512, pure core 21,610. Original WAVs are never shifted,
repaired, filtered, trimmed or renormalized. Outputs retain the source length,
mono RIFF/WAVE PCM_16 at 22,050 Hz and exact decoded samples outside the edit.
The archive-derived editing schema is `vocopatch_partial_v2`; this release's
`v1` dataset name does not rename the annotation schema.

## Contents and correct use

| Artifact | Purpose |
|---|---|
| `audio/*.wav` | Primary recordings; only files referenced by manifest examples are training examples |
| `manifest.json`, `manifest.csv` | Opaque output IDs, original groups/splits, class/vocoder/source identities, edit and donor indices, PCM properties, hashes and complete labels |
| `labels.json` | Complete zero-based half-open raw-audio sample labels: 0 real, 1 pure synthetic, -1 transition/ignore |
| `sources.json` | Original source paths/hashes and recorded upstream evidence, not copies of the original dataset |
| `selection.json`, `variant_exclusions.json` | Full frozen-source census and exclusions; only one variant is requested |
| `provenance.json`, `validation.json` | Generation configuration, actual code/HEAD/environment hashes and recomputed integrity summary |
| `artifacts.sha256.json` | Complete local file inventory except this root receipt's own self-hash |
| `demo/` | Offline review only; duplicate WAVs are not extra observations and must never be globbed into training |
| `PROTOCOL.md` | Proposed train/dev/eval isolation, targets, mask treatment, grouped metrics and confound checks |
| `preparation.json`, `upload_plan.json` | Original generation inventory, preparation/doc hashes, private-only intention and unresolved publication requirements |
| `dataset-metadata.template.json` | Non-executable owner/license placeholder; no runnable `dataset-metadata.json` is supplied |

Metadata, source hashes, filenames and edit/ignore positions must not be model
inputs. All derivatives/crops/controls stay grouped by source utterance and
inherit its frozen split. Transition samples never become negatives. Read
`PROTOCOL.md` before future fitting/scoring. Some eval groups were inspected
during Phase 1 review, so this is a legacy development benchmark rather than
an untouched final research test. No training or model selection occurs here.

Full-band Griffin-Lim bandwidth and Vocos resampling cues remain. The original
INV-17/18 analysis transforms/membership and all other invariants are unchanged.
Loudness can drift after editing; RMS activity is not phonetic annotation.
Structural pairing and recorded alignment gates do not establish authenticated
historical sample alignment. These single-speaker known-vocoder examples cannot
establish unknown-source, multi-speaker, codec or real-world generalization.

## Local verification and publication review

`partial validate` needs the original btpvf-v0 source manifest/tree to independently
reconstruct outputs; those source WAVs are intentionally not duplicated here.
`partial validate-upload` additionally checks preparation and all copied bytes.
The expanded source package is immutable; preparation creates a separate copy
and a new inventory covering the additional documentation and review receipts.
`preparation.json` preserves the generation inventory's identity separately.

The preparation tool never imports Kaggle, reads credentials or uses the
network. Future publishing must resolve source/derivative redistribution
rights and exactly one license, review the authenticated owner and new slug
`btpvf-partial-v1`, obtain explicit user authorization and retain private
visibility. The repository's code license does not establish audio rights.
Use the existing Kaggle wrapper/harness conventions, not a second runner.
The installed CLI defaults directory handling to **skip**: explicitly include
`audio/` and `demo/` using a reviewed archive/directory plan (for example its
`zip` directory mode), and verify retrieved paths/hashes before any consumption.
Privacy must be checked in the upload operation and remote result; this
template alone cannot enforce it. No remote identity, dataset availability,
license, authentication or successful future upload is claimed.
