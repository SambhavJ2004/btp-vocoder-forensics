# VocoTrace Phase 4F: real multi-speaker acquisition gate and compatibility audit

**Real-audio pilot BLOCKED at acquisition. Development-only and non-citable.**
No authorized real VCTK source was found in the inspected local locations.
Actual genuine VCTK speakers accessed: **0**; source utterances ingested: **0**;
paired archive/derived outputs generated: **0/0**. No quality, alignment,
reconstruction, empirical compatibility or real-output determinism result is
claimed. Existing fixtures are explicitly excluded as evidence of real ingestion.

2026-11-07, Asia/Calcutta. Starting manually created HEAD:
`aa624d2e763566f8fd576b9acd5fe5c52f6901ca`
(`feat: add VocoTrace multi-speaker ingestion foundation`). The pre-existing
unstaged `.gitignore` edit is preserved. No source/test behavior was changed.
This report and ignored audit receipts complete the checks possible without
real data; they do not complete the conditional real-audio experiment.

## 1. Official acquisition, integrity and rights

The selected source is **CSTR VCTK Corpus 0.92**, September 2019 release,
published by University of Edinburgh/CSTR, DOI **10.7488/ds/2645**. Use the
[official release](https://datashare.ed.ac.uk/handle/10283/3443), not a lossy
mirror, speech-enhancement variant or synthetic derivative. It describes 110
English speakers; distributed recordings are downsampled to 48 kHz/16-bit and
manually end-pointed from the publisher's 96-kHz/24-bit recordings. Those prior
operations are source history, not processing this project performed.

The [official full item metadata](https://datashare.ed.ac.uk/items/30e7453c-9ea8-48b4-8e18-f96d0dc62928/full)
lists these files and integrity values:

| File | Listed size | Publisher-listed MD5 |
|---|---:|---|
| VCTK-Corpus-0.92.zip | 10.94 GB, rounded listing | `8a6ba2946b36fcbef0212cad601f4bfa` |
| README.txt | 5.11 KB | `31b61beec43565e27237dbb738566a7a` |
| license_text.txt | 17.01 KB | `2946b37a07baeba80cea628909e28cae` |

The maintained [PyTorch VCTK loader](https://raw.githubusercontent.com/pytorch/audio/main/src/torchaudio/datasets/vctk.py)
also records archive SHA-256
`f96258be9fdc2cbff6559541aae7ea4f59df3fcaf5cf963aae5ca647357e359c`.
This is a maintainer's published checksum, not a publisher signature or a new
local computation. MD5 is a legacy transfer check; retain SHA-256 as the byte
identity. No local archive bytes, size or checksum were verified in this phase.
Metadata API endpoints were unreadable through the web tool; the official full
item page above supplied the published MD5 values without downloading audio.

Required mic2 layout below the extracted original corpus root:

```text
speaker-info.txt
txt/pNNN/pNNN_UUU.txt
wav48_silence_trimmed/pNNN/pNNN_UUU_mic2.flac
```

Keep supplied release notices, with an explicit separate evidence directory if
README/license are outside the extracted root. The official README documents
mic2 technical problems for p280/p315, and the release page documents missing
p315 transcripts. The maintained loader additionally identifies missing p362
recordings. These are known potential exclusions; **no new counts are measured**
here. Do not substitute mic1 or copy another utterance to fill missing pairs.
Source FLAC is acquired original material only; intake writes no processed
FLAC, and all future delivered audio must use the sanctioned PCM16 WAV writer.

### Verified grant and unresolved release questions

The [official README](https://datashare.ed.ac.uk/server/api/core/bitstreams/bb7edd96-5d96-4c0e-8989-1e45597e7b72/content)
and [release license](https://datashare.ed.ac.uk/server/api/core/bitstreams/956a1688-0b59-428c-8a2f-10837433dde3/content)
identify **CC BY 4.0**. It permits reproduction, technical transformations,
adaptation and sharing of material within the licensor's grant, subject to its
terms. Preserve the creators, copyright/license/disclaimer notices, source and
license links, and modification indications. Credit Junichi Yamagishi,
Christophe Veaux and Kirsten MacDonald, University of Edinburgh/CSTR, corpus
title, version and DOI; identify resampling/resynthesis/normalization changes.
Do not imply endorsement. [CC BY 4.0 sections 2–3](https://creativecommons.org/licenses/by/4.0/legalcode.en)
do not automatically grant privacy, publicity or all personality rights, and
prohibit incompatible downstream restrictions on licensed material.

These published terms support conditional local processing; they do not
authenticate an arbitrary local copy or resolve every recording/transcript/
third-party right. Preserve source-text notices and review the actual acquired
release. Check checkpoint/software notices separately before distributing
resynthesized audio. Private Kaggle visibility does not resolve licensing
restrictions. No Kaggle-specific prohibition was identified in CC BY 4.0;
current platform terms and an actual derived release's complete rights ledger
are not cleared here. No upload, slug selection or publication is authorized.
Research non-citability describes evidence status, not a new upstream license.

## 2. Local search and gate disposition

Read-only filename searches covered the user's **Desktop** (including BTP and
BTP_Handoff_Final), **Downloads**, **Documents** and **Music**, using `rg --files
--hidden --no-ignore` with case-insensitive VCTK, speaker-info and mic2 patterns.
Git internals, virtual environments, Python caches, pytest temporary trees and
the already identified Phase 4E fixture source were excluded. Common paths
`C:/Users/samak/datasets`, `C:/Datasets`, and explicitly named VCTK roots under
Desktop/Downloads were also checked and absent. No archive contents were
extracted or searched under an unrelated archive name.

Only the VCTK fixture helper and resampling reference JSON matched outside the
excluded fixture tree. The known Phase 4E source carries
`kind=synthetic_audio_fixture`; its 24 native audio files are not genuine VCTK
speech. They were not used as real intake or generation input. This bounded
search is not proof that no copy exists anywhere on the machine, other drives,
cloud storage or arbitrarily named archives. No real VCTK waveform was opened.

| Gate/result | Actual Phase 4F status |
|---|---|
| Published source/license documentation | Reviewed; not blanket release clearance |
| Authorized local real corpus / archive integrity | Not located / not verified; **BLOCKED** |
| Real speakers, sources, frozen assignments | 0, 0, none |
| Measured real exclusions or speech activity | Not assessed; no real candidate pool |
| Real-data vocoder compatibility/alignment | Not run |
| Paired archive / 16-kHz files | 0 / 0 |
| Real reconstruction, silence, clipping, LUFS, bandwidth measurements | Not measured |
| Real-data deterministic regeneration | Not run |
| Detector training / partial-audio generation | None / none |

The requested real pilot stops at acquisition. No gate was waived, no fixture
outcome was promoted, and no acquisition failure was recorded as an empirical
vocoder failure. Static specifications and CPU fixture/stub tests below remain
separate evidence categories.

## 3. Static vocoder compatibility and remaining integration work

The new `resampling_record()` passes against the current environment:
numpy 2.5.3, scipy 1.18.1, librosa 1.0.0, soxr 1.1.0, soundfile 0.14.0,
libsoxr `0.1.3-14-ga66f3ee`, libsndfile 1.2.2. The existing explicit intake
profile remains `vctk48_archive_soxr_hq_v1`: 48,000 -> 22,050 Hz, HQ,
`fix=True`, `scale=False`, mono float32, exact ceil length, reference-vector
absolute/relative tolerance 1e-6. It is not silently substituted for Vocos's pin
or a legacy bundle's preprocessing. No dependencies were installed/upgraded.

| Path | Declared input/native output and front end | Current local feasibility |
|---|---|---|
| `bigvgan_112m` | 22,050 Hz; upstream extractor; FFT/window 1024, hop 256, 80 mels, fmax 8 kHz | Specification audit passes, but torch, huggingface_hub and matplotlib are absent. No weights loaded or generator called. |
| `vocos` | 24,000 Hz; checked upstream extractor; FFT/window 1024, hop 256, 100 mels, fmax 12 kHz; pinned Kaiser input/output resampling | Specification audit passes, but torch, torchaudio, vocos and huggingface_hub are absent. No weights loaded or generator called. |
| `griffin_lim` | 22,050 Hz; FFT/window 1024, hop 256, 80 mels, fmax 8 kHz; natural-log magnitude, 60 iterations, seed 20260907 | Required CPU numerical libraries are present and adapter tests pass. No real-audio compatibility or generation performed because acquisition is blocked. |

BigVGAN's declared checkpoint is
`nvidia/bigvgan_v2_22khz_80band_fmax8k_256x`, revision
`a9d8b711c344f3c49cc1a05d0c8b6741cf733aa8`, weight SHA-256
`5011c15a31a8d04fb511c1813ef4b0c2fcc0c48043f3952ce06a3b65201a5aaa`.
Vocos is `charactr/vocos-mel-24khz`, revision
`0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21`; its declared admission pin is the
repository revision, with actual weight-file byte identities still to be
recorded on future authorized loading. Griffin-Lim has no learned checkpoint.
These values are read from repository specifications, not newly verified
checkpoint files. A pin-presence audit is not a runtime/empirical pass.

All three resynthesis APIs consume waveform/sample rate without a speaker-ID
or speaker embedding argument. That supports a common processing path, not a
claim of speaker-invariant quality or unseen-speaker generalization. Actual
per-speaker reconstruction and failure rates remain unknown.

Static audit confirms these unresolved integration requirements:

1. `preflight` and `run_gated` in `experiments/phase_a_resynthesis.py` require an
   LJSpeech `metadata.csv`/`wavs` root and instantiate `LJSpeech`. The dormant
   legacy VCTK stub is unimplemented. Do not rename FLACs or forge LJSpeech
   metadata to bypass that boundary. The Phase 4E `IngestedVCTK` bridge alone
   is not a gated VCTK generation driver.
2. Generic `run` reassigns splits over the supplied Corpus. Passing only the
   training subset would create new dev/eval labels inside training speakers,
   rather than inherit the saved map. A metadata-only hypothetical check gives
   24 all-training sources becoming 16/4/4 under that incorrect reuse. No real
   source was reassigned; this demonstrates a future integration trap. The
   driver must accept the frozen map and preserve every generated row as train.
3. Real's 48->22.05 resample is explicit in the intake profile. Existing
   non-Vocos input-side generation calls omit `res_type`, relying on librosa's
   current default. An explicit, tested profile connection is required for
   VCTK before generation; this is a prospective gap, not a reproduced
   correctness defect in historical 22.05-kHz LJSpeech runs.
4. Vocos must take a separate native-source 48->24 path with its existing
   torchaudio `sinc_interp_kaiser`, width 64, rolloff 0.9999,
   beta 14.769656459379492, then one 24->22.05 output conversion. Never use an
   unrecorded 48->22.05->24 input chain. Historical LJSpeech resampler-control
   evidence does not establish this new real-data path's neutrality.
5. The shared postprocessor permits bounded right length padding/truncation
   (`align_length`, at most `sr//20` samples) before the common real-derived
   trim. That existing geometry policy is not permission to compensate delay.
   A new driver must record native/raw, resampled/pre-alignment and final
   lengths/deltas so equal final lengths cannot conceal a mismatch. Preserve
   the existing zero-lag alignment gate and reject excessive drift.
6. Keep all alignment, manifest, pairing, band, calibrated sanity, loudness,
   clipping, membership and publication gates intact. The gated driver currently
   creates its own staging and can delete failed staging by default; a future
   bounded pilot should explicitly preserve failed receipts/staging. No such
   driver was invoked or modified during this acquisition-blocked phase.

## 4. Predetermined bounded plan after acquisition

This is an **unexecuted plan**, not achieved coverage. Use existing Phase 4E
selection with **8 eligible speakers x 4 eligible utterances = 32 sources**,
fixed seed 20260907, existing hash ordering, unchanged 4.1–8.0-s trimmed bounds
and unchanged speaker-disjoint split fractions. Existing rounding gives
**6/1/1 speakers and 24/4/4 sources**. Actual IDs, coverage and exclusions must
come from real intake and be frozen before vocoder generation. No LJSpeech
assignment changes; no new ratios.

Eight speakers rather than the earlier six-speaker fixture topology leave at
least twenty training sources without using dev/eval for generation. Require
all source-format, decoding, finite/saturation, transcript, duplicate, duration
and trim checks. Review declared speaker IDs/mic2 origin against the official
release and listen/review speech activity on training candidates: quiet fraction
and a nonzero waveform are not proof of speech, identity or correct transcript.
Capture exact original hashes and measured exclusions. Assess near duplicates,
shared text/session history and any speaker aliases; current exact hash/text
checks cannot prove their absence.

Freeze the approved full intake map. Choose **20 training sources by round-robin
over sorted training speaker IDs, taking each speaker's source IDs in lexical
order**, independently of vocoder success, for all three conditions. With the
planned six training speakers this gives 4/4/3/3/3/3 sources and represents every
training speaker. No new dev/eval resynthesis or model/feature selection. Do not
backfill failures with held-out or newly chosen speakers. If global exclusions
leave fewer than twenty, stop and report the failed probe budget.

The existing INV-16 gate requires **20 files per vocoder**, exactly zero modal
envelope lag, 50-ms envelope window and the unchanged conditional waveform gate
(correlation threshold 0.30, modal-share floor 0.60). Four or six illustrative
pairs cannot establish compatibility. When all upstream gates can actually run,
the planned unique budget is 20 sources x (real + three vocoders) = **80 archive
WAVs**, plus **80 separately derived 16-kHz WAVs**. These are upper planned
counts, not generated files or independent sample counts. All retain source
and speaker train grouping and development-only/non-citable status.

Process every condition through native-output -> full-band 22,050 archive,
real-derived trim, unchanged -27-LUFS analysis-band normalization, unchanged
true-peak/drop policy, PCM16 WAV; derive 16 kHz once from the archive without
renormalization. Each global utterance drop affects every condition/tier. No
adapter denoising, EQ, individual trim, time shift or gain exception. Use the
sanctioned I/O and manifest path resolver. Preserve checkpoint/frontend/code/
environment identities, source/output SHA-256, raw/final lengths, trim/gain,
alignment arrays, gate records and exact failure reasons.

Repeat the first **four** IDs in that fixed selection order in a fresh output directory for
byte/sample comparison after compatibility admission. A four-file replay is a
determinism check only; it cannot replace the original twenty-file gate or be
published as a separately admitted four-file gated run. Keep original outputs
immutable. Record GPU device/runtime/kernel settings and numeric residuals if
byte identity fails; do not alter tolerances, seeds or accepted outputs to hide
nondeterminism. Separate vocoder environments and compare real output identities
on merge. Additional runtime/model installation or remote compute needs its own
approved environment/compute setup; none was initiated here.

### Required quality and confound evidence before scaling

Measure raw duration error, actual geometry adjustments, finite/peak/clipping
statistics, waveform/envelope correlations, calibrated edge-silence readings,
analysis-band versus full-band LUFS, gain differences and per-band energy for
each training source/condition. Keep real and synthetic transforms symmetric.
Use unchanged physical sanity gates and existing design exceptions. Report
per-vocoder/per-speaker failures and paired counts; do not pool failures away.
Reconstruction differences are expected; distinguish acoustic distortion from
sample misalignment, missing content or processing errors.

Griffin-Lim's above-8-kHz cliff, Vocos resampling-edge cues, the source mic2
transfer function, publisher endpointing/downquantization and normalization's
band dependence remain known concerns. Retain INV-17/18's existing primary
band/membership and separate full-band diagnostics. Do not extrapolate old
single-speaker measurements into a VCTK confound-gate pass. No new detector,
attribution, robustness or unknown-source performance claim follows from a
generation/quality pilot.

## 5. Manual acquisition instructions

No acquisition command below was executed. The next action is for the user to
obtain an authorized original source, not for the agent to launch a download.

1. Open the official release/full metadata links in section 1. Manually download
   the named `VCTK-Corpus-0.92.zip` only if desired; the listing is **10.94 GB**.
   Also save the **original bytes** of `README.txt` and `license_text.txt` from
   that release, not browser-reformatted copies. Preserve URL, version, download
   date, actual byte size, all original notices and rights-review notes.
2. Before extraction compute the archive MD5 and SHA-256. Compare against
   section 1's publisher MD5 and maintainer SHA-256; any mismatch blocks intake.
   Check the README/license MD5 too. Record actual values rather than copying
   expected values into a purported local receipt. MD5 equality alone is not
   authenticity or legal clearance.
3. Extract safely into a fresh directory outside Git/legacy datasets, with
   enough space for archive and extracted originals. Retain a read-only original
   copy. Choose the directory containing `speaker-info.txt`, `txt` and
   `wav48_silence_trimmed` as `--source-root`; do not pass the ZIP or its parent.
   Keep acquisition receipts outside this strict source inventory root.
4. Alternatively supply an authorized unmodified subset in that layout with
   original file identities and release evidence. Record exactly how the
   candidate pool/speakers were chosen. Never claim the full archive checksum
   was verified if only a subset was obtained. All selection/duplicate checks
   then apply only to that supplied candidate pool; they do not prove global
   corpus coverage or duplicate absence. Do not cherry-pick detector-friendly
   or good-sounding resyntheses.
5. Provide the local source/evidence location for a new real intake. A full-root
   Phase 4E intake currently hashes the complete supplied inventory and decodes
   every mic2 candidate before ranking; selecting 32 does not make that scan
   cheap. Review the candidate-pool scope before launch. If fewer than eight
   qualifying speakers with four eligible sources exist, report the shortfall.
   Do not weaken gates, fabricate coverage or silently expand/reshuffle it.

Future-only PowerShell checksum and intake commands:

```powershell
$archivePath = '<downloaded-original-VCTK-Corpus-0.92.zip>'
Get-Item -LiteralPath $archivePath | Select-Object FullName,Length
Get-FileHash -Algorithm MD5 -LiteralPath $archivePath
Get-FileHash -Algorithm SHA256 -LiteralPath $archivePath
Get-FileHash -Algorithm MD5 -LiteralPath '<original-README.txt>'
Get-FileHash -Algorithm MD5 -LiteralPath '<original-license_text.txt>'

& '.venv/Scripts/python.exe' -B -X utf8 -m experiments.cli ingest vctk `
  --source-root '<reviewed-real-VCTK-root>' `
  --evidence-dir '<official-release-evidence-directory>' `
  --out-dir '<fresh-real-pilot-intake-directory>' `
  --source-kind local_dataset --speakers 8 --utterances-per-speaker 4
& '.venv/Scripts/python.exe' -B -X utf8 -m experiments.cli ingest verify `
  --ingestion-dir '<fresh-real-pilot-intake-directory>'
```

The current ingestion profile remains unmodified and marks scientific admission,
source authenticity and redistribution clearance as incomplete/false. Human
acquisition review and later empirical gates must remain explicit. There is no
valid current VCTK `resynth` command to provide: driver integration must first
preserve the frozen map and all existing gates. A generation command that only
looks plausible would misrepresent this blocker.

## 6. Reproduction, tests and preservation

Read-only audit artifacts are under
`runs/vocotrace-phase4f-audit-20261107/`: initial snapshot `before.json`,
`static_compatibility.json`, the correct-root `gate_review-v2.json`,
`audit_phase4f.py`, focused/full/Ruff logs and the final preservation receipt.
An initial helper receipt `gate_review.json` used an incorrect parent-derived
search root; it is preserved as a superseded debugging artifact and is not
acquisition evidence. The corrected helper uses the actual user home and
requires valid search roots. Independent direct searches and the corrected
receipt agree; no real audio was processed in either case.

Actual invocation names follow. Choose **new** receipt filenames and pytest
basetemp paths on replay; never reuse a basetemp containing previous artifacts.

```powershell
& '.venv/Scripts/python.exe' -B -X utf8 `
  runs/vocotrace-phase4f-audit-20261107/audit_phase4f.py `
  --receipt gate_review-v2.json

& '.venv/Scripts/python.exe' -B -m pytest -o addopts='' -q -ra `
  tests/test_source_ingestion.py `
  tests/test_invariants.py::TestINV16Alignment `
  tests/test_phase_a_run.py::TestPreflight `
  tests/test_phase_a_run.py::TestGriffinLimAdapter `
  tests/test_phase_a_run.py::TestBigVGANAdapter `
  tests/test_phase_a_run.py::TestVocosAdapter `
  --basetemp runs/vocotrace-phase4f-audit-20261107/pytest-focused-1

& '.venv/Scripts/python.exe' -B -m pytest -o addopts='' -q -ra `
  --basetemp runs/vocotrace-phase4f-audit-20261107/pytest-full-1
& '.venv/Scripts/python.exe' -B -m ruff check src tests
& '.venv/Scripts/python.exe' -B -m ruff check .
& '.venv/Scripts/python.exe' -B -X utf8 `
  runs/vocotrace-phase4f-audit-20261107/audit_phase4f.py `
  --receipt final_review.json
git --no-optional-locks rev-parse HEAD
git --no-optional-locks status --short --untracked-files=all
git --no-optional-locks diff --cached --name-only
git --no-optional-locks diff --check
```

Focused CPU fixture/stub tests: **99 passed, 2 skipped**, 89.74 s. Windows denied
actual symlink and hardlink creation (WinError 1314/5). These tests validate
ingestion determinism, duplicate/split/provenance guards, format errors,
alignment enforcement and adapter contracts; they do not establish empirical
real-VCTK inference. Full pytest: **1,016 passed, 4 failed, 9 skipped, 1 xfailed**
in **788.79 s**. The four failures are unchanged from Phase 4E:

- `test_kaggle_harness.py::TestGeneratedNotebook::test_the_committed_notebook_is_exactly_what_the_harness_generates`
- `test_phase_a_run.py::TestKaggleNotebook::test_the_generated_notebook_is_the_committed_one`
- `test_phase_a_run.py::TestUpstreamImportGuardThroughADirectoryLink::test_namespace_package_through_the_link_passes`
- `test_phase_a_run.py::TestUpstreamImportGuardThroughADirectoryLink::test_namespace_package_from_elsewhere_fails`

The notebook failures compare CRLF/LF bytes; the directory-link tests fail with
Windows junction permission errors. The nine existing skips concern link
privileges, and the expected failure remains the unimplemented INV-17 ablation
quantization contract. No new failure appears. Logs are `focused.log`,
`full.log`, `ruff-src-tests.log` and `ruff-all.log` in the audit directory.
`ruff check src tests` passes. Repository-wide Ruff retains **11 existing
generated-notebook findings**; none were edited or suppressed.

Final preservation checks passed for all **135 pre-existing tracked files** and
inventory/size/nanosecond-mtime records for **4,722 files across 20 protected
roots**, including prior pilots and Phase 4E sources/receipts. No historical
audio or feature array bytes are read by this audit; no new cryptographic
historical-WAV verification is claimed. Source code, test behavior, model
bundles, checkpoint pins, CLAUDE, invariants, LJSpeech split/data files and the
user's existing `.gitignore` edit remain unchanged. No frozen legacy eval
waveform or feature analysis occurs. No detector training, partial generation,
dependency/model download, remote job, publication or Git write was performed.
The completed receipt is `runs/vocotrace-phase4f-audit-20261107/final_review.json`;
its live search covers the four user directories listed above, and its split
topology demonstration is explicitly hypothetical, not a real-data result.

Only new reviewable repository file: **this report**. New ignored artifacts:
the audit script, snapshots/receipts and test logs/temp fixtures under the new
Phase 4F audit directory. All changes remain unstaged; HEAD/index are preserved.

## 7. Scaling decision

**Do not scale.** The real experiment has not started, and no quality or
compatibility pass exists. Next acquire/review an authorized original VCTK
sample, run the predetermined bounded intake, freeze its actual speaker/source
assignments, and resolve the explicitly identified generation integration and
runtime requirements. Only then run the twenty-training-source compatibility
pilot and fixed replay with unchanged gates. Reassess scaling from those actual
measurements. The 50-speaker/500-source cohort remains conditional.
