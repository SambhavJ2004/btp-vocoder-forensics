# VocoTrace Phase 4E: multi-speaker ingestion foundation

**Development-only; non-citable. Fixture validation, not real-speech ingestion or
scientific admission.** This phase implements read-only VCTK 0.92 mic2 intake,
explicit resampling previews, deterministic speaker-disjoint selection and
independent artifact verification. No detector or vocoder was trained or run.

Audit date: 2026-11-07, Asia/Calcutta. Starting manually created HEAD:
`d35408304f62e4bf3065c2766a1b71c9afe29946`. The pre-existing unstaged `.gitignore`
edit is preserved. No Git write operations, dependency changes, audio-archive
downloads, Kaggle operations or old frozen-eval measurements are part of this
phase. The new fixture partition named `eval` is unrelated to the frozen
LJSpeech evaluation population.

## 1. Dataset decision and verified public facts

Recommend **VCTK 0.92 mic2 as the primary** and **AISHELL-3 as the conditional
backup**. AISHELL-3 would require a separately reviewed Mandarin cohort, native
44.1-kHz resampling profile and format adapter. It is not an interchangeable
fallback inside the implemented VCTK loader. Hi-FiTTS is unsuitable for the
proposed pristine 50-speaker cohort: it has ten speakers and MP3 acquisition
history. This choice does not admit any of these sources as research evidence.

| Candidate | Official source/download | Speakers, identity and format | Suitability and decision |
|---|---|---|---|
| VCTK 0.92 | [Edinburgh release and 10.94-GB ZIP](https://datashare.ed.ac.uk/handle/10283/3443), DOI 10.7488/ds/2645 | 110 English speakers; published speaker IDs and speaker-info metadata. Distributed recordings are 48-kHz/16-bit, downsampled/end-pointed by the publisher from 96-kHz/24-bit studio recordings. | Primary. Fixed mic2 only; do not mix microphone transfer functions. Published mic2 problems for p280/p315 and lost p315 transcripts require measured exclusions, not replacements from mic1. Shared prompts and prior processing remain material. |
| AISHELL-3 | [OpenSLR 93, 19-GB data_aishell3.tgz and mirrors](https://www.openslr.org/93/) | 218 Mandarin speakers; speaker IDs plus declared gender, age-group and accent metadata; 44.1-kHz/16-bit recording specification, WAV layout in the maintained ESPnet recipe, transcripts and pinyin. Actual container/channel headers have not been inspected locally. | Conditional backup. Quiet indoor acquisition is promising; language, demographic imbalance, speaker overlap and checkpoint-domain compatibility need new review. No AISHELL adapter or real-file validation is claimed. |
| Hi-FiTTS | [OpenSLR 109, 41-GB hi_fi_tts_v0.tar.gz and mirrors](https://www.openslr.org/109/) | Ten English readers with reader/book metadata, 44.1 kHz. The authors acquired 128-kbps MP3 chapters; a later container conversion does not remove that history. No local distributed-file header verification. | Reject for this pristine 50-speaker cohort. Reader/channel differences, clean/other selection and compression history could dominate forensic evidence. |

The [VCTK README](https://datashare.ed.ac.uk/server/api/core/bitstreams/bb7edd96-5d96-4c0e-8989-1e45597e7b72/content)
identifies the release and license. The maintained
[PyTorch VCTK implementation](https://raw.githubusercontent.com/pytorch/audio/main/src/torchaudio/datasets/vctk.py)
confirms the `wav48_silence_trimmed/<speaker>/<utterance>_mic2.flac` and
`txt/<speaker>/<utterance>.txt` layout and additional missing p362 recordings.
This phase uses repository audio I/O, not torchaudio's downloader/loader. Only
the acquired original may be FLAC: intake writes no FLAC intermediate or
processed audio. Future delivered audio must use the sanctioned PCM16 WAV path.

AISHELL-3 recording conditions, speaker attributes and demographic imbalance
are described by the [dataset authors](https://arxiv.org/html/2010.11567v2).
The maintained [ESPnet data preparation](https://raw.githubusercontent.com/espnet/espnet/master/egs2/aishell3/tts1/local/data_prep.py)
uses `wav/<speaker>/<recording>` and `content.txt`, with the first seven filename
characters as the speaker ID. This is layout evidence, not local header or
identity validation; its training recipe's trimming/splits are not adopted here.
Hi-FiTTS compression history and quality selection are described by its
[authors](https://arxiv.org/html/2104.01497v3); perceptual quality does not make
compressed source audio a lossless genuine reference.

### License, attribution and publication ledger

These are verified published license declarations and their stated permissions,
not certification of every recording, transcript or personality right.

| Candidate | Local transformation and derived redistribution | Attribution/notices and Kaggle status |
|---|---|---|
| VCTK | Official README and [release license](https://datashare.ed.ac.uk/server/api/core/bitstreams/956a1688-0b59-428c-8a2f-10837433dde3/content) identify CC BY 4.0. The grant permits reproduction, adaptation and sharing subject to its terms. | Credit Junichi Yamagishi, Christophe Veaux, Kirsten MacDonald / University of Edinburgh, corpus title, version, source and DOI; retain applicable notices/license link and mark resampling, normalization and resynthesis changes. Check transcript/third-party notices. No Kaggle-specific ban was identified in the license, but no publication clearance is issued. |
| AISHELL-3 | OpenSLR declares Apache License 2.0. Its copyright grant includes reproduction, derivative works and distribution subject to section 4. | Preserve the actual release license, relevant attribution/copyright notices, any NOTICE, and prominent modification notices; credit Beijing Shell Shell Technology and the dataset authors. Confirm the archive's notices and scope for speech/transcripts before any redistribution. No license-specific Kaggle prohibition identified; not cleared. |
| Hi-FiTTS | OpenSLR declares CC BY 4.0; the same conditional transformation/sharing grant applies. | Credit Bakhturina, Lavrukhin, Ginsburg and Zhang and preserve source/license/reader/book notices as applicable; mark changes. The authors discuss speaker consent. Neither public accessibility nor the declared license resolves all source/text/voice rights. Not selected or cleared for Kaggle. |

Under [CC BY 4.0 sections 2–3](https://creativecommons.org/licenses/by/4.0/legalcode.en),
preserve supplied attribution/copyright/license/disclaimer notices, source links
and modification indications, avoid implied endorsement and do not impose
incompatible downstream restrictions. Privacy, publicity and other rights are
not automatically granted. Under [Apache-2.0 sections 2 and 4–6](https://www.apache.org/licenses/LICENSE-2.0),
include the license, retain required notices and identify modified files;
trademark rights are not granted. Check actual release contents before relying
on a landing-page license label. Neither license makes a private upload exempt
from its conditions.

The [Kaggle terms page](https://www.kaggle.com/terms) returned no readable text
to this audit's web reader. Current platform terms and the exact upload license
therefore remain **unverified**, requiring separate review before any upload.
No new owner/slug or publishing configuration is selected here. Earlier owner
confirmation for btpvf-partial-v1 does not authorize this cohort's publication.
Repository software licensing does not license the source audio. Research
`non-citable` status is an evidence limitation, not an additional legal license
restriction. Pinned vocoder/checkpoint rights and attribution need review before
later distribution of resynthesized outputs.

## 2. Implementation and scope

`src/data/source_ingestion.py` adds the VCTK-specific adapter and reusable
inventory, safety, hashing, measurement, selection and verification functions.
`btpvf ingest vctk` invokes intake; `btpvf ingest verify` independently remeasures
it. Explicit `--source-kind fixture|local_dataset` prevents silent promotion of
marked fixtures. A real dataset's README/license text is required, but recognizing
license/version strings does not authenticate its origin. Both authenticity and
redistribution clearance remain false in provenance.

Originals are read through `data.audio_io.probe/read_native`. Headers are checked
**before** decoding, preventing the generic reader's downmix from concealing
stereo. Required native inputs: FLAC magic, mono, 48,000 Hz, PCM_16, consistent
header/decoded sample count, finite samples, positive duration at most 60 s,
nonzero signal and no PCM endpoint samples. Missing/malformed IDs, speaker
records, transcripts, bad UTF-8, empty/oversized transcripts, corrupt decoding
and unsupported formats are rejected with observed evidence. No missing pair,
microphone, sample rate, text or alignment is repaired.

Selection measures the existing real-reference trim on an in-memory 22,050-Hz
preview. It uses the unchanged relative 30-dB, 512-frame, 128-hop, 20-ms-context
policy and the predetermined **4.1–8.0-s** initial-cohort bounds from Phase 4D.
Raw duration/peak, native samples, preview geometry/hash, trim span and relative
quiet-sample fraction are recorded. Quiet fraction is descriptive, not a new
silence threshold; all-zero audio is rejected. These checks do not establish
speech activity, SNR, speaker authenticity or transcript correctness. Existing
scientific invariant exceptions propagate rather than becoming exclusions that
allow a failed scientific gate to pass.

Every member of an exact source-file, decoded-float32-audio or normalized-text
SHA-256 duplicate group is excluded **before** ranking/splitting. Text matching
uses Unicode NFKC, case folding and whitespace collapse. This conservatively
removes repeated common prompts; it does not detect paraphrases, punctuation
variants, near-duplicate audio or cross-corpus speaker aliases. Missing metadata
or incompatible source layout fails explicitly. All source files are inventoried,
including unselected recordings and mic1 when present; only mic2 candidates are
decoded/selected. Unexpected files are rejected, not silently skipped. A proper
extracted release root is required; unrelated acquisition files belong outside it.

Paths must remain below their declared root. Symlinks, junctions and multiply
linked files are rejected. Output must be fresh and disjoint from originals and
release-evidence directories. There is no overwrite/delete/repair mode. Failed
selection leaves a failed validation receipt and measured reasons; it emits no
fabricated successful manifest. Filesystem concurrent-writer protection is not
a transactional lock: use immutable, access-controlled source directories.

### Frozen new-cohort grouping

IDs are namespaced, for example `vctk092_p201` and
`vctk092_mic2_p201_001`. Qualify speakers with the requested number of eligible
utterances, rank by SHA-256 of
`20260907:vocotrace-ms-v1:speaker:<speaker_id>`, and rank each speaker's sources
by `20260907:vocotrace-ms-v1:utterance:<source_id>`; canonical IDs break ties.
No acoustic-model result enters selection.

Reuse `assign_splits(..., speaker_disjoint=True, seed=20260907)` and the existing
70/10/20 policy **at speaker level**. Fifty speakers imply 35/5/10 speakers,
and 350/50/100 sources if ten survive per speaker. These are planned counts,
not acquired data. At least six speakers are required to give all three
partitions nonempty under existing rounding. Insufficient qualified speakers
produce a failure and shortfall/selection receipts; there is no automatic smaller
cohort or split reassignment. Ranking is stable for identical inventories, but
adding/changing eligibility can change a newly generated cohort; retain and
freeze the approved manifest. Verification recomputes the saved selection and
rejects discrepancies. Post-generation exclusions must retain that frozen map.

`IngestedVCTK(Corpus)` exposes checked utterances and a copy of `frozen_splits`.
This bridge is **not** integrated into the gated LJSpeech-only Phase A CLI.
The dormant legacy `VCTK` stub is unchanged (its mic1 suggestion is obsolete for
this separately named mic2 intake). Future integration must consume the saved
map explicitly rather than regenerate it after exclusions. Speaker IDs, text,
source paths, demographics, microphone and vocoder labels are provenance or
supervision metadata, never inference features. No feature extractor/model is
invoked by intake.

## 3. Resampling reproducibility

The new profile is `vctk48_archive_soxr_hq_v1`. It fails closed unless its exact
runtime identity matches. No dependencies were installed or changed.

| Component/setting | Pinned value |
|---|---|
| Python packages | numpy 2.5.3; scipy 1.18.1; librosa 1.0.0; soxr 1.1.0; soundfile 0.14.0 |
| Native libraries | libsoxr `0.1.3-14-ga66f3ee`; libsndfile 1.2.2 |
| Call | `librosa.resample(..., orig_sr=48000, target_sr=22050, res_type="soxr_hq", fix=True, scale=False, axis=-1)` |
| Samples | mono finite float32 input/output; no loudness normalization or alignment compensation |
| Exact length | `ceil(N * 22050 / 48000) = (N*22050 + 47999)//48000`; librosa's fix=True right-padding/truncation only |
| Numerical acceptance | reference vectors `atol=1e-6, rtol=1e-6`; repeated local calls must be sample-identical; no cross-platform bitwise guarantee |

The wrapper explicitly supplies all parameters; it does not replace
`resample_once`, `resample_with_spec`, legacy generation calls or saved model
implementation identities. A version mismatch requires review of a new profile,
not substitution of a resampler or an automatic dependency upgrade. Current
runtime pins are not evidence that old bundles used this environment.

`tests/fixtures/vctk_resampling_v1.json` stores complete input/output samples for
impulse (257 samples), ramp (513) and mixed signal (1,025) regression vectors
captured in this environment. They are synthetic regression references, not
independent empirical proof of resampler fidelity. Independent length arithmetic,
unchanged-input checks, 1-kHz passband and 12-kHz stopband assertions complement
them. Existing `resample_once` agrees on tested 48-kHz vectors; no legacy source
file was changed. [soxr's API documentation](https://python-soxr.readthedocs.io/en/latest/soxr.html)
describes the HQ non-streaming interface.

## 4. Artifact contract and integrity

Successful intake contains exactly twelve metadata/evidence files:

| File | Meaning |
|---|---|
| plan.json | Version, microphone, explicit cohort policy, source kind, URLs and non-citable flags |
| source_inventory.json | Relative path, size and original byte SHA-256 for every allowed source file |
| source_manifest.csv | Selected IDs/splits, source/transcript/decoded hashes, native format, resampling preview and trim measurements |
| speakers.json | Declared native and namespaced speaker identity/attribute records, not verified personal identities |
| selection.json | Candidate/eligible counts, qualified/ranked speakers, per-speaker availability, selected and unselected decisions |
| exclusions.json | Measured rejection reasons, observations and collision memberships |
| resampling.json | Exact versioned algorithm/environment/length/tolerance settings |
| provenance.json | Original roots, original release-evidence byte hashes, Git HEAD/status, Python/platform and implementation SHA-256; rights/authenticity false |
| validation.json | Actual source/speaker/split counts, unchanged-source result, zero processed audio/models, incomplete scientific admission |
| release_readme.txt, release_license.txt | UTF-8 copies of supplied release evidence; fixture copies explicitly disclaim real source rights |
| checksums.json | Sizes and SHA-256 for the eleven other files; not an authority to add content |

This is a **new native-intake schema**, not a production `data.manifest` archive.
Relative paths resolve against the explicit source root in provenance; relocation
requires an explicit override and identical inventories. Verification checks the
exact schema, links, all file sizes/hashes, implementation/profile/claims, live
source inventory and original release evidence; then decodes and recomputes
measurements, exclusions, ranking and speaker splits and compares canonical rows
and receipts. Resealing checksums alone cannot authorize a changed split, source
hash, selection, scientific status or extra content. Local SHA-256 receipts prove
consistency with supplied bytes, not publisher authenticity or legal rights.
Source inventory is rechecked after measurement. Output verification does not
require or consume legacy LJSpeech eval audio.

## 5. Actual fixture pilot

No suitable VCTK local sample was found in the inspected BTP trees, named
Downloads locations or common dataset directories. This was not an exhaustive
whole-machine search. No external speech archive was acquired. Test-first intake
was followed by a saved synthetic tones/noise fixture dry-run with explicit
`synthetic_audio_fixture` markers. **Real speech ingested: 0; real speakers: 0.**

Final saved artifacts: `runs/vocotrace-phase4e-fixture-20261107/intake-v2/`;
repeat: `intake-v2-repeat/`; original synthetic source: sibling `source/`.

| Quantity | Actual result |
|---|---:|
| Simulated speakers / qualified / selected | 6 / 6 / 6 |
| Candidate source identities | 25 |
| Native audio files present | 24, including deliberately invalid files; one candidate lacks audio |
| Complete source inventory | 53 files including transcripts and fixture evidence |
| Eligible unique sources | 18 |
| Selected / eligible unselected | 12 / 6 (utterance rank limit) |
| Rejected candidates | 7 |
| Selected native audio duration | 55.2 s |
| Selected in-memory trimmed-preview duration | 54.04843537414966 s |
| Processed/paired audio outputs / models | 0 / 0 |
| Original fixture files / final intake metadata, logical bytes | 5,671,130 / 27,808 |

| New fixture split | Speakers | Selected sources |
|---|---:|---:|
| train | 4 | 8 |
| dev | 1 | 2 |
| eval | 1 | 2 |

| Exclusion | Candidates |
|---|---:|
| silent_source | 1 |
| trimmed_duration_outside_cohort_bounds | 1 |
| unreadable_audio | 1 |
| missing_audio | 1 |
| unsupported_native_format (44.1-kHz fixture) | 1 |
| duplicate_source_or_content (both members excluded) | 2 |

Independent count inspection caught an intake bookkeeping defect: exclusions
aliased the malformed-candidate list, inflating the reported candidate count
from 25 to 32. The lists are now independent; a regression assertion requires
25 = 18 eligible + 7 rejected. Selected sources, exclusions and split assignments
were unaffected. Initial `intake/` and `intake-repeat/` receipts are preserved as
superseded debugging artifacts; their code identity differs and current verification
rejects them. The v2 receipts are the final fixture evidence for this phase.

Final verification and byte-identical repeat evidence are recorded in
`runs/vocotrace-phase4e-audit-20261107/final_review.json`. Counts and unchanged
checks remain development-only. No alignment, vocoder checkpoints, loudness,
paired-duration or confound gates are claimed to have passed on real data.

## 6. Future pairing plan and next steps

1. With separate acquisition authorization, obtain the **original** Edinburgh
   VCTK-Corpus-0.92.zip using the named download on the official landing page.
   Preserve its URL, acquisition date, byte size, SHA-256 and supplied publisher
   digest evidence. Retain the official README and license bytes from the same
   release separately. Do not use a compressed/reencoded mirror, noisy variant,
   enhanced derivative or synthetic corpus. Verify safe extraction outside old
   dataset roots; record the released 96-to-48-kHz/downquantization history.
2. First ingest a bounded lawful local multi-speaker source subset with the
   same format/identity gates and explicit receipts. Do not interpret arbitrary
   fixture identities or strings as real-data rights evidence. Inspect real
   inventory/headers, shared prompts, near duplicates, session/recording history,
   source authenticity and speaker aliases. Do not infer eligibility of 50
   speakers from the published total of 110.
3. Review the measured exclusions and license/attribution ledger. Then run the
   predetermined 50 x 10 selection only if enough admissible sources remain.
   Inventorying a complete release rehashes all allowed files, and selection
   currently decodes all mic2 candidates before ranking; expect substantial CPU
   and I/O work even though only 500 are selected. No performance estimate was
   benchmarked on the real corpus. A future optimization must preserve the same
   duplicate evidence and scientific selection, not bypass validation.
4. Freeze the approved manifest and speaker map in a **manual** commit before
   any research record-mode generation. Add a narrowly tested Phase A adapter
   that consumes this map and the reviewed resampling profile explicitly. The
   current gated CLI remains LJSpeech-only; generic `run_phase_a` presently
   assigns splits anew and is not a substitute for this integration.
   Research admission must reject `source_kind=fixture`, require the external
   source/rights review and retain all downstream gates. The Corpus bridge by
   itself grants no scientific or publication admission.
5. Initially use the same admitted BigVGAN, Griffin-Lim and Vocos conditions.
   Preserve each upstream mel/front end, immutable checkpoint revision/weight
   digest and isolated environment. For real, derive 48,000 -> 22,050 once.
   Vocoder inputs are separate native-source paths: 48,000 -> the 22,050-Hz
   front ends using the reviewed soxr settings; 48,000 -> 24,000 for Vocos with
   its existing pinned Kaiser specification. Do not feed a 48->22.05->24 chain
   as an unrecorded input path. Existing non-Vocos Phase A input code relies on
   a default librosa res_type; explicit profile wiring and an end-to-end audit
   are still required before this new corpus is generated.
6. Admit each vocoder only after the existing **20-file** length/alignment
   preflight, required zero sample lag and existing correlation/modal-share
   gates. Fixture counts do not satisfy or replace this gate. No shifting,
   stretching, denoising or per-condition repair. Reject unsuitable recordings
   or halt incompatible conditions with evidence. Apply the real-derived trim
   span to every archive counterpart, unchanged loudness/true-peak processing,
   full-band 22,050-Hz mono PCM16 WAV writing and 16-kHz derivation once from
   the completed archive without renormalization. No source FLAC intermediates
   enter the processed pipeline.
7. Record every source/condition/tier path, original and output SHA-256,
   resampling counts/specifications, mel configuration, checkpoint/environment,
   raw/output lengths, trim span, gain, alignment evidence, frozen split,
   code identity and exclusion reason. Check exact paired membership before
   analysis; preserve frozen grouping if entire groups are excluded later.
8. Run unchanged scientific gates symmetrically, with existing calibration and
   design exceptions, before relying on results. Include real resampling controls,
   measured below-8-kHz checks and full-band diagnostics. Native mic2 bandwidth,
   endpointing, downquantization, Vocos roundtrip and Griffin-Lim bandwidth cues
   remain confounds; retain existing INV-17/18 membership rather than treating
   full-band shortcuts as source robustness. Add controlled partial audio only
   after valid paired full utterances exist, with both null/real-to-real controls.
9. Keep all conditions, tiers, windows and augmentations of each source in its
   frozen speaker partition. Limit experimental selection to group-disjoint
   training folds; calibrate on the prescribed development partition once and
   preserve the new final evaluation population. Check upstream training speaker/
   corpus overlap before claiming unseen-speaker/generalization performance.
   Do not transfer conclusions about the exposed legacy dev set to a clean test.

No current result supports stronger robustness, genuine-speech recognition,
unknown-source detection, speaker identity verification or deployment. Real-data
acquisition/rights review, near-duplicate/session review, integration, empirical
alignment/processing gates and approved generation compute remain blockers.
Sampling bounds, mic2-only eligibility, exact-text duplicate removal and hash
ranking can bias accent, content and duration coverage; report these biases.
All 18 invariants remain binding; intake is an earlier layer, not a new gate
exception or a completed research dataset.

## 7. Reproduction

PowerShell from the repository root, using a **fresh** output tree and pytest
basetemp for every rerun. The original invocation used the indicated fixture
source path; do not overwrite it.

```powershell
& '.venv/Scripts/python.exe' -B -m pytest -o addopts='' -q -ra `
  tests/test_source_ingestion.py `
  --basetemp runs/vocotrace-phase4e-audit-20261107/pytest-focused-3

& '.venv/Scripts/python.exe' -B tests/vctk_fixture.py `
  --out-dir runs/vocotrace-phase4e-fixture-20261107/source --include-invalid

& '.venv/Scripts/python.exe' -B -m experiments.cli ingest vctk `
  --source-root runs/vocotrace-phase4e-fixture-20261107/source `
  --out-dir runs/vocotrace-phase4e-fixture-20261107/intake-v2 `
  --speakers 6 --utterances-per-speaker 2 --source-kind fixture

& '.venv/Scripts/python.exe' -B -m experiments.cli ingest verify `
  --ingestion-dir runs/vocotrace-phase4e-fixture-20261107/intake-v2

& '.venv/Scripts/python.exe' -B -m experiments.cli ingest vctk `
  --source-root runs/vocotrace-phase4e-fixture-20261107/source `
  --out-dir runs/vocotrace-phase4e-fixture-20261107/intake-v2-repeat `
  --speakers 6 --utterances-per-speaker 2 --source-kind fixture

& '.venv/Scripts/python.exe' -B -m pytest -o addopts='' -q -ra `
  --basetemp runs/vocotrace-phase4e-audit-20261107/pytest-full-2
& '.venv/Scripts/python.exe' -B -m ruff check src tests
& '.venv/Scripts/python.exe' -B -m ruff check .
& '.venv/Scripts/python.exe' -B runs/vocotrace-phase4e-audit-20261107/verify_review.py
git --no-optional-locks diff --check
git --no-optional-locks status --short
```

**Future-only, not executed:** after authorized acquisition, preserve the ZIP
with `Get-FileHash -Algorithm SHA256 -LiteralPath <downloaded-zip>` and keep the
official README.txt/license_text.txt in a separate evidence directory. Point
`--source-root` at the extracted directory containing `speaker-info.txt`, `txt`
and `wav48_silence_trimmed`, not its parent or the ZIP. Then:

```powershell
& '.venv/Scripts/python.exe' -B -m experiments.cli ingest vctk `
  --source-root '<reviewed-original-VCTK-Corpus-0.92-root>' `
  --evidence-dir '<official-release-evidence-directory>' `
  --out-dir '<fresh-local-intake-directory>' `
  --speakers 50 --utterances-per-speaker 10 --source-kind local_dataset
& '.venv/Scripts/python.exe' -B -m experiments.cli ingest verify `
  --ingestion-dir '<fresh-local-intake-directory>'
```

Keep source/evidence roots read-only. Intake records paths/digests and evidence
but does not verify a publisher signature or archive acquisition automatically.
Retain that external acquisition ledger. Do not execute a 50-speaker intake,
download command, generation, training or upload as an implicit consequence of
this plan.

## 8. Tests, changed files and preservation

Preservation verification passed for **4,621 files across 19 historical roots**:
identical inventories, byte sizes and nanosecond modification times, plus matching
non-audio file SHA-256 records. Historical WAV content was not read; this is not
a new cryptographic verification of historical WAV bytes. Saved non-audio arrays
were hashed opaquely, not loaded or evaluated. Only `src/experiments/cli.py`
differs among pre-existing tracked files; CLAUDE, invariants, old resampling,
checkpoint pins, source/model implementations and the pre-existing `.gitignore`
edit match the initial snapshot. HEAD is unchanged and the index is empty.

The final focused run passed **54 tests, with 2 skipped** in 133.45 s. Windows
denied actual symlink creation (WinError 1314) and hardlink creation (WinError 5);
these are explicit platform-limited skips, not successful live link tests.
Unsafe relative paths and schema/tamper rejection are exercised independently.
Full pytest completed in **1,290.49 s**: **1,016 passed, 4 failed, 9 skipped,
1 xfailed**. The four failures match the Phase 4D baseline exactly:

- `test_kaggle_harness.py::TestGeneratedNotebook::test_the_committed_notebook_is_exactly_what_the_harness_generates`
- `test_phase_a_run.py::TestKaggleNotebook::test_the_generated_notebook_is_the_committed_one`
- `test_phase_a_run.py::TestUpstreamImportGuardThroughADirectoryLink::test_namespace_package_through_the_link_passes`
- `test_phase_a_run.py::TestUpstreamImportGuardThroughADirectoryLink::test_namespace_package_from_elsewhere_fails`

The first two compare CRLF and LF notebook bytes; the latter two fail creating
Windows directory junctions with WinError 5. The nine skips comprise the seven
previous link-privilege skips plus the two new live-link tests above. The
unimplemented INV-17 ablation quantization contract remains an expected failure.
No new failure appears. An earlier full run was stopped after the receipt fix
superseded its code; `pytest-full-2.log` is the completed final-code run.
`ruff check src tests` passes; repository-wide Ruff retains **11 existing
generated-notebook errors**. No notebook or enforcement test was weakened.
Logs are in `runs/vocotrace-phase4e-audit-20261107/` (`focused-3.log`,
`pytest-full-2.log`, `ruff-src-tests.log`, `ruff-all.log`, `final_review.json`).

Changed implementation: `src/experiments/cli.py`; new files:
`src/data/source_ingestion.py`, `tests/test_source_ingestion.py`,
`tests/vctk_fixture.py`, `tests/fixtures/vctk_resampling_v1.json`, and this report.
The new tests cover speaker isolation/counting, deterministic ranking and
resampling, all-member duplicate exclusion, original immutability, native-format
errors, provenance and schema integrity, resealed tampering, relocation, unsafe
paths/links, shortfalls and CLI verification. Test-only FLAC writes simulate
external acquisition; production ingestion writes metadata only.

The foundation is ready for independent code review and a manual commit with
these baseline/platform limitations disclosed. It is not a validated real-speech
dataset, a complete multi-corpus generation driver or publishing authorization.
All changes remain unstaged, including the user's pre-existing `.gitignore`
change; no Git write was performed.
