# VocoRoute-P N2B.0 corpus preparation

Local staging uses synthetic audio. The real archive already exists in the private Kaggle
notebook `sambhavjainsj/vocoroute-vctk092-acquisition`, version **2**. The preparation
and held-out replay commands below are approval-gated. [Acquisition notebook](https://www.kaggle.com/code/sambhavjainsj/vocoroute-vctk092-acquisition)

## Source and inventory

Attach version 2 as a read-only notebook output. The worker locates exactly one
`vctk-0.92/VCTK-Corpus-0.92.zip` and pins `acquisition-receipt.json` to SHA-256
`b578c8878525ff3db87e25b4fd659d3962d75155295c15ed59e6ca07a34393fe`. It hashes both
sidecars and streams the mounted ZIP once to verify **11,747,302,977 bytes**, MD5
`8a6ba2946b36fcbef0212cad601f4bfa` and SHA-256
`f96258be9fdc2cbff6559541aae7ea4f59df3fcaf5cf963aae5ca647357e359c`. It preserves the
original README, license, receipt and `speaker-info.txt` in every release. The
archive remains in its existing Kaggle mount. [Edinburgh release and attribution](https://doi.org/10.7488/ds/2645)

The inventory reads the ZIP index and transcript bytes without unpacking the
archive. It rejects unsafe paths, duplicate case-insensitive names, symlinks,
special files, encrypted members and file/directory collisions. Selected FLACs
are read by ZIP member into bounded buffers and written under generated release
paths. The inventory reports unrecognized audio/text paths, missing speaker
metadata, transcripts without audio, selected-mic audio without a transcript,
invalid text, and paired or missing microphones per speaker.

Use `mic2` consistently. The publisher identifies the MKH 800 as the wideband
microphone and documents technical issues for `p280` and `p315`. TorchAudio's
loader documents the missing `p280` mic2 audio, missing `p315` transcriptions
and partial recordings for `p362`. The worker records actual availability and
uses eligibility checks rather than hard-coded exclusions. It does not switch
individual utterances to another mic. [Edinburgh release](https://doi.org/10.7488/ds/2645) · [TorchAudio VCTK loader](https://docs.pytorch.org/audio/0.7.0/datasets.html)

## Frozen allocation

Seed `20261125` and protocol `vocoroute-vctk092-n2b0-v1` select speakers with
at least 200 structurally matched mic2/transcript pairs. In order, allocate
held-out eval, dev, router fit, router diagnostics, and base/expert fit from
the remaining speakers. Each allocation spreads the available accent/gender
strata by ranking within each stratum by salted SHA-256, interleaving by
fractional position and using a salted hash for ties. Use the same procedure
for the nested pilot speaker subsets. The receipt records the chosen IDs and
accent, gender and age coverage. This procedure has no per-accent guarantee.

| Group | Speaker target | Recording target |
|---|---:|---:|
| Base and expert fit | 60 | 200 per speaker |
| Development | 10 | 200 per speaker |
| Held-out evaluation | 10 | 100 per speaker |
| Router fit reserve | 10 | Speaker reservation only |
| Router diagnostics reserve | 5 | Speaker reservation only |
| Pilot train / dev | 20 / 4 | First 50 valid expanded clips per speaker |
| Train-only overfit | 8 from pilot train | First valid clip per selected speaker |

These are proposed quotas, pending the verified census. A speaker or audio
shortfall produces a failure receipt with the measured inventory and rejection
reasons. Within each allocated speaker, candidates follow a fixed hash order.
Invalid audio is skipped in that order with a reason. A duplicate native FLAC,
decoded native waveform, or prepared WAV among accepted rows fails the freeze.
An utterance has one source group `vctk-0.92/<speaker>_<utterance>` across
microphones. The manifests retain train/dev/eval assignments, and the router
reserves are absent from the model-facing releases. Shared transcript content
is expected because VCTK includes common prompts. The audit hashes Unicode
NFKC/casefolded alphanumeric tokens and records shared distinct texts and the
number of affected recordings on each side of each role pairing. Text is
audited as metadata and is not an input to this mel-conditioned model.

The historical forensic measurements record LJSpeech and the repository's
VocoTrace VCTK generation artifacts inspected here are synthetic fixtures. No
historical corpus or split feeds this protocol. Review any separately held real
VCTK forensic evaluation speaker/source registry before approving the freeze.
If such a registry exists, reconcile it before choosing the final speakers.

## Audio and loader verification

Require finite, nonempty mono 48-kHz PCM16 FLAC with header and decoded sample
counts in agreement. Decode one utterance at a time, with a 60-second maximum
source header and 32-MiB ZIP member limit. Preserve the exact FLAC bytes in the
designated role release. Run one float32 `scipy.signal.resample_poly` pass with
`up=147`, `down=320`, Kaiser beta `5.0`, zero constant padding, and no gain
normalization or added trim. Encode mono 22,050-Hz PCM16 WAV and verify exact
`ceil(native_samples * 147 / 320)` length, valid headers and at most one
PCM16 step of quantization error. Input PCM extrema, digital silence, invalid
decodes, and resampled overflow are logged as exclusions. Low RMS, atypical
duration, DC offset, and a prepared PCM endpoint are recorded separately.
The pinned resampling receipt records package, platform and libsndfile versions.
It belongs to generator training. The historical forensic audio writer,
normalization gates and SoXR preparation profiles keep their current behavior.

The expanded and pilot releases use `vocoroute-generator-manifest-v1` with
the held-out metadata rows required by `load_manifest`. Each row records native
and prepared bytes, counts, rates, identities and resampling receipt pin.
`recording-provenance.json` adds original ZIP member, transcript and decoded
PCM hashes, durations and audio statistics. The worker opens every train/dev
file through `TrainingAudio` byte and header checks. It uses the actual
`prepare_example` and `SpectralFrontend` on one train and one dev recording per
release and every overfit recording, comparing the mono target samples and 80
mel bands at the exact expected frame count. The train-only `overfit` selection
pins its pilot manifest and uses `load_overfit` to expose only the approved
training rows to `TrainingAudio`. It is a diagnostic projection. The regular
base trainer requires a development role for validation and checkpoint selection.

## Storage and resource estimates

Assume 3–4 s per clip until the worker measures actual duration. PCM16 WAV
uses 44,100 bytes/s plus a small header. The last column budgets native FLAC
at the equivalent 48-kHz PCM payload rate of 96,000 bytes/s plus prepared WAV,
with modest metadata and container overhead additional. These are estimates.

| Release | Clips | Audio duration | Prepared WAV | Native + WAV budget |
|---|---:|---:|---:|---:|
| Overfit | 8 | 24–32 s | 1.1–1.4 MB | 3.4–4.5 MB |
| Pilot train/dev | 1,200 | 1.00–1.33 h | 159–212 MB | 0.50–0.67 GB |
| Expanded train/dev | 14,000 | 11.67–15.56 h | 1.85–2.47 GB | 5.88–7.85 GB |
| Secured eval | 1,000 | 0.83–1.11 h | 132–176 MB | 0.42–0.56 GB |

The train/dev job caps audio output at 12 GB, with a 13 GB free-space
preflight. Held-out replay caps audio at 3 GB and requires 4 GB free. The
read-only ZIP is streamed and never copied to `/kaggle/working`. The notebook
output limit documented by Kaggle is 20 GB. A hypothetical 20–100 MB/s
mount yields roughly 2–10 minutes for one checksum pass. Selective decoding,
tens of thousands of small output files, mount latency and output saving
dominate the remaining uncertainty. Allow one CPU thread for frontend probes,
an output-saving margin and up to six hours for each approved job. The saved
receipt measures elapsed time, release bytes and the actual corpus durations.
[Kaggle notebook storage documentation](https://www.kaggle.com/docs/notebooks)

A local in-memory benchmark used approximately 0.48 CPU seconds for 32
synthetic 3.5-second clips after warm-up. That includes ZIP read, FLAC decode,
resampling, WAV encode and re-read, statistics and hashes. It excludes Kaggle
mount latency, disk validation and output saving.

1. The private train/dev preparation job writes `n2b0-train-dev/{expanded,pilot,overfit}`.
   It computes held-out metadata and exact hashes in memory and writes zero
   eval audio bytes to that job's output. `freeze.json` pins the manifest,
   allocation, source archive, resampling, provenance and audit identities.
2. Once the freeze and version are reviewed, a separate approved private CPU
   replay job attaches the acquisition output and that exact train/dev
   version. It reproduces only held-out members, checks their manifest rows
   and provenance byte identities, and publishes `n2b0-held-out` with
   `eval-index.json`, the canonical expanded manifest, and only eval audio.
3. For the existing Kaggle GPU runner, use the finished train/dev notebook's
   **Output → Create Dataset** and select one release directory. Create it as
   private `sambhavjainsj/vocoroute-vctk092-n2b0-pilot` or
   `sambhavjainsj/vocoroute-vctk092-n2b0-expanded`, version 1, matching its
   generated `dataset-receipt.json`. Verify the resulting mount preserves
   relative paths and hashes before fitting. Keep held-out notebook output
   or any subsequent private eval dataset separate. Publication of these
   resources requires approval. No archive transfer through the laptop is
   needed. [Kaggle dataset output connector](https://www.kaggle.com/docs/datasets)
4. The training plan attaches only approved code, one train/dev release and
   an optional checkpoint. Check the saved source metadata for the matching
   allowlist and an empty notebook source list. The VocoRoute runtime also rejects an
   accidental raw archive mount, held-out index or `native/eval` or
   `prepared/eval` directory before opening the train/dev manifest. It
   separately rejects the individual eval paths inside the data root.
   Use the updated launcher to obtain that protection.

## Exact Kaggle commands after approval

These commands assume PowerShell at the repository root and Kaggle CLI 2.2.4.
The listed local stage is an iteration snapshot. For a record release, set
`$commit` to an existing user-created commit containing the preparation files,
and use `--sha $commit` when staging. The kernel embeds an allowlisted,
deterministically packed source subset and verifies both payload and tree SHA-256.
It requires no remote code dataset or dependency installation. Private, GPU
off, TPU off, and internet off are scripted settings. Confirm the owner, new
slugs, source versions and CPU accelerator before any submit. CLI 2.2.4 sends
notebook slugs. Confirm they resolve to the intended exact versions and retain
the server attachment evidence. If the slug points to a different version,
attach the exact saved version in the UI before running. The in-worker hashes
are mandatory in both cases.

```powershell
$env:PYTHONPATH = (Resolve-Path -LiteralPath src).Path
$vocoPython = 'C:\Users\samak\Desktop\BTP\vocoroute-env\Scripts\python.exe'
$kaggle = (Get-Command kaggle -ErrorAction Stop).Source
$run = (Resolve-Path -LiteralPath 'runs/vocoroute-n2b0').Path
$acquisition = 'sambhavjainsj/vocoroute-vctk092-acquisition/2'

# Pull the exact source metadata. The file listing is for the current latest version.
& $kaggle kernels pull $acquisition --metadata -p "$run/acquisition-metadata"
& $kaggle kernels files 'sambhavjainsj/vocoroute-vctk092-acquisition' --format json

# Use the verified existing stage, or render a new record stage with --sha.
$stage = "$run/stage-train-dev"
# $stage = "$run/stage-train-dev-record"
# & $vocoPython -B -m experiments.cli kaggle vocoroute-corpus --sha $commit --out-dir $stage
Get-FileHash -Algorithm SHA256 "$stage/prepare.py", "$stage/kernel-metadata.json"

# First remote execution. Run only after approval, with no accelerator flag.
& $kaggle kernels push -p $stage --timeout 21600
$trainRef = 'sambhavjainsj/vocoroute-vctk092-n2b0-train-dev'
& $kaggle kernels status $trainRef
& $kaggle kernels logs $trainRef
```

After successful worker and receipt status, set `$trainVersion` to the returned
version number. Open that exact saved version's **Output** tab and retrieve
only `n2b0-reports/prep-receipt.json`, `inventory.json`, `allocation.json`,
`n2b0-train-dev/freeze.json` and `leakage-audit.json`. Review rejection and
failure files when present. In CLI 2.2.4, `kernels files` and `kernels output`
can parse a version suffix but still request the latest result. The saved
version's Output tab is the authority for recovery. A worker can finish while
its prep receipt records a validation failure. There is no automatic retry.

```powershell
$trainSaved = "$trainRef/$trainVersion"
& $kaggle kernels pull $trainSaved --metadata -p "$run/train-server-metadata"
# Download only the two JSON files to these paths from the exact version's Output tab.
$reported = Get-Content -Raw -LiteralPath "$run/train-reports/prep-receipt.json" | ConvertFrom-Json
$freezeDigest = (Get-FileHash -Algorithm SHA256 "$run/train-reports/freeze.json").Hash.ToLowerInvariant()
if ($reported.status -ne 'verified' -or $reported.release.freeze_sha256 -ne $freezeDigest) {
    throw 'The saved train/dev output did not pass the freeze check.'
}
```

Read the verified worker receipt's `release.freeze_sha256` and compare it to
the exact saved `freeze.json`. Stage held-out replay only with that digest
and exact train/dev version. Reuse the same record commit and Kaggle CPU
runtime. Launch after separate evaluation storage/execution approval.

```powershell
$evalStage = "$run/stage-held-out-record"
& $vocoPython -B -m experiments.cli kaggle vocoroute-corpus --sha $commit --out-dir $evalStage --freeze-source $trainSaved --freeze-sha256 $freezeDigest
& $kaggle kernels push -p $evalStage --timeout 21600
$evalRef = 'sambhavjainsj/vocoroute-vctk092-n2b0-held-out'
& $kaggle kernels status $evalRef
& $kaggle kernels logs $evalRef
```

The target mount layout, real FLAC decoding, actual speaker and transcript
counts, cloud byte-for-byte replay, and final dataset connector behavior
remain to be verified on the approved CPU jobs. No model fit is part of N2B.0.

## Local verification

The existing VocoRoute CPU environment ran the corpus, manifest, Kaggle
preflight and audio I/O tests with Torch and TorchAudio 2.9.0+cpu. The
completed focused run reported **60 passed, 6 deselected in 19.62 s**.
The six deselections exercise optimizer steps. A synthetic corpus hit the
same loader checks as a future speech corpus. It also verified native and
prepared header geometry, repeated text reporting, exact held-out replay,
nested manifests, and refusal of accidental raw or held-out mounts. The
staged script bootstrapped its embedded source in an isolated local Python
process and produced the expected failure receipt for an empty input mount.

```powershell
$skipOptimizer = 'not bounded_synthetic_optimizer_smoke and not interrupted_resume_matches_uninterrupted and not update_cap_is_total and not wall_budget_resumes and not resume_refuses_changed and not signal_saves_safe_boundary'
& $vocoPython -B -m pytest -ra -o pythonpath=src -p no:cacheprovider --basetemp "$run/pytest-corpus-FRESH" tests/test_vocoroute_n2b0.py tests/test_vocoroute_n2a.py tests/test_vocoroute_n2k.py tests/test_audio_io_stream.py -k $skipOptimizer
& '.venv\Scripts\python.exe' -B -m pytest -ra -o pythonpath=src -p no:cacheprovider --basetemp "$run/pytest-project-regression-FRESH" --ignore=tests/test_temporal_cnn.py --ignore=tests/test_learned_temporal.py -k $skipOptimizer
& '.venv\Scripts\python.exe' -m ruff check --no-cache src tests
& '.venv\Scripts\python.exe' -m ruff check --no-cache . --output-format concise
```

Use fresh basetemp paths. The initial pytest command needed its workspace
parent directory created. A Windows test fixture also needed to write a
literal ZIP backslash because Python's ZIP writer normalizes it on Windows.
The focused result above is the rerun after both corrections. The broader
VocoRoute-environment collection encountered missing pandas and pyloudnorm.
The project environment supplies them and the command above leaves out
historical tests that fit neural detectors. The broad project run reported
**1,057 passed, 4 failed, 13 skipped and 1 xfailed in 18 min 45 s**. The four
failures match the N2K.1 baseline, with two generated-notebook CRLF/LF comparisons
and two Windows junction permission failures. Four skips were Torch modules
and nine were Windows link permissions. The xfail was the existing INV-17
ablation. Ruff passed on `src tests`. Repository-wide Ruff found 11 existing
diagnostics in `notebooks/kaggle_v0.ipynb`.
