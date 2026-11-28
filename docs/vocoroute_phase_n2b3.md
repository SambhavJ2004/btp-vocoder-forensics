# VocoRoute-P N2B.3 pilot preparation

N2B.3 prepares a fresh base-backbone fit on frozen VCTK pilot train/dev only.
The preparation uses saved N2B.0 metadata and N2B.2 diagnostic artifacts.
Remote attachment creation and the GPU attempt require separate authorization.

## N2B.2 listening closeout

The following are the supplied observations, verbatim.

```text
Pair 1:
Both clearly understandable, high voice similarity,
second preferred. Slight echo at end of first.

Pair 2:
Both clearly understandable, high voice similarity,
second preferred. First sounds like two identical
people speaking together.

Pairs 3-6:
No formal assessments recorded.

Pair 7:
Slight echo on first recording.

Pair 8:
General observation that the first recording in
each pair sounds slightly echoey.
```

| Pair | Utterance | First | Second |
| --- | --- | --- | --- |
| 1 | p229_128 | Reconstruction | Reference |
| 2 | p287_015 | Reconstruction | Reference |
| 3 | p314_284 | Reconstruction | Reference |
| 4 | p351_135 | Reference | Reconstruction |
| 5 | p364_041 | Reference | Reconstruction |
| 6 | p265_267 | Reference | Reconstruction |
| 7 | p310_355 | Reconstruction | Reference |
| 8 | p281_325 | Reconstruction | Reference |

The seeded N2B.2 procedure shuffled utterance positions and independently
randomized side. The realized order was five reconstructions first and three
references first. All 16 blind WAV hashes match the listening key and their
corresponding source WAV. The specific comments on pairs 1, 2 and 7 refer to
reconstructions. A literal reading of the broad observation recorded under
pair 8 also covers first-position references in pairs 4–6. It supplies no
individual assessment for those pairs or a separate rating for p281_325.

## Offline echo probe

`runs/vocoroute-n2b3/echo/results.json` gives all eight comparisons. The steps
can be reproduced with `echo_probe.py` and `echo_plot.py` in the same run root.
The probe used the step-16,000 unnormalized float predictions and paired PCM16
reference WAVs. The diagnostic receipt and all its listed artifact hashes
verified, including checkpoint SHA-256
`fed13a069a0b9326bcac6d3603756339ac2c569f0990a5c8007304a367153c7a`.
CPU weights-only replay differed by at most 0.00000601 in absolute amplitude.
Each WAV had exactly the expected length at 22,050 Hz. The sample difference
between each reconstruction's PCM16 export and raw prediction was below one LSB.
The saved snapshot reports zero clipped and zero nonfinite samples.

Full-waveform cross-correlation peaked at zero samples within ±20 ms on all
eight comparisons. To test for a fixed delayed copy, the probe band-passed
both signals at 1.8–8 kHz, projected out the aligned reference, and scored
positive delays of 10–200 ms across disjoint 550 ms windows. It also projected
the aligned reference out of each delayed predictor. An injected 60 ms,
0.15-amplitude copy was recovered at 60 ms with correlation 1.00 in all four
tested windows. The signal probe selected the lag with the highest median
signed partial correlation, so these exploratory maxima can reflect periodicity
or shared speech structure.

| Utterance | Selected lag, ms | Median partial correlation | Windows |
| --- | ---: | ---: | ---: |
| p229_128 | 12.11 | 0.120 | 4 |
| p287_015 | 20.27 | 0.049 | 8 |
| p310_355 | 19.05 | 0.145 | 4 |

The plots compare reference, reconstruction and residual at a common
spectrogram scale. Much of the residual coincides with speech. With boundaries
defined by reference 10 ms RMS at −35 dB relative to peak frame, p229_128
had residual RMS 0.00316 in the first 100 ms after the last active frame.
Reference and reconstruction there had RMS 0.00232 and 0.00235. The comparable
p287_015 values were 0.00341 residual, 0.00265 reference and 0.00275
reconstruction. p310_355 had only 8.1 ms available after its last active
frame. This provides evidence of residual distortion around parts of
the boundary, with little increase in aggregate tail energy for the two
measurable examples. The fixed-copy probe cannot exclude local,
time-varying or pitch-related effects.

The saved step-16,000 absolute circular phase errors for the three specific
reports were 0.07605, 0.08734 and 0.11256, respectively. Those errors establish
imperfect phase reconstruction and do not identify a perceptual cause.
The source comparisons leave phase or harmonic distortion and local temporal
smearing as hypotheses. Export errors at the verified filenames are limited to
PCM16 quantization. Playback chain effects remain untested. No conclusive echo
mechanism was identified.

## Frozen pilot and private attachment

Source is saved notebook version 1 of
`sambhavjainsj/vocoroute-vctk092-n2b0-train-dev`, notebook ID 147338652.

| Role | Utterances | Speakers | Prepared seconds | Native FLAC bytes | Prepared WAV bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 1,000 | 20 | 3,374.443 | 122,710,492 | 148,856,924 |
| Dev | 200 | 4 | 645.827 | 24,571,168 | 28,489,784 |

Each speaker contributes 50 recordings. Train speakers are p229, p233, p237,
p243, p251, p265, p266, p268, p281, p287, p299, p302, p304, p310, p314, p326,
p343, p351, p360 and p364. Dev speakers are p227, p257, p275 and p297.
The existing manifest validator passed. The frozen speaker and recording-source
intersections are both empty. There are 25 shared normalized transcript hashes
across 28 train and 27 dev utterances. Dev measures speaker-disjoint
development performance with some repeated text.

The attachment projection contains 2,405 files totaling 325,981,238 bytes
(310.88 MiB), based on saved file inventory and frozen recording provenance.
It includes each train/dev native FLAC and prepared WAV, the unchanged
manifest and dataset receipt, the freeze receipt, resampling receipt and
license. Eval rows remain in the original manifest as metadata. Audio for eval,
expanded sets, router sets and the raw archive is excluded. The local
verification used saved metadata and file listings. Physical audio bytes
will be independently hashed during authorized cloud projection and again by
the training loader.

| Identity | SHA-256 |
| --- | --- |
| N2B.0 freeze | `39ae136dbf240762a8c2abd38e39f8c56f9d246cd191c6f7dae014400808ec6b` |
| Pilot manifest | `5ed731b8c47dcd0331f14ab1b5d877200fb6f7e09500fbb7507f94d998167d2f` |
| Pilot attachment inventory | `2383fe519cd2ceb8b6b416e2031a975681392012599cb3e6251ed44363af2787` |
| Training configuration | `eb34334848910ef649900ec173df860415406d22a9246da9e4b3b8e40d621052` |
| Pilot protocol | `c12e257b186ab089a610c9048965925541453a5f08880434e09bc32c5e209b19` |

Draft artifacts are `pilot-attachment-inventory.json`,
`pilot-config.json`, `pilot-spec.json` and `pilot-local-audit.json` under
`runs/vocoroute-n2b3`. After approval, provision the exact saved N2B.0
version in an isolated CPU worker. Invoke
`vocoroute.pilot_diagnostics.prepare_attachment(source, output)` with the
mounted `n2b0-train-dev` directory and a fresh output directory. It
verifies the listed bytes and copies only that projection. Use version 1
of the private dataset `sambhavjainsj/vocoroute-vctk092-n2b0-pilot`.
Reuse it if its existing private version and bytes match the inventory.
Confirm the exact remote private ID, version and mount path. The training
worker receives only that projection, the private RECORD source snapshot,
and an optional two-file verified resume input. Training refuses unexpected
mounts, unexpected pilot contents and changed metadata or audio.

## Frozen training and development protocol

The existing N2K v2 trainer constructs fresh weights for the
566,144-parameter backbone, seed 20260907, FP32, batch size one, existing
AdamW and `ExponentialLR(gamma=1.0)` at LR 0.0002. The common objective
sums waveform, linear magnitude and log-magnitude loss, each with weight 1.
Both phase-difference weights are 0.05. There are at most 20 complete
epochs and 20,000 optimizer updates. Each fit epoch shuffles sorted
train IDs with the existing seed/epoch PRNG and visits every train row once.
Use `NvidiaTeslaT4` provisioning, verify both visible T4s and train on
`cuda:0`. The draft pins the previously observed Kaggle software stack
with Python 3.13.16, Torch and TorchAudio 2.11.0+cu130, CUDA build 13.0,
SoundFile 0.14.0, NumPy 2.1.3 and expected librosa 0.11.0. Verify the
current remote image and availability before launch.

All 200 dev utterances are synthesized at each completed epoch in eval
mode with gradients disabled. Training metrics are pre-update per-utterance
objective components. Dev adds spectral convergence, energy-weighted
absolute circular phase error, RMS, peak, PCM16 clipping and nonfinite
sample counts. Save dev measurements individually and as equal-utterance
macro and per-speaker summaries. Save the lowest equal-utterance mean
**weighted** dev objective observed on a completed epoch, with earliest
epoch on an exact tie. Common objective is reported separately. A budget stop
after completed dev saves the best weights-only inference artifact and the
latest resumable state. A first-epoch stop before complete dev has no best
checkpoint yet.

The following panel is selected before inference by SHA-256 ranking within
speaker, using the protocol namespace and seed above.

| Dev speaker | Fixed utterances |
| --- | --- |
| p227 | p227_002, p227_033, p227_138 |
| p257 | p257_248, p257_320, p257_366 |
| p275 | p275_036, p275_045, p275_240 |
| p297 | p297_012, p297_170, p297_262 |

Save initialization, completed epochs 1, 5, 10 and 20, and the best
checkpoint. Separate seeded hashes choose pair order and side at every stage,
with six references first and six reconstructions first. The listening
directory for a stage contains only neutral WAV pairs. Reference WAVs are
exact byte copies. Model exports use the existing PCM16 quantization with
no gain adjustment, with separate float arrays and keys. Each attempt's
metric report hashes its keys. Retain previous attempt outputs to preserve
earlier panels across resumes.

## Budget and continuation

Train record lengths are 1.452–14.716 s (median 3.167 s, p95 5.038 s)
and average 291.132 STFT frames. Dev averages 278.625 frames and peaks at
8.445 s. The saved N2B.2 trace measured 326.375 mean frames, 33.44
updates/s, 29.52 ms median update and 32.76 ms p95. Scaling by mean frames
gives 534 s using mean update time or 584 s using p95 for 20,000 pilot
updates. The N2B.2 trace covers already-loaded tensors and is a proxy for
the fit computation only. The additional costs below are allowances.
On the local CPU, a cached read, SHA-256 check, decode and mel calculation
on the eight N2B.2 reference WAVs took 5.65 ms median and 9.05 ms p95 over
64 measurements. Kaggle mount I/O and device transfer remain unmeasured.

| Component | Estimated seconds |
| --- | ---: |
| Fit compute | 534–584 |
| Train loading, SHA-256, decoding, mel features and telemetry | 160–300 |
| 4,000 dev evaluations, including I/O and metrics | 60–120 |
| Setup, input verification and CUDA preflight | 30–120 |
| Checkpoints, panel rendering and output saving | 40–120 |
| **Total** | **824–1,244 (13.7–20.7 min)** |

The internal 1,500 s admission clock starts in the notebook bootstrap.
The proposed Kaggle timeout is 1,680 s, within a 1,800 s planned GPU
budget. Startup before Python and platform persistence time remain
unmeasured. Admission checks each new train update and dev evaluation.
Checkpoint cadence is every 1,000 updates, at the scheduled listening
milestones and at controlled stop. The final drain may exceed the admission
clock. A hard termination can prevent publication of outputs. The longest
training record is about 3.9 times the N2B.2 mean frame count, so its
actual peak GPU memory and per-update duration still need observation.

The existing resume retains weights, optimizer, scheduler, RNG, train/dev
cursor and best completed-epoch weights. The opt-in pilot persists dev
diagnostics through a partially evaluated epoch. Exposure counts are
reconstructed from completed epochs and the exact seeded prefix of the
current epoch. Resume uses the original configuration and protocol digests
and existing runtime identity gate. It gets a fresh attempt clock with
unchanged cumulative epoch and update caps and requires separately approved
GPU time. The run records `convergence_claimed: false`, including after
20 completed epochs.

## Local verification

The saved corpus check used metadata and file listings. Synthetic CPU
checks exercised the projection and digest refusal, blocked extra mounts,
and verified that dev leaves model state and RNG unchanged. A simulated
wall-time stop held one dev example of epoch 2 pending. The first attempt
published the best completed-epoch checkpoint, then a resumed attempt
matched the uninterrupted final state, diagnostics and exposure counts.
The actual staged draft produces six reference-first pairs at all six
listening stages.

```powershell
$env:PYTHONPATH = 'src'
$vocoPython = 'C:\Users\samak\Desktop\BTP\vocoroute-env\Scripts\python.exe'
$stamp = [guid]::NewGuid().ToString('N')
& $vocoPython -B -m pytest -o 'addopts=' -q -ra tests/test_vocoroute_n1.py tests/test_vocoroute_n2a.py tests/test_vocoroute_n2k.py tests/test_vocoroute_n2b0.py tests/test_vocoroute_n2b1.py tests/test_vocoroute_n2b1p.py tests/test_vocoroute_n2b2.py --basetemp "runs/vocoroute-n2b3/pytest-focused-$stamp"
$stamp = [guid]::NewGuid().ToString('N')
& '.venv/Scripts/python.exe' -B -m pytest -q --basetemp "runs/vocoroute-n2b3/pytest-project-$stamp"
& '.venv/Scripts/ruff.exe' check src tests --no-cache
& '.venv/Scripts/ruff.exe' check . --no-cache --output-format concise
```

The full suite in the isolated CPU environment stopped at collection with
20 missing-dependency errors for pandas or pyloudnorm. The project
environment supplies the historical dependencies and skips Torch-dependent
modules. The focused run in the Torch-capable environment passed 149 tests
in 68.70 s. Ruff passed on `src tests`. Repository-wide Ruff reported
the 11 existing findings in `notebooks/kaggle_v0.ipynb`.
The current project-environment regression completed with 1,069 passed,
four known Windows failures, 16 skips and one xfail. The failures were
the two CRLF/LF notebook comparisons and two junction permission errors.
Seven skips occurred during collection for missing Torch and nine occurred
during the run.

## RECORD staging

Local staging against `42516643db00787d0a36b912a0e39e3c7dd8cb36`
correctly refuses the pilot because the commit lacks its implementation.
After the user creates a commit, verify its full identity and live
`origin/main`, then stage using the current local draft.

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
$vocoPython = 'C:\Users\samak\Desktop\BTP\vocoroute-env\Scripts\python.exe'
$commit = Read-Host 'User-created N2B.3 commit SHA'
git --no-optional-locks rev-parse --verify "${commit}^{commit}"
git --no-optional-locks ls-remote --exit-code origin refs/heads/main
$root = (Resolve-Path 'runs/vocoroute-n2b3').Path
$stage = Join-Path $root ('record-' + $commit.Substring(0, 7))
& $vocoPython -B -m experiments.cli kaggle vocoroute-plan --sha $commit --spec "$root/pilot-spec.json" --config "$root/pilot-config.json" --out-dir $stage
Get-FileHash -Algorithm SHA256 "$stage/plan.json", "$stage/kernel/train.py", "$stage/kernel/kernel-metadata.json"
```

Compare the staged protocol and configuration digests with the frozen table
above. After authorization, verify Kaggle authentication, live quota, hardware,
mounts, private flags and recorded dataset versions. The prepared submission
uses one private, internet-disabled kernel with accelerator
`NvidiaTeslaT4` and timeout `1680`. Verify source, dataset and output receipts
from the exact saved version before treating the results as RECORD evidence.
