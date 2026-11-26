# VocoRoute-P N2B.2 eight-utterance overfit diagnostic

## N2B.1-P listening closeout

The private notebook `147530439` version 1 saved the following mapping for the
three blinded WAVs. The listening notes and preference below are the user's
observations.

| Clip | Resolved condition | User observation |
| --- | --- | --- |
| `clip-01.wav` | Run A, phase terms disabled, update 4,000 | clearly understandable, natural human speech, slight background buzzing. |
| `clip-02.wav` | Run B, both phase terms weighted 0.05, update 4,000 | clearly understandable, natural human speech, no obvious artifacts. |
| `clip-03.wav` | Reference | clearly understandable, natural human speech, no obvious artifacts. |

```text
Preference: clip-03 = clip-02 > clip-01.
```

The notes form an informal intelligibility gate for this one example. Measured
at 4,000 matched updates, Run B reduced common reconstruction loss by 6.50%,
waveform MAE by 22.21%, spectral convergence error by 26.55%, and absolute
circular phase error by 42.16%. Run B took 14.82% more update time. The
listening observations and numerical results support carrying the fixed Run B
objective into the eight-utterance diagnostic. A wider listening study and
speaker-disjoint evaluation remain future work.

## Frozen input and minimum attachment

The source is `sambhavjainsj/vocoroute-vctk092-n2b0-train-dev/1`, notebook ID
`147338652`. Its previously saved N2B.0 preparation receipt was verified. The
local metadata audit recomputed the freeze, overfit manifest, pilot manifest,
resampling receipt and license digests. The freeze is
`39ae136dbf240762a8c2abd38e39f8c56f9d246cd191c6f7dae014400808ec6b` and the
selection is `20601c92e22cb203c38dc1ed85d5827c8814e9dfdf46cc7310aab978056f9f5c`.
The selected rows belong to eight distinct pilot TRAIN speakers and source
groups. None of the speakers occur in the frozen dev, pilot-dev, router-fit,
router-diagnostic or evaluation groups.

| Utterance | Speaker | Native samples at 48 kHz | Prepared samples at 22.05 kHz | Mel frames | Prepared seconds |
| --- | --- | ---: | ---: | ---: | ---: |
| `p229_128` | `p229` | 166,580 | 76,523 | 299 | 3.47043 |
| `p310_355` | `p310` | 190,035 | 87,298 | 342 | 3.95909 |
| `p287_015` | `p287` | 289,920 | 133,182 | 521 | 6.04000 |
| `p281_325` | `p281` | 138,904 | 63,810 | 250 | 2.89388 |
| `p351_135` | `p351` | 102,433 | 47,056 | 184 | 2.13406 |
| `p314_284` | `p314` | 209,941 | 96,442 | 377 | 4.37379 |
| `p364_041` | `p364` | 164,379 | 75,512 | 295 | 3.42458 |
| `p265_267` | `p265` | 191,058 | 87,768 | 343 | 3.98041 |

The bundle totals 3,852,084 bytes. It contains eight source FLACs
(1,163,650 bytes), eight prepared WAVs (1,335,534 bytes), and five provenance
files (1,352,900 bytes). Those five files are `freeze.json`,
`overfit/manifest.json`, `overfit/parent-manifest.json`,
`overfit/source/resampling-receipt.json`, and `overfit/source/license_text.txt`.
The parent manifest contains split metadata necessary to verify membership.
The attachment has only the selected TRAIN audio bytes. The inventory receipt
at `runs/vocoroute-n2b2/frozen-corpus-evidence.json` lists every saved source path,
size and SHA-256. It also records the absence of preparation flags for all rows.

The saved N2B.0 metadata and inventory were reconciled locally with the
published recording paths, exact audio pins, and the no-trim, no-gain,
ceil-length resampling contract. Fresh checks on all 16 real audio objects are
deferred to the approved private attachment. The existing `p229_128` WAV was
rehashed locally and its frontend round-trip maximum absolute error was
`1.19e-7`. Its native pin also matched the earlier N2B.1 byte receipt. Before
submitting, copy just the
21 approved files from the exact saved version to a private dataset with slug
`sambhavjainsj/vocoroute-vctk092-n2b2-overfit-eight`. Record the assigned ID,
verify version 1 and each saved byte digest and retain source credit and the
included license. If a cloud-side selector can import the exact inventory,
prefer that path. Never attach the source notebook or full corpus to training.
The worker independently checks the whole input tree, both audio digests per
row, encoding, mono rate and length, and same-waveform target/mel alignment.
It rejects extra, duplicate, or linked files before opening audio. It can
read canonical folders or a flattened attachment with unique basenames.

## Training protocol

The worker reuses the existing backbone, frontend, loss, `_step`, AdamW,
waveform export, reconstruction metrics, and weights-only checkpoint writer.
It seeds a fresh 566,144-parameter model with `20260907`, and uses deterministic
FP32 on `cuda:0` of T4 ×2. AdamW uses constant LR `0.0002`, weight decay `0.01`,
betas `0.9` and `0.999`, epsilon `1e-8`, and no foreach or fused path. Common
loss sums waveform, linear magnitude, and log magnitude L1 with unit weights.
The optimization objective adds temporal and frequency phase terms at 0.05
each. Batch size is one complete utterance, up to 16,000 optimizer updates.

For zero-based round `r`, rank all eight IDs by the SHA-256 bytes of the UTF-8
string `vocoroute-n2b2-round-v1:20260907:{r}:{utterance_id}`, breaking any tie by
utterance ID. Visit each exactly once in that rank order. The first round is
`p314_284`, `p229_128`, `p265_267`, `p351_135`, `p281_325`, `p310_355`,
`p287_015`, `p364_041`. The 16,000-update cap gives 2,000 complete rounds and
2,000 visits per utterance and per speaker. The training history records the
actual row and speaker for every update, and the receipt includes exposures
and the SHA-256 of the newline-delimited actual order.

Snapshots at `0, 800, 4000, 8000, 16000` evaluate all eight examples with one
model state and restore the training mode afterwards. They record per-utterance
and per-speaker metrics, equal-utterance macro means, medians and best/worst
utterances ranked by common loss. The worker copies the original prepared WAVs
once and writes exact-length float arrays and PCM16 snapshots at every step.
It records any clipping without applying gain changes and refuses a nonfinite
reconstruction. The receipt includes gradient norms, parameter changes, GPU
peaks and throughput. The final frozen checkpoint is loaded weights-only into
a fresh backbone and checked against all eight raw outputs. Eight final
reference/reconstruction pairs are copied in seeded blind order with a separate
condition key.

The protocol SHA-256 is
`64b9a387b6331d1fac0228411acfbb83741717edc0bd03534a7ee9dd844fab20`.
The input inventory SHA-256 is
`3014d74ad86a2ef6892cf7092be2086cb330656e3feee214dbeff6d9c991f145`.
The full order SHA-256 is
`9f5dacdb68455e41a4e5e180b3c20f0d768f517f360ed620a9286cd246635fca`.
The precommit draft containing these identities is
`runs/vocoroute-n2b2/proposed-record-plan.json`.

## GPU budget and stopping

Run B's saved T4 median and p95 update timings were 35.120 ms and 39.027 ms
for 299 mel frames. The eight clips average 326.375 frames and range from 184
to 521. Linear frame scaling gives 613 s at the observed median or 682 s at
the observed p95 for 16,000 updates. Applying a 25% allowance to the latter and
adding 60 s for startup, snapshot evaluation and checkpoint replay gives a
planning estimate of 912 s. Mixed-length kernel and scheduling overhead on the
new worker remain unmeasured. The estimate leaves about 48 s to the 960 s
admission deadline, 120 s from that deadline to the 1,080 s Kaggle timeout,
and another 120 s to the 1,200 s planned GPU limit.

Before admitting a round, the worker checks for an orderly stop and reserves
at least five seconds or twice the longest observed round, whichever is
larger. It completes an admitted round and writes the last balanced snapshot
and checkpoint even when that step is off schedule. A hard platform stop can
interrupt output saving. There is one private notebook and no retry.

## Local verification and RECORD staging

The focused CPU tests use generated fixtures. They check the exact attachment,
freeze and audio-pin refusal, the full 16,000-position schedule, phase gradients,
snapshot isolation, balanced budget stopping, weights-only replay, and the
embedded RECORD stage. The isolated staging test caught a mismatched notebook
title and slug. The corrected title renders the intended slug successfully.
The old `fae5355` archive correctly refused staging because it lacks the new
source files.

The focused run passed 146 tests in 76.50 s with the existing CPU environment.
The full Windows run finished in 17 min 41 s with 1,069 passed, four known
failures, 16 skipped and one xfail. The failures were two notebook line-ending
comparisons and two junction permission errors. Seven skips were due to missing
Torch in the project environment, and nine were due to Windows link permissions.
Ruff passed on `src tests`. Repository-wide Ruff reported the 11 existing
findings in the historical Kaggle notebook.

```powershell
$env:PYTHONPATH = (Resolve-Path -LiteralPath src).Path
$vocoPython = 'C:\Users\samak\Desktop\BTP\vocoroute-env\Scripts\python.exe'
$env:NUMBA_CACHE_DIR = (Resolve-Path -LiteralPath 'runs/vocoroute-n2b2').Path
$stamp = [guid]::NewGuid().ToString('N')
& $vocoPython -B -m pytest -ra -o pythonpath=src -p no:cacheprovider --basetemp "runs/vocoroute-n2b2/pytest-focused-$stamp" tests/test_vocoroute_n2b2.py tests/test_vocoroute_n2b1p.py tests/test_vocoroute_n2b1.py tests/test_vocoroute_n2b0.py tests/test_vocoroute_n2a.py tests/test_vocoroute_n2k.py tests/test_vocoroute_n1.py
$stamp = [guid]::NewGuid().ToString('N')
& '.venv\Scripts\python.exe' -B -m pytest -ra -o pythonpath=src -p no:cacheprovider --basetemp "runs/vocoroute-n2b2/pytest-regression-$stamp"
& '.venv\Scripts\python.exe' -B -m ruff check --no-cache src tests
& '.venv\Scripts\python.exe' -B -m ruff check --no-cache . --output-format concise
```

After the user-created commit, stage locally from the same recovered metadata.

```powershell
$source = 'runs/vocoroute-n2b0/record-f934882/metadata/n2b0-train-dev'
$commit = Read-Host 'User-created N2B.2 commit SHA'
git --no-optional-locks rev-parse --verify "${commit}^{commit}"
git --no-optional-locks ls-remote --exit-code origin refs/heads/main
$stage = 'runs/vocoroute-n2b2/record-' + $commit.Substring(0, 7)
& $vocoPython -B -m experiments.cli kaggle vocoroute-eight-plan --sha $commit --source-metadata $source --out-dir $stage
Get-FileHash -Algorithm SHA256 "$stage/kernel/overfit_eight.py", "$stage/kernel/kernel-metadata.json"
```

After the commit, compare the staged protocol, inventory and order digests
with the frozen draft and verify live `origin/main`, the private saved dataset
and current account quota. Submit only after explicit approval with
`--accelerator NvidiaTeslaT4 --timeout 1080`. Inspect the private server
metadata and retrieve the exact saved version for verification.
