# VocoRoute-P N2B.1-P phase reconstruction diagnostic

N2B.1 finished 1,000 updates on the fixed training utterance `p229_128`. The
user heard a speech-like but unintelligible reconstruction, with a different
voice, metallic sound, buzzing and background noise. The user reported that
the snapshots improved over training. The controls below measure properties
of that same saved waveform and provide new clips for a listening comparison.

## Local oracle controls

The local command checks the N2B.1 receipt, reference WAV, final raw float32
array, checkpoint file and flattened checkpoint tensor hashes. It loads the
checkpoint weights-only and compares its CPU replay to the saved GPU waveform.
The Griffin–Lim controls use 64 plain iterations, seed `20260907` and identical
initial angles. The analysis and synthesis both use the existing frontend.
It uses 22,050 Hz, a periodic Hann window of length 1,024, FFT length 1,024,
hop 256, centered zero padding and an unnormalized one-sided spectrum. The
DC and Nyquist signs come from the phase donor. Every waveform has 76,523
samples. Raw arrays retain original amplitude and PCM16 export counts any
saturation. No gain changes are applied.

| Imposed magnitude and phase | Waveform MAE | Spectral convergence | Linear L1 | Log L1 | Circular phase | Time phase diff. | Freq. phase diff. | Imposed spectrum residual |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Reference and reference | 2.79e-9 | 1.26e-7 | 4.27e-8 | 5.72e-6 | 2.75e-8 | 1.30e-14 | 1.94e-14 | 1.64e-7 |
| Predicted and predicted | 0.02703 | 0.45490 | 0.06769 | 0.62562 | 0.63662 | 0.56705 | 0.28601 | 1.63e-7 |
| Predicted and reference | 0.01188 | 0.41241 | 0.05145 | 0.45993 | 0.02401 | 0.02966 | 0.03005 | 0.43723 |
| Reference and predicted | 0.02723 | 0.40089 | 0.05367 | 0.54109 | 0.63537 | 0.45012 | 0.24104 | 0.51908 |
| Reference and Griffin–Lim | 0.03440 | 0.12734 | 0.01864 | 0.32809 | 0.79388 | 0.01807 | 0.02077 | 0.07760 |
| Predicted and Griffin–Lim | 0.03429 | 0.46432 | 0.06934 | 0.66607 | 0.90790 | 0.58851 | 0.32107 | 0.05187 |

The identity controls have maximum absolute errors `1.19e-7` and `8.94e-8`
with a float32 tolerance of `atol=2e-6`, `rtol=1e-5`. All six controls have
zero clipped and nonfinite samples. Both swaps use an imposed spectrum that
is far from its iSTFT then STFT projection. Interpret the results as paired
waveform controls, with magnitude and phase both measured again after synthesis.
The relative residual is `||STFT(iSTFT(S)) - S||_F / max(||S||_F, 1e-4)`.

The magnitude losses and spectral convergence use the three existing loss
resolutions. Pointwise phase uses `1 - cos(delta)` weighted by reference
energy on the native grid, omitting DC, Nyquist and reference bins below
`1e-4`. A predicted bin below that floor receives distance 1 and counts as
undefined in the reported phase-coverage metric. Temporal and frequency
phase differences use the actual N2A loss. It normalizes with a magnitude
floor, weights each eligible pair by its lower reference endpoint energy and
floors the energy denominator. For bins below the predicted floor the phasor
product can be shorter than one, so this pair term is a squared chord distance
with magnitude-dependent damping. Silent and low-energy tests verified finite
gradients. Phase difference does not recover a constant phase offset. In the
reference-magnitude Griffin–Lim control, the adjacent errors are low while
pointwise phase and waveform MAE remain high.

- **H1.** Reference phase improves waveform MAE by 56% on this example, with
  nearly all pointwise circular phase error removed. This supports a phase
  bottleneck. Intelligibility still requires listening.
- **H2.** In N2B.1, common reconstruction loss fell from 0.83234 at step 500
  to 0.72034 at step 1,000. The extended no-phase arm tests continued fitting.
- **H3.** Spectral convergence remains 0.41241 with reference phase and
  predicted magnitude. The matched Griffin–Lim controls also show a substantial
  magnitude gap. These results leave the role of model capacity unresolved.

Reproduce with the existing isolated CPU Torch interpreter. Use a fresh output
path. Locally generated arrays, WAVs and plots live under ignored `runs/`.

```powershell
$env:PYTHONPATH = (Resolve-Path -LiteralPath src).Path
$vocoPython = 'C:\Users\samak\Desktop\BTP\vocoroute-env\Scripts\python.exe'
$baseline = 'runs/vocoroute-n2b1/record-7788489/output'
& $vocoPython -B -m vocoroute.phase_diagnostic $baseline 'runs/vocoroute-n2b1p/oracles-fresh'
```

The focused CPU verification covers the exact endpoints, seeded Griffin–Lim,
low-energy backward behavior, Run A replay, receipt hash checks and saving
matched checkpoints after a simulated deadline. The same diagnostic inputs
regenerated all six PCM16/raw pairs and their receipt byte-for-byte.

The focused CPU run passed 114 tests in 98.67 s. The Windows project `.venv`
regression reported 1,069 passed, four known failures, 15 skips and one xfail
in 24 min 16 s. Two failures were CRLF/LF notebook comparisons and two were
junction permission errors. Six skips were missing Torch and nine were link
permissions. Ruff passed on `src tests`. Its repository-wide run reported
the 11 existing findings in the historical Kaggle notebook.

```powershell
$stamp = [guid]::NewGuid().ToString('N')
& $vocoPython -B -m pytest -ra -o pythonpath=src -p no:cacheprovider --basetemp "runs/vocoroute-n2b1p/pytest-focused-$stamp" tests/test_vocoroute_n2b1p.py tests/test_vocoroute_n2b1.py tests/test_vocoroute_n2a.py tests/test_vocoroute_n2k.py tests/test_vocoroute_n1.py
$stamp = [guid]::NewGuid().ToString('N')
& '.venv\Scripts\python.exe' -B -m pytest -ra -o pythonpath=src -p no:cacheprovider --basetemp "runs/vocoroute-n2b1p/pytest-regression-$stamp"
& '.venv\Scripts\python.exe' -B -m ruff check --no-cache src tests
& '.venv\Scripts\python.exe' -B -m ruff check --no-cache . --output-format concise
```

## Frozen matched Kaggle protocol

One private GPU notebook attaches only `sambhavjainsj/vocoroute-vctk092-n2b1-p229-128`,
dataset ID `13671242`, saved version 1. The worker reuses the N2B.1 loader,
which rejects extra mounted files and verifies the pinned corpus, manifests
and recordings. GPU provisioning is `NvidiaTeslaT4` with two Tesla T4s
visible and only `cuda:0` used. Internet and TPU are disabled.

| Setting | Both arms |
| --- | --- |
| Backbone | Unchanged 566,144 parameters, independent instances initialized with seed `20260907` |
| Optimizer | Independent AdamW states, constant LR `0.0002`, decay `0.01`, betas `0.9` and `0.999`, epsilon `1e-8`, foreach and fused disabled |
| Reconstruction | Waveform L1, magnitude L1 and log-magnitude L1 each weighted 1 |
| Phase | A disabled, effective weights 0. B enabled, temporal and frequency weights `0.05` each |
| Precision | Deterministic FP32, TF32 disabled |
| Update limit | 4,000 per arm, fixed single-utterance ordering, matched pairs |
| Snapshots | 0, 100, 250, 500, 1,000, 2,000, 4,000 and any last matched off-schedule step |
| Budget | 660 s worker deadline, 780 s Kaggle timeout, 900 GPU seconds planned |

The phase weights are the existing documented N2A settings. A CPU backward
probe on the final N2B.1 checkpoint measured weighted combined phase loss
`0.04265` and gradient L2 `0.20888`, against common loss `0.72034` and gradient
L2 `6.47636`. This verifies a finite nonzero signal at the fixed weights. It
does not calibrate a universal optimum.

The worker checks identical initial tensor bytes and step-zero waveforms. On
an exact original runtime match, it also requires the original initial tensor
hash. It compares Run A step 1,000 with the original raw artifact hash and
common reconstruction loss when that snapshot exists. A local numerical check
after retrieval reports max absolute waveform difference and MAE and records
runtime and initial hash matches. The diagnostic tolerance is `atol=2e-4`,
`rtol=1e-3` and both the numerical and exact results are kept.

The two arms have identical common evaluation metrics, regardless of their
training weights. Step CSVs separate pre-update common reconstruction from the
weighted optimization objective and record gradient norms and update timings.
The post-update snapshots also include RMS, clipping and phase errors. Output
folders `A/` and `B/` identify conditions. `blinded/` holds three neutral clips
for the reference and last matched A/B pair. `listening-key.json` maps clips to
conditions. The blind clips copy the PCM16 bytes without gain changes.

Before admitting a pair, the worker reserves the larger of five seconds and
twice the longest observed pair. Once admitted, the pair completes before the
worker checks its stop flag again. On wall-time or orderly signal stop, both
checkpoints and snapshots use the same completed update. Per-arm peak memory
is measured during that arm's updates in the shared CUDA process. Resident
state for the other arm contributes to process memory and reservation. The
worker also records the increase over the pre-step allocator baseline.

N2B.1 took 26.02 s for 1,000 updates and snapshots, and Kaggle charged 54.255 s.
A conservative projection using that pace for A and twice that pace for B
gives about 312 s for 8,000 updates combined. Adding 30 s for the prior
observation of overhead gives about 342 s. A seven-update synthetic CPU
profile at full utterance length measured warm median update times of
0.154 s in A and 0.191 s in B. This profile checks the introduced work on
CPU. T4 phase-loss runtime and actual quota are unmeasured in N2B.1-P. The
worker and Kaggle limits leave 120 s for saving and another 120 s of planned
allowance.

## Commit and approval gate

Only the user creates and pushes commits. The existing `7788489` commit lacks
the new files and fails RECORD staging. Once a new commit exists, verify it
against live `origin/main`, regenerate the stage, verify the staged hashes and
compare `protocol_sha256` with the frozen draft. Verify the existing private
dataset version and its five-file inventory, current quota and T4 ×2 option.
Submit one job only after explicit GPU approval, using `--timeout 780` and
`--accelerator NvidiaTeslaT4`. Recover the same saved version, reconcile both
arm checkpoint steps, output and receipt hashes, and verify actual quota.

```powershell
$commit = Read-Host 'Approved user-created N2B.1-P commit SHA'
$stage = 'runs/vocoroute-n2b1p/record-' + $commit.Substring(0, 7)
& $vocoPython -B -m experiments.cli kaggle vocoroute-phase-plan --sha $commit --baseline-receipt "$baseline/diagnostic-receipt.json" --out-dir $stage
Get-FileHash -Algorithm SHA256 "$stage/kernel/phase_ablation.py", "$stage/kernel/kernel-metadata.json"
# Only after a completed authorized job has been retrieved locally.
& $vocoPython -B -m vocoroute.phase_diagnostic $baseline 'path/to/retrieved/vocoroute-n2b1p' --compare-baseline
```
