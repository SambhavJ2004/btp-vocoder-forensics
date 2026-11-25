# VocoRoute-P N2B.1 single-utterance diagnostic

The regular base trainer requires a disjoint development split and selects its
checkpoint using dev loss. N2B.1 fits only `p229_128` and saves the backbone
at the actual stopping step. It reuses the base model, optimizer construction,
frontend, loss, and waveform loader. The protocol measures reconstruction of
the training utterance. Speech quality and generalization require later evaluation.

## Corpus and minimum private attachment

The source is `sambhavjainsj/vocoroute-vctk092-n2b0-train-dev/1`, notebook ID
`147338652`. `p229_128` is the first entry in its eight-item diagnostic manifest
and a member of pilot train. The prepared mic2 waveform has 76,523 samples at
22,050 Hz. The committed corpus descriptor contains all content pins.

| Artifact | SHA-256 |
|---|---|
| Corpus freeze | `39ae136dbf240762a8c2abd38e39f8c56f9d246cd191c6f7dae014400808ec6b` |
| Eight-item selection | `20601c92e22cb203c38dc1ed85d5827c8814e9dfdf46cc7310aab978056f9f5c` |
| Parent pilot manifest | `5ed731b8c47dcd0331f14ab1b5d877200fb6f7e09500fbb7507f94d998167d2f` |
| Original 48 kHz mic2 FLAC | `dee0b6e5e34c2682e9dad351edbe6988b231aa9c423f228c8b428b81cc1117d7` |
| Prepared 22.05 kHz WAV | `5648c74a7014cbc335417687b32d819209a2e0ee77b85b90687ce66b771d10f8` |

After approval, use **Output → Create Dataset** on the exact N2B.0 saved
version. Use private dataset `sambhavjainsj/vocoroute-vctk092-n2b1-p229-128`,
version 1, and select exactly these five output files. The connector can
retain the folder structure or flatten the filenames. The worker verifies
both layouts. [Kaggle Dataset documentation](https://www.kaggle.com/docs/datasets)

```text
n2b0-train-dev/freeze.json
n2b0-train-dev/overfit/manifest.json
n2b0-train-dev/overfit/parent-manifest.json
n2b0-train-dev/overfit/native/train/p229/p229_128_mic2.flac
n2b0-train-dev/overfit/prepared/train/p229/p229_128.wav
```

The parent manifest supplies metadata for verifying the original role and
speaker assignments. The worker validates the freeze and both manifests,
then projects the approved training row and checks both recordings through
`TrainingAudio`. It rejects any additional mounted file before opening audio.
The GPU job attaches only the five-file dataset. Code is embedded from the
record commit, and the corpus stays in Kaggle storage during preparation.

## Training and timing protocol

| Setting | Value |
|---|---|
| Backbone | Existing 566,144 trainable parameters, seed `20260907` |
| Device | T4 ×2 provisioned, `cuda:0` used, FP32 with deterministic algorithms and TF32 disabled |
| Input | One complete, untrimmed example, with mel and training target derived from the same WAV |
| Loss | Existing waveform L1, multi-resolution magnitude L1 and log-magnitude L1, each weighted 1. Phase difference disabled |
| Optimizer | AdamW with learning rate `0.0002`, weight decay `0.01`, betas `[0.9, 0.999]`, epsilon `1e-8`, foreach and fused off |
| Learning rate | Constant over the diagnostic. The normal trainer's ExponentialLR runs at dev-backed epoch boundaries |
| Snapshots | 0, 25, 100, 250, 500 and 1,000 completed updates, plus the last completed step after an early stop |
| Limits | 1,000 updates, 840 s internal deadline, 1,080 s Kaggle session timeout, 1,200 GPU seconds total planned |

The internal deadline starts when the Kaggle driver enters the worker, before
the training imports and input checks. An update is admitted only before the
deadline. The worker finishes the in-flight update and saves the last waveform,
checkpoint and receipt on an orderly stop. The platform timeout leaves
240 seconds beyond the training deadline and the 20-minute plan reserves another
120 seconds for platform overhead. Before submission, verify live quota, that
Kaggle offers T4 ×2, and that the requested kernel slug is unused. Monitor the
only submitted version including output saving. Escalate to Kaggle cancellation
if needed to stay within the authorized quota. Reconcile actual used and
remaining quota from before and after receipts. A scheduler delay or a hard
platform stop can affect output recovery. There is one submission.

## Artifacts and reading the result

`reference.wav` is the exact prepared input. Each saved `step-NNNN` has raw
float32 `.npy`, PCM16 `.wav`, and metrics `.json` files. Waveform MAE is the
waveform component of the reconstruction loss. Log-magnitude STFT error is
the existing mean absolute log-magnitude loss. Spectral convergence is the
mean across the three loss resolutions of `||S_pred - S_ref||_F / max(||S_ref||_F, 1e-4)`
for magnitude spectrograms, using the same STFT settings as the loss.

Metrics use the raw waveform. The WAV export saturates samples outside
`[-1, 32767/32768]`, records the changed sample count, and applies no gain
normalization. A nonfinite snapshot retains the original float array and the
nonfinite count and fails before WAV export. `train-loss.csv` records each
successful update's pre-update terms, gradient L2 and elapsed time. Snapshot
metrics use the weights after the labeled update. `backbone.pt` contains the
final backbone state dict for `torch.load(..., weights_only=True)`. Its receipt
records changed scalar count, L2 and maximum absolute parameter delta, and
hashes of both initial and final flattened FP32 parameters.

`diagnostic-receipt.json` binds the corpus, dataset version plan, commit, exact
embedded code, configuration, runtime, both visible accelerators, selected
device, peak allocated and reserved GPU memory, elapsed times, update count,
and output hashes. Treat a finite completed run with changed weights as a
successful execution. Use the initial and final reconstruction metrics to
state whether training reduced error. Report any plateau or clipping with
the actual snapshots. A budget-limited run is partial. The waveform variation
check uses a standard deviation above one PCM16 step as a numerical sanity
check. It carries no listening judgment. Listen to the reference, untrained,
intermediate and final clips before making any perceptual observation.

For recovery, use an exact-version Kaggle API call or the saved version's
Output tab. Retrieve only `vocoroute-n2b1/` diagnostic artifacts and verify
their hashes. The regular base runner's dev-selected checkpoint schema still
applies to later speaker-disjoint training. Before the eight-utterance
diagnostic, review completed-run loss, clipping, duration-scaled GPU cost and
listening evidence, then keep per-utterance snapshots so aggregate loss does
not mask an individual failure.

## Record staging and launch after approval

Commit these implementation files manually and push the commit to `origin/main`.
The staging command checks the committed source archive and rejects modified
source bytes or a commit missing the diagnostic. After verifying the live
commit and exact private dataset version, stage locally and use the accepted
Kaggle accelerator identifier. [Official Kaggle CLI accelerator documentation](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md)

```powershell
$repo = 'C:\Users\samak\Desktop\BTP\btp-vocoder-forensics'
$env:PYTHONPATH = "$repo\src"
$vocoPython = 'C:\Users\samak\Desktop\BTP\vocoroute-env\Scripts\python.exe'
$commit = Read-Host 'Approved user-created N2B.1 commit SHA'
$stage = "$repo\runs\vocoroute-n2b1\record-$($commit.Substring(0, 7))"
& $vocoPython -B -m experiments.cli kaggle vocoroute-overfit-plan --sha $commit --out-dir $stage
Get-FileHash -Algorithm SHA256 "$stage\kernel\overfit.py", "$stage\kernel\kernel-metadata.json"
# Submit only the approved version after the private dataset and stage checks.
kaggle kernels push -p "$stage\kernel" --accelerator NvidiaTeslaT4 --timeout 1080
```

The planned job is `sambhavjainsj/vocoroute-vctk092-n2b1-overfit`. The stage
sets private visibility and internet and TPU off. Confirm the server metadata
and downloaded stage hashes before citing results. Use only the new diagnostic
version, and retain the exact job and dataset IDs in the verification record.

## Local verification before the commit gate

The recovered N2B.0 metadata matched the three corpus pins. The fixed row
matched both pinned recording hashes, its TRAIN role and its sample count.
The planned diagnostic configuration SHA-256 is
`491b37741eb94e56d23ac755a53b0d9a84362a8bac265ca39543f03149e4fa90`.

The focused CPU suite passed **121 tests in 84.69 s**. A short synthetic run
completed three finite updates and changed all 566,144 parameters. Its total
reconstruction loss went from 13.4936 to 9.1388 and its waveform MAE went from
0.55041 to 0.29584. The saved checkpoint reproduced the final float waveform
within the test tolerance. A separate synthetic test completed one update at
the real utterance's full 76,523-sample geometry. Tests also covered exact
attachment contents, fail-closed source and GPU checks, export clipping and
raw waveform preservation, and checkpointing the last step at a budget stop.

The new stager refused current commit `f9348828bedfffdb5e18e6a776fb00cf00d90782`
because the archive lacks the new implementation. The generated worker also
ran against an empty local mock mount and saved a zero-update failure receipt.

```powershell
& $vocoPython -B -m pytest -ra -o pythonpath=src -p no:cacheprovider --basetemp runs/vocoroute-n2b1/pytest-focused-FRESH tests/test_vocoroute_n2b1.py tests/test_vocoroute_n2b0.py tests/test_vocoroute_n2a.py tests/test_vocoroute_n2k.py tests/test_vocoroute_n1.py tests/test_audio_io_stream.py
$env:NUMBA_CACHE_DIR = (Resolve-Path -LiteralPath 'runs/vocoroute-n2b1').Path
& '.venv\Scripts\python.exe' -B -m pytest -ra -o pythonpath=src -p no:cacheprovider --basetemp runs/vocoroute-n2b1/pytest-regression-FRESH
& '.venv\Scripts\python.exe' -B -m pytest -ra -o pythonpath=src -p no:cacheprovider --basetemp runs/vocoroute-n2b1/pytest-relocation-FRESH tests/test_phase_a_run.py::TestManifestPathsSurviveRelocation::test_a_renamed_directory_still_resolves
& '.venv\Scripts\python.exe' -m ruff check --no-cache src tests
& '.venv\Scripts\python.exe' -m ruff check --no-cache . --output-format concise
```

The broad project suite reported **1,068 passed, 5 failed, 14 skipped, and
1 xfailed in 24 min 40 s**. The failures were two generated-notebook line
ending comparisons, two Windows junction permission tests, and the recurring
Windows directory rename error in Phase A. The rename test passed in isolation
in 18.27 s. Nine skips were Windows link permissions and five were missing
Torch in the project virtual environment. The xfail was the existing INV-17
ablation. The focused CPU suite used the separate Torch installation.

Ruff passed for `src tests`. Repository-wide Ruff reported 11 existing findings
in `notebooks/kaggle_v0.ipynb`. The local measurements are synthetic. The
Kaggle dataset creation, real-speech training, and GPU usage are pending the
user commit and approval.
