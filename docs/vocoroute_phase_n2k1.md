# VocoRoute-P N2K.1 Kaggle acceptance plan

Scope is deterministic synthetic training mechanics on the existing `da4a7ac`
trainer. The private jobs, uploads, retrieval and GPU time in this plan require
approval. Read-only Kaggle account queries and local staging are complete.

## Readiness checked 24 November 2026

- Installed Kaggle CLI **2.2.4** authenticated via access token for
  `sambhavjainsj`. Authenticated `datasets list --mine`, `kernels list --mine`,
  and `quota` succeeded. The filtered inventory showed historical `btpvf-v0`,
  `btpvf-code`, and six `btpvf-*` notebooks. It showed no VocoRoute acceptance
  resources. Direct status checks did not confirm the three proposed dataset
  refs. The existing `btpvf-code` returned `ready`, version 11. Confirm the
  new refs are free before creation and keep the historical resources separate.
- `kaggle quota` reported **30.00 GPU hours remaining of 30.00**, zero consumed,
  and next reset `2026-11-28T00:00:00` as returned by Kaggle. This is an account
  allowance. Actual accelerator availability is checked in the signed-in
  notebook UI under **Settings → Accelerator**, without starting a session.
  The **GPU quota** indicator in the notebook's Settings/session panel
  provides the UI cross-check when the CLI endpoint is unavailable.
- Installed `kernels push --help` supports `--accelerator` and `--timeout`
  and immediately schedules execution on a successful push. Versioned
  `kernels pull` and `kernels output` are supported. Request **one NVIDIA L4**
  (`NvidiaL4X1`) in both sessions after checking availability on the account.
  If unavailable, agree on a single supported GPU type before either push.

[Kaggle CLI kernel commands](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md)
· [Kaggle quota announcement](https://www.kaggle.com/discussions/product-feedback/173129)
· [Kaggle accelerator selection](https://docs.nvidia.com/datascience/deployment/latest/platforms/kaggle/)

Kaggle publishes a GPU image build and package recipe. The actual selected
GPU image's Torch, TorchAudio, CUDA, Python, SoundFile, NumPy and librosa builds
remain unverified. A imports the provisioned packages and runs the pinned
trainer's FP32 complex STFT/iSTFT forward/backward before any update. Missing
packages, unavailable CUDA or nonfinite gradients stop the experiment. If exact
image versions are available beforehand, render A with `--runtime` and the
observed runtime JSON. The staged A instead explicitly observes the provided
GPU runtime at startup. B is generated with exact pins from A; the trainer's
existing checkpoint admission independently requires matching runtime, CUDA
device provenance, code, config and manifest. The job has internet disabled and
uses the provisioned environment. [Kaggle GPU image recipe](https://github.com/kaggle/docker-python)

## Fixture and resources

The fixture is `vocoroute-n2k1-fixture-v1`. Integer arithmetic generates five
mono PCM16 WAVs at 22,050 Hz, with distinct triangle periods, lengths and
single-sample impulses. There are three train examples of 1,409, 1,537 and
1,665 samples, and two dev examples of 1,793 and 1,921 samples. A sixth eval
row has a deterministic byte identity and no corresponding file. Every row
has an independent synthetic speaker and source group. Native and prepared
paths and SHA-256s agree within a row, with zero resampling. Preparation
verifies every audio digest, header and SoundFile decoded length through
`TrainingAudio`. `fixture-provenance.json` records the fixture-only scope,
generator hash and manifest hash. No historical or VCTK audio is read.

| Private resource | Contents and use |
|---|---|
| `sambhavjainsj/vocoroute-n2k1-code` v1 | Existing N2K harness record snapshot at `da4a7ac`, archive and provenance |
| `sambhavjainsj/vocoroute-n2k1-fixture` v1 | Five WAVs, frozen manifest, dataset receipt, fixture provenance |
| `sambhavjainsj/vocoroute-n2k1-session-a` | First script version, writes control, interrupted and budget probes |
| `sambhavjainsj/vocoroute-n2k1-handoff` v1 | Two verified saved A checkpoints, train traces, and handoff receipt |
| `sambhavjainsj/vocoroute-n2k1-session-b` | Second script execution, restores A's published handoff and writes comparison |

Allow 256 MiB under `/kaggle/working` per session and 64 MiB for the three
private datasets combined. These are planning margins from the CPU run.
Recovered files sit under `vocoroute-run/` in the saved notebook output.
Keeping a separate versioned handoff lets B verify the exact A output even if
Kaggle's notebook source attachment selects a latest version by slug.
[Kaggle notebook output limits](https://www.kaggle.com/docs/notebooks)

## Exact experiment

Both control and resume use the same canonical v2 configuration with FP32,
seed `20260907`, two epochs over three train rows each, at most
six optimizer updates, 120 seconds per trainer attempt, and a checkpoint
every two updates. The original `da4a7ac` code snapshot bootstrap verifies the
archive or expanded tree before import. The staged script adds instrumentation
around `vocoroute.train_base.main` and checkpoints. The production runner and
inference checkpoint schema are the ones in the Git snapshot. Keep the staged
script hashes and Kaggle pull metadata with the receipts.

1. **Session A** records GPU name, compute capability, CUDA and library
   versions, available and total memory, and driver details. The actual
   trainer checks CUDA FP32 forward/backward before updates and validates
   finite losses and gradients at every update. A runs the six-update control
   to epoch completion. Its second attempt uses `--attempt-updates 4` and saves
   the validated step-four checkpoint with one completed dev epoch. A saves
   both runs, their profiles, and the checkpoint receipts.
2. **Session A budget probes** use distinct, labeled configurations. At a
   cumulative cap of four, the runner stops after one completed dev epoch and
   withholds its conventional `inference/` export. The probe uses the existing
   strict `read_resume` interface to recover the best validated weights,
   loads them into a fresh backbone, and checks a finite dev loss. It records
   that recovery while leaving `convergence_claimed=false`. A separate
   five-second wall probe injects a controlled six-second delay inside the
   first example and reports elapsed time and overshoot. Probe timings are
   separate from acceptance performance measurements.
3. **Session B** receives a private v1 dataset built from A's exact saved
   output version. It checks the outer handoff pin, every included file hash,
   both checkpoint receipts, the config identity, and the frozen runtime.
   It resumes at update four and performs exactly two optimizer updates.
   Its final weights, best weights, AdamW state, scheduler, cursor and loss
   history are compared to the uninterrupted control. It also compares the
   complete update order, Python/Torch CPU/CUDA sentinel draws and final RNG
   states. Sentinel draws deliberately advance the RNGs once per update in
   every leg to exercise actual restoration. The report distinguishes exact
   equality from tolerance-only agreement (`atol=1e-7`, `rtol=1e-6`, floats
   only). Updates, optimizer step counters, scheduler state, order and RNG
   comparisons require exact agreement. Report any floating-point discrepancy
   with its field and maximum absolute gap. A tolerance-only result is evidence
   of observed run-to-run numerical variation. Check both reported driver
   versions before attributing variation to a GPU operation.

Per-attempt profiling synchronizes CUDA before and after updates and
checkpoint operations. It records allocated/reserved memory peaks, per-update
wall time, train audio seconds per timed update second, checkpoint byte counts
and save/load seconds, trainer preflight time, and script setup elapsed time.
Record Kaggle queue, provisioning and output-save timestamps separately from
the script's process clock. The short fixtures exercise mechanics. Their
throughput is unsuitable for extrapolating real speech training duration.

## Approval-gated commands

Local staging is already available under `runs/vocoroute-n2k1/gpu-approval`.
It can be reproduced with a fresh root from the repo.

```powershell
$vocoPython = 'C:\Users\samak\Desktop\BTP\vocoroute-env\Scripts\python.exe'
$env:PYTHONPATH = (Resolve-Path -LiteralPath src).Path
$stage = (Resolve-Path -LiteralPath 'runs/vocoroute-n2k1/gpu-approval').Path
# Reproduce locally using --root with a fresh directory.
& $vocoPython -B scripts/vocoroute_acceptance.py prepare --root 'runs/vocoroute-n2k1/gpu-reproduction' --owner sambhavjainsj --commit da4a7ac
```

Run the next blocks only after explicit approval of these private uploads,
job submission and retrieval, observed-runtime admission for A, and the GPU
budget. Verify the CLI output after each action. The new dataset refs are
private by CLI default, and both staged kernels set `is_private=true`.

```powershell
$kaggle = Join-Path $env:APPDATA 'Python/Python313/Scripts/kaggle.exe'
$owner = 'sambhavjainsj'
$accelerator = 'NvidiaL4X1'
& $kaggle quota
# Confirm the two refs do not exist before creating these v1 datasets.
& $kaggle datasets create -p "$stage/n2k-stage/code" --keep-tabular --dir-mode skip
& $kaggle datasets create -p "$stage/fixture" --keep-tabular --dir-mode skip
& $kaggle datasets status "$owner/vocoroute-n2k1-code" --format json
& $kaggle datasets status "$owner/vocoroute-n2k1-fixture" --format json
# Require both READY at version 1, then submit exactly once.
& $kaggle kernels push -p "$stage/kernel-a" --accelerator $accelerator --timeout 600
```

Record the actual saved kernel version from the push result and Kaggle UI.
Record or screenshot the attached dataset versions without secrets. Kernel
metadata names the source slugs, and the script independently checks the
committed code tree, fixture dataset receipt, and byte hashes. Wait for the
exact version to finish before retrieving its saved output. The recovery
command refuses to stage B unless the saved A receipt passed.

```powershell
$aVersion = Read-Host 'Exact saved Session A version number'
$a = "$owner/vocoroute-n2k1-session-a/$aVersion"
& $kaggle kernels status $a
& $kaggle kernels pull $a --metadata -p "$stage/a-code-$aVersion"
& $kaggle kernels output $a --page-size 200 -p "$stage/a-output-$aVersion"
& $vocoPython -B scripts/vocoroute_acceptance.py recover --root $stage --saved "$stage/a-output-$aVersion" --kernel-version $a
# Require successful recovery and a fresh nonexistent handoff ref.
& $kaggle datasets create -p "$stage/handoff" --keep-tabular --dir-mode skip
& $kaggle datasets status "$owner/vocoroute-n2k1-handoff" --format json
& $kaggle quota
# Require handoff READY at version 1 and sufficient remaining approved time.
& $kaggle kernels push -p "$stage/kernel-b" --accelerator $accelerator --timeout 420
```

```powershell
$bVersion = Read-Host 'Exact saved Session B version number'
$b = "$owner/vocoroute-n2k1-session-b/$bVersion"
& $kaggle kernels status $b
& $kaggle kernels pull $b --metadata -p "$stage/b-code-$bVersion"
& $kaggle kernels output $b --page-size 200 -p "$stage/b-output-$bVersion"
```

Read the saved `kernel-receipt.json` in both sessions. The script catches
failures so logs can persist even if the Kaggle process has exit status zero.
Require A and B `passed` receipts and B's `comparison.json` before reporting
acceptance. Preserve all failure information if a job fails. There is no
automatic retry or replacement. [Kaggle dataset metadata](https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets_metadata.md)
· [Kaggle source metadata](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md)

## Maximum 30 GPU minutes

| Allocation | Notebook script timeout | Provisioning and saving allowance |
|---|---:|---:|
| A | 10 minutes | 4 minutes |
| B | 7 minutes | 3 minutes |
| Shared contingency | | 6 minutes |

All fixture generation, hashing, staging, recovery, and package-readiness
review run on CPU outside GPU jobs where possible. Measure charged time from
GPU allocation. The script timeout does not control external provisioning.
Watch the session clock and quota, cancel in Kaggle's running-version UI if an
approved cutoff is reached, and only start B if the remaining allowance
safely covers its envelope. If A fails, misses the window, or fails to preserve
its checkpoint, stop before B. No additional GPU jobs are in the plan.

## Local verification

The fixture manifest SHA-256 is
`60442b6b96f9c59df9756a7754191c65dcf704e35002338937c99c73af88d34f`.
The staged CUDA config digest is
`cd626a6f163b1960cc48a66a17c43ecd4d01abd597eb3c8f23d2ef49bc0ebda2`.
The CPU rehearsal used the same fixture and commit with device-specific
config and runtime pins. A and B ran as separate Python processes, with B's
input copied through the pinned handoff directory. The control and resume
matched exactly with nested and flat mock mounts and with the extracted code
mount. A modified handoff payload was rejected before resume. The cumulative
cap probe restored validated best weights after update four. The wall probe
stopped after one update at 6.287 seconds,
1.287 seconds after the five-second target. Staged fixture, handoff and saved
output sizes were approximately 21 KB, 18.3 MB and 94/39 MB for A/B.

The resumed inference bundle validated against the saved code snapshot.
Validation against the ordinary Windows checkout refused the byte identity
because its `model.py` uses LF and the staged archive uses CRLF. Their text
matched after normalizing line endings. Use the exact archived source to
validate an inference bundle produced by these kernels.

```powershell
$cpu = 'runs/vocoroute-n2k1/cpu-reproduction'
& $vocoPython -B scripts/vocoroute_acceptance.py prepare --root $cpu --owner sambhavjainsj --commit da4a7ac --local
& $vocoPython -B "$cpu/kernel-a/train.py"
& $vocoPython -B scripts/vocoroute_acceptance.py recover --root $cpu --saved "$cpu/local-output-a" --kernel-version sambhavjainsj/vocoroute-n2k1-session-a/1
& $vocoPython -B "$cpu/kernel-b/train.py"
& $vocoPython -B -m pytest -ra -o pythonpath=src -p no:cacheprovider --basetemp runs/vocoroute-n2k1/pytest-focused-20261124 tests/test_vocoroute_n1.py tests/test_vocoroute_n2a.py tests/test_vocoroute_n2k.py tests/test_audio_io_stream.py
$env:NUMBA_CACHE_DIR = Join-Path (Resolve-Path -LiteralPath 'runs/vocoroute-phase-n2k') 'numba-cache'
& '.venv\Scripts\python.exe' -B -m pytest -ra -p no:cacheprovider --basetemp runs/vocoroute-n2k1/pytest-full-20261124
& '.venv\Scripts\python.exe' -B -m ruff check --no-cache scripts src tests
& '.venv\Scripts\python.exe' -B -m ruff check --no-cache . --output-format concise
```

The focused CPU suite passed 90 tests in 37.81 seconds. The project `.venv`
full suite reported **1,069 passed, 4 failed, 12 skipped and 1 xfailed** in
19 minutes 36 seconds. The four failures were the known generated-notebook
line endings and Windows junction permission failures. Nine skips were
filesystem permissions, three were Torch modules exercised by the separate
focused suite, and the xfail was the existing INV-17 ablation. Ruff passed
for `scripts src tests`. Repository-wide Ruff reported the same 11 findings
in the historical generated notebook. GPU and saved Kaggle output metrics
are pending the approved experiment.
