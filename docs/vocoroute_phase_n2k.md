# VocoRoute-P Phase N2K training readiness

N2K extends the 566,144-parameter base runner. All execution in this phase
used short synthetic CPU examples. The Kaggle workflow was staged and tested
locally with generated fixtures. CUDA execution, Kaggle persistence, real data
acquisition, and speech quality remain unverified.

## Device, budget and resume contract

`vocoroute-base-training-v1` keeps its canonical config and CPU inference bundle
layout. `vocoroute-base-training-v2` adds explicit `cpu` or `cuda` device and
three required budget fields in `budget`. Existing loss and optimizer settings
are carried over. Initialize a v2 configuration with `--print-default-config
--device cuda` or `--device cpu` and set a frozen manifest SHA-256.

| Setting | Enforcement |
|---|---|
| `epochs`, `budget.max_updates` | Cumulative ceilings across resumes. An update is counted only after a successful optimizer call. Dev can finish an epoch when the last permitted update completed its train split. |
| `budget.wall_time_seconds` | Per attempt, measured by monotonic clock from runner entry. Checked between train and dev examples. Dataset byte verification and CUDA acceptance occur before runner entry. An in-flight example or checkpoint write may extend wall time. |
| `budget.checkpoint_every_updates` | Immutable resume bundle each configured interval and at controlled exit. Checkpoint retention is append-only within an attempt. Choose cadence with the output storage budget in mind. |
| `--attempt-updates` | Optional per-invocation pause limit inside the cumulative update ceiling. Use this for planned segmentation without changing the pinned config. |
| `SIGINT` and `SIGTERM` | Request a stop between examples. A caught training error records failure and preserves the previous published checkpoint. A forced kill can recover only already persisted bundles. |

CUDA selects visible device 0 and refuses fallback. The model, waveform,
frontend and loss buffers follow the selected device. The CUDA path keeps
FP32, disables TF32 and cuDNN benchmarking, requests deterministic algorithms,
and runs a synthetic STFT/iSTFT forward/backward acceptance check before
training. A detected failure stops the run. The receipt records Python, Torch
and audio versions, device name, architecture, VRAM, CUDA and cuDNN versions
and deterministic settings. Kaggle records `nvidia-smi` output when available.
CUDA determinism, optimizer restore, timing and VRAM still need an approved
GPU acceptance run. [PyTorch deterministic algorithms](https://docs.pytorch.org/docs/main/generated/torch.use_deterministic_algorithms.html)

Each `resume-*` directory contains `training.pt` and a receipt written last.
The caller supplies the receipt SHA-256. The loader verifies the state hash,
run/config/manifest/source/runtime identity, tensor schema, optimizer moments
and step counts, scheduler epoch and learning rate, and the data cursor. It
uses weights-only loading and restores Python and Torch CPU RNG, plus the
selected CUDA RNG when applicable. The epoch-local order is regenerated from
seed and frozen manifest. Partial train/dev losses and cursors avoid replaying
completed updates or dev examples. Use a **new output directory** for resume.

`run.json` records planned progress, latest published receipt, epoch metrics,
and `completed`, `budget_limited`, `interrupted`, or `failed`. It always records
`convergence_claimed=false`. Full planned epoch completion emits an `inference/`
bundle using the existing `base.pt` and checkpoint receipt format. A budget
stop leaves resume state and its progress. No training here produced research
evidence or a public model.

## Local and Kaggle reproduction

Use the existing isolated CPU environment for synthetic validation and config
creation. These commands write only local staging or config output.

```powershell
$vocoPython = 'C:\Users\samak\Desktop\BTP\vocoroute-env\Scripts\python.exe'
$env:PYTHONPATH = (Resolve-Path -LiteralPath src).Path
& $vocoPython -B -m vocoroute.train_base --dry-run --device cpu
& $vocoPython -B -m vocoroute.train_base --print-default-config --device cuda
# After the user has made an N2K commit and approved the config/spec:
& $vocoPython -B -m experiments.cli kaggle vocoroute-plan --sha <existing-N2K-commit> --spec <kaggle-spec.json> --config <training-config.json> --out-dir <fresh-local-dir>
```

The new local-only subcommand uses the harness's record snapshot and kernel
metadata functions. It stages `code/`, `kernel/`, and `plan.json`. The current
historical N2A commit `c9a62e2` predates N2K. Staging for future execution requires
a user-created commit containing the new code. The generated private kernel
starts with internet disabled, requests a GPU and verifies the attached code
tree before importing it. It uses the approved runtime as provisioned and
performs no package installation. Historical forensic notebooks and their
submission path keep their current behavior.

Example staging spec. Replace the owners, versions, mount paths, digests and
runtime with values actually observed and approved for the intended job.

```json
{
  "schema": "vocoroute-kaggle-spec-v1",
  "title": "VocoRoute N2K base pilot",
  "code": {"ref": "owner/vocoroute-n2k-code", "version": 1, "mount": "datasets/owner/vocoroute-n2k-code"},
  "data": {"ref": "owner/vocoroute-train-dev", "version": 1, "mount": "datasets/owner/vocoroute-train-dev", "receipt_sha256": "<64 lowercase hex characters>"},
  "resume": null,
  "runtime": {"python": "<version>", "torch": "<exact build>", "torchaudio": "<exact build>", "cuda_build": "<version>", "soundfile": "<version>", "numpy": "<version>", "librosa": "<version>"}
}
```

- Explicitly attach the approved dataset versions and verify them in the
  Kaggle version metadata. The CLI's `dataset_sources` accepts `owner/slug`,
  so the staging metadata alone does not pin attachment versions. The kernel
  checks the archive or expanded tree, the train/dev dataset receipt SHA-256,
  and the manifest and recording bytes independently. Train/dev mount root
  contains `manifest.json` and `dataset-receipt.json`. The receipt has
  `schema: "vocoroute-kaggle-train-dev-v1"`, `ref`, `version`,
  `manifest_sha256`, and `audio_roles: ["train", "dev"]`. Eval paths in the
  frozen manifest must be absent from this mount. Attach no eval dataset.
- With later upload and execution approval, use the staged snapshot as a
  separate private code dataset and the generated kernel files as a private
  script version. Select the approved accelerator and retain the exact
  attached version metadata. The kernel writes metrics, periodic resume
  bundles, validation receipts and `training.log` under
  `/kaggle/working/vocoroute-run`. Recover the exact saved output version
  before starting another attempt. A script can finish successfully at the
  platform level while its `kernel-receipt.json` records `failed` to preserve
  the error logs. An infrastructure kill may prevent Kaggle from saving the
  current attempt. Keep the last independently saved checkpoint version.
- Resume with the same config and runtime. Replace `resume: null` in the spec
  with an object containing `ref`, `version`, `mount`, `checkpoint` (relative
  directory below that dataset mount), and the externally recorded
  `receipt_sha256`. Attach only the approved resume snapshot. A pinned code
  snapshot establishes code identity, and the kernel receipt alone does not
  establish citable research evidence.

After separate execution and retrieval approval, record the exact kernel
version and recover its code metadata and saved output into fresh directories.

```powershell
kaggle kernels pull 'owner/kernel-slug/42' --metadata -p '.\saved-kernel-42'
kaggle kernels output 'owner/kernel-slug/42' -p '.\saved-output-42'
```

[Kaggle CLI kernel metadata](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md)
· [Kaggle CLI kernel version and output commands](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md)

The 18 synthetic `training.pt` snapshots measured 6,862,109–9,142,197 bytes
each, depending on whether AdamW moments and a best-model copy were present.
Estimate persistent output from the checkpoint cadence and attempt count,
including the controlled-exit snapshot, metrics and logs.

## Verification

The isolated Windows environment verified Python 3.13.9, Torch and TorchAudio
2.9.0+cpu, SoundFile 0.14.0, NumPy 2.5.3, SciPy 1.18.1 and librosa 1.0.0.
`pip check` passed. Direct CPU dry-run reported 566,144 trainable parameters.
The same interpreter's CUDA dry-run exited 2 with `CUDA was requested, but no
usable CUDA device is available`.

```powershell
$vocoPython = 'C:\Users\samak\Desktop\BTP\vocoroute-env\Scripts\python.exe'
& $vocoPython -B -m pytest -ra -o pythonpath=src -p no:cacheprovider --basetemp runs/vocoroute-phase-n2k/pytest-focused2-20261124 tests/test_vocoroute_n1.py tests/test_vocoroute_n2a.py tests/test_vocoroute_n2k.py tests/test_audio_io_stream.py
```

**90 passed in 36.40 seconds.** Six uninterrupted optimizer updates matched
four updates plus a two-update resume exactly for model tensors, AdamW state,
scheduler, metrics and sampled RNG values. The tests also covered a wall stop
inside dev, an update ceiling across attempts, signal handling, preserving a
prior checkpoint after failure, config and manifest mismatches, checkpoint
tampering, and local Kaggle staging and data isolation. Rechecking the saved
historical v0 hash lists found zero mismatches across 800 WAVs.

The isolated Torch environment lacks pandas for legacy harness tests, so the
completed-tree full regression used the project `.venv` instead.

```powershell
$env:NUMBA_CACHE_DIR = Join-Path (Resolve-Path -LiteralPath 'runs/vocoroute-phase-n2k') 'numba-cache'
& '.venv\Scripts\python.exe' -B -m pytest -ra -p no:cacheprovider --basetemp runs/vocoroute-phase-n2k/pytest-full-20261124
& '.venv\Scripts\python.exe' -B -m pytest -ra -p no:cacheprovider --basetemp runs/vocoroute-phase-n2k/pytest-relocation-20261124 tests/test_phase_a_run.py::TestManifestPathsSurviveRelocation::test_a_renamed_directory_still_resolves
```

The full run reported **1,068 passed, 5 failed, 12 skipped and 1 xfailed** in
21 minutes 26 seconds. Four failures matched the N2A.1 baseline: the two
notebook CRLF/LF comparisons and the two Windows junction permission failures.
The fifth was `WinError 5` during `Path.rename` in the unchanged historical
relocation fixture. It passed in isolation on the final tree and in a scratch
archive of `c9a62e2` containing `src`, `tests`, `pyproject.toml` and
`data/measurements`. Nine skips were Windows link permissions and three were
the N1, N2A and N2K modules in the Torch-free project environment. The xfail
was the existing INV-17 ablation.

```powershell
& '.venv\Scripts\python.exe' -B -m ruff check --no-cache src tests
& '.venv\Scripts\python.exe' -B -m ruff check --no-cache . --output-format concise
```

Source and test lint passed. Repository-wide lint reported the 11 historical
findings in `notebooks/kaggle_v0.ipynb`.

## Proposed CPU-only VCTK acquisition

The Edinburgh DataShare record lists `VCTK-Corpus-0.92.zip` at **10.94 GB**
and MD5 **`8a6ba2946b36fcbef0212cad601f4bfa`**. It lists CC BY 4.0 and
separate `README.txt` and `license_text.txt` members. The official archive
has not been downloaded or extracted in this phase. [Edinburgh DataShare record](https://datashare.ed.ac.uk/items/30e7453c-9ea8-48b4-8e18-f96d0dc62928/full)

Kaggle documents up to **20 GB of saved notebook output** in
`/kaggle/working` and separate nonpersistent scratch, with no fixed scratch
capacity in the cited notebook specification. It lists a **12-hour** CPU/GPU
session limit. Its dataset documentation lists **200 GB per dataset**, **200 GB
aggregate private storage**, and **50 top-level files**. Verify actual account
space, scratch free bytes, mount layout and session limits before acquisition.
[Kaggle notebook storage](https://www.kaggle.com/docs/notebooks)
· [Kaggle dataset limits](https://www.kaggle.com/docs/datasets)

| Component | Planned space |
|---|---:|
| Archive in nonpersistent CPU scratch | 10.94 GB, official rounded size |
| Bounded pilot w/ 50 speakers × 10 clips × at most 12 s | Up to 6,000 s total |
| Native mono 48 kHz PCM16-equivalent | 576 MB before headers. Measure actual selected native FLAC bytes from ZIP metadata. |
| Prepared 22.05 kHz mono PCM16 | 265 MB before headers |
| Train/dev vs eval share, 35/5/10 speakers | About 673 MB vs 169 MB combined PCM-equivalent, before receipts |
| Proposed admission thresholds | At least 15 GB free scratch and 2 GB free for the selected persistent outputs, subject to the measured member census |

These are decimal estimates for a provisional cohort. They are not measured
sizes of unextracted FLACs. Apply the estimated free-space thresholds before
downloading. Once the archive is verified, census the ZIP directory and
calculate the output envelope before selection or extraction. On shared
volumes verify the combined peak, including staging and metadata. Keep the
original archive out of persistent training inputs.

1. After explicit approval, use a separate CPU notebook with Internet enabled
   for acquisition only. Check `df -B1` on the actual scratch and output
   mounts, verify the official download target on the cited record, and
   record archive size, URI and release ID. Record exact transfer size if
   available. Stream the archive to scratch with restart support. Run
   `md5sum` and compare the approved value **before opening the ZIP**. Record
   SHA-256 for future checks. Stop on checksum or space mismatch, or if the
   received byte count differs from a verified exact transfer size.
2. Preserve the official README, license, attribution, speaker metadata and
   selected transcripts. Check the official README MD5
   `31b61beec43565e27237dbb738566a7a` and license MD5
   `2946b37a07baeba80cea628909e28cae` against the separately listed files.
   Census ZIP paths and expanded sizes before selecting. Reject unsafe paths,
   links, duplicate names and byte-limit violations. Extract only approved
   members with per-member CRC and SHA-256.
   Use one recorded microphone profile and audited source eligibility. The
   existing VocoTrace intake and pinned 48→22.05 kHz resampling evidence
   provide patterns. Verify the actual CPU image and reference vectors before
   using that recipe on Kaggle. [VocoTrace ingestion profile](vocotrace_phase4e_ingestion.md)
3. Check overlap against historical eval sources and retain measured
   exclusions. Select speakers and clips deterministically after admission.
   Approve and freeze a new generator-specific speaker split; the example
   allocation is 35 train, 5 dev, 10 eval with no speaker shared. Record
   original source and prepared hashes, geometry and the single-step
   resampling receipt. Persist the train/dev and eval audio as separate
   access-controlled outputs. The training manifest carries frozen eval
   metadata, and only the train/dev material and approved code/resume inputs
   enter the GPU notebook. Publication and access settings need separate
   approval.

A separately reviewed fallback is a revision-pinned Hugging Face redistribution
that claims VCTK 0.92 provenance. The cited candidate adds normalized text and
documents missing speakers or takes. Check license and attribution, artifact
hashes, source IDs, microphone, waveform format, preprocessing and selected
record equivalence before proposing a distinct frozen cohort. Its shard hashes
cannot prove the Edinburgh ZIP's MD5. [Candidate dataset card](https://huggingface.co/datasets/thelarryproject/vctk)
