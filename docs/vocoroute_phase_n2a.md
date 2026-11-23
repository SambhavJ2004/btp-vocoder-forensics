# VocoRoute-P Phase N2A base training protocol

N2A adds a CPU entry point for the existing 566,144-parameter spectral
backbone. The checks use synthetic waveforms, forward and backward passes,
and checkpoint serialization of initialized weights. They measure numerical
and data contracts. Speech quality, learning behavior, and real-corpus runtime
remain open. The N1 refinement expert and the historical forensic pipeline
are outside this protocol.

## Tensor and loss flow

Use one full-length, unpadded mono utterance at a time. A snapshot of its
prepared 22,050 Hz samples supplies both the target `[1,L]` and the input to
the N1 frontend. The frontend uses 80 Slaney area-normalized mel power bands
over 0–11,025 Hz, a periodic Hann window of length 1,024, hop 256, centered
constant-zero padding, an unnormalized one-sided FFT, and natural log after a
`1e-5` floor. The mel tensor `[1,80,T]` enters the existing backbone. Its
complex `[1,513,T]` output passes through the matching iSTFT with length `L`.

All terms use the **synthesized waveform** and its aligned target:

- **Waveform.** Mean samplewise absolute error.
- **Spectral.** Equal averages of linear-magnitude L1 and log-magnitude L1
  over `(FFT, hop)` of `(256,64)`, `(512,128)`, and `(1024,256)`. Each uses its
  own periodic Hann and the same centered, zero-padded analysis conventions.
  Log magnitudes use an absolute `1e-4` floor in unnormalized FFT units.
- **Phase.** Optional circular distance between adjacent-bin phase
  differences, separately across time and frequency at 1,024/256. Adjacent
  conjugate phasor products avoid angle wrapping. Half the squared chord
  distance is `1 - cos(delta)` for unit phasors. Reference endpoints and weak
  reference pairs are excluded. The weight for an eligible pair is the
  smaller of its two reference bin energies. Normalization is per utterance
  with a floored denominator. Predicted phasors use a floored normalization
  as well, so even zero predicted energy has a finite distance. An empty time
  axis of phase pairs contributes zero. Setting `phase_enabled=false` skips
  the phase calculation and zeros both phase terms.

The config independently weights the five terms. The defaults enable the
three reconstruction terms at weight 1 and disable phase. The proposed phase
ablation uses the same reconstruction weights and enables the two phase terms
at 0.05 each. These are initial experimental settings.

## Frozen generator data

An approved `vocoroute-generator-manifest-v1` is a standalone JSON object
with `schema`, `dataset_id`, `source_kind: "native_generator"`,
`target_rate: 22050`, and `rows`. Supply its exact byte SHA-256 as
`manifest_sha256` in the training config. Every row includes:

| Fields | Meaning |
|---|---|
| `utterance_id`, `speaker_id`, `source_group`, `role` | Stable release identities, original recording group, and approved `train`, `dev`, or `eval` assignment. Alternate takes must share the recording group where appropriate. |
| `native_path`, `native_sha256`, `native_rate`, `native_samples` | Original mono recording identity and geometry. |
| `prepared_path`, `prepared_sha256`, `prepared_samples` | The exact mono waveform consumed at 22,050 Hz. |
| `resample_steps`, `resampling_profile`, `resampling_receipt_sha256` | `0`, `identity`, and `null` for native-rate byte-identical data. Otherwise `1`, the approved versioned conversion profile, and its receipt SHA-256. Prepared length uses `ceil(native_samples * 22050 / native_rate)`. |

Rows are frozen after loading. Duplicate paths, recording bytes and utterance
IDs are rejected. Source groups stay in one role and a multi-speaker corpus
requires speaker-disjoint roles. Train is iterated in stable ID order before a
deterministic epoch-specific shuffle. Dev stays in stable ID order. The runner
validates eval's assignments from metadata and leaves its audio unopened.
Dry-run with data verifies native and prepared file digests for train and dev.
The future training command checks their headers using the sanctioned reader,
then rechecks each prepared file against its pin when decoding. Missing data
fails closed. Data preparation and independent overlap review belong to intake,
including checking that generator sources do not overlap historical evaluation.
N2A keeps the resampling receipt reference and does not replay conversion.

## Entry point and checkpoint

`python -m vocoroute.train_base --print-default-config` prints the full
`vocoroute-base-training-v1` JSON. Fixed model, frontend, CPU float32,
batch size one, and role settings are validated on load. The config includes
the initialization and sampler seed, epochs, loss weights, AdamW, and an
exponential learning rate scheduler applied per epoch. The default seed is
20260907, learning rate `2e-4`, weight decay `0.01`, and scheduler factor `0.98`.

`--dry-run` validates the configuration and reports the configuration SHA-256
and model parameter count. With `--config`, `--manifest`, and `--data-root`
it also performs the pinned metadata and train/dev byte checks. Dry-run
constructs no optimizer and writes no checkpoint. The implementation of
`--train` requires the explicit approved data arguments and a fresh `--out-dir`
outside the data root. That branch was not exercised with training data or
optimizer steps in N2A.

A completed future run chooses the first epoch with the lowest dev total
and publishes `base.pt` followed by `checkpoint.json`. The checkpoint receipt
records canonical config and manifest digests, dataset ID, source hashes,
Python and package versions, device and thread settings, epoch losses, and
the weight digest. `--validate-checkpoint DIRECTORY --config CONFIG
--receipt-sha256 SHA256` verifies the external receipt pin, inner weight pin,
config and model/frontend compatibility, and calls the restricted weights-only
loader. State keys, shapes, dtypes, and finite values are verified. The schema
supports model verification and inference. Optimizer, scheduler, and RNG
resume state are outside v1. Receipts identify this protocol as development
evidence. Citable runs still require a separately approved record workflow.

## Verification

CPU Torch and torchaudio `2.9.0+cpu` passed 77 focused cases (44 N1 and 33
N2A). These cover exact signal lengths and alignment, loss and gradient
checks, disabled phase computation, role and digest checks, missing inputs,
and checkpoint round-trips and tampering. A constructor probe verified AdamW
and ExponentialLR at the declared initial rate with zero optimizer state
entries. The system Torch runtime lacks SoundFile. The project virtual
environment has SoundFile but lacks Torch. Real-audio I/O was exercised only
through a mocked sanctioned reader with in-memory waveform fixtures.

The full project-environment regression run started before the N2A module was
added. It finished with 1,068 passed, 4 failed, 10 skipped, and 1 xfailed.
The failures were the same two notebook CRLF/LF byte comparisons and two
Windows junction permissions recorded in N1. The skips match the baseline
link privileges and the N1 Torch skip in this environment. The expected
failure is the dormant INV-17 evaluation ablation. Ruff passed on `src tests`
and reported 11 existing generated-notebook findings repository-wide.

Executed from the repository root. Use fresh temporary paths for reruns.

```powershell
& 'C:\Python313\python.exe' -B -m pytest -o pythonpath=src -p no:cacheprovider --basetemp runs/vocoroute-phase-n2a/pytest-torch3-20261123 tests/test_vocoroute_n1.py tests/test_vocoroute_n2a.py
$env:NUMBA_CACHE_DIR = (Join-Path (Resolve-Path -LiteralPath 'runs/vocoroute-phase-n2a') 'numba-cache')
& '.venv\Scripts\python.exe' -B -m pytest -p no:cacheprovider --basetemp runs/vocoroute-phase-n2a/pytest-full-20261123
& '.venv\Scripts\python.exe' -B -m ruff check --no-cache src tests
& '.venv\Scripts\python.exe' -B -m ruff check --no-cache . --output-format concise
```

## N2B proposal

Approve native sources and an isolated CPU-compatible runtime with both the
Torch and sanctioned audio stacks present. Audit identity overlap and source
groups, freeze a small generator cohort with independent dev and eval speakers,
and approve its exact manifest and resampling receipts. Before a longer run,
use a bounded one-epoch base pilot and confirm aligned resynthesis, finite
losses, checkpoint validation, and dev improvement over initialization. Keep
eval sealed. Compare the phase-on and phase-off configurations at the same
data and update budget with common dev metrics and paired listening. Actual
phase benefit, convenient batching, and throughput remain unmeasured.
