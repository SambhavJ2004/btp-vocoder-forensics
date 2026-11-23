# VocoRoute-P Phase N2A.1 training verification

The existing CPU base runner executed eight AdamW updates on one fixed,
1,025-sample synthetic tone plus transient. A separate synthetic perturbation
provided dev input. The test exercised the `--train` entry point, optimizer,
scheduler, dev selection, checkpoint publication, and pinned reload. It used
the default seed, learning rate, and phase-disabled reconstruction objective.
The decoded tensors were supplied by a test double in the CPU Torch process.
A separate SoundFile test exercised the real sanctioned reader on an encoded
synthetic PCM signal. All fixtures and checkpoint files lived in scratch.

## Review and corrections

- Generator header validation used `rate` while the sanctioned probe returns
  `sample_rate`. Prior tests reproduced the incorrect key in their stub. The
  corrected check verifies mono, rate, and exact frame count.
- Training previously decoded from a path and reopened it to check its hash.
  A changing file could produce samples from different bytes. The generator
  path now resolves within the approved root, reads a byte snapshot, and hashes
  it before probing and decoding it. Both decoder operations receive that
  snapshot. Native and prepared files are verified independently. Synthetic
  tamper tests cover both files, a replacement during probing, and refusal to
  decode changed bytes. The sanctioned reader still accepts existing paths.
- The checkpoint loader already verified the external receipt, the weight
  digest, config, model/frontend source, and tensor shape, dtype, and finiteness.
  Restricted loader rejections now use a clear validation error. A test
  verifies that a correctly pinned payload with an unsupported object is
  refused by weights-only loading.

Waveform L1 and equal-resolution spectral averages flow through iSTFT to the
existing 566,144 backbone parameters. The optional phase distance uses
reference-energy normalization per utterance and clamps weak predicted
phasors. Silence and a one-frame input produce finite loss and gradients.
The phase-off test verifies that no phase-pair calculation is reached. A live
reconstruction test compares the versioned frontend settings with the
implemented STFT and mel filterbank. The smoke decoded train and dev only.
The manifest records native identities and a resampling receipt pin. Authenticity,
cross-corpus overlap, and receipt replay still belong to intake review.

## Synthetic results

| Check | Result |
|---|---:|
| AdamW updates | 8 |
| Fixed-example total reconstruction loss, initial → final | 14.0956669 → 6.6940565 |
| Fixed-example waveform L1, initial → final | 0.5700602 → 0.1809761 |
| Largest absolute gradient observed | 2.0036502 |
| Largest absolute parameter change | 0.00150812 |
| Dev checkpoint selected | Epoch 8 |
| Maximum absolute waveform difference before and after reload | 0.0 |

Every parameter had a finite gradient at every update. The measured dev
total decreased from 11.5446167 after the first update to 6.3578577 after the
eighth. These results establish optimization mechanics and exact checkpoint
reload on the synthetic example. Intelligibility and speech quality remain
unmeasured.

## Proposed isolated environment

Observed Windows x64 Python 3.13.9 has Torch and TorchAudio `2.9.0+cpu` in the
system interpreter. It lacks SoundFile. The project `.venv` has SoundFile
0.14.0 with libsndfile 1.2.2 and lacks Torch. CUDA is unavailable in the tested interpreter, and
`nvidia-smi` was absent from PATH. A separate venv avoids mixing the installed
package sets. The official CPU wheels pair Torch and TorchAudio at 2.9.0, and
the SoundFile 0.14.0 Windows x64 wheel ships libsndfile.
[PyTorch wheels](https://pytorch.org/get-started/previous-versions/)
· [TorchAudio compatibility](https://docs.pytorch.org/audio/stable/installation.html)
· [SoundFile release](https://pypi.org/project/soundfile/0.14.0/)

Proposed Windows CPU setup after separate dependency approval. This combined
package set has yet to be installed and verified in one interpreter.

```powershell
& 'C:\Python313\python.exe' -I -m venv 'runs/environments/vocoroute-cpu-313'
$vocoCpuPython = 'runs/environments/vocoroute-cpu-313/Scripts/python.exe'
& $vocoCpuPython -I -m pip install --only-binary=:all: --index-url https://download.pytorch.org/whl/cpu 'torch==2.9.0' 'torchaudio==2.9.0'
& $vocoCpuPython -I -m pip install --only-binary=:all: --index-url https://pypi.org/simple `
  'numpy==2.2.6' 'scipy==1.16.3' 'pandas==2.3.0' 'soundfile==0.14.0' `
  'librosa==1.0.0' 'pyloudnorm==0.2.0' 'PyYAML==6.0.3' 'tqdm==4.67.1' `
  'numba==0.63.1' 'soxr==1.1.0' 'cffi==2.0.0' 'pytest==9.0.1'
& $vocoCpuPython -I -m pip install --no-deps -e .
& $vocoCpuPython -I -m pip check
& $vocoCpuPython -I -B -c "import torch, torchaudio, soundfile; from data.audio_io import read_native; print(torch.__version__, torchaudio.__version__, soundfile.__version__, soundfile.__libsndfile_version__, torch.cuda.is_available())"
& $vocoCpuPython -I -B -m vocoroute.train_base --dry-run
& $vocoCpuPython -I -m pip freeze --all
```

Capture the resolved versions and wheel hashes before training. The existing
versioned runner is CPU-only. GPU training would need a separately reviewed
device configuration, compatible NVIDIA hardware and driver, paired CUDA
wheels, and a verified deterministic forward/backward test for STFT and iSTFT.
Measure peak VRAM and wall time at the approved utterance length and batch size
before sizing a run. Approve the host, storage, compute ceiling, and package
sources first. A suitable wheel family depends on the target device and
driver. [PyTorch wheel options](https://pytorch.org/get-started/previous-versions/)
· [NVIDIA CUDA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/latest/minor-version-compatibility.html)

For example, the official 2.9.0 `cu126` Torch/TorchAudio pair is a candidate
for a compatible approved host. Create a separate environment for it and use
`https://download.pytorch.org/whl/cu126` in place of the CPU wheel index.
Verify the actual device and wheel architecture list before running a
synthetic CUDA acceptance test. [PyTorch wheel options](https://pytorch.org/get-started/previous-versions/)

## Reproduction commands

The smoke was executed once for eight updates. The other Torch tests were
rerun after correcting two tamper-test fixtures that omitted headers for
unrelated rows. The completed-tree project regression used the Torch-free
project venv. Each pytest invocation requires a fresh scratch path.

```powershell
& 'C:\Python313\python.exe' -B -m pytest -s -o pythonpath=src -p no:cacheprovider --basetemp runs/vocoroute-phase-n2a1/pytest-optimizer-20261123 tests/test_vocoroute_n2a.py::test_bounded_synthetic_optimizer_smoke
& 'C:\Python313\python.exe' -B -m pytest -ra -o pythonpath=src -p no:cacheprovider --basetemp runs/vocoroute-phase-n2a1/pytest-focused2-20261123 tests/test_vocoroute_n1.py tests/test_vocoroute_n2a.py -k 'not bounded_synthetic_optimizer_smoke'
& '.venv\Scripts\python.exe' -B -m pytest -p no:cacheprovider --basetemp runs/vocoroute-phase-n2a1/pytest-audio-20261123 tests/test_audio_io_stream.py
$env:NUMBA_CACHE_DIR = (Join-Path (Resolve-Path -LiteralPath 'runs/vocoroute-phase-n2a1') 'numba-cache')
& '.venv\Scripts\python.exe' -B -m pytest -ra -p no:cacheprovider --basetemp runs/vocoroute-phase-n2a1/pytest-final-tree2-20261123
& '.venv\Scripts\python.exe' -B -m ruff check --no-cache src tests
& '.venv\Scripts\python.exe' -B -m ruff check --no-cache . --output-format concise
```

The final-tree project suite reported **1,069 passed, 4 failed, 11 skipped,
1 xfailed**. The four failures matched the N2A baseline: the two generated
notebook CRLF/LF comparisons and the two Windows junction permit tests
(`WinError 5`). Nine skips were filesystem link permissions. The remaining
two were N1 and N2A collection skips in the Torch-free venv. The xfail was the
existing INV-17 ablation. The current Torch run passed **81 cases** across
the smoke and focused commands. The real SoundFile parity test passed in
the project venv. Ruff passed on `src tests` and reported the same 11
generated-notebook findings repository-wide.

All 800 WAVs checked against the historical v0 receipt matched their recorded
sizes and SHA-256 hashes. Sampled historical manifest, provenance, and
invariant-document hashes matched the earlier saved verification.

## N2B readiness checklist

- [ ] Approve the corpus release, license, native checksums, and any resampling receipts.
- [ ] Audit overlap with historical evaluations and freeze generator-specific speaker-disjoint groups.
- [ ] Approve the manifest digest and verify train/dev file hashes, headers, and alignment.
- [ ] Keep held-out eval audio unavailable to fitting and checkpoint selection.
- [ ] Verify an isolated runtime, exact dependency resolution, and the approved CPU or GPU device.
- [ ] Record code identity, config, seed, manifest, runtime, metrics, and checkpoint hashes.
- [ ] Approve a fixed cohort, update and wall-time ceilings, and a stop rule for nonfinite values.
- [ ] Predeclare an initialization baseline, paired dev metrics, and listening checks before viewing eval.
