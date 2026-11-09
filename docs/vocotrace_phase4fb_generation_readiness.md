# VocoTrace Phase 4F-B: multi-speaker generation readiness

**Development-only; non-citable. Fixture execution is not real VCTK or neural
generation.** This phase adds a separate ingestion-backed pilot interface. The
legacy LJSpeech driver, preprocessing, invariants, checkpoints and historical
bundles are not modified. Real acquisition and neural execution remain untested.

Starting manually created HEAD: `24da945eef65cb407e29092fb0e15ec1da69c860`.
Date: 2026-11-07, Asia/Calcutta. The existing unstaged `.gitignore` edit is
preserved. No Git writes, real dataset/model-weight downloads, installations,
Kaggle operations, detector training or frozen real-evaluation access occur.

## 1. Architecture and legacy audit

The old `experiments.phase_a_resynthesis.preflight/run_gated` explicitly require
LJSpeech `metadata.csv` and `wavs/`, instantiate `LJSpeech`, and call generic
`run`, which assigns splits over its supplied Corpus. These remain correct for
their supported protocol. Passing a VCTK training subset to generic `run` would
resplit that subset; renaming VCTK files into a fictitious LJSpeech layout would
bypass the wrong boundary. Neither approach is used.

New module: `experiments.multispeaker_generation`. New namespace:
`btpvf multispeaker plan|compatibility|generate|validate`. Existing `ingest`
commands are reused. The adapter calls shared processing, manifest, alignment,
band and calibrated sanity mechanisms directly, without editing their rules.
There is no option to skip gates, lower the twenty-file minimum, resplit a
cohort or overwrite an existing output.

The fixed pilot is **eight speakers x four eligible sources = 32 intake rows**,
with the original seed 20260907 and speaker-level policy: 6/1/1 speakers,
24/4/4 sources. Select twenty **training** sources by round-robin over sorted
training speakers and lexical source IDs. All six training speakers contribute
(4/4/3/3/3/3). Every generated row retains its original source/speaker identity
and `train` membership. The full 32-source split map remains in the immutable
plan. This is not a change to any LJSpeech assignment or a new split ratio.

### Independent gates

| Stage | Admission requirement | Artifact/evidence |
|---|---|---|
| Acquisition | Explicit source kind; real sources require a reviewed, strictly shaped attestation bound to the completed intake and release evidence | Human review and byte-identity references; not an automatic rights/authenticity certificate |
| Ingestion | Existing Phase 4E full intake validation first; source/version/microphone/profile/metadata integrity, frozen groups and unique identities | Twelve existing intake files, copied unchanged into the plan |
| Plan | Exact bounded population; deterministic twenty-source train-only selection; source hashes; fixed frontend/checkpoint specs and implementation hashes | `plan.json`, copied `ingestion/`, checksums |
| Compatibility | One condition per invocation; native float/mono/rate checks; common processing; twenty-file alignment; manifest, band, calibrated sanity and output checks | Eighty paired-tier WAV files and measured receipts, or retained failed output |
| Limited generation | Separately validated compatibility run for the exact same plan, generator, checkpoint assets and environment | Independent rerun of the twenty-source selection, linked to compatibility |
| Artifact validation | Exact recursive inventory, hashes, source/group/label semantics, format/sample counts, source reconstruction, derived-tier reconstruction, alignment and signal-gate recomputation | Validation summary; never silently deletes, ignores or repairs artifacts |

Compatibility is already a bounded empirical generation run when a real vocoder
is used; `generate` is a separately gated rerun, not 20 additional independent
source groups. A compatibility failure never authorizes limited generation.
Failed outputs stay at their fresh requested path with `status=failed` and
available diagnostics. They are not accepted datasets.

## 2. Identity, provenance and held-out isolation

`source-kind=fixture` is explicit and must match the intake. Such plans identify
the dataset as `synthetic_fixture_vctk_layout`, not genuine VCTK. WAV basenames
start with `fixture__`; reference rows retain the shared schema's condition
`real` solely as the comparison anchor, with explicit
`reference_semantics=synthetic_fixture_input_not_genuine`. Every manifest row
includes source kind, dataset/version, original source SHA-256, generator kind
and development/non-citable flags. The copied source IDs are structural fixture
identifiers, not evidence that an actual VCTK speaker was processed.

`fixture_mock` has a distinct specification and identity. Injection is permitted
only for that fixture-only condition. Fixture requests for BigVGAN/Vocos fail
before loading a generator; real intake cannot use a fixture mock. The loaded
adapter specification must equal the frozen plan. Validation derives the allowed
generator label from the condition: neither mocks nor Griffin-Lim can become
neural evidence by resealing a changed label. A Griffin-Lim invocation records
actual DSP, sixty iterations, seed and library versions, with neural inference
false. Neural labels alone are not empirical evidence; this phase runs none.
The mock is explicitly outside the primary ladder (`primary_ladder=false`),
and is not added to the shared research-vocoder registry.

Both plan and result inventories have fixed recursive schemas. Modified
inventories cannot authorize arbitrary new files, including nested files or
unexpected directories. Source/package symlinks, junctions, hardlinks and
escaping paths use the existing rejection mechanisms. HF checkpoint-cache links
are a separate, intentional cache mechanism behind pinned adapter checks.
Output directories must be fresh and disjoint from source, evidence, plan and
compatibility directories. Manifest paths must match exact relative packaged
paths; the adapter does not invoke the legacy absolute-path recovery behavior.

Source identity means the declared speaker/utterance metadata, file hashes,
dataset version, microphone and frozen split agree. It is not biometric human
speaker verification or proof of consent. The Phase 4E authenticity, rights and
scientific-admission flags remain false/incomplete. No artifact is promoted to
research-record or publication status here.

### Full intake first, scoped real-source revalidation thereafter

The original `verify_ingestion` decodes the complete supplied cohort. It remains
unchanged and is used for the full initial intake gate and for synthetic fixture
verification. The future real-generation path does **not** call it again on a
frozen population. Instead it requires the external acquisition review to bind
the digest of the completed intake checksums, source manifest, source inventory
and copied release evidence. It verifies that exact metadata schema, original
intake implementation/profile and passed receipt, checks original speaker and
release metadata, and mathematically checks the saved speaker map using all
32 metadata rows. It then rehashes and remeasures only the twenty selected
training sources and their transcripts using the existing source inspector.
Generation and output reconstruction read only those training recordings.

The verification scope is recorded in the plan. This is not a claim that
unselected real waveform bytes were revalidated later. The output receipt says
`selected_source_files_unchanged`; it does not claim a fresh full-corpus hash
pass. Original held-out data must remain immutable under the source custodian.
The human-bound intake digest is essential: resealing its inventory alone does
not update the external attestation. This attestation is a local trust boundary,
not a digital signature or independent proof of every upstream right.

## 3. Processing and output validation

The new native-source path explicitly calls the Phase 4E pinned 48->22.05-kHz
soxr HQ profile. Vocos uses its existing separate 48->24-kHz Kaiser input path
and one 24->22.05-kHz output conversion. No 48->22.05->24 chain is introduced.
Generator input contains waveform samples and sample rate only; speaker/source
IDs, transcript, split and condition labels are not inference features.

Native outputs must be finite, nonempty, mono floating-point at the declared
native rate. Raw output samples/hash, generator input samples/rate, pre-alignment
archive length, exact adjustment, reference trim span and final lengths are
recorded. The shared output processor remains responsible for bounded geometry
alignment, real-derived trimming, analysis-band loudness and full-band archive
processing. Excessive drift, nonzero alignment lag or clipping fails; there is
no per-file gain exception, time-shift repair, denoising or condition-specific
trimming. No automatic replacement/backfill after a failed source occurs.

The existing twenty-file zero-lag envelope gate and conditional waveform gate
run before accepting synthetic files. Both pre-encoding and decoded PCM16
alignment reports are retained. `validate_manifest(require_zeroshot=True)`,
`band_gate`, `run_all(split=None)` and unchanged silence calibration apply to the
training comparison set. The floor, one-sided verdict policy and existing
design exceptions are preserved; this task does not resolve their previously
reported scientific limitations.

The new interface derives 16-kHz audio from the **written/read-back PCM16
archive**, without renormalization, so validation can reconstruct its derivation.
The legacy LJSpeech emitter is unchanged (it derives from its archive float
buffer). No bitwise equivalence of those two emission implementations is
claimed; the new path has no historical VCTK outputs to rewrite. Shared
post-trim processing and all invariant targets remain unchanged.

Validation checks actual mono/PCM16 WAV headers and exact samples/durations,
decoded loudness/peaks, original hashes, real-reference reconstruction and the
common archive->16-kHz derivation. Quantized reconstruction allows at most one
PCM16 LSB plus 1e-7 for float arithmetic; independent recorded byte and decoded
SHA-256 checks remain exact. This quantization comparison does not relax any
alignment, loudness, clipping, format or signal gate. Neural raw outputs are
not re-synthesized during validation; their recorded hash/geometry is producer
provenance, not an independent reconstruction of a model's hidden output.
The derived reconstruction supplies the recorded archive full-band loudness to
the existing drift guard, just as emission does; it does not bypass that guard
by omitting the parent measurement. An actual RIFF/WAVE signature is required,
so a resealed FLAC file renamed `.wav` is rejected.

Copied plans/intakes and manifests are portable; use explicit source/evidence
root overrides after relocation. A moved generation prerequisite can be supplied
with `validate --compatibility-dir`. Validation needs the same implementation
bytes, native-library profile and original selected sources. It does not load
neural weights. Hashes attest local consistency, not a cryptographic signature
against a party capable of forging an entire evidence chain.

## 4. Neural runtime requirements and proposed setup

No neural runtime or model weights were installed or downloaded. Current CPU
environment: Python **3.13.9**; torch, torchaudio, vocos, huggingface_hub and
matplotlib are absent in this phase's fresh offline module check
(`runs/vocotrace-phase4fb-audit-20261107/neural_runtime_presence.json`). Package availability
and successful inference are different gates.

The [BigVGAN authors](https://raw.githubusercontent.com/NVIDIA/BigVGAN/main/README.md)
report testing Python **3.10**, PyTorch **2.3.1**, CUDA **11.8/12.1**. This is
a tested upstream combination, not a universal minimum or proof of compatibility
with this project's newer, fixed NumPy profile. The adapter disables the custom
CUDA kernel; NVCC/ninja compilation is unnecessary for that path. Upstream's
[full requirements](https://raw.githubusercontent.com/NVIDIA/BigVGAN/main/requirements.txt)
also contain training/evaluation packages that the narrow inference adapter
does not import.
The pinned Hugging Face model revision's Python text could not be retrieved by
the web reader in this phase. The upstream links describe upstream requirements,
not a newly verified dependency lock for that cache snapshot. Inspect and record
the actual pinned cached sources during authorized runtime setup.

[Vocos 0.1.0 requirements](https://raw.githubusercontent.com/gemelo-ai/vocos/v0.1.0/requirements.txt)
name torch, torchaudio, numpy, scipy, einops, PyYAML, huggingface_hub and
**encodec==0.1.1**, without a torch/torchaudio version pin. Those two packages
must have matching versions/builds and a working binary ABI. Encodec is a
runtime package dependency; this mel model must not initialize/download Encodec
weights. The offline loader checks the expected mel extractor/backbone/ISTFT
component classes before loading Vocos.

### Exact candidate environment, not an empirically verified neural lock

Use separate environments per neural vocoder. Keep the current analysis
environment and historical bundles untouched. This **uninstalled candidate**
retains the exact data-profile packages and proposes a newer matched torch pair
for Python 3.13.9; install/import/NumPy-ABI/model tests must still establish it.
Do not describe the upstream 3.10/2.3.1 test as validation of this candidate.

| Packages | Candidate versions | Scope |
|---|---|---|
| Python | 3.13.9 | Matches this phase's CPU interpreter |
| numpy / scipy / librosa / soxr / soundfile | 2.5.3 / 1.18.1 / 1.0.0 / 1.1.0 / 0.14.0 | Existing enforced intake profile |
| Native libsoxr / libsndfile | `0.1.3-14-ga66f3ee` / 1.2.2 | Runtime check must match; do not silently substitute a system library |
| pandas / pyloudnorm / PyYAML / tqdm | 3.0.5 / 0.2.0 / 6.0.3 / 4.70.0 | Existing processing/report dependencies |
| torch / torchaudio | 2.9.1 / 2.9.1, same CPU/CUDA build | Proposed; torchaudio required for Vocos; not executed here |
| huggingface_hub | 0.36.0 | Proposed cache/API version for both neural adapters |
| matplotlib | 3.10.7 | BigVGAN upstream import dependency |
| vocos / encodec / einops | 0.1.0 / 0.1.1 / 0.8.1 | Vocos-only environment |

[PyTorch's official version instructions](https://pytorch.org/get-started/previous-versions/)
list matched 2.9.1 torch/torchaudio wheels for Linux/Windows CPU and CUDA 12.6,
among other builds. The package releases
[Hub 0.36.0](https://pypi.org/project/huggingface-hub/0.36.0/),
[matplotlib 3.10.7](https://pypi.org/project/matplotlib/3.10.7/) and
[einops 0.8.1](https://pypi.org/project/einops/0.8.1/) exist; existence is not a
resolved or tested combined environment. Transitive dependencies/build artifacts
are not yet locked. Record `pip list --format=json`, native-library identities,
wheel/build information and the successful runtime checks before compatibility.

Future manual setup example, **not executed** (choose CPU or the approved CUDA
build; do not install over the project environment):

```powershell
py -3.13 -m venv '<outside-repository-vocoder-venv>'
$runtimePython = '<outside-repository-vocoder-venv>/Scripts/python.exe'
& $runtimePython -m pip install torch==2.9.1 torchaudio==2.9.1 `
  --index-url https://download.pytorch.org/whl/cu126
& $runtimePython -m pip install numpy==2.5.3 scipy==1.18.1 librosa==1.0.0 `
  soxr==1.1.0 soundfile==0.14.0 pandas==3.0.5 pyloudnorm==0.2.0 `
  PyYAML==6.0.3 tqdm==4.70.0 huggingface_hub==0.36.0
# BigVGAN environment only:
& $runtimePython -m pip install matplotlib==3.10.7
# In a SEPARATE Vocos environment instead:
& $runtimePython -m pip install vocos==0.1.0 encodec==0.1.1 einops==0.8.1
& $runtimePython -m pip install -e . --no-deps
& $runtimePython -m pip check
& $runtimePython -c "import torch,numpy as np; print(torch.__version__); print(torch.from_numpy(np.zeros(8,dtype=np.float32)).numpy())"
& $runtimePython -c "from data.source_ingestion import resampling_record; print(resampling_record())"
# Vocos environment also needs an actual torch/torchaudio import and resampling smoke check.
& $runtimePython -c "import torch,torchaudio; print(torch.__version__,torchaudio.__version__); print(torchaudio.functional.resample(torch.zeros(4800),48000,24000).shape)"
```

Use `--index-url https://download.pytorch.org/whl/cpu` for the CPU alternative.
An import/ABI/version/profile failure blocks execution; it is not permission to
change frozen pins. A different core/native profile needs separate review and
test vectors. Future setup must also install the repository's ordinary core
dependencies (the candidate commands list those used here, not a complete
transitive lock) and its test extras before running the requested tests.

Practical BigVGAN execution should use approved GPU compute; no minimum VRAM,
speed or Windows neural success is measured here. Vocos may be run on CPU or
GPU by its API, but neither is validated here. Windows has known filesystem-link
test limitations; plain torch avoids the custom CUDA compiler requirement but
does not establish an end-to-end Windows pass. `--device cuda` fails if CUDA is
unavailable; there is no silent requested-GPU fallback.

### Checkpoints and offline-only loading

BigVGAN: `nvidia/bigvgan_v2_22khz_80band_fmax8k_256x`, revision
`a9d8b711c344f3c49cc1a05d0c8b6741cf733aa8`, weight SHA-256
`5011c15a31a8d04fb511c1813ef4b0c2fcc0c48043f3952ce06a3b65201a5aaa`.
All pinned upstream code/config/weight assets must already be cached together;
the legacy adapter still verifies the weight digest and frontend configuration.
Vocos: `charactr/vocos-mel-24khz`, revision
`0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21`, cached `config.yaml` and
`pytorch_model.bin`. Actual cached asset sizes and SHA-256 values are recorded
before execution. A revision declaration is not a newly measured weight hash.

Both neural paths require `HF_HUB_OFFLINE=1` **before Hub import**, check the Hub's
effective offline constant and use `try_to_load_from_cache` before calling the
existing adapters. Missing cache files cause failure without a download. No
model download command is executed or provided as an implicit setup step;
authorized cache provisioning is separate. Reproducibility remains dependent
on actual model/runtime/device/kernel behavior. Seed and cuDNN flags are not a
cross-device bitwise guarantee; compare the two bounded runs' output hashes and
retain differences rather than hiding them.

BigVGAN and Griffin-Lim use 22,050-Hz, FFT/window 1024, hop 256, 80 mels and
fmax 8 kHz; BigVGAN uses its upstream extractor. Vocos uses its checked upstream
24-kHz, 100-mel, fmax-12-kHz extractor and existing pinned Kaiser resampler
(width 64, rolloff 0.9999, beta 14.769656459379492). No speaker embedding is
passed. Unseen-speaker quality and processing neutrality must be measured, not
assumed from those signatures.

## 5. Exact future real-pilot commands

These are **unexecuted** commands for an authorized real cohort. First complete
the acquisition/rights review in Phase 4F, preserve original source bytes and
verify the published archive checksum. No real source is obtained in 4F-B.

```powershell
& '.venv/Scripts/python.exe' -B -m experiments.cli ingest vctk `
  --source-root '<authorized-original-vctk-root>' `
  --evidence-dir '<original-release-evidence>' --source-kind local_dataset `
  --speakers 8 --utterances-per-speaker 4 --out-dir '<fresh-intake>'
if ($LASTEXITCODE -ne 0) { throw 'Ingestion blocked; preserve diagnostics and stop.' }
& '.venv/Scripts/python.exe' -B -m experiments.cli ingest verify --ingestion-dir '<fresh-intake>'
if ($LASTEXITCODE -ne 0) { throw 'Ingestion validation failed; stop.' }
```

This full initial source-eligibility verification precedes model experimentation.
Freeze its 32-source map. A human reviewer then supplies an acquisition JSON
with **exactly** the following keys. Hash fields must be computed from the
verified intake, and approval/reviewer/date must reflect an actual review.
Do not copy expected values and call them locally verified measurements.

```json
{
  "dataset": "vctk_0.92_mic2",
  "version": "0.92",
  "microphone": "mic2",
  "license": "CC-BY-4.0",
  "source_url": "https://datashare.ed.ac.uk/handle/10283/3443",
  "archive_sha256_reviewed": "f96258be9fdc2cbff6559541aae7ea4f59df3fcaf5cf963aae5ca647357e359c",
  "local_processing_authorized": false,
  "reviewed_by": "<actual human reviewer>",
  "reviewed_at": "<actual ISO datetime including timezone>",
  "source_inventory_sha256": "<sha256 of intake/source_inventory.json>",
  "source_manifest_sha256": "<sha256 of intake/source_manifest.csv>",
  "ingestion_checksums_sha256": "<sha256 of intake/checksums.json>",
  "release_readme_sha256": "<sha256 of intake/release_readme.txt>",
  "release_license_sha256": "<sha256 of intake/release_license.txt>"
}
```

The template deliberately does not grant approval. Real access is blocked until
the completed review explicitly authorizes local processing. This is neither
an upload permission nor a software-issued legal clearance.

In each approved per-vocoder environment, with the exact validated code bytes,
matching core profile and already provisioned checkpoint cache:

```powershell
$env:HF_HUB_OFFLINE = '1'
$runtimePython = '<approved-condition-environment>/Scripts/python.exe'
$condition = 'bigvgan_112m'  # Repeat separately for vocos and griffin_lim.
& $runtimePython -B -m experiments.cli multispeaker plan `
  --ingestion-dir '<verified-intake>' --source-kind local_dataset `
  --acquisition-receipt '<completed-human-review.json>' --condition $condition `
  --source-root '<immutable-original-root>' --evidence-dir '<original-release-evidence>' `
  --out-dir '<fresh-condition-plan>'
if ($LASTEXITCODE -ne 0) { throw 'Acquisition/plan gate blocked; stop.' }
& $runtimePython -B -m experiments.cli multispeaker compatibility `
  --plan-dir '<fresh-condition-plan>' --out-dir '<fresh-condition-compatibility>' --device cuda
if ($LASTEXITCODE -ne 0) { throw 'Compatibility gate blocked; stop.' }
& $runtimePython -B -m experiments.cli multispeaker validate `
  --directory '<fresh-condition-compatibility>'
if ($LASTEXITCODE -ne 0) { throw 'Compatibility artifact validation failed; stop.' }
& $runtimePython -B -m experiments.cli multispeaker generate `
  --plan-dir '<fresh-condition-plan>' --compatibility-dir '<fresh-condition-compatibility>' `
  --out-dir '<fresh-condition-generation>' --device cuda
if ($LASTEXITCODE -ne 0) { throw 'Limited generation failed; stop.' }
& $runtimePython -B -m experiments.cli multispeaker validate `
  --directory '<fresh-condition-generation>'
if ($LASTEXITCODE -ne 0) { throw 'Generated artifact validation failed; stop.' }
```

Use `--device cpu` for Griffin-Lim and for any deliberately approved CPU neural
diagnostic. Per condition the budget is 20 source groups, 80 compatibility WAVs
and 80 generation/replay WAVs. These are stored repetitions, not 40 independent
sources. Comparing all eighty hashes is a stronger local replay check than a
four-source illustration; no empirical GPU equality is claimed here.

Run each condition separately. This interface does not yet merge three
condition artifacts into a release dataset; any later merge must compare real
references and preserve extra source-kind/non-citable metadata. Do not apply a
legacy manifest writer/merge that drops those fixture labels. No 500-source
expansion, partial-audio generation or publishing is authorized.
These commands describe an approved host execution payload. The existing Kaggle
harness remains LJSpeech-specific; no new remote runner or end-to-end remote
automation is implemented or claimed here.

## 6. Fixture evidence, reproduction and tests

The fixture helper reuses the existing eight-speaker-capable native-layout
generator and the existing speech/room-tone fixture. Test-only resampling and
FLAC writes simulate external acquisition; production writers remain sanctioned
PCM16 WAV only. These are synthetic signals, not recorded human speech. This
room/silence fixture is needed to exercise the unchanged real-reference silence
calibration honestly; constant tones are not used to justify a waived gate.

The saved native-layout intake contains **32 synthetic source files and eight
simulated speakers**, with **zero exclusions**. Frozen source counts are
24 train / 4 dev / 4 eval; speaker counts are 6 / 1 / 1. Only twenty training
sources from the six training speakers reach generation. No real VCTK speaker
or utterance and no neural-vocoder output is processed in this phase.

Final saved CLI plans are `plan-v3/` (mock) and `griffin-plan-v2/` (DSP).
Earlier plans remain preserved but their old implementation fingerprints are
superseded; they must not be resealed or used as current-code evidence.

| Accepted mock artifact | Source groups / simulated speakers | Train / dev / eval WAVs | Reference / mock WAVs | Archive / zeroshot WAVs | Files / logical bytes |
|---|---|---|---|---|---|
| `compatibility/` | 20 / 6 | 80 / 0 / 0 | 40 / 40 | 40 / 40 | 103 / 14,594,552 |
| `generation/` | Same 20 / 6 | 80 / 0 / 0 | 40 / 40 | 40 / 40 | 103 / 14,594,773 |

Each accepted artifact stores **378.922668 seconds** of audio counting both
conditions and tiers. These are repeated processing of the same source groups,
not independent recordings. Both passed twenty-file alignment, all seven sanity
rows and the existing band check (five probe sources; worst residual
1.886303035888185e-16 against allowance 1e-6). Source and output byte/decoded
hashes, fixed inventory, sample counts, grouping, reference reconstruction and
the archive-derived tier are checked again by the standalone validator.
The saved independent review revalidated both artifacts and confirmed **all
80 relative WAV paths byte-identical** between the two mock stages. Across the
two stages there are 160 accepted fixture WAVs from the same twenty groups.
All seven mock sanity EERs are 0.5, as expected for an identity mock. This
exercises the validation plumbing; it does not establish acoustic neutrality
or detectability of any real multi-speaker vocoder cohort.

Fixture-only CLI reproduction (use new output paths on replay):

```powershell
& '.venv/Scripts/python.exe' -B tests/multispeaker_fixture.py `
  --out-dir runs/vocotrace-phase4fb-fixture-20261107/source
& '.venv/Scripts/python.exe' -B -m experiments.cli ingest vctk `
  --source-root runs/vocotrace-phase4fb-fixture-20261107/source `
  --out-dir runs/vocotrace-phase4fb-fixture-20261107/intake `
  --source-kind fixture --speakers 8 --utterances-per-speaker 4
& '.venv/Scripts/python.exe' -B -m experiments.cli multispeaker plan `
  --ingestion-dir runs/vocotrace-phase4fb-fixture-20261107/intake `
  --source-kind fixture --condition fixture_mock `
  --out-dir runs/vocotrace-phase4fb-fixture-20261107/plan-v3
& '.venv/Scripts/python.exe' -B -m experiments.cli multispeaker compatibility `
  --plan-dir runs/vocotrace-phase4fb-fixture-20261107/plan-v3 `
  --out-dir runs/vocotrace-phase4fb-fixture-20261107/compatibility
& '.venv/Scripts/python.exe' -B -m experiments.cli multispeaker generate `
  --plan-dir runs/vocotrace-phase4fb-fixture-20261107/plan-v3 `
  --compatibility-dir runs/vocotrace-phase4fb-fixture-20261107/compatibility `
  --out-dir runs/vocotrace-phase4fb-fixture-20261107/generation
& '.venv/Scripts/python.exe' -B -m experiments.cli multispeaker validate `
  --directory runs/vocotrace-phase4fb-fixture-20261107/generation
```

The actual sixty-iteration Griffin-Lim determinism probe resynthesizes **one
training fixture twice**: input 110,250 samples, native output 110,080 samples,
finite and sample-identical, maximum replay difference **0**. Both output
float32 SHA-256 values are
`d1b5ff8b0a1e1bbdd36ce92a709b53c40b8de35cc72bf90bd9cfcdbc7295f682`.
This one-file probe emits no WAVs and cannot satisfy a twenty-source gate.

A separate **twenty-training-source Griffin-Lim compatibility attempt actually
ran** and was **blocked by unchanged INV-12**. Pre-encoding alignment passed on
all twenty files (all envelope lags zero); archive and zeroshot
`silence_duration` each had EER **0.0**, below the unchanged **0.40** floor with
no design exception. Duration EER was 0.5 and RMS EER 0.65 in both tiers.
Archive bandwidth EER 0.0 retains its existing Griffin-Lim design exception.
The attempt retains **80 failed fixture WAVs**, a failed validation receipt,
lengths, pre-encoding alignment and sanity diagnostics. It did not reach the
decoded-alignment gate or receive a complete success inventory, and authorizes
no `generate` stage. No gate or fixture was changed to force this run to pass.
This is a property of synthetic test signals under DSP processing, not an
estimate of VCTK quality or eligibility. Neither DSP execution nor the passing
mock pipeline establishes neural generation.

Reproduce that expected gate failure on fresh output paths:

```powershell
& '.venv/Scripts/python.exe' -B -m experiments.cli multispeaker plan `
  --ingestion-dir runs/vocotrace-phase4fb-fixture-20261107/intake `
  --source-kind fixture --condition griffin_lim `
  --out-dir runs/vocotrace-phase4fb-fixture-20261107/griffin-plan-v2
& '.venv/Scripts/python.exe' -B -m experiments.cli multispeaker compatibility `
  --plan-dir runs/vocotrace-phase4fb-fixture-20261107/griffin-plan-v2 `
  --out-dir runs/vocotrace-phase4fb-fixture-20261107/griffin-compatibility --device cpu
# Expected: nonzero exit, INV-12; preserve failed diagnostics and stop.
```

The saved audit scripts `probe_griffin_fixture.py` and
`attempt_griffin_compatibility.py` record the independent one-file replay and
expected full-attempt rejection under the fresh audit directory. The mock
dataset never uses `griffin_lim`, `bigvgan` or `vocos` as its condition label.

```powershell
& '.venv/Scripts/python.exe' -B -m pytest -o addopts='' -v --durations=10 `
  tests/test_multispeaker_generation.py `
  --basetemp runs/vocotrace-phase4fb-audit-20261107/pytest-focused-review-fresh
# Exact final focused regression/legacy selection (nine cases):
& '.venv/Scripts/python.exe' -B -m pytest -o addopts='' -v --durations=10 `
  tests/test_multispeaker_generation.py::test_fixture_end_to_end_uses_exact_training_groups_and_shared_gates `
  'tests/test_multispeaker_generation.py::test_strict_output_schema_hashes_and_semantics_reject_tampering[container_resealed]' `
  tests/test_multispeaker_generation.py::test_derived_reconstruction_uses_the_archive_parent_loudness `
  tests/test_invariants.py::TestINV07SingleOrdering `
  tests/test_phase_a_run.py::TestGatedRun `
  --basetemp runs/vocotrace-phase4fb-audit-20261107/pytest-focused-complete
& '.venv/Scripts/python.exe' -B -m pytest -o addopts='' -q -ra `
  --basetemp runs/vocotrace-phase4fb-audit-20261107/pytest-full-complete
& '.venv/Scripts/python.exe' -B -m ruff check src tests
& '.venv/Scripts/python.exe' -B -m ruff check .
git --no-optional-locks diff --check
git --no-optional-locks status --short --untracked-files=all
```

Use a fresh unique basetemp for every rerun. Tests include native-rate/shape/
finite failures, source/speaker/split and hash mismatches, resealed plan/output
tampering, wrong relative paths, stage prerequisites, real/fixture separation,
offline cache failure, actual twenty-file delay rejection, deterministic mock
replay, reconstruction and the training-only revalidation read scope.

The final focused selection passed **9 tests in 362.13 seconds**, including
the six existing ordering/gated-run tests. All **52 new cases passed** in the
final full suite. Full pytest: **1,068 passed, 4 failed, 9 skipped, 1 xfailed**
in **2,192.16 seconds (36:32)**. This is the Phase 4F baseline of 1,016 passes
plus 52 new passes, with the same four failures:

- `test_kaggle_harness.py::TestGeneratedNotebook::test_the_committed_notebook_is_exactly_what_the_harness_generates`
- `test_phase_a_run.py::TestKaggleNotebook::test_the_generated_notebook_is_the_committed_one`
- `test_phase_a_run.py::TestUpstreamImportGuardThroughADirectoryLink::test_namespace_package_through_the_link_passes`
- `test_phase_a_run.py::TestUpstreamImportGuardThroughADirectoryLink::test_namespace_package_from_elsewhere_fails`

The first two compare CRLF/LF notebook bytes; the last two fail when Windows
denies junction creation (WinError 5). The nine pre-existing skips concern
filesystem-link privileges. The pre-existing expected failure is the dormant
INV-17 ablation quantization contract. No failures were suppressed or tests
weakened. Earlier interrupted/intermediate runs are retained but not counted as
final results. Final logs are `focused-complete.log` and `full-complete.log`.

Ruff `src tests`: **passes**. Repository-wide Ruff: the same **11 findings in
the generated `notebooks/kaggle_v0.ipynb`**, with no notebook changes.
`git diff --check` passes. Logs are under
`runs/vocotrace-phase4fb-audit-20261107/`.

Changed files: `src/experiments/cli.py`; new
`src/experiments/multispeaker_generation.py`, `tests/multispeaker_fixture.py`,
`tests/test_multispeaker_generation.py`, and this report. Fresh ignored audit
receipts/logs and synthetic fixture outputs stay under new `runs/` directories.
All changes remain unstaged. No historical source, dataset, model or checkpoint
pin is deliberately changed; final preservation receipts distinguish metadata
checks from cryptographic WAV rereads.

The independent `final_review.json` confirms unchanged HEAD and empty index.
Of the 136 pre-existing tracked files, this phase changes only the CLI; its
initial `.gitignore` edit remains byte-identical to the starting working tree.
All **4,722 files across twenty protected historical roots** retain the same
inventory, size and modification time. This includes the original v0 data,
partial-audio datasets/pilots, exposed historical model bundles and previous
fixtures. These are filesystem-metadata checks, **not** freshly computed hashes
of historical waveform/feature bytes. Historical audio and frozen evaluation
data were not opened for this preservation audit. The new fixture source bytes
are independently hashed and verified unchanged by generation and replay tests.
The read-only finalization script refreshes report/log hashes after recording
these results and rechecks the same protected inventories without audio reads.

## 7. Remaining limitations and decision

Real acquisition remains unresolved. Neural runtimes, checkpoint caches,
Windows/Linux behavior, speaker generalization and GPU determinism have no new
empirical validation. The candidate environment is a documented experiment
setup, not a proven lockfile. Human source/rights review and official acquisition
evidence remain necessary; hashes cannot prove genuine human speech or consent.
The twenty-source Griffin-Lim fixture cohort failed INV-12, so only the mock
cohort is admitted. It supplies useful interface and rejection evidence without
establishing a successful real or DSP-cohort generation protocol.

Exact duplicate detection does not cover every near duplicate, shared session
or cross-corpus speaker alias. Existing calibration, one-sided sanity verdicts,
Griffin-Lim bandwidth cliff, Vocos resampling signature and full-band processing
confounds retain their previous limitations. No robustness or detector improvement
is inferred from passing fixture gates. All 18 invariants remain binding.

Proceed to the real eight-speaker compatibility pilot only after acquisition,
approved runtime/cache setup and successful validation of that environment.
Do not scale or publish from fixture or one-file Griffin-Lim evidence.
