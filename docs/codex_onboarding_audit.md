# Codex onboarding and environment audit

Audit dates: 2026-10-11 to 2026-10-12 (Asia/Calcutta). Scope: local inspection, CPU
verification, persistent instructions and one small Kaggle retrieval correction. No new model, API, UI,
training, dependency installation, checkpoint download or remote mutation.

Evidence labels below distinguish implementation, local tests, saved historical
artifacts and current remote verification. A historical measurement is not a new
experiment or an automatically citable research finding.

## 1. Starting Git state

HEAD was `59002be40e67ea8711206f37f1db8e34ee4bfef0`, matching the supplied short
SHA. Branch was `main`, tracking `origin/main`; `git status --short --branch`
showed a clean working tree. No root or applicable ancestor `AGENTS.md` existed.
Only read-only Git inspection/object export was used. All onboarding edits remain
unstaged; the user performs every Git write.

## 2. Research understanding

Paired resynthesis holds speaker, words and the underlying recording fixed while
changing the vocoder. The research compares quality estimates with EER under two
different questions: pretrained mismatched detection measures generalization;
per-vocoder matched training measures available artifact information. Their
default sample-rate tiers also differ, so a protocol gap cannot be attributed
solely to generalization. The quality ladder is a hypothesis, not measured rank.

Frequency error profiles describe differences; the retrained band-limited
ablation is needed for the mechanism argument. Controls, analysis exclusions,
resampling, quantization, silence, loudness and alignment matter because a
detector can learn processing history instead of synthesis artifacts.

The expanded system adds binary detection, known-source attribution and eventually
unknown-source rejection, controlled resynthesis and an application. It preserves
this research rather than replacing it. No empirical quality/detection correlation
or trained-system accuracy was established by this audit.

## 3. Repository architecture and inspected sources

All requested documents are present and were inspected, along with
`btp-project-idea.md`, package implementations, configuration, notebook generation
and the four test modules. Fixtures are primarily synthetic audio, temporary
corpora, stub upstream modules and mocked dependencies inside those tests; there
is no committed separate fixture directory. Ignored stale bytecode is not evidence
of additional implemented source modules.

| Location | Responsibility |
|---|---|
| `src/data/invariants.py`, `measurements.py` | Shared contracts/constants, analysis membership, checked-in evidence verification |
| `src/data/corpus.py`, `audio_io.py`, `mel.py`, `preprocess.py`, `alignment.py` | Corpus/splits, sanctioned audio I/O, native mel front ends, processing, alignment gates |
| `src/data/manifest.py`, `merge.py` | Portable paths, provenance columns, table validation, loudness reports, isolated-run merge |
| `src/vocoders/` | Registry, lazy upstream adapters, checkpoint identities and narrowly scoped compatibility shims |
| `src/metrics/` | MCD, spectral distances, F0/voicing, PESQ and lazy UTMOS wrapper/runner |
| `src/detectors/` | Adapter/crop interface, EER, protocol guards, filter utilities, attribution scaffolding |
| `src/experiments/` | CLI, Phase A/B/C drivers, confound probes, configuration, analysis, Kaggle runner |
| `notebooks/kaggle_v0.ipynb` | Generated Phase A notebook, sourced from the harness |
| `data/measurements/`, `docs/`, `tests/` | Small committed measurement evidence, design/enforcement history, CPU tests |
| ignored `runs/` | Existing downloaded datasets/run records plus new local audit logs |

The distribution is `btp-vocoder-forensics` 0.1.0; imports are the five top-level
packages, not a `btpvf` Python package. Console entry point:
`btpvf = experiments.cli:main`. Current commands: `audit`, `resynth`, `validate`,
`provenance`, `loudness`, `sanity`, `silence-diag`, `kaggle` (run/code/notebook),
`merge`, `quality`, `plot`. There is no train/detect/attribute/infer CLI.

## 4. Implemented and reusable components

Phase A preflights corpus shape, pins, runtime dependencies, output paths and
environment isolation. It selects deterministic utterances/splits, processes the
real reference once, synthesizes with native upstream mel front ends, checks
alignment, applies the shared trim/length/loudness/encoding pipeline, emits both
tiers and records provenance. `run_gated` uses staging and validates the manifest,
an analysis-band sample, confound probes and loudness records before publishing;
failure can preserve a `.failed` directory. Unit tests exercise refusal paths
with synthetic fixtures. They do not certify every neural condition.

Merging reuses isolated outputs, validates pairing and requires identical real
file bytes. It records source runs in `manifest.merge.json`; it does not itself
rerun every signal gate on the merged analysis population.

The filter implementation and comparison-set transform are useful existing
building blocks: zero-phase two-edge Chebyshev-II filtering, real included, fixed
measured specification/floors, and rate-aware cutoff utilities. Crop helpers
preserve 4.0375 seconds: 64,600 samples at 16 kHz and 89,027 at 22.05 kHz, with
repeat padding. Measurement references and analysis exclusions have unit tests.

MCD (MFCC-style cepstra, no DTW), band-wise LSD/high-band distance and F0/voicing
functions exist. PESQ-WB and UTMOS wrappers exist and use the derived 16 kHz tier;
MCD/F0/spectral analysis uses the archive in the quality runner. UTMOS is a MOS
predictor, with hub tag `tarepan/SpeechMOS:v1.2.0`, not a measured listening MOS.
The quality runner joins paired files and aggregates per-condition results.
No complete new real-data quality run or pretrained reference benchmark was
performed; optional dependencies are absent from the project environment.

EER/bootstrap, labeled `DetectionResult`, scoring orchestration, protocol/tier
guards, Spearman reporting and paired-bandwidth contrasts exist. Attribution
fold/label/energy-score utilities also exist. These are reusable infrastructure,
subject to the evaluation defects and missing implementations below.

## 5. Partial and missing work

- AASIST, RawNet2 and wav2vec2-AASIST loading/scoring are stubs; training is absent.
  Their usable checkpoint pins and reference benchmark reproduction are pending.
- `phase_b_detection.run_mismatched` and `run_matched` orchestrate adapters but
  cannot run with the supplied stubs. The matched driver lacks primary-band
  transform plumbing; its default full-band result cannot feed the band-limited
  headline guard unchanged. `run_bandlimit_ablation` is explicitly unimplemented.
- `phase_c_attribution` closed/open/family training and `open_set_metrics` are
  unimplemented. There is no trained multiclass system or unknown-source rejector.
- No shared forensic feature cache, binary AUROC reporting, confusion-matrix
  artifact, per-file predictions, single-file inference or model bundle loader.
  Existing AUROC measurement probes are not a deployed binary evaluation pipeline.
- Periodicity RMSE is mentioned in the F0 module description but not returned by
  its implementation. VCTK loading is a stub; multi-speaker split safeguards exist.
- No detector training notebook/CLI workflow, structured checkpoint/metric
  retrieval contract, API or frontend. Those remain future work.
- UTMOS checkpoint-byte provenance and empirical benchmark validation have not
  been established by merely pinning a hub tag.

## 6. Dataset and vocoder status

The pre-existing local directory `runs/onboarding/v0-kaggle-input` contains the
reported v0: 200 LJSpeech utterances, real plus `bigvgan_112m`, `vocos` and
`griffin_lim`, archive and zeroshot tiers, 1,600 referenced audio files. Frozen
utterance splits are 140 train / 20 dev / 40 eval.

This audit independently passed current table validation with both tiers
required, checked all 1,600 headers (mono PCM_16, recorded rate/duration), and
found no within-pair frame-length discrepancy. All 1,606 receipt-listed files
matched their saved size/SHA-256; receipt and manifest digests matched the local
source-audit record. See `runs/codex-audit-20261011/v0-local-audit.json`.

Saved `gates.json` reports a pass with zero leaked probes. **Signal gates were
not rerun in this audit, and historical generation/gate bytes are not proven.**
The original sidecar has no code-snapshot identity; the pre-existing receipt
explicitly says `mode=legacy`, `sha=unknown`, `citable=false`,
`historical_gate_bytes_proven=false`, `signal_gates_rerun=false`. This verifies
local inventory/consistency, not record-mode citability. Its saved source audit
identifies published v0 version 1; current remote state was not queried.

| Condition | Current code/pins | Available empirical evidence |
|---|---|---|
| `griffin_lim` | Deterministic adapter, no checkpoint | Saved 200-utterance CPU iteration marked COMPLETE; 800 paired two-tier files for real/GL locally present |
| `bigvgan_112m` | Adapter, HF revision and weight digest | Included in historical v0; no new record-mode verification |
| `vocos` | Adapter, HF revision, frozen measured Kaiser resampler | Included in historical v0; high-band exclusion remains binding |
| `bigvgan_base` | Adapter and pins now populated | Saved iteration failed bandwidth probe: EER 0.395 versus unchanged 0.40 floor; 800 real/base files in failed output |
| `hifigan_v1` | Upstream adapter, code/revision/digest populated | Saved alignment failure: modal lag -141 samples, 2/20 votes; candidate front-end diagnosis remains unverified |
| `melgan` | Official adapter, code/digest populated | Saved alignment failure: modal lag -1 sample, 4/20 votes; fidelity/alignment diagnosis remains unverified |
| `bigvgan_v2_22khz_fullband` | Shared BigVGAN adapter, control identity | Preflight blocked by missing checkpoint SHA-256; no validated control run |
| `specdiff_gan` | Declared unavailable stub | Excluded from generable conditions; no obtainable-weight verification |

The six ladder rungs now pass the pin/mel readiness audit (`blocked=false`), but
that does not override the unresolved empirical failures or local missing neural
dependencies. Historical prose saying base/HiFi-GAN/MelGAN still lack pins is
stale. The v0 shared-environment exception is explicitly a team-reported exception,
not independent proof of general dependency compatibility. No pins were changed.

## 7. Summary of the 18 authoritative invariants

This is a non-normative navigation summary. `CLAUDE.md`, including its measured
qualifications, remains authoritative; this table does not replace its rules.

| Invariant | Protected property |
|---|---|
| 01 | Primary 22.05 kHz archive; separate single-step native/archive and archive/16 kHz derivations, documented resampler exception |
| 02 | Use each checkpoint's verified native mel front end; document differences |
| 03 | Compute trim indices from real once and reuse across paired conditions |
| 04 | One analysis-band-referenced loudness gain at the archive; paired drops for clipping; derived drift recorded without renormalization |
| 05 | Sanctioned mono WAV PCM_16 encoding and common I/O |
| 06 | Identical utterance sets and lengths; exclusions/drops stay paired |
| 07 | One fixed processing order and derivation path |
| 08 | Fixed seeds, checkpoint/code pins, environment and provenance records |
| 09 | Frozen shared splits; deliberate speaker-disjoint policy for multi-speaker data |
| 10 | Raw upstream vocoder output; no condition-specific cleanup |
| 11 | Identical detector crop duration, padding and input policy across conditions/tiers |
| 12 | Confound gates before believing Phase B/C results; fixed floors and real-derived silence calibration |
| 13 | Isolated vocoder environments except the explicitly reported v0 exception; merge checks real identity |
| 14 | Matched and mismatched protocols are labeled and never pooled/confused |
| 15 | No credentials in code, repository, artifacts or logs |
| 16 | Alignment probe requires exact zero modal lag; do not compensate by shifting audio |
| 17 | Archive remains full-band; shared measured band limit is an analysis transform, real included, with the PCM_16 detector-input contract |
| 18 | Analysis-specific membership and measured exclusions; controls stay separate; rejected equalization alternatives remain rejected |

Current primary analysis is band-limited at `LADDER_FMAX` (8 kHz). High-band
analysis excludes Griffin-Lim and Vocos for documented measured confounds; the
fullband BigVGAN control is not a headline ladder rung. Do not transfer an
exclusion to another analysis or undo it based on convenience.

## 8. Persistent Codex instructions

Created root `AGENTS.md`: research/expanded scope, architecture and extension
points, documentation authority, all-invariant binding by reference, scientific
integrity/leakage controls, provenance/benchmark requirements, minimal development
and checks, user-only Git writes, and explicitly authorized Kaggle automation.
It resolves older staging instructions in favor of the user's absolute Git rule.
It requires valid record-mode evidence and does not claim historical datasets or
models are newly verified. `CLAUDE.md` was not edited.

## 9. Kaggle capability audit

Classification concerns the capability's present implementation and available
verification, not a promise of current end-to-end service readiness. Local saved
CPU iteration evidence supports a historical Phase A round trip; no new Kaggle
job, upload or artifact download occurred. No successful record/GPU-training
round trip is proven.

| # | Capability | Classification | Evidence and limits |
|---|---|---|---|
| 1 | Prepare code artifacts | Implemented and tested | Working-tree and existing-commit snapshots, deterministic zip/tree identity and exclusions tested |
| 2 | Upload/version private code dataset | Implemented but not verified | Create/version/status code and saved iteration records; current service unverified; existing visibility not checked |
| 3 | Launch notebook/kernel | Implemented but not verified | Push/generated metadata and historical CPU run; no current remote submission/orchestration test |
| 4 | Monitor execution | Implemented but not verified | Poll loop exists, enum parsing tested, historical terminal records; error/recovery paths not comprehensively tested |
| 5 | Detect success/failure | Partially implemented | COMPLETE/ERROR/cancel/timeout handling; early failures can leave no durable run record |
| 6 | Retrieve complete logs | Partially implemented | JSON/plain-text parsing tested; progress/timestamps removed, no raw archive; nonzero log command exit ignored |
| 7 | Download outputs | Implemented and tested | Historical local output evidence; new mocked CLI tests reject failed pages; no remote integrity test today |
| 8 | Output pagination | Implemented and tested | Token following, stderr token, repeated/cyclic token, page cap and CLI failure regressions now tested |
| 9 | Experiment provenance | Partially implemented | Snapshot/sidecar merge tested; failure sidecars, mounted-tree verification and output identity gaps remain |
| 10 | Iteration runs | Implemented and tested | Dirty working-tree snapshot/non-citable guard tested; saved iteration records |
| 11 | Record runs | Partially implemented | Existing-commit archive/citable flag tested; full remote execution identity and successful record evidence unverified |
| 12 | Safe retries | Missing | No durable resume/idempotency/retry state machine; mutable kernel/dataset references and second-resolution directory IDs |
| 13 | GPU training jobs | Partially implemented | GPU metadata/quota reporting and guard tested with stubs; generated job is Phase A, no training entry point |
| 14 | Model checkpoints | Partially implemented | Generic output transport can retrieve files emitted by a job; no training/save/checkpoint validation contract |
| 15 | Metrics/predictions collection | Partially implemented | Phase A manifest/gates/loudness artifacts; no detector-training metrics/per-file prediction schema |

Authentication remains managed by the external Kaggle CLI, which inherits the
environment and reads its own credentials. The harness does not open or rewrite
credential files. Historical docs record CLI 2.2.4 with access-token authentication;
the installed global CLI is also 2.2.4. A sanitized `config view` attempt returned
1 and contained a connection error, so current method/account/auth freshness is
**unverified**, not evidence that credentials are absent. Secret values were
neither inspected nor printed.

Adapters accept the existing environment/CLI token mechanisms. The notebook's
optional `UserSecretsClient`/`HF_TOKEN` example is commented out; automatically
enabling/associating that per-notebook secret is not currently implemented. No
authentication redesign, credential file edits or secret relocation occurred.

## 10. Defects, inconsistencies and deferred fixes

**Corrected:** `download_output` used `check=False` and treated a failed CLI page
without a continuation token as successful completion. A repeated token was also
treated as completion. It now uses the existing checked CLI wrapper and raises on
repeated tokens/cycles; bounded pagination remains. Seven tests mock subprocess
responses through the real wrapper. Partial files remain for diagnosis; this is
not retry support and does not change generation or measurement logic.

**Evaluation issues, reproduced and left unchanged:**

- `detectors.eer.compute_eer` processes tied samples individually in stable
  class order. Four identical real and synthetic zero scores return EER 1.0,
  instead of chance under a tie-aware threshold ROC. Intermediate steps split
  identical scores and cannot correspond to a real threshold. Add grouped-tie,
  constant-score, permutation and score-direction regression tests before using
  baseline EER. This audit did not change a shared scientific metric.
- `experiments.sanity_checks._verdict` only flags `eer < 0.40`. A perfectly
  reversed synthetic example (real zeros, synthetic ones) gives EER 1.0 and
  `leaked=false`: a trivially separable confound can evade this one-sided test.
  Review orientation handling separately with evidence and unchanged acceptance
  stringency; do not silently change the established gate during onboarding.
- `bootstrap_eer_ci` independently resamples real/spoof arrays. A paired study
  should explicitly review uncertainty resampling by utterance (and speaker when
  relevant). This is a methodological question, not an already implemented fix.
- The F0 no-jointly-voiced branch hardcodes VDE 1.0, even if both voicing masks
  agree. This is a code-inspection finding; real optional-backend execution was
  not attempted. Add mocked-F0 edge-case tests in a later metric task.

**Automation/provenance gaps, left for focused follow-up:**

- `run.json` is written only after server verification, polling, logs and
  download. Authentication/upload/timeout/retrieval failures can lose durable
  failure state. Preserve an incremental receipt and raw logs before safe retry.
- `read_log` ignores return code and irreversibly transforms log records. Parsed
  display is useful, but does not satisfy complete/raw log retention.
- Snapshot mounting locates the first provenance file rather than verifying its
  content hash against the requested snapshot; the code dataset/kernel refs are
  mutable. Pin/verify input version and code bytes on the remote side.
- In record mode, packaged project code comes from the requested commit but the
  launch notebook is rendered by the current checkout. The notebook bytes/driver
  identity are not a single requested-commit execution contract yet.
- Successful resynthesis merges snapshot provenance in a later notebook cell.
  Gate failure skips that cell; observed failed sidecars lack `code_snapshot`.
- `assert_citable` checks the snapshot flag, not success, gates or artifact
  integrity, and is not a mandatory result-consumption gate. A citable flag is
  necessary but insufficient for a valid research record.
- `.gitignore` excludes untracked secrets, not files already committed. Snapshot
  `git ls-files -c ...`/commit tree export can include a tracked secret despite an
  ignore rule. The current inspected tracked paths contain no credential files;
  the existing untracked-dummy test does not establish the stronger claim in the
  module documentation. Add fail-closed artifact checks without reading secrets.
- Dataset creation is private by default; versioning does not verify existing
  visibility. Dataset status errors are conflated with nonexistence.

**Documentation/contract discrepancies:**

- Historical pin blockers, the old generation-time low-pass/control design and
  older v0 condition descriptions remain in prose/comments. Current code and
  dated measurements must be consulted together; no historical record was erased.
- README's environment-only credential statement omits current CLI-managed auth.
- `_emit_pair` downsamples the in-memory archive waveform after writing it,
  rather than rereading the finished PCM_16 archive described in
  `DERIVATION_ORDER`. This is a code/contract discrepancy to review, not permission
  to modify validated Phase A during onboarding.
- Compatibility shim events are recorded on the adapter/log path but are not an
  explicit persistent sidecar field; retain before/after digests in future record
  provenance. Existing shims remain narrowly matched and fail closed.
- `Detector.score/score_batch` lacks an explicit rate parameter even though crop
  helpers are rate-aware. Future adapters need a reliable configured rate/tier
  contract; never infer it from waveform length.
- Full-repo notebook lint errors and Windows notebook-byte/link test issues are
  described below. They predate the retrieval fix.

## 11. Files changed and boundaries

- Added `AGENTS.md` and this report.
- Changed only retrieval error/pagination handling in
  `src/experiments/kaggle_harness.py`.
- Added seven offline regression cases in `tests/test_kaggle_harness.py`.
- Added retrieval behavior/verification limits to `docs/kaggle_harness.md`.
- New ignored `runs/codex-audit-20261011/` holds test logs/temp fixtures, local
  dataset checks and synthetic EER reproductions. Existing `runs/onboarding/`
  receipts/dataset were read, not rewritten.

No Phase A logic, pins, authoritative invariant rules or dependencies changed.
No Git write, remote upload, kernel launch, training or large download occurred.

## 12. Environment and Ruff

Windows Python is 3.13.9 (`C:\Python313\python.exe`). Existing `.venv` is also
3.13.9 and has editable project 0.1.0 pointing at this workspace. Its creation
metadata names the older checkout location; use the environment's Python with
`-m`, rather than relying on potentially relocated console launchers.

| Environment | Observed package availability |
|---|---|
| Project `.venv` | numpy 2.5.3, scipy 1.18.1, soundfile 0.14.0, librosa 1.0.0, pyloudnorm 0.2.0, pandas 3.0.5, PyYAML 6.0.3, tqdm 4.70.0, pytest 9.1.1, Ruff 0.16.6, scikit-learn 1.9.0 |
| Project optional packages | torch, torchaudio, PESQ, pyworld, vocos, huggingface_hub, matplotlib, seaborn and Kaggle are not installed here |
| Global Python/PATH | Kaggle CLI 2.2.4 available; global torch/torchaudio also present but global Python cannot import the editable project and lacks some core audio dependencies |

`python -m pip check` in `.venv`: **No broken requirements found.** Dependency
floors/extras are declared in `pyproject.toml`; no locked full environment exists.
Core/dev CPU work is possible locally. No dedicated training GPU was assumed or
used. Heavy neural/quality work requires an approved isolated remote environment.

README documents installation/CLI usage but no explicit Ruff/full-test command.
Used module invocations with the repository's Ruff/pytest configuration. Ruff:
100 columns, Python 3.10 target, E/F/I/UP/B families; only `vocoders/compat.py`
has E501 ignored to preserve verbatim upstream matching.

- `.venv\Scripts\python.exe -m ruff check src tests`: **passed**, before and after
  the retrieval correction.
- `.venv\Scripts\python.exe -m ruff check .`: **11 pre-existing errors**, all in
  generated notebook cells: E401 (2), I001 (4), UP031 (2), E402 (3). The correction
  does not modify generated cells. No automatic lint fixes were applied.

## 13. Full test verification

The initial default-temp full run failed because this sandbox could not create
pytest's numbered temp directory under its redirected Windows temporary path.
This was an environment/setup error, not a source regression. Repeated with a
fresh workspace-owned `--basetemp`, without changing tests or acceptance rules.

Baseline command (PowerShell, existing environment):

```powershell
& '.venv\Scripts\python.exe' -m pytest -o addopts='' -q -ra --basetemp runs/codex-audit-20261011/pytest-baseline
```

Baseline: **396 passed, 4 failed, 4 skipped, 1 xfailed** (405 collected;
212.51 seconds). The two notebook equality tests compare checkout CRLF bytes with
generator LF bytes; normalizing line endings gives identical notebook bytes and
JSON objects. `git ls-files --eol` reports index LF / worktree CRLF;
`core.autocrlf=true`. The other two failures are directory-link fixture creation:
symlink fallback calls `_winapi.CreateJunction`, denied with WinError 5 in this
sandbox. Four HF file-symlink cases skip with WinError 1314. The explicit expected
failure covers the still-unimplemented PCM_16 ablation input contract.

New retrieval regressions: **7 passed**, 48 harness cases deselected, 0.17 seconds;
mocked subprocess only, no network.

Post-change full suite: **403 passed, 4 failed, 4 skipped, 1 xfailed**
(412 collected; 228.52 seconds), exit code 1. The same four failures remained;
all seven new regression cases passed and no new failure appeared. Command was
the baseline command with the distinct `pytest-final` basetemp. Logs are
`runs/codex-audit-20261011/pytest-baseline.log` and `pytest-final.log`. No known
failure was skipped or weakened to obtain a green result.

## 14. Detection/attribution pilot architecture (plan only)

| Concern | Reuse and proposed extension |
|---|---|
| Dataset admission | `data.manifest.read_manifest/validate_manifest`, portable path resolver, `data.measurements`, existing signal gates; require a verified dataset receipt/code/gate identity before a research run |
| Splits | Existing manifest train/dev/eval; group every condition/tier/crop by `utt_id`, speaker-disjoint when applicable; never random-split rows or regenerate splits |
| Preprocessing/features | `data.audio_io.read_processed`, `Detector.crop_for`, comparison-set band transform; add one common rate-aware forensic LFCC/log-spectrum statistics extractor/cache, separate from vocoder-native mel front ends |
| Binary baseline | New lightweight scikit-learn linear/logistic baseline implementing the detector contract, with scaler fit on training only, real/synthetic balancing, explicit score direction and save/load; no neural download required |
| Attribution baseline | Same common features; synthetic-only multiclass head over admitted known conditions; persist condition-to-class mapping and per-vocoder/confusion-matrix metrics |
| Research protocols | Keep per-condition `run_matched` and pretrained `run_mismatched`; add explicit band/tier plumbing. A pooled binary pilot is a separate task/protocol artifact, not a replacement for per-vocoder matched research |
| Evaluation | Extend existing EER after tied-score repair; AUROC, dev-calibrated threshold, per-condition metrics and paired/grouped confidence intervals; attribution macro metrics and confusion matrix |
| Predictions/bundles | Per-file rows with utterance/condition/split/tier/band/model identity; bundle classifier/scaler/feature/crop config, class map, seed, dataset/code digest and environment; hash outputs |
| Training driver/CLI | Extend `experiments` and `experiments.cli` with a distinct pilot driver; paths/compute knobs in `RunConfig`, scientific controls remain in invariants |
| Kaggle | Extend existing `KernelSpec`/`notebook_source`/`pip_extras` for an explicit training job, mounted manifest/output paths and CPU/GPU selection; reuse snapshot/push/poll/retrieve with durable receipts and raw logs |
| Inference | A shared `predict_file`/bundle loader used by both evaluation and later API; validate uploads and report rate/channel/preprocessing provenance and unsupported input cases |
| Application later | Thin API around the same inference/resynthesis services, then upload/results/waveform/spectrogram/paired-audio UI; no duplicated preprocessing or premature dashboard |

Start with the admitted v0 conditions, at the archive tier with the **same primary
8 kHz analysis transform for real and every synthetic member**. Materialize and
read processed PCM_16 analysis files before detector input; avoid Phase A float32
arrays, per-condition front ends, per-file renormalization or ad hoc shifts.
Content-matched real copies must not be overrepresented when pooling synthetic
classes; report class weighting and per-vocoder scores. Fit all learned feature
scaling/selection on train only, select hyperparameters/thresholds on dev, evaluate
once on eval. With 40 eval utterance groups and one speaker, this is an exploratory
pilot, not evidence of broad generalization.

Retain full-band archive measurements as separately labeled secondary work.
High-band membership/exclusions remain measured and analysis-specific; do not
train or claim attribution of excluded classes from known resampling/bandwidth
signatures. Frequency sweeps retrain at each cutoff, preserve unfiltered results
separately and record tier/filter/cutoff on every output. Keep external pretrained
mismatched evaluation isolated until checkpoint provenance and its reference
benchmark are reproduced. Unknown-source work later reuses leave-one-vocoder-out
folds with the held-out vocoder excluded from training **and calibration**.

Kaggle training must write bounded checkpoints, configuration, structured metrics,
predictions and complete provenance under `/kaggle/working`. Mounted input paths
must be supplied/discovered via the existing manifest/corpus/snapshot mechanisms,
not embedded laptop paths. A CPU linear baseline needs no GPU reservation;
GPU-heavy follow-ups require explicit job authorization. Record runs require an
existing user-created commit, remote code/driver verification and complete output
receipts. No job was launched to demonstrate this proposed architecture.

## 15. Recommended next task

Before model implementation, approve a focused **evaluation and dataset-admission
repair**: fix EER ties with independent threshold/ROC regression cases; review
probe score orientation without relaxing gate stringency; add an offline receipt
that verifies manifest/splits/file hashes and explicitly requires matching code
and freshly run gates for research eligibility. Recheck the existing v0 signal
gates locally under the unchanged methodology and record any failure honestly.
Legacy v0 remains useful for non-citable integration debugging even if its
generation provenance cannot be recovered.

Then, with separate explicit approval, implement the shared-feature linear binary
baseline and synthetic-only attribution pilot through the extension points above.
Add durable/raw-log/provenance Kaggle hardening before approving a record training
run. The user reviews these unstaged onboarding changes and creates any required
commit; Codex does not proceed into the new forensic system during onboarding.
