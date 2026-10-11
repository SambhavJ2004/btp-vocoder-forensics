# Codex instructions: BTP Vocoder Forensics

## Purpose and scope

The original question is: **does better-sounding vocoder-generated speech become
harder for deepfake detectors to detect?** Resynthesize the same real utterances
with different vocoders, preserving speaker, words, recording, pairing and
processing controls; compare perceptual quality with forensic detectability.

The expanded objective adds binary real/synthetic detection, known-vocoder
attribution, frequency analysis and eventually unknown-source rejection and an
interactive audio application. The quality-versus-detectability study remains
part of this work. Develop incrementally, beginning with lightweight baselines
and reusable inference/preprocessing. Introduce API and UI work only when
requested and supported by an evaluated backend.

## Read the authoritative sources

- `CLAUDE.md`: all **18 numbered experimental invariants are binding**. Read the
  applicable rules before changing experimental behavior. Do not duplicate,
  redefine or weaken them here. Modify `CLAUDE.md` only with explicit user
  approval for that separate change.
- `docs/measurements.md`: measured evidence and unresolved experimental issues.
- `docs/mel_configs.md`: upstream vocoder front ends and their provenance.
- `docs/enforcement_audit.md`: enforcement locations and remaining gaps.
- `docs/kaggle_harness.md`: existing automation and iteration/record workflow.
- `README.md`, `pyproject.toml`: usage, package layout and dependency groups.
- `docs/codex_onboarding_audit.md`: dated onboarding evidence and integration plan.

Distinguish current implementation, unit-test evidence, saved historical runs and
newly verified remote experiments. Historical status prose can lag code; inspect
both and report discrepancies without silently revising the methodology.

## Architecture and extension points

`src/` contains the top-level Python packages `data`, `vocoders`, `metrics`,
`detectors` and `experiments`; the distribution is `btp-vocoder-forensics`.
The `btpvf` entry point is `experiments.cli:main`.

- `data/invariants.py`: shared rules/constants and measured analysis membership.
  `manifest.py`, `audio_io.py`, `preprocess.py`, `alignment.py` and `merge.py`
  implement dataset contracts, processing, alignment gates and merging.
- `vocoders/`: lazy, isolated upstream adapters, registry, immutable checkpoint
  descriptors and narrowly recorded compatibility shims.
- `metrics/` and `experiments/phase_b_quality.py`: paired quality measurements.
- `detectors/`: detector interfaces, crop/filter utilities, EER, protocol and
  attribution scaffolding. Inspect stubs before promising model functionality.
- `experiments/phase_a_resynthesis.py`: preflight, generation, staging, gates and
  publication; `experiments/sanity_checks.py` runs confound probes. Phase B
  drivers handle quality/detection; Phase C is attribution.
- `experiments/config.py`, `analysis.py`: experiment configuration and reporting.
- `experiments/kaggle_harness.py`: the sole Kaggle runner. The committed
  `notebooks/kaggle_v0.ipynb` is generated from it, not hand edited.
- `tests/`: CPU tests with synthetic fixtures/stubs and platform-dependent import
  tests. `data/measurements/` holds small checked-in evidence; generated audio,
  checkpoints and run outputs remain outside version control.

## Scientific integrity and reproducibility

Preserve paired utterances, frozen grouped splits and speaker-disjoint splits
when multiple speakers are involved. Prevent leakage between conditions, tiers,
feature transforms, training, calibration and evaluation. Keep sample-rate tiers,
frequency bands, analysis membership, matched/mismatched protocols and controls
explicit in artifacts and reports. Every exclusion requires measurement evidence.
Never lower validation gates to make a run pass, silently change established
processing, invent numbers, or equate a pin/preflight pass with empirical validity.

Use existing manifest path resolution and sanctioned audio I/O. Preserve seeds,
configuration, dataset identities/splits, checkpoint/code revisions and digests,
environment, gate records, exclusions and output identities. Reproduce reference
benchmarks before relying on pretrained models for research conclusions.
Debugging/iteration output is not final evidence: only successful, valid
**record-mode** results tied to an existing user-created commit and verified
dataset/model/output provenance may be cited as final research findings.

## Development and local checks

Reuse working modules and package patterns; make minimal, focused, testable
changes. Do not install/upgrade dependencies, download models or run training
without authorization. Heavy GPU work belongs on explicitly approved remote
compute. Update relevant documentation with behavior/evidence changes; do not
turn hypotheses or planned experiments into reported results.

Use the project environment explicitly on Windows, for example
`& '.venv\Scripts\python.exe' -m experiments.cli --help`.
Run Ruff (`python -m ruff check .`, also `src tests` to isolate source findings)
and the full suite (`python -m pytest`). Ruff and pytest settings are in
`pyproject.toml`. For a restricted Windows temp directory, use a fresh, unique
workspace `runs/<audit>/pytest-<name>` with `--basetemp`; do not reuse an existing
directory that pytest could delete. Report existing failures, sandbox/platform
limitations and regressions separately. Do not hide failures or weaken tests.

## Git: user controls every write

Codex may inspect status, diffs, history and existing objects with read-only Git
commands. **All Git writes are performed manually by the user.** Never stage,
commit, push, reset, clean, restore, checkout, switch, stash, rebase, merge,
cherry-pick, alter branches/history/configuration/index, or create/publish pull
requests. Leave every file change unstaged for review. These user instructions
override any older documentation suggesting that an agent stage changes.

## Kaggle: reuse the existing harness

Use `experiments/kaggle_harness.py` and the `btpvf kaggle` commands. Automated
submission, monitoring, retrieval and failure reporting are authorized only when
explicitly requested; never silently launch expensive jobs or mutate remote
resources during an audit. Preserve CLI-managed credentials and per-notebook
Kaggle secrets. Never print, hardcode, track, document or package secret values.
The generated optional secret example is not proof of an active association.

Iteration ships working-tree bytes and is debugging-only. Record mode ships code
from an **existing manually created commit**; Codex must never create that commit.
Verify gates, code/dataset identities, complete outputs and provenance before
calling a record valid. Unit tests or a historical CPU run do not establish
current authentication, end-to-end automation, GPU training or model validity.
