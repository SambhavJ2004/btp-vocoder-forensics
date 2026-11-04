# Does better-sounding mean harder-to-detect?

Measuring whether perceptual quality and forensic detectability are orthogonal
properties of a neural vocoder. B.Tech project, NSUT.

Design: [btp-project-idea.md](btp-project-idea.md).
**Invariants you must not violate: [CLAUDE.md](CLAUDE.md).**

## Layout

| Package | Owner | Contents |
|---|---|---|
| `src/data` | Person A | Corpus, confound enforcement, manifest |
| `src/vocoders` | Person A | Vocoder adapters, the quality ladder |
| `src/metrics` | Person B | UTMOS, PESQ, MCD, F0, band-wise LSD (X-axis) |
| `src/detectors` | Person C | AASIST/RawNet2, both protocols, source tracing (Y-axis) |
| `src/experiments` | shared | Phase drivers, sanity gate, main plot |

## Install

Base install stays light; vocoders are never co-installed (CLAUDE.md, INV-13).

```bash
pip install -e .              # core: manifest, audio pipeline, analysis
pip install -e '.[quality]'   # Person B
pip install -e '.[detect]'    # Person C
pip install -e '.[pilot]'     # CPU logistic detection and known-vocoder attribution
pip install -e '.[dev]'       # tests + ruff
```

## Commands

```bash
btpvf audit                                    # which conditions are runnable yet
btpvf resynth --corpus-root ... --conditions griffin_lim hifigan_v1
btpvf validate --manifest manifests/dataset.csv
btpvf sanity   --manifest manifests/dataset.csv   # MUST pass before B or C run
btpvf quality  --manifest manifests/dataset.csv
btpvf plot     --quality results/quality_by_condition.csv --detection results/detection.csv
btpvf pilot train --manifest ... --out-dir ...  # fresh directory, non-citable development
btpvf pilot evaluate --bundle ... --manifest ... --out-dir ...
btpvf pilot predict --bundle ... --wav ...
btpvf partial build --manifest ... --out-dir ...  # local non-citable partial-audio pilot
btpvf partial validate --dataset-dir ... --source-manifest ...
```

The working CPU forensic pilot uses shared archive-tier 8 kHz log-spectral
features, frozen train/dev/eval groups and two logistic classifiers. Legacy v0
pilot results are always non-citable. See [the pilot guide](docs/forensic_pilot.md)
for supported audio, provenance, exact commands and planned Kaggle integration.

The [VocoPatch partial-audio pilot](docs/vocopatch_partial_pilot.md) creates bounded,
sample-labeled real/vocoder splices and matched real-only null controls from the
existing archive. It validates source/output integrity, preserves frozen groups
and provides an offline audio/timeline review. Publishing remains a separate step.

The [Phase 1 scientific review](docs/vocopatch_phase1_review.md) adds opt-in
real-to-real controls and explicit `--split-count train=N` selections while
preserving the first pilot's legacy recipe and original frozen assignments.

The [Phase 2 expansion](docs/vocopatch_phase2_expansion.md) scans every frozen
source group with `partial build --all-utterances --real-to-real-controls`.
It retains explicit exclusions and prepares a separate local private-upload
review folder. The [model protocol](docs/vocopatch_training_protocol.md) defines
training/evaluation separation. Phase 2 performs no upload or model training.

The [Phase 3A temporal baseline](docs/vocopatch_temporal_baseline.md) adds
`btpvf temporal train|predict`: CPU logistic localization with acoustic windows,
explicit ignore/support masks, safe numeric bundles and source-grouped train/dev
metrics. Its first development run has poor real-only control specificity;
it is non-citable and does not run final eval.

The [Phase 3B failure analysis](docs/vocopatch_temporal_failure_analysis.md) audits
the saved baseline and compares four fixed acoustic representations with grouped
validation inside train and control-FPR thresholds. Dev comparisons are
descriptive; eval remains unused. Reproduction uses the `temporal_review` and
`temporal_ablation` experiment modules.

## Order of work

1. `btpvf audit` — pin checkpoints and verify mel configs (INV-02, INV-08).
2. Phase A per condition, one isolated Kaggle session each (INV-13).
3. `btpvf validate` then `btpvf sanity` — the gate. A leak blocks the condition.
4. Phase B quality and detection, in parallel.
5. `btpvf plot` — the main figure. Rough version due December.

Credentials live in environment variables only; see `.env.example` and INV-15.
