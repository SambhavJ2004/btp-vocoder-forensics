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
```

## Order of work

1. `btpvf audit` — pin checkpoints and verify mel configs (INV-02, INV-08).
2. Phase A per condition, one isolated Kaggle session each (INV-13).
3. `btpvf validate` then `btpvf sanity` — the gate. A leak blocks the condition.
4. Phase B quality and detection, in parallel.
5. `btpvf plot` — the main figure. Rough version due December.

Credentials live in environment variables only; see `.env.example` and INV-15.
