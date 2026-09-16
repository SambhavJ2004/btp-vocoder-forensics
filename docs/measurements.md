# Committed measurements

`data/measurements/` holds evidence produced **outside this repository** — by
measurement runs on the real corpus on Kaggle — and committed here because
invariants cite it by row. It is the one path under `data/` that git tracks; see
the negation rule in [`.gitignore`](../.gitignore).

**Why committed.** INV-18 requires every analysis exclusion to name a
measurement rather than an argument. A citation that points at someone's Kaggle
scratch disk is an argument. `src/data/measurements.py` recomputes each cited
statistic from these files, and `TestINV18PerAnalysisMembership` fails if a
recorded value and the file disagree — so the evidence cannot drift underneath a
claim, and a claim cannot quote a number the evidence does not contain.

**Do not hand-edit these files.** They are outputs. If a measurement is re-run,
replace the file and let the resolver tests tell you which citations moved.

---

## `band_measurements.csv`

Per-file, per-condition high-band energy fractions.

| | |
|---|---|
| **Rows** | 300 = 60 utterances × 5 conditions |
| **Corpus** | LJSpeech-1.1 (13,100 files available; 60 measured). Example path: `/kaggle/input/datasets/mathurinache/the-lj-speech-dataset/LJSpeech-1.1/wavs/LJ001-0001.wav` |
| **Rate** | 22,050 Hz (archive) |
| **Quantisation** | PCM_16 — values include the 16-bit quantisation floor |
| **Produced** | Kaggle measurement run; timestamp not recorded in the artifact |

### Columns

| column | meaning |
|---|---|
| `utt_id` | LJSpeech utterance id |
| `condition` | see below — **the artifact's own naming**, not the registry's |
| `samples` | length in samples; identical across conditions for every utterance (checked) |
| `hb_8k_nyq` | fraction of total energy in **[8000, 11025) Hz** — `LADDER_FMAX` to Nyquist |
| `hb_8k_rtedge` | fraction of total energy in **[8000, 10900) Hz** — `LADDER_FMAX` to `rt_edge` |
| `rms` | RMS level |

`hb_8k_nyq − hb_8k_rtedge` is the energy in **[10900, 11025) Hz**, the sliver
where a resampler's transition band leaves its fingerprint. The resolver accepts
this difference directly as the metric `hb_8k_nyq-hb_8k_rtedge`.

### Conditions

| artifact name | what it is | registry condition |
|---|---|---|
| `real` | LJSpeech at the archive rate | `real` |
| `real_roundtrip` | real, round-tripped 22050 → 24000 → 22050 with the **torchaudio default** resampler | none — a measurement control |
| `bigvgan` | BigVGAN v2 fmax8k resynthesis (see `meta.json`) | `bigvgan_112m` |
| `vocos` | Vocos resynthesis, resampled to the archive rate | `vocos` |
| `griffin_lim` | Griffin-Lim resynthesis | `griffin_lim` |

**Which resampler produced `real_roundtrip`** is not stated in the file. It is
identified as the torchaudio default because its high-band ratio to `real`
(ratio of means, **0.8198**) matches the default-resampler figure reported with
the run, and no other resampler is represented in the artifact.

### Summary (recomputed from the file)

| condition | `hb_8k_nyq` mean | ratio to real | energy in [10900, 11025) mean |
|---|---|---|---|
| real | 0.020169 | 1.0000 | 2.24e-08 |
| real_roundtrip | 0.016535 | **0.8198** | 9.03e-07 |
| bigvgan | 0.016246 | 0.8055 | 7.30e-09 |
| vocos | 0.016216 | 0.8040 | **8.63e-07** |
| griffin_lim | 5.1e-09 (median) | ~0 | ~0 |

---

## `meta.json`

Run metadata.

| key | value | used by |
|---|---|---|
| `bigvgan_repo` | `nvidia/bigvgan_v2_22khz_80band_fmax8k_256x` | `vocoders.bigvgan.LARGE_SPEC.checkpoint` |
| `bigvgan_sha256` | `5011c15a…aaa` (64-hex) | `vocoders.bigvgan.LARGE_SPEC.checkpoint_sha256` |
| `bigvgan_cfg` | sr 22050, n_fft 1024, hop 256, win 1024, 80 mels, fmin 0, fmax 8000 | `vocoders.bigvgan.LARGE_SPEC.mel`, cross-checked against the config audit |
| `vocos_repo` | `charactr/vocos-mel-24khz` | informational; matches `vocos` spec |
| `corpus_example`, `corpus_files` | LJSpeech path, 13100 | informational |
| `n_measured` | 60 | informational |
| `archive_sr` | 22050 | informational; matches `ARCHIVE_SR` |
| `ladder_fmax` | 8000 | informational; matches derived `LADDER_FMAX` |
| `rt_edge` | 10900 | defines the `hb_8k_rtedge` band edge |
| `resample_probe_auroc` | `{"full": 1.0, "high_band": 1.0}` | informational — see limits below |

**`bigvgan_sha256` is a weight-file digest, not a repo revision.** It fills
`checkpoint_sha256`. The spec's `checkpoint_revision` (40-hex commit
`a9d8b711…`) is a different identifier from the earlier config audit and is left
unchanged. Which file was hashed is not recorded in `meta.json`.

**The BigVGAN mel config has two sources and they are cross-checked.** On import,
`vocoders.bigvgan` compares `meta.json`'s `bigvgan_cfg` field-by-field against the
config audit ([`mel_configs.md`](mel_configs.md)) and against
`AUDITED_MEL_FMAX`. A disagreement raises rather than letting one source win:
either the run measured a different checkpoint than the ladder declares, or the
audit is wrong, and both invalidate the measurements.

---

## What the artifacts do NOT contain

Several numbers reported with this measurement run are **not present** in either
file. They are quoted in the codebase only where clearly marked as reported, and
no exclusion depends on them.

| reported | status |
|---|---|
| torchaudio default: ratio 0.8198, AUROC 1.0000 | **ratio verified** from the CSV; AUROC consistent with `meta.json` (1.0) |
| torchaudio **kaiser**: ratio 0.9998, AUROC 0.9958 | **absent** — no kaiser condition in the CSV |
| **soxr_vhq**: ratio 0.9936, AUROC 1.0000 | **absent** — no soxr condition in the CSV |
| one kaiser round trip vs two: AUROC **0.9903** | **absent** — recorded as `REJECTED_ALTERNATIVES["uniform_round_trip"]` with `in_committed_artifacts=False` |
| resample probe through an 8 kHz band limit: **0.502** | **absent** — `meta.json` records only `full` and `high_band`, both 1.0 |

**`resample_probe_auroc` is not reproducible from the CSV columns.** An
*unpaired* AUROC on `hb_8k_nyq` alone separates `real` from `real_roundtrip` at
only **0.558**; the round trip is lower on 100% of files *paired*. The recorded
1.0 therefore comes from a paired or richer-feature probe whose method is not
recorded. It is kept as a reported result.

---

## Locally reproduced measurements

Not from the Kaggle run; produced in this repository and reproducible from the
test suite.

### Resample probe below vs above `LADDER_FMAX`

Synthetic speech-like audio (harmonic stacks + broadband noise), round trip with
`RESAMPLE_METHOD` (`soxr_hq`) — **not** the pinned torchaudio kaiser, which is
absent here by INV-13. Logistic regression on log band energies, **5-fold
cross-validation grouped by utterance**.

| configuration | below `LADDER_FMAX` | above `LADDER_FMAX` |
|---|---|---|
| committed 0.502 config, band limit **before** loudness (old INV-17 pipeline) | 0.505 | — |
| same config, band limit **after** loudness (current pipeline) | **0.558** | **1.000** |
| noisier signal, n_fft 1024, filter passband only, 5 independent draws | 0.66 – 0.69 | 1.000 |

The signature is **robustly concentrated above** `LADDER_FMAX` — AUROC 1.0 in
every configuration. Whether it is **at chance below** is **not robust**: 0.50 to
0.69 depending on configuration. `TestINV18ResampleProbe` asserts the robust
claim and records the stronger one as a strict `xfail`.

### Cross-validation must be grouped by utterance

The two paths share an utterance, and utterances differ far more than the paths
do. Shuffled `StratifiedKFold` puts one version in train and the other in test,
and the classifier is systematically **inverted**: on the same data it reports
**0.14** against the grouped **0.56**. An unshuffled `cv=5` happens to group
correctly because of how the arrays are stacked, which is why the original 0.502
was sound — but that is an accident of layout, not a method.
`TestINV18ResampleProbe.test_paired_probe_must_group_by_utterance` pins this.

---

## Claims that rest on these files

| claim | where | evidence |
|---|---|---|
| Griffin-Lim emits no high-band content | CLAUDE.md INV-17, INV-18; `ANALYSIS_EXCLUSIONS` | `griffin_lim` `hb_8k_nyq` max 5.49e-08 vs `real` min 6.38e-04 |
| Vocos carries a resampler signature above `LADDER_FMAX` | CLAUDE.md INV-18; `ANALYSIS_EXCLUSIONS` | [10900, 11025) energy: `vocos` 8.63e-07 vs `bigvgan` 7.30e-09; AUROC 0.9997 vs bigvgan, **0.507 vs `real_roundtrip`** |
| the torchaudio default resampler costs 18% of the high band | `vocoders.vocos.VOCOS_RESAMPLER.justification` | `real_roundtrip` / `real` mean `hb_8k_nyq` = 0.8198 |
| BigVGAN v2 fmax8k checkpoint identity and mel config | `vocoders.bigvgan.LARGE_SPEC` | `meta.json` |
| BigVGAN produces ~80% of real's high-band energy | [`mel_configs.md`](mel_configs.md) | `bigvgan` / `real` mean `hb_8k_nyq` = 0.8055 |
| Griffin-Lim, Vocos included in the band-limited primary analysis | CLAUDE.md INV-18; btp-project-idea.md Part 2 | **not established by these files** — see above |
