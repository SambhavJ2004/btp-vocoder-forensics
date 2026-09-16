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

**`vocos` was very likely resampled with the default too, not the pin.** In the
[10900, 11025) sliver it carries ~40× real's energy, like `real_roundtrip`, and is
indistinguishable from it (AUROC 0.507). A round trip that *adds* energy just
below Nyquist is folding content back, and the default's short kernel (6 zero
crossings, cutoff 0.99 × Nyquist ≈ 10.9 kHz) is the kind that does. The pin (64
zero crossings, cutoff 0.9999 × Nyquist) is a different kernel with a different
imprint, and the CSV contains no Vocos made with it. The Vocos high-band
exclusion still stands under the pin, because `probes.json` shows the kaiser round
trip separable above the band at 0.9935, and the exclusion now cites that too. But
the sliver figures describe a Vocos generated before the pin existed. This is
inferred from the file, not recorded in it.

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

**`resample_probe_auroc` is not reproducible from the CSV columns.** An
*unpaired* AUROC on `hb_8k_nyq` alone separates `real` from `real_roundtrip` at
only **0.558**; the round trip is lower on 100% of files *paired*. The recorded
1.0 therefore comes from a paired or richer-feature probe whose method is not
recorded. It is kept as a reported result; `probes.json` supersedes it.

---

## `probes.json`

Paired resample probes: does a round trip leave a learnable imprint, and in
which band?

| | |
|---|---|
| **Files** | 200 LJSpeech utterances |
| **Probe** | `LogisticRegression(max_iter=5000)` on the log mean magnitude spectrum (n_fft 1024, hop 256), no feature scaling recorded |
| **CV** | `GroupKFold(5)` grouped by utterance — both versions of an utterance always in the same fold |
| **Quantisation** | PCM_16 grid |
| **Bands** | `below` = [0, `LADDER_FMAX`), `above` = [`LADDER_FMAX`, Nyquist), `full` |
| **Kaiser parameters** | `sinc_interp_kaiser`, width 64, rolloff 0.9999, beta 14.769656459379492 — **identical to `VOCOS_RESAMPLER`** (checked by `TestMeasurementArtifacts`) |
| **Not recorded** | loudness normalisation, log base, which soxr quality, scikit-learn version, how `ratio_highband` is aggregated, per-fold coefficients |

| probe | high-band ratio | AUROC below `LADDER_FMAX` | AUROC above | AUROC full |
|---|---|---|---|---|
| real vs torchaudio default round trip | 0.8005 | **0.5291 ± 0.0024** | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| real vs **kaiser (pinned)** round trip | 0.9997 | **0.5000 ± 0.0000** | 0.9935 ± 0.0056 | 0.9924 ± 0.0071 |
| real vs soxr round trip | 0.9922 | **0.5000 ± 0.0000** | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| kaiser round trip once vs twice | 0.9998 | **0.5000 ± 0.0000** | 0.9928 ± 0.0073 | 0.9918 ± 0.0066 |

Read with its controls (next section), this establishes what INV-18 needed:
**the pinned kaiser round trip leaves no below-band imprint this probe can
detect, and a clear one above.** The torchaudio default does leave a small
below-band imprint, so the pin is load-bearing for Vocos's membership in the
*primary* analysis, not only for high-band energy.

### Why 0.5000 ± 0.0000 is not a bug

Three probes read exactly 0.5000 below `LADDER_FMAX`, with zero spread across
five folds. Seen cold, that looks like a broken pipeline: a constant score, an
empty feature slice, a label mix-up. It is a *degenerate* result with a specific
mechanism, and two controls rule out the pipeline faults.

**It is not noisy chance.** Noisy chance has spread. Uninformative labels at this
n give a fold std near 0.06 (measured locally: 0.4984 ± 0.0628). Even a
non-degenerate probe on paired data spreads by 0.003–0.005, and its test folds
hold both versions of the same utterances. Exactly 0.5000 in every fold is a
different thing.

**What exactly 0.5 means.** In a grouped fold, each test utterance appears twice,
once per label. AUROC counts every positive-versus-negative score comparison.
Comparisons between *different* utterances come in mirror pairs (i against j,
j against i). If the two versions of every utterance get the same score, each
mirror pair contributes exactly one win, and each same-utterance comparison is a
tie worth 0.5. The total is exactly 0.5, whatever the scores are. So
0.5000 ± 0.0000 means the decision function gave both versions of every
utterance the same score, in every fold.

**Why the classifier does that.** lbfgs starts from zero coefficients. There, the
gradient for each coefficient is 0.25 × the mean paired difference in that
feature (scikit-learn's mean-loss objective, balanced labels). If every component
is under the solver tolerance (`tol = 1e-4`), lbfgs declares convergence at
iteration 0. The coefficients stay exactly zero, every score is the intercept,
and the AUROC is exactly 0.5. This is the solver never leaving its starting
point, not L2 shrinkage: the penalty's gradient is also zero there. Reproduced by
`TestINV18ProbeControls.test_exact_half_is_a_solver_that_never_left_zero`
(n=200 paired, 372 features, scikit-learn 1.9):

| consistent paired shift, every feature (log units) | AUROC | max \|coef\| | lbfgs iterations |
|---|---|---|---|
| 0, 1e-6, 1e-5, 1e-4, 3e-4 | 0.5000 ± 0.0000 | 0 | 0 |
| 1e-3 | 0.5130 ± 0.0005 | 2.8e-5 | 3 |
| 1e-2 | 0.5142 ± 0.0011 | 0.13 | 57–107 |

**What it establishes, and what it does not.** It does not show the two paths are
identical below the band. It shows that no below-band feature has a mean paired
difference above about 4 × tol, i.e. 4e-4 in the feature's log units, *for this
linear probe on mean spectra*. In amplitude that is ~0.0035 dB for natural-log
magnitudes or ~0.008 dB for log10 (`probes.json` does not say which). That is a
resolution, not a proof of zero. It is also fine enough: the torchaudio
default's below-band difference clears it.

**Why the pipeline is not at fault — two controls.**

1. **Above the band, same probe.** The same features, labels, folds and
   classifier separate the same pairs at 0.9935 (kaiser), 1.0000 (soxr) and
   0.9928 (once vs twice). A broken label join, a constant feature matrix, or a
   scoring bug would break these too.
2. **Same band, a lossier resampler.** Below the band the torchaudio default
   reads 0.5291 ± 0.0024. That is not degenerate, and it is above the 0.5125
   that within-pair ordering alone can reach at 40 utterances per fold. So the
   below-band slice carries live features and the probe can report a non-trivial
   value there. A fault confined to that slice would have made this 0.5000 too.

`data.measurements.probe_reading` refuses to return a below-band reading whose
above-band control is under `PROBE_CONTROL_MIN_AUROC` (0.95), so the number
cannot travel without its control.

**What would make it airtight.** Record per-fold `n_iter_` and coefficient
norms, the per-bin mean paired differences, the log base, whether loudness was
normalised, and the scikit-learn version, all in `probes.json`.

---

## Reported numbers against the committed artifacts

Several numbers were reported with the earlier 60-file run before `probes.json`
existed. They are superseded by the committed 200-file figures, and **none of the
earlier values appears in any artifact**. The disagreements are recorded here
rather than reconciled.

| earlier report (60 files) | committed | status |
|---|---|---|
| torchaudio default: ratio 0.8198, AUROC 1.0000 | CSV: 0.8198 (ratio of means; 0.8062 mean of per-file ratios). `probes.json`: **0.8005**, full 1.0000 | 0.8198 verified in the CSV; the 200-file ratio differs, and its aggregation is not recorded |
| torchaudio **kaiser**: ratio 0.9998, AUROC 0.9958 | ratio **0.9997**, above **0.9935**, full **0.9924** | committed; 0.9998 and 0.9958 appear nowhere |
| **soxr_vhq**: ratio 0.9936, AUROC 1.0000 | `real_vs_rt_soxr`: ratio **0.9922**, above 1.0000 | committed; quality level not recorded, so "vhq" is unconfirmed |
| one kaiser round trip vs two: AUROC **0.9903** | above **0.9928**, full **0.9918**, below 0.5000 | committed; `REJECTED_ALTERNATIVES` now cites 0.9928 |
| resample probe through an 8 kHz band limit: **0.502** | no counterpart. Kaiser below 0.5000; default below **0.5291** | superseded; if 0.502 was the default resampler, the 200-file run disagrees with it |

---

## Locally reproduced measurements

Not from the Kaggle run; produced in this repository and reproducible from the
test suite.

### The local probe and the loudness coupling

Earlier local measurements put the below-band probe anywhere from 0.50 to 0.69,
which kept Vocos's primary membership open. The committed probe settles the
membership. The local spread now has an explanation: **loudness normalisation.**

The pipeline normalises loudness on the full band (INV-04) before any
analysis-time band limit. A round trip that removes above-band energy lowers the
K-weighted loudness, so the normaliser applies a slightly larger gain, and that
gain shifts every below-band bin the same way. A paired probe is sensitive to
exactly that kind of consistent shift (above). `probes.json` does not record a
loudness step.

Isolated on the local synthetic probe (`soxr_hq` round trip, n=60,
`GroupKFold(5)`, current band-limit filter):

| loudness normalised | features | with StandardScaler | without |
|---|---|---|---|
| yes (pipeline) | 32 log band energies, band-limited | 0.5694 ± 0.0158 | 0.5458 ± 0.0034 |
| yes (pipeline) | log mean \|STFT\| < 8 kHz | 0.5514 ± 0.0056 | 0.5431 ± 0.0028 |
| no | 32 log band energies, band-limited | **0.5000 ± 0.0000** | **0.5000 ± 0.0000** |
| no | log mean \|STFT\| < 8 kHz | 0.4986 ± 0.0028 | **0.5000 ± 0.0000** |

Without loudness normalisation the local probe is degenerate, like the committed
one. With it, the probe reads 0.54–0.57. The earlier 0.505 / 0.558 / 0.66–0.69
spread is consistent with this, in each case more high-band loss meaning a
larger gain difference, but was not individually re-run.

**Does the coupling reach the pinned kaiser resampler? Measured: no.** Synthetic
soxr_hq loses far more high-band energy than LJSpeech loses to the pin, so the
coupling was measured as a resampler-free dose-response. Synthetic audio was set
to real LJSpeech's high-band fraction (0.0202, committed CSV), and
[8 kHz, Nyquist) was scaled to each committed `ratio_highband`. It then went
through loudness → PCM_16 → analysis band limit (n=200, same probe as
`probes.json`, no scaler):

| high-band ratio | matches | gain difference | AUROC, own gain (pipeline) | AUROC, shared gain |
|---|---|---|---|---|
| 1.0000 | — | 0 | 0.5000 ± 0.0000 | 0.5000 ± 0.0000 |
| **0.9997** | **kaiser pin** | **+6.4e-5 dB** | **0.5000 ± 0.0000** | 0.5000 ± 0.0000 |
| 0.9922 | soxr | +1.7e-3 dB | 0.5000 ± 0.0000 | 0.5000 ± 0.0000 |
| 0.8005 | torchaudio default | +4.3e-2 dB | **0.5320 ± 0.0061** | 0.5096 ± 0.0037 |

At the default's loss the coupling is real. Giving each path its own gain moves
the probe from 0.510 to 0.532, and a shared gain removes most of that. At the
pinned ratio the gain difference is 6.4e-5 dB and the probe stays degenerate
through the full pipeline order. `TestINV18LoudnessCoupling` pins the pinned and
default rows at n=60, the default row serving as the positive control.

Limits: synthetic audio, and an idealised loss rather than the kaiser kernel.
Loudness depends on how much K-weighted energy is removed, not on the kernel's
shape, which is why a resampler-free test answers this question. An idealised
loss starting exactly at 8 kHz also leaks into the top STFT bins just below it
(0.513 on `bins < 8 kHz` without the band limit, 0.5000 with it). That comes from
the idealisation; a real resampler's loss sits near Nyquist.

### Signature concentration (local, `soxr_hq`)

`TestINV18ResampleProbe` still pins what the local synthetic probe shows robustly:
**AUROC 1.000 above the band** in every configuration, far above the below-band
figure.

### Cross-validation must be grouped by utterance

The two paths share an utterance, and utterances differ far more than the paths
do. Shuffled `StratifiedKFold` puts one version in train and the other in test,
and the classifier is systematically **inverted**: on the same data it reports
**0.13** against the grouped **0.57**. An unshuffled `cv=5` happens to group
correctly because of how the arrays are stacked — an accident of layout, not a
method. `TestINV18ResampleProbe.test_paired_probe_must_group_by_utterance` pins
this. `probes.json` records `GroupKFold(5)` by utterance.

### Band-limit filter response

Measured **directly** — impulse through `band_limit_to_ladder`, pure tones, and
analytic `sosfreqz`, agreeing to 0.1 Hz — never energy-weighted. Zero-phase
(filtfilt) response:

| design | order | −0.1 dB | −1 dB | −3 dB | −20 dB | −100 dB | full −200 dB stopband |
|---|---|---|---|---|---|---|---|
| old: order 12, stopband edge only | 12 | 6144 Hz | **6419 Hz** | 6555 Hz | **6876 Hz** | 7616 Hz | 8000 Hz |
| **current: passband 7900 Hz / 0.1 dB, stopband 8000 Hz / 100 dB per pass** | **53** | **7900.0 Hz** | **7915.5 Hz** | **7922.9 Hz** | **7940.3 Hz** | **7979.1 Hz** | **7998.8 Hz** |

The old design gave scipy only the stopband edge. `cheby2`'s `Wn` is where
rejection *first reaches* `rs`, so the passband edge fell wherever order 12 and
100 dB put it, 1.6 kHz below `LADDER_FMAX`. The one passband figure ever quoted,
"0.004 dB below 7 kHz", was energy-weighted, and speech's low-frequency energy
makes that small regardless of the response near the edge.

Candidates for the same stopband requirement, measured on this machine (harmonic
test signal, `ms / 10 s` = filter time per 10 s of audio):

| design | size | −1 dB | −20 dB | passband dev < 6 kHz | IR 99.99% | residual f64 / f32 | ms / 10 s |
|---|---|---|---|---|---|---|---|
| Chebyshev II o12 (old) | 6 SOS | 6419 | 6876 | 0.031 dB | 2.8 ms | 1.3e-23 / 1.7e-16 | 19 |
| Chebyshev II, pass 7800 | o38 | 7830 | 7879 | ~0 | 15 ms | 2.9e-22 / 1.8e-16 | 52 |
| **Chebyshev II, pass 7900 (chosen)** | **o53** | **7915** | **7940** | **~0** | **24 ms** | **4.8e-21 / 1.8e-16** | **72** |
| Chebyshev II, pass 7950 | o75 | 7958 | 7970 | ~0 | 37 ms | 4.6e-20 / 1.8e-16 | 108 |
| elliptic, pass 7900 | o17 | 7903 | 7914 | **0.1 dB ripple** | 41 ms | 1.1e-21 / 1.7e-16 | 24 |
| Kaiser FIR once, 7900–8000 | 1415 taps | 7931 | 7970 | ~0 | 17 ms | **6.8e-16** / 8.6e-16 | 21 |
| Kaiser FIR fwd-bwd, 7900–8000 | 647×2 taps | 7915 | 7961 | 0.002 dB | 16 ms | 2.6e-16 / 4.4e-16 | 27 |

Chosen for staying in the family INV-17's residual argument was made for: a flat
passband, and a stopband residual unchanged at the float32 storage floor. Its
cost is order 53 instead of 12, a 24 ms impulse response instead of 2.8 ms (a
linear smear, identical for every member of a comparison set), and ~3.8× the
compute. The FIRs are cheaper but leave the filter's own residual above
`MEASUREMENT_FLOOR`; the elliptic ripples.

---

## Claims that rest on these files

| claim | where | evidence |
|---|---|---|
| Griffin-Lim emits no high-band content | CLAUDE.md INV-17, INV-18; `ANALYSIS_EXCLUSIONS` | `griffin_lim` `hb_8k_nyq` max 5.49e-08 vs `real` min 6.38e-04 |
| Vocos carries a resampler signature above `LADDER_FMAX` | CLAUDE.md INV-18; `ANALYSIS_EXCLUSIONS` | [10900, 11025) energy: `vocos` 8.63e-07 vs `bigvgan` 7.30e-09; AUROC 0.9997 vs bigvgan, **0.507 vs `real_roundtrip`**; pinned kaiser above-band 0.9935 (`probes.json`) |
| the torchaudio default resampler costs 18–20% of the high band | `vocoders.vocos.VOCOS_RESAMPLER.justification` | CSV `real_roundtrip` / `real` = 0.8198; `probes.json` 0.8005 |
| the kaiser pin keeps the high band | `VOCOS_RESAMPLER.justification` | `probes.json` `real_vs_rt_kaiser` ratio 0.9997 |
| **Vocos belongs in the band-limited primary analysis** | CLAUDE.md INV-18; btp-project-idea.md Part 2 | `probes.json` kaiser below 0.5000 ± 0.0000 with above-band control 0.9935; default below 0.5291 as same-band control; loudness path checked locally |
| a uniform round trip would not equalise Vocos | `REJECTED_ALTERNATIVES` | `probes.json` `kaiser_1x_vs_2x` above 0.9928, full 0.9918 |
| BigVGAN v2 fmax8k checkpoint identity and mel config | `vocoders.bigvgan.LARGE_SPEC` | `meta.json` |
| BigVGAN produces ~80% of real's high-band energy | [`mel_configs.md`](mel_configs.md) | `bigvgan` / `real` mean `hb_8k_nyq` = 0.8055 |
