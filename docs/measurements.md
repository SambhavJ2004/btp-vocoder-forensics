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
| **Bands** | `below` = [0, `LADDER_FMAX`), `above` = [`LADDER_FMAX`, Nyquist), `full` — bin selections, not the analysis band-limit filter |
| **Pipeline order** | resample → loudness normalise to −27.0 LUFS (probes ending `_norm` only) → quantise. No trim step; the pipeline trims before normalising |
| **Kaiser parameters** | `sinc_interp_kaiser`, width 64, rolloff 0.9999, beta 14.769656459379492 — **identical to `VOCOS_RESAMPLER`** (checked by `TestMeasurementArtifacts`) |
| **Not recorded** | log base, which soxr quality, scikit-learn version, how `ratio_highband` and `mean_gain_delta_db` are aggregated, per-fold values or coefficients |

| probe | loudness | high-band ratio | mean gain delta | AUROC below `LADDER_FMAX` | AUROC above | AUROC full |
|---|---|---|---|---|---|---|
| real vs torchaudio default round trip | raw | 0.8005 | — | **0.5291 ± 0.0024** | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| | **normalised** | 0.8005 | 0.02986 dB | **0.5348 ± 0.0018** | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| real vs **kaiser (pinned)** round trip | raw | 0.99974 | — | **0.5000 ± 0.0000** | 0.9935 ± 0.0056 | 0.9924 ± 0.0071 |
| | **normalised** | 0.99976 | 4.62e-05 dB | **0.5000 ± 0.0000** | 0.9989 ± 0.0011 | 0.9972 ± 0.0043 |
| real vs soxr round trip | raw | 0.99222 | — | **0.5000 ± 0.0000** | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| | **normalised** | 0.99223 | 0.001534 dB | **0.5000 ± 0.0000** | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| kaiser round trip once vs twice | raw | 0.99979 | — | **0.5000 ± 0.0000** | 0.9928 ± 0.0073 | 0.9918 ± 0.0066 |
| | **normalised** | 0.99979 | 4.046e-05 dB | **0.5000 ± 0.0000** | 0.9941 ± 0.0065 | 0.9925 ± 0.0076 |

Read with its controls (next section), this establishes what INV-18 needed:
**the pinned kaiser round trip leaves no below-band imprint this probe can
detect, and a clear one above — with and without the pipeline's loudness
normalisation.** The torchaudio default does leave a small below-band imprint,
so the pin is load-bearing for Vocos's membership in the *primary* analysis, not
only for high-band energy.

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

### Loudness coupling: a threshold with a margin

Loudness is normalised on the **full** band (INV-04). A round trip that removes
high-band energy lowers the K-weighted loudness, so the normaliser applies a
slightly larger gain, and that gain shifts every bin *below* `LADDER_FMAX` by the
same amount. The `_norm` probes measure this on real LJSpeech in the pipeline's
order:

| round trip | mean gain delta | AUROC below, raw → normalised | verdict |
|---|---|---|---|
| torchaudio default | **0.02986 dB** | 0.5291 → **0.5348** | coupling detected |
| soxr | **0.001534 dB** | 0.5000 → 0.5000 | not detected — highest measured null |
| kaiser (pinned) | **4.62e-05 dB** | 0.5000 → 0.5000 | not detected — the operating point |
| kaiser once vs twice | 4.046e-05 dB | 0.5000 → 0.5000 | not detected |

**The threshold is a bracket.** Onset lies somewhere in (0.001534, 0.02986] dB.
Three points do not locate it more finely than that factor of ~19. "Roughly
0.0015 dB" is the highest measured null, the conservative end, not a located
threshold.

**The bracket is this probe's resolution, not a property of the coupling.** A
gain delta *g* moves every log-magnitude feature by the same amount, and lbfgs
never leaves zero while 0.25 × shift < tol (see *Why 0.5000*). That puts the
probe's detection limit at *g* ≈ 0.0035 dB for natural-log features or ≈ 0.008 dB
for log10. Either value falls inside the bracket, and
`TestINV18LoudnessCoupling.test_bracket_is_this_probes_resolution` reproduces
both verdicts from the mechanism alone. The gain shift itself does not vanish
below 0.0015 dB. It falls below what a linear probe on mean spectra resolves, and
a stronger detector could resolve less.

**The operating point's margin.** The pinned resampler's gain delta is
**33× below the highest null** (1.5 orders of magnitude) and **646× below the
lowest detected delta** (2.8 orders). `data.measurements.loudness_coupling`
derives these from every normalised probe on file, rather than from constants,
and raises if the evidence is not monotone in gain delta.
`LOUDNESS_COUPLING_MIN_MARGIN` requires at least 10× below the highest null. The
committed 33× clears 10× and does **not** clear 100×. A future resampler change
cannot cross it silently, because the pin is also checked equal to the parameters
`probes.json` measured, so changing the pin forces a re-measurement.

**Corroboration.** Before this was committed, a resampler-free synthetic
dose-response estimated the same three gain deltas at 6.4e-05, 1.7e-03 and
4.3e-02 dB, with identical verdicts. The real-audio values are 0.72×, 0.92× and
0.69× of those. That run is superseded and no longer tested.

**Beyond resamplers — estimated, not measured.** Nothing about the coupling is
specific to resampling. Any condition with less high-band energy than real gets a
larger gain, and so a below-band level offset. Scaling from the committed deltas
gives about 7.4–9.8 dB of gain delta per unit of total energy removed above
`LADDER_FMAX`, using real's high-band fraction of 0.0202 from the 60-file CSV.
The same energy was removed from each condition's total, which isolates only the
above-band-driven part:

| condition (CSV) | high-band ratio to real | estimated gain delta |
|---|---|---|
| `bigvgan` | 0.8055 | ~0.03–0.04 dB — the size the default was detected at |
| `vocos` | 0.8040 (row predates the pin) | ~0.03–0.04 dB |
| `griffin_lim` | ~0 | ~0.15–0.20 dB |

If these hold, the band-limited primary analysis carries a projection of each
condition's above-band deficit as a level offset. That is the mechanism INV-18
relies on the band limit to remove, including a trace of Griffin-Lim's cliff.
This is an open item in CLAUDE.md.

**Instrumented for the next Phase A run.** Every archive manifest row now records
the normaliser's input (`lufs_in`), the gain the file carries
(`loudness_gain_db`), the target (`loudness_target_lufs`), and the
pre-normalisation loudness on the analysis band (`lufs_in_band`), which is
recorded and never used. `btpvf loudness --manifest ...`
(`data.manifest.loudness_gain_report`) reports two paired deltas for each
condition against real, each with n, mean, std, 5th/50th/95th percentiles:

| column | what it is | reaches the primary analysis? |
|---|---|---|
| `gain_delta_db` | gain received minus real's | only partly — it includes plain level differences the normaliser is meant to remove |
| `band_offset_db` | `(lufs_in_band + gain)` minus real's: the in-band level after normalisation | **yes** — this is the offset the band-limited analysis sees |

The report also gives `band_offset_x_resolution`, the mean offset in units of the
committed probe's ~0.0035 dB resolution, and the mean `high_band_fraction` delta
alongside.

The two deltas separate cleanly on controlled synthetic audio carrying real's
2.02% high band (`TestINV04LoudnessRecord`, pipeline records):

| difference from real | `gain_delta_db` | `band_offset_db` |
|---|---|---|
| 1 dB quieter, identical spectrum | 1.00000 | 0.00000 |
| 20% less energy above `LADDER_FMAX` | 0.04375 | 0.04375 |
| no energy above `LADDER_FMAX` | 0.22331 | 0.22331 |

These synthetic magnitudes sit near the estimates above (0.03–0.04 and
0.15–0.20 dB), slightly higher. They are not measurements of the ladder. The
ladder's numbers come from the next real Phase A run.

### Normalisation and the high-band evidence

With the pipeline's loudness normalisation, above-band separability is **at
least** as high as the un-normalised probe showed. So the Vocos high-band
exclusion and the uniform-round-trip rejection are, if anything, better supported
under the real pipeline:

| probe | above, raw | above, normalised | rise | raw fold std |
|---|---|---|---|---|
| kaiser (pinned) | 0.9935 ± 0.0056 | **0.9989 ± 0.0011** | +0.0054 | 0.0056 |
| kaiser once vs twice | 0.9928 ± 0.0073 | **0.9941 ± 0.0065** | +0.0013 | 0.0073 |

Both non-saturated probes rise, and so do both full-band figures. But each rise
is at most about one un-normalised fold std, and per-fold values are not
recorded. So what the evidence supports is "not weaker under normalisation", not
"normalisation sharpens it" as an established effect. The mechanism is not
recorded either. One untested hypothesis is that per-file normalisation removes
between-utterance level variation, a nuisance the unpaired AUROC otherwise has to
overcome. Griffin-Lim's high-band exclusion does not depend on a resampler and is
unaffected. `TestINV18PerAnalysisMembership.test_high_band_evidence_holds_in_the_pipeline_order`
asserts the evidence clears the control threshold, not that it rose.

---

## `silence_calibration.csv`, `silence_floors.csv`, `silence_threshold_sweep.csv`

`btpvf silence-diag` output from the first 200-utterance v0 run, the one whose
INV-12 gate raised. Committed because they record two separate things: the
**refutation** of the reconstruction-noise hypothesis (below), and the **floor
estimator's pooled-percentile defect** that the pad-agreement check caught.

| | |
|---|---|
| **Files** | 200 LJSpeech utterances, archive tier, all four conditions |
| **Produced by** | `btpvf silence-diag --manifest <run>/manifest.csv` |
| **Status** | the run RAISED: `pad_agreement_ok = False`, so no silence EER from it is a silence measurement |

`silence_calibration.csv` is the gate's own record: the frame-RMS cut, the floor,
the margin, the threshold, and what real actually read against `2 × TRIM_PAD_MS`.
`silence_floors.csv` is per condition, including the two quiet-frame statistics
the threshold is derived from. `silence_threshold_sweep.csv` is the probe's own
feature at five fixed thresholds plus the derived one, per condition.

**They disagreed with each other, which is how the defect surfaced.** Both report
the same named quantity for `real`:

| | frame-RMS cut | peak floor |
|---|---|---|
| `silence_floors.csv` (per file, then median) | −54.273 dBFS | **−46.854 dBFS** |
| `silence_calibration.csv` (as the gate computed it) | −54.273 dBFS | **−39.808 dBFS** |

The cut agreed because both paths took a median of per-file values. The peak did
not, because the gate concatenated every file's quiet samples and took one
percentile of the **pool**. Recorded in
`TestSilenceCalibrationArtifacts`; the mechanism and the fix are under
*The replacement* below.

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

### The local synthetic probe (`soxr_hq`)

`TestINV18ResampleProbe` still runs a synthetic probe, round-tripped with
`soxr_hq` because torchaudio is absent by INV-13. With loudness normalisation it
reads 0.54–0.57 below the band. That is the coupling above: synthetic
flat-spectrum audio loses far more high-band energy to `soxr_hq` than LJSpeech
loses to the pin. It explains the earlier "configuration-sensitive, 0.50–0.69":

| loudness normalised | features | with StandardScaler | without |
|---|---|---|---|
| yes (pipeline) | 32 log band energies, band-limited | 0.5694 ± 0.0158 | 0.5458 ± 0.0034 |
| yes (pipeline) | log mean \|STFT\| < 8 kHz | 0.5514 ± 0.0056 | 0.5431 ± 0.0028 |
| no | 32 log band energies, band-limited | **0.5000 ± 0.0000** | **0.5000 ± 0.0000** |
| no | log mean \|STFT\| < 8 kHz | 0.4986 ± 0.0028 | **0.5000 ± 0.0000** |

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

### Ablation sweep filter response

`detectors.bandlimit.lowpass` used a zero-phase Butterworth order 8. It now uses
the analysis band limit's two-edge Chebyshev II design at every sweep cutoff
(`data.preprocess.band_limit_sos`): passband to cutoff − 100 Hz at ≤ 0.1 dB,
stopband from the cutoff at 100 dB per pass. At `LADDER_FMAX` the sweep point is
byte-identical to the primary analysis. Measured **directly** (impulse through
the filter, both passes), offsets relative to each cutoff:

| cutoff | old order | old −1 dB | old −3 dB | old −20 dB | old at cutoff | old residual | **new order** | **new −1 dB** | **new −3 dB** | **new −20 dB** | new at/above cutoff | new residual (f32) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1000 | 8 | −122 | −53 | **+145** | −6.0 dB | 2.0e-04 | 31 | −85.7 | −78.6 | −62.0 | ≤ −200 dB | 5.5e-16 |
| 2000 | 8 | −236 | −102 | **+276** | −6.0 dB | 3.1e-04 | 44 | −85.1 | −77.9 | −61.0 | ≤ −200 dB | 4.6e-16 |
| 3000 | 8 | −333 | −144 | **+380** | −6.0 dB | 2.6e-04 | 52 | −84.7 | −77.3 | −60.0 | ≤ −200 dB | 4.4e-16 |
| 4000 | 8 | −407 | −174 | **+450** | −6.0 dB | 2.1e-04 | 58 | −85.0 | −77.7 | −60.8 | ≤ −200 dB | 4.1e-16 |
| 5000 | 8 | −451 | −191 | **+480** | −6.0 dB | 1.6e-04 | 60 | −84.5 | −77.1 | −59.7 | ≤ −200 dB | 3.6e-16 |
| 6000 | 8 | −460 | −193 | **+471** | −6.0 dB | 1.3e-04 | 60 | −84.4 | −76.9 | −59.4 | ≤ −200 dB | 2.9e-16 |
| 7000 | 8 | −431 | −179 | **+426** | −6.0 dB | 5.6e-05 | 58 | −84.6 | −77.1 | −59.8 | ≤ −200 dB | 2.4e-16 |
| 8000 | 8 | −365 | −150 | **+349** | −6.0 dB | 3.0e-05 | 53 | −84.5 | −77.0 | −59.6 | ≤ −200 dB | 1.8e-16 |
| 9000 | 8 | −266 | −108 | **+248** | −6.0 dB | 1.7e-05 | 45 | −84.4 | −76.9 | −59.5 | ≤ −200 dB | 1.2e-16 |
| 10000 | 8 | −142 | −57 | **+130** | −6.0 dB | 7.1e-06 | 33 | −84.4 | −76.9 | −59.6 | ≤ −200 dB | 5.7e-17 |

(Offsets in Hz. Residual = fraction of total energy above the cutoff, harmonic
test signal, Blackman window.)

The Butterworth failed the sweep both ways it can. Its transition **straddled**
every cutoff: −6 dB at the cutoff itself and −20 dB only 130–480 Hz *above* it.
It also **attenuated without equalising**, leaving 7e-06 to 3e-04 of total energy
above the cutoff, 350× to 15000× the PCM_16 floor. A retrained detector could
still learn the band a cutoff was meant to remove, which pushes the measured
collapse point too high. That is the same flaw INV-17 rejected Butterworth for.

**Cost.** Order 31–60, depending on how wide 100 Hz is relative to the cutoff
(60 at 5–6 kHz, 31 at 1 kHz), against 8. The impulse response 99.99%-energy span
is 22–44 ms against 1.5–5.8 ms, longest at low cutoffs. Compute is 40–90 ms per
10 s of audio against ~13 ms. All of it is linear and identical for every member
of a sweep point, and negligible next to retraining a detector per cutoff.
Pinned by `TestBandlimit`, which also mutation-checks that Butterworth-8 fails
the leak assertion at every cutoff.

---

## The first v0 run, 200 LJSpeech utterances

Not committed here: the audio and its manifest live in the Kaggle working
directory (`resynth_v0.failed`). These are the numbers it produced, recorded
because two invariants moved because of them.

### Loudness: the coupling, measured on real audio

`btpvf loudness` on that run, per condition against real, paired by utterance:

| condition | gain delta | in-band offset | × probe resolution | high-band fraction delta |
|---|---|---|---|---|
| `vocos` | 0.184 dB | −0.0022 dB | 0.6 | +0.0010 |
| `bigvgan_112m` | 0.231 dB | **0.0506 dB** | 14.4 | −0.0064 |
| `griffin_lim` | 0.763 dB | **0.3018 dB** | 86.2 | −0.0324 |

The in-band offset tracks the high-band fraction delta across all three, which is
the coupling's signature: full-band normalisation turns a high-band energy
difference into a level offset inside the band-limited primary analysis. Vocos has
no high-band deficit and reads clean; Griffin-Lim has the largest and reads worst.

The gain delta and the in-band offset are different quantities, which is why the
report carries both: `vocos` received 0.184 dB more gain than real and still
landed level with it inside the band, because that gain was equalising an ordinary
level difference, not a bandwidth one.

**What changed.** INV-04's target is now defined on the band-limited signal. The
offset is zero by construction, the high-band analysis measures against a
low-band-matched level, and both loudness measurements stay on every manifest row
so `target − lufs_in` (the old rule's gain) remains computable. The archive must
be regenerated.

**What it closed, measured rather than argued.** Under the band-referenced rule
`band_offset_db` is ~1e-9 dB for all three conditions —
`band_offset_x_resolution` ~1e-6 — against 0.3018 dB and 86.2× for
`griffin_lim` under the old rule. That residual is the arithmetic of measuring a
gain and applying it, not a level difference: the in-band level is now the same
number for every condition by construction, which is what the change was for.

### The rms_level probe improved, and the prediction was the opposite

The expectation was that referencing the gain to the band would leave each
condition's FULL-band level free to differ, and that the broadband `rms_level`
probe would therefore read worse. Every value moved the other way — closer to
chance, which for a two-sided EER means the distance from 0.500 shrank:

| probe | condition | tier | full-band rule | band rule | distance from 0.500 |
|---|---|---|---|---|---|
| rms_level | `griffin_lim` | archive | 0.630 | **0.480** | 0.130 → 0.020 |
| rms_level | `griffin_lim` | zeroshot | 0.680 | **0.555** | 0.180 → 0.055 |
| rms_level | `bigvgan_112m` | archive | 0.565 | **0.530** | 0.065 → 0.030 |
| rms_level | `vocos` | archive | 0.495 | 0.500 | 0.005 → 0.000 |

**Why, and it is not a coincidence.** RMS and LUFS weight the high band very
differently. RMS is unweighted broadband energy, and the high band is only ~2% of
total energy (INV-17: real 0.01803), so RMS is dominated by in-band content and is
very nearly an in-band measure already. LUFS is K-weighted — a high-frequency
shelf — so it gives that same 2% far more say. Matching full-band LUFS across
conditions therefore had to push the in-band level around to compensate for the
high-band difference, and RMS, being in-band, saw that compensation directly.
Referencing the gain to the band aligns the normaliser's weighting with what RMS
mostly measures, so the compensation is gone.

The ordering says the same thing as the loudness table above, in a different
measurement: `griffin_lim` has the largest high-band deficit and the largest
improvement, `vocos` has none and does not move. The two are the same effect
seen through a K-weighted meter and through an unweighted one.

Recorded, not acted on. Every value was above the `SANITY_EER_FLOOR` of 0.40
under both rules, so no verdict changed either way — this is evidence about the
mechanism, not a gate result.

### Silence: griffin_lim trips the INV-12 probe, and the probe is misspecified

| probe | condition | tier | EER | floor |
|---|---|---|---|---|
| silence_duration | griffin_lim | archive | **0.350** | 0.40 |
| silence_duration | griffin_lim | zeroshot | **0.355** | 0.40 |

Trim spans are reference-derived (INV-03), so leading and trailing silence are
identical in LENGTH by construction. What differs therefore has to be the level
inside those regions. `btpvf silence-diag` on the run's own audio says that level
difference is **a tenth of a decibel**, and that the probe separates only at
thresholds where it is reading noise rather than silence.

#### The run refutes the synthetic diagnosis, by about a hundredfold

Level inside the regions the real reference calls silent, median dBFS:

| condition | median | vs real | synthetic prediction |
|---|---|---|---|
| `real` | −53.556 | 0.000 | 0.000 |
| `bigvgan_112m` | −53.524 | +0.033 | — |
| `griffin_lim` | −53.448 | **+0.108** | **+13.2** |
| `vocos` | −53.628 | −0.072 | — |

The synthetic corpus predicted +13.2 dB for `griffin_lim`; the real figure is
+0.108 dB, a factor of about 120. **There is no elevated noise floor.** The
phase-reconstruction-smearing hypothesis, which those synthetic numbers were
consistent with, is refuted on the real corpus.

The synthetic corpus overstated it because of how it was built: speech between
pads holding only a −65 dBFS room tone, with a broadband noise floor added to the
`griffin_lim` member. That *is* an elevated floor, injected by hand, and the
measurement recovered the size of the injection. Real Griffin-Lim inverts a mel
that already contains LJSpeech's own room tone, so what it reconstructs in those
regions lands within 0.1 dB of what real has. A synthetic corpus returns the
structure that was built into it; this is the same failure INV-18's second clause
exists for, in a diagnostic rather than in an exclusion.

**One real effect survives**, and it is in the trim pad rather than in the file's
silent regions — the pad is the 20 ms of pre-onset context INV-03 keeps, so it
holds the onset ramp, not room tone. That is why it sits 6.3 dB above the silent
regions (−47.265 dBFS against −53.556). Median level in the pad:

| condition | median | vs real |
|---|---|---|
| `real` | −47.265 dBFS | 0.000 |
| `bigvgan_112m` | −47.351 | −0.086 |
| `griffin_lim` | −46.378 | **+0.887** |
| `vocos` | −48.068 | −0.803 |

Roughly 1 dB, in a 40 ms region, and it is a level difference rather than a
duration one. It is now **measured on every run and gated by nothing** — see
*The trim-pad level is recorded and ungated* below.

#### The probe's threshold is −50 dBFS, which is inside the corpus floor

The probe's own feature, swept over all three conditions (real's column is the
same audio the gate reads):

| threshold | real | `griffin_lim` | `bigvgan_112m` | `vocos` |
|---|---|---|---|---|
| −60 dBFS | 0.006703 s | 0.001600 s / **0.290** | 0.005945 s / 0.435 | 0.005146 s / 0.445 |
| **−50 dBFS** | 0.015524 s | 0.010500 s / **0.370** | 0.015601 s / 0.500 | 0.015080 s / 0.480 |
| −40 dBFS | 0.040759 s | 0.039600 s / 0.460 | 0.041819 s / 0.530 | 0.042160 s / 0.510 |
| −30 dBFS | 0.088867 s | 0.092700 s / 0.540 | 0.087170 s / 0.495 | 0.090848 s / 0.535 |
| −20 dBFS | 0.247841 s | 0.278500 s / 0.535 | 0.241532 s / 0.495 | 0.262309 s / 0.525 |

(seconds of leading + trailing silence / EER against real.)

**The dip is specific to `griffin_lim`.** At −50 dBFS `bigvgan_112m` reads
0.015601 s against real's 0.015524 s — indistinguishable, EER 0.500 — and `vocos`
0.015080 s at 0.480. Only `griffin_lim` reads materially less silence, 0.010500 s.
So this is not a general property of putting audio through a vocoder, and not a
property of the pipeline: it is one condition's pad crossing a threshold that sits
inside the corpus's noise, which is the same +0.887 dB pad level showing up as a
duration.

The failing gate reads 0.350/0.355 against the −50 row's 0.370, which identifies
the threshold: `_leading_trailing_silence` defaults to **−50 dBFS**, an absolute
constant with no recorded derivation (see the note below).

**What the rows mean.** INV-03 keeps `TRIM_PAD_MS = 20 ms` of context at each
end, so a written file holds **2 × 20 = 40 ms** of leading plus trailing silence,
by construction and identically for every condition. Read against that bound:

- at −50 dBFS the probe reads **15.5 ms of the 40 ms pad**, so it is finding
  "voiced" samples *inside* the pad — which can only be the pad's own noise;
- at −40 dBFS it reads **40.8 ms**: the pad, and nothing else;
- at −30 dBFS, 88.9 ms — more than twice the pad, so it is into speech onsets.

The separation lives entirely at or below −50 dBFS, where the probe reads noise,
and is gone at −40 dBFS, where it reads the pad. With the corpus's silent regions
sitting at −53.5 dBFS, a −50 dBFS threshold is ~3 dB inside the floor, so the
feature is a statistic of how often the signal's own noise crosses a level near
its own floor — not a silence duration. That is a specification error whichever
way the gate then reads.

**`reference_silent_fraction` is not a fourth measurement.** It reads 0.172786 on
every condition's row because it is computed from `real` alone, by INV-03's
criterion, and applied by index. It also counts silence **anywhere** in the file,
inter-word pauses included, so it is not comparable with the probe's 15 ms: the
probe reads inward from the two ends and never looks at the interior. The
quantity that *is* comparable with it is `2 × TRIM_PAD_MS`.

**Where −50 dBFS came from.** Nowhere recorded. It entered as a bare default
argument on `_leading_trailing_silence` in the first commit (`ac0cb5a`, the bulk
"Phase A pipeline" commit) and was never edited afterwards. It was not in
`data/invariants.py`, not in INV-03's or INV-12's constants, not in
`btp-project-idea.md`, and no measurement in this file or in
`docs/enforcement_audit.md` compared it to a corpus noise floor. The underlying
mismatch is that INV-03 defines silence *relatively* — `TRIM_TOP_DB = 30 dB`
below the loudest frame — while the probe that checks INV-03 defined it
*absolutely*. That the two agree on any particular corpus was never checked, and
on LJSpeech they did not. The default is gone: `_leading_trailing_silence` now
requires the threshold, so no caller can obtain one without choosing it.

#### The replacement: a threshold derived from the measured floor

Implemented (INV-12). Per dataset and per tier, from the **real reference alone**:

```
floor     = 99.9th percentile of |sample| inside the quietest 10% of real's RMS frames
threshold = floor + 8 dB
```

**Why a sample-peak floor.** The probe compares `|sample|` to the threshold. For a
noise-like floor the p99.9 of `|x|` sits **+10.35 dB** above its RMS, and the
maximum over a corpus sits **+14.5 dB** (simulated: +13.1 dB over 200 × 441 pad
samples, +14.5 dB over full silent regions). A floor quoted as an RMS would hide
that `sqrt(2 ln N)` term inside the margin, where it scales with region length and
sample rate — so the same constant would mean something different on another tier.

**Why 8 dB.** Clearing the corpus-wide maximum of a noise-like floor above its own
p99.9 needs 3.2–4.1 dB; 6 dB is about twice that; 8 dB puts the threshold near the
middle of the band LJSpeech admits rather than at its edge. The margin comes from
the noise statistics, and the band below is used to check where it lands, not to
derive it.

**Why the quietest decile rather than `reference_silent_mask`.** INV-03's relative
criterion (`TRIM_TOP_DB` below the loudest frame) admits quiet speech — its median
is −53.5 dBFS, essentially its own cutoff. The loudest samples of a "quiet" band
are not a floor.

**The band this corpus admits.** INV-03 writes `2 × TRIM_PAD_MS` = 40 ms of
leading plus trailing silence into every file. Reading real's sweep column against
that bound and interpolating gives the thresholds at which the probe reads the pad
and nothing else:

| threshold | real reads | vs the 40 ms pad |
|---|---|---|
| −50 dBFS | 15.5 ms | 0.39× — inside the pad's noise |
| **−43 dBFS** | ~30 ms | 0.75× — lower edge |
| −40 dBFS | 40.8 ms | **1.02× — the pad, and nothing else** |
| **−37 dBFS** | ~50 ms | 1.25× — upper edge |
| −30 dBFS | 88.9 ms | 2.2× — into speech onsets |

**The first implementation got the aggregation wrong, and the pad check caught
it.** The rule above says "of |sample| inside the quietest 10% of real's RMS
frames" — but *whose* frames. The gate concatenated all 200 files' quiet samples
and took one p99.9 of the pool; the diagnostic took each file's p99.9 and the
median across files. Those are different quantities, and on the real run they
differed by **7.05 dB**: −39.808 dBFS pooled against −46.854 dBFS per-file.

The pooled version is the level exceeded by 0.1% of all quiet samples *anywhere
in the corpus*, so its upper tail is supplied by whichever files have the loudest
quiet moments — and it climbs as files are added. Measured on nested synthetic
corpora with a 5 dB spread of per-file floors, through the real code path:

| n (nested subsets) | median of per-file p99.9 | pooled p99.9 |
|---|---|---|
| 8 | −50.10 dBFS | −48.07 dBFS |
| 40 | −50.77 | −44.72 |
| 120 | −51.03 | −45.47 |
| 240 | −51.88 | **−43.01** |

The median drifts 1.8 dB downward across a 30× change in n — ordinary sampling
noise on a converging estimator. The pooled figure climbs **5.1 dB**. On
identical files the two agree to 0.1 dB, which is exactly why the suite did not
catch it: the fixtures were homogeneous and n = 12, and the defect only appears
with a heterogeneous corpus at scale. **A statistic whose meaning depends on how
much was measured** is the `ASVSPOOF_CROP_SAMPLES` failure again, in a new place.

The consequence on the run: threshold −31.808 dBFS, about 7 dB into speech, where
real read **79.2 ms** against the 40 ms INV-03 leaves — 98% off, so the probe
raised. Worth stating plainly: at that threshold every condition read ~0.5 EER,
so **the gate would have passed** on a threshold that was not measuring silence.
The pad-agreement check is what refused it, and this is the case it was added for.

**Fixed: per file first, then median across files.** `quiet_stats` computes one
file's two numbers and is now the single implementation — `silence_floor_dbfs`
takes the median of it across files, and `silence_diagnostic` reports it per
condition, so the two paths cannot disagree again.

Median rather than a lower percentile: the threshold is one number applied to
every file, so the floor it clears should be the one a *typical* file's quiet
frames reach. A low percentile would target the quietest files and push the
threshold under the noisier ones, whose pads would then read as voiced. The
median has a 50% breakdown point, so a minority of noisy files cannot move it,
and it does not move with n. The spread *above* the median is what the margin
covers, and a spread too wide for the margin is what the pad check catches.

**The margin is unchanged, and the +7.4 dB figure does not argue against it.**
The per-file floor sits 7.42 dB above `quiet_frame_rms_cut_dbfs`, against the
+10.35 dB a Gaussian floor gives — but those are not the same ratio. The cut is
the *10th percentile of frame RMS*, an upper bound on the masked samples' own
RMS, not that RMS. The masked samples sit below it, so +7.4 dB against the cut is
consistent with ~+10 dB against their own level. Nothing in the margin's
derivation changes.

**What the corrected threshold is predicted to read.** Computed from the
committed files, because the run's audio is not in this repository:

| | |
|---|---|
| corrected floor (`silence_floors.csv`, real) | −46.854 dBFS |
| + `SILENCE_FLOOR_MARGIN_DB` | **−38.854 dBFS** |
| inside the −43…−37 band the sweep admits | yes |
| real's pad, interpolated from its own sweep column | 44.7 ms (log-linear) to 46.1 ms (linear) |
| against `2 × TRIM_PAD_MS` = 40 ms | 11.8% to 15.4% off, inside the 25% tolerance |

A prediction, not a measurement: it is an interpolation between the −40 and −30
dBFS sweep rows. The next run reports the real figure, and if it lands outside
the tolerance the run stops and says by how much.


**The pad-agreement check is what makes it safe, and it raises.** At the derived
threshold real must read `2 × TRIM_PAD_MS` within 25%, or the probe stops. Below
the band it is counting the corpus's noise as signal; above it, eating speech
onsets. A silence EER measured at either is a number nobody can interpret, which
is worse than no number. The rule therefore carries a range requirement: a corpus
whose floor is not far enough below speech is **refused rather than measured**
(`TestDerivedSilenceThreshold` builds one and pins the refusal).

**What it does and does not decide.** Across the whole admissible band the
`griffin_lim` silence EER is ≥ 0.43, so the gate passes there. That is a
consequence of fixing the specification, not its justification: the calibration
uses real alone plus `TRIM_PAD_MS`, both fixed before any condition existed, and
the diagnosis found no floor difference (+0.108 dB) for the old threshold to have
been detecting. `griffin_lim` is still not marked `expected_by_design` and
`SANITY_EER_FLOOR` is unchanged.

#### The trim-pad level is recorded and ungated

The derived threshold sits above the pad's noise by design, so the silence probe
can no longer see a level difference inside the pad. Nothing else covers it:
`rms_level` reads whole-file RMS, which is dominated by speech, and the pad is
40 ms of a multi-second file. So the measurement is kept as a column,
`trim_pad_above_real_db`, on every silence row of the sanity CSV — +0.887 dB for
`griffin_lim`, −0.086 for `bigvgan_112m`, −0.803 for `vocos` — and it gates
nothing.

Ungated because a gate needs a floor and there is no basis here for choosing one.
What choosing one would take, in order:

1. **A null distribution.** The pad level's spread across utterances *within* a
   condition, and across independent real recordings, so "how big is big" has an
   answer that is not an intuition. `vocos` reads −0.803 dB and `griffin_lim`
   +0.887 dB; without a spread those are two numbers, not two effects.
2. **A demonstration that it is learnable.** A probe on the pad level alone,
   grouped by utterance like every other paired probe (INV-18), showing it
   separates real from a condition. A 1 dB difference in 40 ms may well sit below
   what any detector can use, in which case it is a curiosity rather than a
   confound.
3. **A cause.** Onset smearing in phase reconstruction predicts `griffin_lim`
   high; it does not predict `vocos` 0.8 dB low. Until the sign pattern is
   explained, a floor would be fitted to three points.

Only then is there something to gate, and the floor would come from (1) rather
than from the values in (2). Recording it now costs nothing and means the next
run's numbers are already there when someone does the work.

**Not decided.** `btpvf silence-diag --manifest <run>/manifest.csv` produces the
tables above, plus the calibration record, from a completed run.

---

## Claims that rest on these files

| claim | where | evidence |
|---|---|---|
| Griffin-Lim emits no high-band content | CLAUDE.md INV-17, INV-18; `ANALYSIS_EXCLUSIONS` | `griffin_lim` `hb_8k_nyq` max 5.49e-08 vs `real` min 6.38e-04 |
| Vocos carries a resampler signature above `LADDER_FMAX` | CLAUDE.md INV-18; `ANALYSIS_EXCLUSIONS` | [10900, 11025) energy: `vocos` 8.63e-07 vs `bigvgan` 7.30e-09; AUROC 0.9997 vs bigvgan, **0.507 vs `real_roundtrip`**; pinned kaiser above-band 0.9935, 0.9989 normalised (`probes.json`) |
| the torchaudio default resampler costs 18–20% of the high band | `vocoders.vocos.VOCOS_RESAMPLER.justification` | CSV `real_roundtrip` / `real` = 0.8198; `probes.json` 0.8005 |
| the kaiser pin keeps the high band | `VOCOS_RESAMPLER.justification` | `probes.json` `real_vs_rt_kaiser` ratio 0.9997 |
| **Vocos belongs in the band-limited primary analysis** | CLAUDE.md INV-18; btp-project-idea.md Part 2 | `probes.json` kaiser below 0.5000 ± 0.0000, raw and loudness-normalised, with above-band controls 0.9935 / 0.9989; default below 0.5291 / 0.5348 as same-band control |
| the pinned resampler's loudness coupling is below this probe's resolution, with margin | INV-18; `loudness_coupling` | `_norm` gain deltas: kaiser 4.62e-05 dB, 33× below the soxr null (0.001534), 646× below the default detection (0.02986) |
| a uniform round trip would not equalise Vocos | `REJECTED_ALTERNATIVES` | `probes.json` `kaiser_1x_vs_2x` above 0.9928, full 0.9918; 0.9941 above normalised |
| the ablation sweep cannot leak across a cutoff | `detectors.bandlimit.lowpass`; CLAUDE.md INV-17 | measured response above; `TestBandlimit` |
| BigVGAN v2 fmax8k checkpoint identity and mel config | `vocoders.bigvgan.LARGE_SPEC` | `meta.json` |
| BigVGAN produces ~80% of real's high-band energy | [`mel_configs.md`](mel_configs.md) | `bigvgan` / `real` mean `hb_8k_nyq` = 0.8055 |
