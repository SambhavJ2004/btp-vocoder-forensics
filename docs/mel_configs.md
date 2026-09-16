# Mel front-end audit

Every row below was read from the checkpoint's **own** config file or, where the
model ships no JSON config, from the **source that constructs the front-end**.
Nothing here is from a paper or from memory. Fetched 2026-09-07, Vocos added
2026-09-08.

This table is the authority for `LADDER_FMAX` (INV-17) and for the `MelConfig`
entries in [`src/data/mel.py`](../src/data/mel.py). If a row changes, the
analysis band changes.

**`fmax` here is what each model was TOLD, not what it EMITS.** Those differ for
every condition except Griffin-Lim — see *Measured bandwidth* below, which
overturned the original INV-17. Do not reason from this table to a claim about
generated audio without measuring.

## The table

| Condition | Role | n_fft | hop_size | win_size | n_mels | fmin | fmax | sampling_rate | Status |
|---|---|---|---|---|---|---|---|---|---|
| `griffin_lim` | ladder | 1024 | 256 | 1024 | 80 | 0.0 | *free — ours to set* | 22050 | N/A (no checkpoint) |
| `melgan` | ladder | 1024 | 256 | 1024 | 80 | 0.0 | `None` → **11025** | 22050 | VERIFIED |
| `hifigan_v1` | ladder | 1024 | 256 | 1024 | 80 | 0 | **8000** | 22050 | VERIFIED |
| `vocos` | ladder | 1024 | 256 | 1024\* | **100** | 0.0\* | `None` → **12000**\* | **24000** | VERIFIED |
| `bigvgan_base` | ladder | 1024 | 256 | 1024 | 80 | 0 | **8000** | 22050 | VERIFIED |
| `bigvgan_112m` | ladder | 1024 | 256 | 1024 | 80 | 0 | **8000** | 22050 | VERIFIED |
| `bigvgan_v2_22khz_fullband` | control | 1024 | 256 | 1024 | 80 | 0 | `null` → **11025** | 22050 | VERIFIED |
| `specdiff_gan` | **unavailable** | 1024 | 256 | 1024 | 80 | 0 | **8000** | 22050 | config VERIFIED, **weights never released** |

\* Vocos passes none of `win_length`, `f_min` or `f_max` to torchaudio, so the
library defaults apply: `win_length = n_fft`, `f_min = 0.0`, `f_max = sr/2`.

## Sources

| Condition | Source |
|---|---|
| `griffin_lim` | No checkpoint exists. Analysis/synthesis with no learned prior, so its front-end is a **project decision, not a model property**. Set to `LADDER_FMAX` so it neither constrains nor escapes the ladder band. |
| `melgan` | [`descriptinc/melgan-neurips`](https://github.com/descriptinc/melgan-neurips) → `mel2wav/modules.py`, `Audio2Mel.__init__` defaults. Confirmed against `scripts/train.py`, which instantiates `Audio2Mel(n_mel_channels=args.n_mel_channels)` — only `n_mel_channels` is passed. |
| `hifigan_v1` | [`jik876/hifi-gan`](https://github.com/jik876/hifi-gan) → `config_v1.json` |
| `vocos` | [`charactr/vocos-mel-24khz`](https://huggingface.co/charactr/vocos-mel-24khz) → `config.yaml` (`sample_rate: 24000, n_fft: 1024, hop_length: 256, n_mels: 100, padding: center`), plus [`gemelo-ai/vocos`](https://github.com/gemelo-ai/vocos) → `vocos/feature_extractors.py`, `MelSpectrogramFeatures`, which constructs `torchaudio.transforms.MelSpectrogram` **without** `f_min` or `f_max`. |
| `bigvgan_base` | [`nvidia/bigvgan_base_22khz_80band`](https://huggingface.co/nvidia/bigvgan_base_22khz_80band) → `config.json` |
| `bigvgan_112m` | [`nvidia/bigvgan_v2_22khz_80band_fmax8k_256x`](https://huggingface.co/nvidia/bigvgan_v2_22khz_80band_fmax8k_256x) → `config.json` |
| `bigvgan_v2_22khz_fullband` | [`nvidia/bigvgan_v2_22khz_80band_256x`](https://huggingface.co/nvidia/bigvgan_v2_22khz_80band_256x) → `config.json` |
| `specdiff_gan` | [`SpecDiff-GAN/SpecDiff-GAN`](https://github.com/SpecDiff-GAN/SpecDiff-GAN) → `configs/config_ljspeech.json`. Retained for the audit trail only. |

## Notes that matter

**`null` / `None` fmax resolves to Nyquist, not to "unbounded".** MelGAN, the
full-band BigVGAN v2, and Vocos all leave fmax unset. BigVGAN's `meldataset.py`
passes it straight to `librosa_mel_fn` and documents the `sr / 2.0` default;
MelGAN's `Audio2Mel` reaches librosa the same way; Vocos reaches torchaudio,
whose `MelSpectrogram` defaults `f_max=None → sample_rate // 2`. So the
effective values are 11025, 11025 and **12000**. Code reading these configs must
resolve `null` rather than treat it as missing.

**Vocos breaks the pattern the other five share.** Five ladder configs are
numerically identical (1024/256/1024, 80 mels, fmin 0, 22050 Hz) and differ only
in `fmax`. Vocos differs in sample rate (24000), mel band count (100) and fmax
(12000). This partly **re-widens INV-02**: "same numbers, different extractors"
describes five conditions and is false of the sixth. Vocos also takes one
resample (24000 → 22050) on the way to the archive, which no other condition
does — permitted by INV-01 as a single native→archive step, and recorded in the
manifest's `source_rate`.

**The 7600 Hz risk did not materialise, but not for the expected reason.** The
`descriptinc` MelGAN release uses the librosa default, 11025, not 7600. Other
LJSpeech pipelines of that era (ESPnet, `kan-bayashi/ParallelWaveGAN`) do use
7600. **If the MelGAN condition is ever re-sourced from one of those, this row
changes to 7600 and `LADDER_FMAX` drops with it**, requiring every condition to
be regenerated.

**SpecDiff-GAN was replaced, not deleted.** Its repo ships `configs/` but no
pretrained weights; inference expects a user-supplied `--checkpoint_file`. Vocos
took the rung. The row stays here so the substitution is an auditable decision
rather than an unexplained gap, and so Vocos can be compared against what it
replaced.

## Measured bandwidth — generated audio, not config

**Everything above this section is config-derived. This section is measured, and
where the two disagree the measurement wins.** Committed evidence lives in
[`data/measurements/`](../data/measurements/) and is described in
[`measurements.md`](measurements.md); citations to it are recomputed by the test
suite (INV-18).

### Committed: 60 LJSpeech files, PCM_16 quantised

Source: `data/measurements/band_measurements.csv`. Fraction of total energy in
[8000, 11025) Hz (`hb_8k_nyq`) and in the resampler-edge sliver [10900, 11025) Hz
(`hb_8k_nyq − hb_8k_rtedge`, with `rt_edge = 10900` from `meta.json`).

| condition | `hb_8k_nyq` mean | ratio to real | [10900, 11025) mean |
|---|---|---|---|
| real | 0.020169 | 1.0000 | 2.24e-08 |
| real_roundtrip (torchaudio default) | 0.016535 | **0.8198** | 9.03e-07 |
| BigVGAN v2 fmax8k | 0.016246 | 0.8055 | 7.30e-09 |
| Vocos | 0.016216 | 0.8040 | **8.63e-07** |
| Griffin-Lim | median 5.1e-09, max 5.5e-08 | ~0 | ~0 |

Separability on the [10900, 11025) sliver (unpaired AUROC, orientation-free):

| pair | AUROC | reads as |
|---|---|---|
| Vocos vs BigVGAN | **0.9997** | Vocos carries something BigVGAN does not |
| real vs real_roundtrip | 0.9928 | a resample round trip leaves this imprint |
| **Vocos vs real_roundtrip** | **0.507** | and Vocos's imprint IS that one — the resampler |

### Earlier: 20 LJSpeech files

The first measurement, which overturned the original INV-17. Superseded in
precision by the 60-file set above, consistent with it in direction.

| condition | >8 kHz fraction |
|---|---|
| real | 0.01803 |
| BigVGAN | 0.01480 |
| Griffin-Lim | 0.00000 **to five decimals** — see below |

**Griffin-Lim is not exactly zero.** "0.00000" was a five-decimal table. The
60-file measurement gives median 5.1e-9 and max 5.5e-8: the PCM_16 quantisation
floor on no signal content, with 4 of 60 files above `PCM16_OOB_FLOOR` (2e-8)
itself. The structural claim holds — real's **minimum** (6.38e-4) is four orders
of magnitude above Griffin-Lim's **maximum** — but "exactly zero" does not.

### Committed: resample probes, 200 LJSpeech files

Source: `data/measurements/probes.json`. `GroupKFold(5)` by utterance, PCM_16.
The 60-file figures reported earlier are in no artifact and differ; they are
listed for the record, not used.

| round trip vs direct | ratio (8 kHz–Nyquist) | AUROC below 8 kHz | above | earlier 60-file report |
|---|---|---|---|---|
| torchaudio default | 0.8005 | **0.5291 ± 0.0024** | 1.0000 | ratio 0.8198 (**verified** in the CSV above) |
| torchaudio kaiser (width 64, rolloff 0.9999, beta 14.7697) | **0.9997** | **0.5000 ± 0.0000** | 0.9935 | 0.9998 / 0.9958 |
| soxr (quality not recorded) | 0.9922 | 0.5000 ± 0.0000 | 1.0000 | soxr_vhq 0.9936 / 1.0000 |
| one kaiser round trip vs two | 0.9998 | 0.5000 ± 0.0000 | 0.9928 | 0.9903 |
| through an 8 kHz band limit | — | — | — | 0.502 (no counterpart) |

Exactly 0.5000 ± 0.0000 is a degenerate probe, not noisy chance. It is read only
next to its above-band control; see [`measurements.md`](measurements.md).

### What this overturned

INV-17 originally low-passed every condition at generation, on the config-derived
premise that an `fmax = 8000` mel front-end yields no output above 8 kHz.
**That premise is false.** `fmax` constrains the *analysis* a vocoder consumes,
not the *synthesis* it performs: a time-domain upsampling vocoder produces
content across the full band whatever the mel carried. BigVGAN has `fmax = 8000`
and emits 0.01480 above 8 kHz on 20 files, 0.016246 on 60.

Griffin-Lim is the sole exception and the source of the error. It inverts the mel
to a linear spectrogram and runs ISTFT, so it structurally cannot exceed `fmax`.
The cliff was measured on Griffin-Lim, then generalised by reading configs to
conditions where it does not hold.

### The rule this produces

**A config-derived bandwidth expectation must be validated against generated
audio before anything is built on it.** Reading `fmax` from a config tells you
what the model was *told*. It does not tell you what the model *emits*, and for
every condition in this ladder except Griffin-Lim those are different things.

The failure was expensive in a specific way: the band that was filtered away is
the band above `LADDER_FMAX`, where the mel carried nothing and the vocoder's
output is therefore *hallucinated*. That is the most forensically interesting
content the archive holds, and it was discarded to remove a cliff that only one
condition actually had.

Concretely, before relying on a bandwidth claim:

1. Generate audio from the condition.
2. Measure the energy fraction above the band in question
   (`data.invariants.out_of_band_energy`).
3. Compare against real on the same utterances, per file, not in aggregate —
   BigVGAN tracking real *per file* is stronger evidence than a matching mean.
4. Only a synthesis method that reconstructs through an inverse transform of the
   mel itself (Griffin-Lim, and any future ISTFT-style condition) should be
   expected to show a hard zero.

### The same lesson, twice more

The config-to-audio rule above has now been confirmed by two further cases where
reasoning pointed one way and measurement another:

- **Vocos's resampler.** Argued erased by the (then generation-time) 8 kHz band
  limit. True below the band. False in a full-band archive, where Vocos is
  indistinguishable from resampled real audio at [10900, 11025) Hz.
- **Griffin-Lim's zero.** Read as exact from a rounded table; it is the
  quantisation floor.

Both became INV-18: analysis membership is decided per band, and every exclusion
names a committed measurement.

### Where the numbers are used

**Per-analysis membership (INV-18),** in `invariants.ANALYSIS_EXCLUSIONS`:

| condition | primary (band-limited to 8 kHz) | high-band [8 kHz, Nyquist) | evidence |
|---|---|---|---|
| `griffin_lim` | **included** | excluded | GL max 5.49e-08 vs real min 6.38e-04 |
| `vocos` | **included** | excluded | [10900, 11025) energy 8.63e-07 vs BigVGAN 7.30e-09; AUROC 0.9997 vs BigVGAN, 0.507 vs real_roundtrip |
| every other rung | included | included | — |

Griffin-Lim is back **in** the primary correlation: its cliff is above the band
the primary analysis keeps. Vocos is in it because the pinned resampler reads
0.5000 below the band with a 0.9935 above-band control (committed, 200 files; see
INV-18 and [`measurements.md`](measurements.md)).

**The resampler pin,** `vocoders.vocos.VOCOS_RESAMPLER`, cites the committed
real_roundtrip / real ratio (0.8198) for the cost of the torchaudio default, and
`probes.json` for the pin's benefit (ratio 0.9997) and for why it matters to the
primary analysis (below-band 0.5000 against the default's 0.5291).

`metrics.spectral.high_band_distance` exists because BigVGAN's 0.01480 answers
only *how much* high-band energy there is. Whether the **content** is right is a
separate and better question — right amount with wrong structure is a strong
detection cue that an energy fraction cannot see.

Phase A records `high_band_fraction` per file in the manifest, so this
measurement is regenerated with every dataset rather than living only here.

## Derived value

```
LADDER_FMAX = min(fmax over constraining primary-ladder conditions)
            = min(11025, 8000, 12000, 8000, 8000)   # melgan, hifigan_v1, vocos,
            = 8000.0                                 # bigvgan_base, bigvgan_112m
```

Excluded from the `min`:
- `griffin_lim` — front-end is ours to choose; set *to* the result.
- `specdiff_gan` — unavailable, never generated.
- `bigvgan_v2_22khz_fullband` — a paired control whose wider fmax is the variable
  it exists to isolate.

**`LADDER_FMAX` did not move** when Griffin-Lim was reclassified as a floor
reference: it was already excluded from the `min` as an fmax-free condition, so
the value is unchanged at 8000.0.

**What `LADDER_FMAX` now means.** It is no longer a delivery cutoff — the archive
is full-band. It is the **analysis band**: the frequency above which at least one
ladder condition had no mel information, so anything it emits above that is
invented. That makes it the natural boundary for
`metrics.spectral.high_band_distance` and the natural centre for the
band-limited sweep.

Implemented as `data.invariants._derive_ladder_fmax()`, reading
`AUDITED_MEL_FMAX` — which is this table, transcribed. Change a row here and the
ladder band follows; there is no literal to forget to update.

**Adding Vocos did not move the band.** Its `fmax` of 12000 was never the
minimum. Because INV-17 equalises the delivered band, the ladder is
**fmax-agnostic**: a replacement rung may have any native band, and only one
coming in *below* 8000 would move `LADDER_FMAX`.

### Which conditions currently set the value

`LADDER_FMAX = 8000` is set **jointly by three conditions**, all tied at the
minimum. No single one determines it, which means removing any one of them
changes nothing — and removing all three moves the band by 3 kHz.

| Condition | fmax | Is it setting the band? |
|---|---|---|
| `hifigan_v1` | **8000** | **yes — tied at the minimum** |
| `bigvgan_base` | **8000** | **yes — tied at the minimum** |
| `bigvgan_112m` | **8000** | **yes — tied at the minimum** |
| `melgan` | 11025 | no — 3025 Hz of headroom |
| `vocos` | 12000 | no — 4000 Hz of headroom |
| `griffin_lim` | *follows* | no — set *to* the result (and now a floor reference, excluded from the correlation) |

What the band would become if the tied conditions were removed:

| Remove | New `LADDER_FMAX` | Effect |
|---|---|---|
| any one of the three | **8000** (unchanged) | the other two still hold it |
| any two of the three | **8000** (unchanged) | the remaining one still holds it |
| **all three** | **11025** | +3025 Hz — every condition regenerates |
| `melgan` or `vocos` | **8000** (unchanged) | they were never the minimum |

**The asymmetry is the thing to notice.** Dropping a *non*-tied condition is
free. Dropping the whole tied group — the plausible version of which is
retiring HiFi-GAN and both BigVGAN rungs together, e.g. moving the ladder to
newer checkpoints — silently raises the band to 11025 and invalidates every
generated file. Conversely, **adding** a condition below 8000 lowers the band
and invalidates everything just as thoroughly; the live risk there is
re-sourcing MelGAN from a ParallelWaveGAN/ESPnet LJSpeech recipe at
`fmax = 7600` (see above).

Neither direction announces itself: `_derive_ladder_fmax()` simply returns a
different number and the pipeline carries on. Before changing the ladder's
membership, recompute this table.

## Checkpoint pins (INV-08)

Two mechanisms, because the ladder has two kinds of checkpoint. See
`vocoders.checkpoints`.

**HuggingFace revision SHAs** — content-addressed over the whole repo,
verifiable without downloading. Pass explicitly at load time; omitting the
revision tracks the branch head.

| Condition | Repo | Revision |
|---|---|---|
| `vocos` | `charactr/vocos-mel-24khz` | `0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21` |
| `bigvgan_base` | `nvidia/bigvgan_base_22khz_80band` | `760f0c694aba68b8c200dd52ffd3b4ecafa7028c` |
| `bigvgan_112m` | `nvidia/bigvgan_v2_22khz_80band_fmax8k_256x` | `a9d8b711c344f3c49cc1a05d0c8b6741cf733aa8` |
| `bigvgan_v2_22khz_fullband` | `nvidia/bigvgan_v2_22khz_80band_256x` | `633ff708ed5b74903e86ff1298cf4a98e921c513` |

**Weight-file SHA-256** — for checkpoints with no revision to pin. Computed
after download, compared on every subsequent load, raises on mismatch.

| Condition | Source | Digest |
|---|---|---|
| `hifigan_v1` | `jik876/hifi-gan` LJ_V1 (Google Drive) | *not yet recorded* |
| `melgan` | `descriptinc/melgan-neurips` (torch.hub) | *not yet recorded* |

To record one: run `vocoders.checkpoints.fetch_and_verify(path, spec,
allow_first_use=True)` once, paste the printed digest into the spec's
`checkpoint_sha256`, and commit before generating any audio. `btpvf audit`
reports these conditions blocked until then.
