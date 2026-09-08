# Mel front-end audit

Every row below was read from the checkpoint's **own** config file or, where the
model ships no JSON config, from the **source that constructs the front-end**.
Nothing here is from a paper or from memory. Fetched 2026-09-07, Vocos added
2026-09-08.

This table is the authority for `LADDER_FMAX` (INV-17) and for the `MelConfig`
entries in [`src/data/mel.py`](../src/data/mel.py). If a row changes, the ladder
band changes, and every condition is regenerated.

## The table

| Condition | Role | n_fft | hop_size | win_size | n_mels | fmin | fmax | sampling_rate | Status |
|---|---|---|---|---|---|---|---|---|---|
| `griffin_lim` | ladder | 1024 | 256 | 1024 | 80 | 0.0 | *free — ours to set* | 22050 | N/A (no checkpoint) |
| `melgan` | ladder | 1024 | 256 | 1024 | 80 | 0.0 | `None` → **11025** | 22050 | VERIFIED |
| `hifigan_v1` | ladder | 1024 | 256 | 1024 | 80 | 0 | **8000** | 22050 | VERIFIED |
| `vocos` | ladder | 1024 | 256 | 1024\* | **100** | 0.0\* | `None` → **12000**\* | **24000** | VERIFIED |
| `bigvgan_base` | ladder | 1024 | 256 | 1024 | 80 | 0 | **8000** | 22050 | VERIFIED |
| `bigvgan_112m` | ladder | 1024 | 256 | 1024 | 80 | 0 | **8000** | 22050 | VERIFIED |
| `melgan_fullband` | control (exempt) | 1024 | 256 | 1024 | 80 | 0.0 | **11025** | 22050 | VERIFIED |
| `bigvgan_v2_22khz_fullband` | control (exempt) | 1024 | 256 | 1024 | 80 | 0 | `null` → **11025** | 22050 | VERIFIED |
| `specdiff_gan` | **unavailable** | 1024 | 256 | 1024 | 80 | 0 | **8000** | 22050 | config VERIFIED, **weights never released** |

\* Vocos passes none of `win_length`, `f_min` or `f_max` to torchaudio, so the
library defaults apply: `win_length = n_fft`, `f_min = 0.0`, `f_max = sr/2`.

## Sources

| Condition | Source |
|---|---|
| `griffin_lim` | No checkpoint exists. Analysis/synthesis with no learned prior, so its front-end is a **project decision, not a model property**. Set to `LADDER_FMAX` so it neither constrains nor escapes the ladder band. |
| `melgan`, `melgan_fullband` | [`descriptinc/melgan-neurips`](https://github.com/descriptinc/melgan-neurips) → `mel2wav/modules.py`, `Audio2Mel.__init__` defaults. Confirmed against `scripts/train.py`, which instantiates `Audio2Mel(n_mel_channels=args.n_mel_channels)` — only `n_mel_channels` is passed. Both conditions are the **same checkpoint**. |
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

## Derived value

```
LADDER_FMAX = min(fmax over constraining primary-ladder conditions)
            = min(11025, 8000, 12000, 8000, 8000)   # melgan, hifigan_v1, vocos,
            = 8000.0                                 # bigvgan_base, bigvgan_112m
```

Excluded from the `min`:
- `griffin_lim` — front-end is ours to choose; set *to* the result.
- `specdiff_gan` — unavailable, never generated.
- `melgan_fullband`, `bigvgan_v2_22khz_fullband` — INV-17 exempt by design;
  including them would drag the band to 11025 and reinstate the cliff they exist
  to measure.

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
| `griffin_lim` | *follows* | no — set *to* the result |

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
| `melgan_fullband` | same file as `melgan` | *must match `melgan`* |

To record one: run `vocoders.checkpoints.fetch_and_verify(path, spec,
allow_first_use=True)` once, paste the printed digest into the spec's
`checkpoint_sha256`, and commit before generating any audio. `btpvf audit`
reports these conditions blocked until then.
