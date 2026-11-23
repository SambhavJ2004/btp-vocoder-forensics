# VocoRoute-P Phase N1 CPU prototype

`vocoroute-p-n1-v1` is a tensor-only prototype with random initial weights and
caller-provided routing scores. It has no data loader, trainer, checkpoint,
forensic adapter, or audio writer. Its versioned frontend is local to `vocoroute`.
The prototype tests exercise synthetic in-memory signals. Results describe
reconstruction and execution, with no speech-quality or latency claim.

## Tensor and reconstruction contract

The frontend takes a batch of unpadded real mono waveforms `[B,L]` at 22,050 Hz.
It uses a 1,024-sample periodic Hann window, FFT size 1,024, and hop 256.
Analysis is centered with zero padding and uses an unnormalized, one-sided FFT.
The transform yields `[B,513,T]` complex values with `T = 1 + L // 256`.
Conditioning applies 80 area-normalized Slaney mel filters over 0–11,025 Hz to
spectral power and takes the natural logarithm after flooring at `1e-5`.
The matching iSTFT uses the same window and requests the original sample length.
It projects DC and Nyquist onto the real axis before synthesis. Every row in a
batch shares the same explicit sample length.

The base maps `[B,80,T]` through a 128-channel temporal stem, six dilated residual
blocks, and a per-frame head. The head emits an interior log magnitude and phase
for each of 511 bins plus two signed real endpoints. The expert sees the base
log magnitude, phase sine and cosine, sine and cosine of adjacent time and
frequency phase differences, and normalized frequency. Weak-bin phases and
missing phase differences contribute zero features. The expert predicts log gain
bounded by `±0.35` and phase offset bounded by `±π/4`.

Tile IDs run in time-major order across `16×32` cores. Stable descending sort
breaks score ties in ascending tile order. Each core grows to `20×40` with two
extra frames and four extra bins per side. A four-cell convolution halo then
produces `28×48` expert inputs. Partial tiles and physical borders are zero
extended in the common feature canvas. The dense path performs one expert call
on that canvas. The sparse path gathers exactly the selected inputs into
`[B*K,8,28,48]`, runs the expert, and crops off the convolution halo. At `K=0`,
even the expert feature canvas is bypassed.

Neighboring raised-cosine tapers add to one at half-cell sample positions.
Where a neighbor is outside the physical grid, its weight is assigned to the
edge tile. Their separable products therefore sum to one at every valid cell
when all candidates are counted. Selecting a subset keeps those weights fixed
and omitted tiles contribute zero correction. The fused log gain and phase
offset multiply the base complex bin by `exp(gain + i*phase)`. A phase-only
correction preserves magnitude before iSTFT. The synthesized waveform can have
a different magnitude on reanalysis because arbitrary edited spectra need not
belong to the range of the STFT.

`measure_patch_effects` compares base, A, B, and A+B at one explicit reference
length under waveform MSE. Interaction is joint benefit minus the sum of the
two independent benefits. It also measures waveform reconstruction error after
STFT and iSTFT for all four conditions. This diagnostic can compare adjacent
and distant patch pairs, with no router training or label persistence.

## Reproducing the tests

Run from the repository root using the verified CPU interpreter. The project
virtual environment can run the existing suite and skips the prototype tests
when the optional Torch runtime is absent.

```powershell
& 'C:\Python313\python.exe' -B -m pytest -o pythonpath=src -p no:cacheprovider tests/test_vocoroute_n1.py
& '.venv\Scripts\python.exe' -B -m ruff check --no-cache src tests
```

The CPU prototype suite passed all 44 cases. The full project-environment
suite reported 1,068 passed, 4 failed, 10 skipped, and 1 xfailed. The four
failures match the documented Windows baseline: two CRLF/LF notebook byte
comparisons and two denied directory junctions. The tenth skip is the N1
test module in the Torch-free project environment. Source and test Ruff
checks passed. The repository-wide check retained 11 generated-notebook
findings.

## Measured execution

The synthetic CPU probe used Torch and torchaudio `2.9.0+cpu`, one Torch thread,
float32, seed 705, one 16,384-sample input, 65 frames, and 80 candidate tiles.
Counting the instantiated parameters gives 566,144 for the base and 21,098
for the expert, or **587,242 total**. The Phase N0 estimate also counted a
577-parameter router. N1 uses supplied scores and has no router parameters.

Forward hooks counted the actual convolution and linear output elements
multiplied by weights per output. The base executed 36,424,960 MAC at every
budget. The CPU profiler's `aten::conv2d` FLOP counts independently matched
twice the expert MACs. Each complete forward called iSTFT once.

| Execution | Expert input | Conv2D calls | Expert GMAC | Profiled expert GFLOP |
|---|---|---:|---:|---:|
| `K=0` | none | 0 | 0 | 0 |
| `K=4` | `[4,8,28,48]` | 6 | 0.112767 | 0.225534 |
| `K=20` | `[20,8,28,48]` | 6 | 0.563835 | 1.127670 |
| `K=80` | `[80,8,28,48]` | 6 | 2.255340 | 4.510679 |
| Single dense pass | `[1,8,92,528]` | 6 | 1.018930 | 2.037860 |

The expert cost is 20,976 MAC per spatial cell, or 28,191,744 MAC per gathered
tile. The dense canvas has different edge padding and does not duplicate halos
between tiles. The arithmetic above excludes bias, nonlinearities, feature
formation, sorting, gather, fusion, and iSTFT. `K=20` used 55.3% of the dense
expert MACs on this input. At `K=80`, tiling used more than twice the dense
expert MACs. CPU wall time was outside this probe.

For the float32 end-to-end probe, the largest absolute all-tile versus dense
difference was `3.81e-5` in the complex spectrum and `9.54e-7` in the waveform.
Across separate synthetic spectra covering degenerate, full, and partial time
tiles, the largest complex-bin difference was `7.54e-7` in float32 and
`9.93e-16` in float64. Tests cover float64 forward equivalence with tolerances
for convolution and summation order, dense/all-tile backward agreement, and a
selected-path gradient check against a central difference. Constant corrections
test the overlap separately so a normalization or seam error cannot hide in an
untrained expert's output.

The maximum waveform round-trip error over 28 float32 signal/length pairs was
`1.19e-7`. In a float64 transient sanity check, a fixed residual on neighboring
patches 4 and 5 produced an interaction of `-3.96e-6` waveform MSE units. The
chosen distant comparison, patches 4 and 13, gave an interaction within
`7e-19` of zero. The reconstructed waveform round-trip error was `1.11e-16`
for both comparisons. The distant tile had negligible signal energy in this
fixture, so this check exercises the measurement without testing a general
hypothesis about distance or perceptual quality.

## Next research milestone

With a separately approved native multi-speaker corpus, freeze a new generator
split and its TRAIN-only router roles before fitting. Establish paired waveform
quality and alignment for the base and dense expert first. Then measure whether
single-patch benefit and neighboring interactions leave enough headroom over
equal-budget fixed or random routing. Keep historical forensic artifacts outside
fitting and selection. A later latency study should compare selected execution
against the single dense pass and a smaller dense expert.
