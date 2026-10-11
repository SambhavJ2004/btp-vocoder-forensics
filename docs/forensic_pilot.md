# First forensic ML pilot

Step 3, 2026-10-12. Starting HEAD: `825bbfd79dbd16321fc46639c5193c8201c4efc8`.
This is a CPU development pilot. **All models, predictions and metrics are
non-citable**, including runs against the local legacy `btpvf-v0`. It is separate
from the per-vocoder matched and pretrained mismatched research protocols. It
must not be fed into the thesis correlation as a matched research result.

## Components and architecture

`detectors.forensic_features` implements the common audio-to-vector path.
`experiments.forensic_pilot` loads the existing manifest, trains/selects two
linear heads and writes evaluation artifacts. `detectors.forensic_pilot` persists
their numerical state, checks compatibility and exposes `predict_file`.
`experiments.cli` adds `pilot train`, `pilot evaluate` and `pilot predict`.

Reused components:

- `data.manifest.read_manifest`, path relocation, `validate_manifest`,
  `select_tier`, `require_primary` and primary-analysis membership.
- `data.audio_io.probe`, `read_processed` and `write_audio`.
- `data.preprocess.band_limit_comparison_set` and the existing analysis-band check.
- `detectors.base.crop_samples_for_rate` and `fixed_length_crop`.
- Corrected, directional `detectors.eer.compute_eer`.
- The existing Kaggle harness as the future remote integration point.

There is no Phase A, signal-gate, orientation policy, 0.40 floor, checkpoint-pin,
split-generation, dataset or `CLAUDE.md` change. The pre-existing user edit to
`.gitignore` is untouched. A `pilot` optional dependency group declares
scikit-learn without requiring torch; the existing environment already has it.
No package was installed or upgraded.

## Data and split contract

The default dataset label is `btpvf-v0`; it is an annotation, not evidence of
provenance. The manifest determines the groups. The local dataset at
`runs/onboarding/v0-kaggle-input/manifest.csv` has 200 utterances, one LJ speaker,
four conditions and two tiers. The frozen groups remain 140 train, 20 dev and
40 eval. The pilot selects only archive rows: 560 training files, 80 development
files and 160 evaluation files. Real appears once per utterance, rather than
being repeated for each synthetic condition.

The loader requires exactly `real`, `bigvgan_112m`, `griffin_lim`, `vocos`, and
existing train/dev/eval assignments. It rejects duplicate recording rows/paths,
incomplete utterance groups, conflicting speaker/split identities, missing
identities, unknown conditions/splits, identical audio bytes across different
splits and unequal paired sample lengths. Multiple speakers must have disjoint
existing splits; the loader never repairs or generates a split.

Actual archive files are hashed incrementally. The identity records the raw
manifest SHA-256, canonical utterance/speaker/split map SHA-256, canonical audio
inventory SHA-256, row/group counts and saved provenance/alignment/gate/sanity
artifact hashes when present. These checks establish which local bytes were
consumed. They **do not certify generation provenance or current signal gates**.
The saved receipt explicitly records legacy/unverified provenance, no fresh
signal-gate rerun, unproven historical gate bytes and inactive orientation repair.
No historical record is rewritten.

## Shared features

Protocol: `multi_vocoder_pilot_archive_8khz`.
Feature version: `archive_log_spectral_v1`, 64 float64 values.

1. Require an existing mono PCM_16 WAV at 22,050 Hz. Read through sanctioned
   processed I/O; reject unsupported headers, empty/non-finite/clipped audio and
   durations outside 0.1–60 seconds. The duration bound limits inference memory;
   it is not a feature or class-dependent selection.
2. Read one complete utterance comparison group at a time. Apply the existing
   zero-phase Chebyshev II transform to **real and every synthetic member**,
   with stopband edge `LADDER_FMAX` (currently 8,000 Hz), existing transition
   and rejection specification, and verification enabled. No full-band or
   16 kHz alternative is introduced.
3. Materialize temporary analysis WAVs through `write_audio` as PCM_16, reread
   through `read_processed`, then verify the decoded analysis band. The detector
   receives decoded analysis audio rather than Phase A in-memory arrays. Only
   these temporary analysis derivatives are written; archived source audio is
   never regenerated or replaced.
4. Use the shared 4.0375-second crop policy: 89,027 samples at archive rate,
   first crop for long recordings and repeat-padding for short ones. No zero
   padding, random crop, class-specific trimming, rate conversion or second
   loudness normalization.
5. Frame without centering: 1,024-sample periodic Hann window, 256-sample hop.
   Compute squared rFFT magnitudes divided by window squared-energy. Average
   power in 32 equally spaced linear frequency bands over [0, 8,000) Hz,
   excluding each upper edge. No bin at/above the analysis cutoff is included.
6. Apply `log10(max(band_power, 1e-12))` per frame. Concatenate 32 temporal means
   and 32 population standard deviations, in that order. Their exact names,
   floor, window, crop and configuration are saved in the bundle.

No vocoder-native mel front end is used. Paths, labels, speaker, sample rate,
file length, manifest loudness, generator identity and header metadata never
enter the feature vector. The loader uses metadata only for integrity checks.
All conditions receive identical processing. Features can still reflect content,
silence or remaining historical confounds; this representation does not prove
that the legacy data is scientifically eligible.

Training holds only one utterance group's audio and the small feature matrix in
memory. Temporary decoded analysis files are reused inside a private temporary
directory and removed afterwards. Single-file inference uses the same transform
on a singleton: the input occupies the comparison API's reference slot regardless
of its unknown class. The filter itself is independent per waveform. Tests assert
exact feature equality with every condition's grouped training path.

## Training and selection

Binary labels: `0=synthetic`, `1=real`. Pooled synthetic training contains 420
rows and real contains 140. Each candidate uses `StandardScaler` and L2
`LogisticRegression(solver="lbfgs", max_iter=2000, tol=1e-6)`, with
`class_weight="balanced"`. This gives real weight 2 and each synthetic row
weight 2/3, without duplicating real recordings. Scikit-learn computes weights
from training class counts; see its
[LogisticRegression documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).

Attribution is a second scaler plus multinomial L2 logistic regression on the
420 **synthetic training rows only**, using the same feature definition. Source
classes have equal counts and use no extra class weighting. The exact ordered
mapping is `0=bigvgan_112m`, `1=griffin_lim`, `2=vocos`. There is no real attribution
class and no unknown-source rejection.

For each head, select C from `{0.01, 0.1, 1, 10}` on dev. Binary selection minimizes
corrected directional EER, breaking ties by higher dev AUROC and then lower C.
Attribution selection maximizes dev macro-F1, breaking ties by lower C.
Every scaler/classifier fit receives training rows only. No refit on dev occurs.
Convergence warnings fail the fit rather than silently saving an unfinished model.
The seed defaults to `GLOBAL_SEED=20260907`; the CLI can record another training
seed without changing any frozen dataset split.

The binary score is the positive-class logistic decision margin: higher means
real. Select the operating threshold on dev using minimum balanced error
`(FRR+FAR)/2` at **attainable grouped score boundaries**, with `score > threshold`
meaning real. Ties prefer smaller `|FRR-FAR|`, then the lower boundary.
The operating threshold is separate from the interpolated EER crossing boundary.
No eval threshold or hyperparameter fitting occurs. The chosen bundle is written
before eval feature extraction; eval is then scored once during training.

## Evaluation and artifacts

`metrics.json` reports pooled binary AUROC, directional EER, EER crossing
boundary, confusion matrix (rows=true, columns=predicted; synthetic then real),
FRR/FAR at the frozen dev threshold, class counts and one EER per vocoder against
the same evaluation real population. An EER boundary need not reproduce an
interpolated EER, as documented in `evaluation_correctness.md`.

Attribution macro-F1, per-class precision/recall/F1/support and ordered confusion
matrix are evaluated on **all ground-truth synthetic eval rows**, independently
of binary routing. Additional cascade routing recall and correct-source fraction
show how the two-stage interface performs when detector mistakes prevent or
misroute attribution. Multiclass decision logits are uncalibrated scores.
No softmax confidence or calibrated-probability claim is made.

The pilot emits no confidence intervals: paired/grouped bootstrap methodology is
an explicit separate task. These exploratory metrics do not replace research
protocol artifacts, quality measurements, uncertainty or benchmark reproduction.

Training output (fresh directory required):

| File | Content |
|---|---|
| `bundle/models.npz` | Binary/attribution scaler mean/scale, coefficients and intercepts |
| `bundle/bundle.json` | Schema, feature definition, mappings, threshold, selection, seed, dataset identities, source/Git/environment provenance, non-citable status, model SHA-256 |
| `bundle/bundle.sha256.json` | Model and metadata integrity hashes |
| `features.npz`, `feature_rows.csv` | Train/dev and eval vectors, and their exact row ordering/identities |
| `dataset_identity.json` | Frozen group and actual input identities, provenance limitations |
| `selection.json` | All dev candidates, selected C values, weights and threshold operating point |
| `predictions.csv` | Raw eval margins/logits, true labels, head/routed predictions, file/group/split/tier/band/feature/model identities |
| `metrics.json` | Evaluation, dataset/model/code identities and explicit non-citable label |
| `artifacts.sha256.json` | Hash inventory of all saved output files |

Numerical state is saved in NPZ and loaded with `allow_pickle=False`, rather than
deserializing arbitrary Python objects. Load only trusted local bundles. Hashes
detect changed bytes but are not signatures or proof of trust. The loader checks
hashes, schema/protocol, feature definition/names, class mapping, finite threshold,
finite array shapes/positive scales, exact dependency versions and preprocessing
source hashes. A changed environment or processing implementation is refused.
The saved training source hashes, Git HEAD and dirty status identify working-tree
code; a Git commit alone is not asserted to contain these unstaged changes.

`pilot evaluate` validates the same manifest/group/audio/source-artifact identity,
loads the frozen bundle and scores only eval without fitting. It writes a new
metrics/predictions/hash directory. Existing outputs are never overwritten.

## Inference API and local commands

```python
from detectors.forensic_pilot import load_bundle, predict_file

bundle = load_bundle("runs/pilot-v0-first/bundle")  # trusted local artifact
result = predict_file(bundle, "archive_input.wav", work_dir="runs")
```

`result` contains `prediction`, `real_score`, `operating_threshold`, score meanings,
protocol and `citable=False`. If predicted synthetic, it also contains
`known_source` and a source-name-to-uncalibrated-logit mapping. For predicted real,
source fields are null. The API does not expose an unknown class. Arbitrary
unprocessed uploads, other rates/channels/encodings and general cross-domain
performance are outside this pilot. It does not guess reference trim/loudness
history or silently convert an unsupported WAV.

From the repository in PowerShell, using new output names:

```powershell
& '.venv\Scripts\python.exe' -m experiments.cli pilot train --manifest runs/onboarding/v0-kaggle-input/manifest.csv --out-dir runs/pilot-v0-first --dataset-id btpvf-v0
& '.venv\Scripts\python.exe' -m experiments.cli pilot evaluate --bundle runs/pilot-v0-first/bundle --manifest runs/onboarding/v0-kaggle-input/manifest.csv --out-dir runs/pilot-v0-first-eval
$pilotPrediction = Import-Csv 'runs/pilot-v0-first/predictions.csv' | Where-Object { $_.condition -eq 'vocos' } | Select-Object -First 1
& '.venv\Scripts\python.exe' -m experiments.cli pilot predict --bundle runs/pilot-v0-first/bundle --wav $pilotPrediction.path --work-dir runs
```

The prediction path comes from the pilot's resolved per-file output, so it works
with the legacy manifest's relocated paths without guessing an utterance name.

On a separately prepared environment the declared CPU extra is `[pilot]`; this
task did not run an install. `btpvf pilot ...` is the equivalent installed entry
point. No remote job is needed for this baseline.

## Smallest Kaggle integration (planned, not implemented)

Keep `experiments.kaggle_harness` as the sole runner. Its current `KernelSpec`
and generated `notebook_source` are Phase A-specific; passing `run_kernel`'s
`notebook` argument only writes another copy of the generated notebook and does
not select a training workflow. Current `btpvf kaggle run` therefore does **not**
launch this pilot.

The smallest future extension is a job discriminator with Phase A as its
unchanged default, plus an input-manifest locator/path, dataset label and training
seed in the same spec. Add a generated pilot driver branch after the existing
snapshot bootstrap. Install `[pilot]`, request CPU (`gpu=False`) and attach the
existing audio dataset; no vocoder/checkpoint installation or resynthesis cell.
Use the existing manifest resolver on the mounted manifest, fail on ambiguous
discovery, call `train_pilot` into a fresh `/kaggle/working/pilot_*` directory and
retain its complete bundle/features/metrics/predictions/hash receipt.

Reuse `stage_snapshot`, code dataset push, metadata validation, polling/log/output
pagination and run receipt. Preserve the existing v0 notebook bytes and Phase A
tests; test the new generated driver/spec/dependency choices separately. Bind
the mounted code and exact generated driver to their hashes in the pilot's code
provenance. For remote record work, the user must first create the commit;
legacy v0 pilot output stays non-citable even from record-mode training code.
Durable failure receipts, complete raw logs, mounted-code verification and
downloaded-output integrity remain harness hardening requirements from onboarding.
No second runner, new notebook, upload, download, authentication mutation or
Kaggle launch is part of Step 3.

## Verification and remaining limits

Ruff `src tests`: **passed**. Repository-wide Ruff: **the same 11 generated
notebook errors**, with no notebook/lint-policy edits. Final focused selection:
**139 passed**, 22.39 seconds (57 new pilot cases plus 82 corrected-EER cases).
Full suite: **542 passed, 4 failed, 4 skipped, 1 xfailed**, 231.35 seconds.
The four failure identities exactly match Step 2: two committed-notebook CRLF/LF
byte comparisons and two sandbox-denied Windows directory junctions. The four
file-symlink privilege skips and existing ablation expected failure remain.
No new test failure or skip remains. An initial new-test failure came from moving
a relative-path manifest without its audio; the fixture was corrected to move
the audio as well, preserving the intended changed-dataset rejection check.

Tests cover grouped and speaker splits, duplicate/content leakage, label mappings,
feature equality/immutability/filename independence, unsupported audio/config,
class weighting without real duplication, training-only scalers, dev threshold
ties, independently derived perfect/tied/reversed metrics, numeric state versus
scikit-learn, saved evaluation/inference equality, integrity/compatibility failures,
repeated training, bundle freeze before eval access and all three CLI dispatches.

Verification commands used the existing environment and fresh basetemp directories:

```powershell
& '.venv\Scripts\python.exe' -m ruff check src tests
& '.venv\Scripts\python.exe' -m ruff check .
& '.venv\Scripts\python.exe' -m pytest -o addopts='' -q -ra tests/test_forensic_pilot.py tests/test_evaluation_correctness.py --basetemp runs/forensic-pilot-step3-20261012/pytest-focused-review
& '.venv\Scripts\python.exe' -m pytest -o addopts='' -q -ra --basetemp runs/forensic-pilot-step3-20261012/pytest-full-final
```

Logs are under ignored `runs/forensic-pilot-step3-20261012/`. Choose new basetemp
paths for a future run: pytest may delete an existing basetemp's contents.
`git diff --check` passes; there are no staged changes. All new changes remain unstaged.

### Local legacy v0 smoke run: non-citable

Final training artifacts: `runs/forensic-pilot-step3-20261012/v0-final/`.
The run consumed the existing frozen 140/20/40 utterance groups and evaluated
160 archive files (40 real, 40 of each known source). It selected binary C=0.01,
attribution C=10 and dev operating threshold `0.07470214195092881`, with
`GLOBAL_SEED=20260907`. No model choice changed after examining eval.

| Local development metric | Value |
|---|---|
| Pooled binary AUROC | 0.6791666667 |
| Pooled binary EER | 0.3833333333 |
| BigVGAN-112m EER | 0.4500000000 |
| Griffin-Lim EER | 0.2000000000 |
| Vocos EER | 0.4250000000 |
| Known-source macro-F1, all 120 true synthetic files | 0.8826733710 |
| Cascade correctly detected and attributed fraction, all true synthetic files | 0.5833333333 |

Binary confusion matrix, ordered `[synthetic, real]`, rows=true/columns=predicted:
`[[80,40],[18,22]]`. At the dev threshold, eval FAR is 1/3 and FRR is 0.45.
Attribution matrix, ordered `[bigvgan_112m, griffin_lim, vocos]`:
`[[36,0,4],[0,40,0],[10,0,30]]`. Per-source precision/recall/F1/support are in
`metrics.json`; all raw margins/logits and 160 predictions are preserved.
These are **new development measurements**, not revised historical gate results.

An earlier integration smoke directory `v0-smoke/` is retained separately. Final
training after implementation completion reproduced its numerical model arrays
and model SHA-256 exactly:
`e12b94dd4aa5f375968cc7180feda42294fe8d7f184fd692e6c4c1507ce8eb76`.
The final bundle records current source-tree identity rather than relying on the
earlier integration receipt. This rerun checked reproducibility; no hyperparameter
or preprocessing changes were made in response to its evaluation numbers.

Standalone CLI evaluation into `v0-final-evaluation/` reproduced `metrics.json`
exactly and produced byte-identical `predictions.csv`. All saved training and
evaluation artifact hashes were independently checked. The binary and attribution
scaler means were also checked against only the saved training feature rows.
Single-file CLI smoke calls exercised both routing branches: one predicted real
with source fields null, and one predicted synthetic with known-source logits.
Their margins matched the corresponding saved per-file predictions to 1e-12.
Receipts/logs are `verification.json`, `v0-final-train.log`,
`v0-final-evaluate.log` and `v0-final-predict-{real,synthetic}.json` under the same
ignored run directory. No temporary analysis WAV directory remains in the
successful final training/evaluation outputs.

Research blockers remain independent: legacy generation provenance, historical
signal-gate revalidation and one-sided probe orientation, single-speaker/few-group
evaluation, small dev-set selection, no cross-corpus/unseen-generator validation,
uncalibrated scores and no paired uncertainty. This task does not activate gate
repair or establish thesis eligibility. AASIST/neural training, F0 repair,
unknown-source rejection, API/frontend and remote execution remain separate work.
