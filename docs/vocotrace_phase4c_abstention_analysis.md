# VocoTrace Phase 4C: abstention failure analysis and robustness feasibility

**Decision:** abstention mainly reflects weak synthetic-speech routing. The
selected confidence layer does not reject any withheld dev recording on its
own. Common processing also breaks the frozen system, sometimes producing
confidently wrong source labels on genuine recordings. Defer an upload-facing
forensic evidence interface. Prioritize **C: additional source-diverse, eligible
data**, with **B: a revised, separately evaluated training protocol** as the next
bounded modeling step. A local diagnostic viewer could expose research scores
and decision reasons; it would not establish forensic validity.

2026-11-06; starting manually created Git HEAD
`ccaa3c5e37b239d3509943728b5ed082644fe6a4`.
**Development-only, non-citable.** No new fitting, parameter selection, dataset
generation, eval access, Kaggle work or Git writes. The user's pre-existing
`.gitignore` modification is preserved. The Phase 4B implementation and both
historical experiment outputs are unchanged.

## Evidence and reproducibility boundary

Authoritative historical input:
`runs/vocotrace-phase4b-verification-v2-20261105/`.
New sealed analysis output:
`runs/vocotrace-phase4c-abstention-20261106/`.
Supplementary scripts, before/after evidence and test logs:
`runs/vocotrace-phase4c-audit-20261106/`.

Read AGENTS.md, all eighteen invariant rules and applicable details in CLAUDE.md,
the Phase 4A feasibility review, Phase 4B report, measurement/mel/enforcement
documentation, and the feature, numeric-head, verifier, calibration and metric
implementations. The full-band archive, original processing, analysis membership,
scientific gates, frozen source splits and inference feature contract remain
unchanged. Stress derivatives are explicitly named analysis conditions, never
replacement archive rows or a new generation pipeline.

The original whole-audio corpus has 140 train / 20 dev / 40 eval source groups.
The 17 dev sources mentioned in partial-audio work describe its donor-eligible
subset, not this whole-audio population. Phase 4C replays saved train/dev feature
arrays and prediction CSVs. Its only newly decoded original audio consists of
48 recordings from twelve predetermined **training** sources. No dev WAV is
opened, and no eval waveform, feature array or prediction row is consumed.
Structural manifest rows identifying frozen groups are not model observations.

The locked plan is saved before this runner inspects dev predictions. Its seven
stress cells come directly from the Phase 4A proposed matrix, not from a search
over Phase 4C results. Models, selected scores and every cutoff are inherited
unchanged from Phase 4B. No favorable dev policy or transform is promoted.

## 1. Decision-path reconstruction

For each feature vector x, the binary head returns an uncalibrated real margin
`m = w_b ((x-mu_b)/sigma_b) + intercept_b`. Higher means more real-like according
to this particular fitted binary classifier. The synthetic route passes iff
`m <= binary_threshold`. It is not a probability or an authenticity test.

The independent attribution scaler/head produces source logits, an argmax
candidate and the three saved confidence scores: negative squared Mahalanobis
distance, maximum softmax, and negative energy. All confidence scores are oriented
higher known-like. Two-class logits retain the Phase 4B `[-margin/2,+margin/2]`
mapping. Source support passes iff the chosen score is at least its cutoff.
Equality passes both gates. A source label requires both gates to pass.

| Context | Primary confidence | Binary threshold (`<=`) | Confidence cutoff (`>=`) |
|---|---|---:|---:|
| closed3 | Negative energy | -1.3822691832116711 | 1.2577363521126246 |
| Without BigVGAN | MSP | -1.1160593582143206 | 0.6924059034990455 |
| Without Griffin-Lim | MSP | -1.8353746195360507 | 0.5399678162486429 |
| Without Vocos | Negative squared distance | -1.4665601620336461 | -82.9033075644128 |

The replay checks all numeric margins/logits/scores against the saved values,
then checks every returned source and decision against both original gates.
It checks the saved metric counts too. There are **2,880 dev policy rows**:
80 recordings x four contexts x nine pre-existing policies. The primary policy
has 320 context-recording decisions on **80 unique recordings / 20 sources**.
It also replays **20,160 training-OOF policy rows** on 560 recordings / 140 sources.
Repeated policies and contexts do not create independent observations.
The eight dev/OOF prediction CSVs in the preserved first cold Phase 4B run and
the corrected v2 run are rechecked as byte-identical, so neither saved version
contains an additional conflicting prediction population.

`dev_decision_paths.csv` and `oof_decision_paths.csv` contain source grouping,
input digest, both thresholds, raw binary evidence, class logits, all confidence
scores, selected score/cutoff/margin, gate booleans, acceptance, final source,
decision, exact first rejection reason and truth-based error category. A separate
counterfactual confidence-pass field is recorded even for binary-blocked inputs.
The replay never uses truth to make a prediction. Truth is used afterward for
error analysis. Missing class-logit columns across contexts mean that class was
excluded from that context, not a zero model score.

The first operative rejection is `binary_real_like_gate`; if that passes but
source support fails, it is `source_support_below_cutoff`. The reject-all reference
has `policy_reject_all`. Both-gate failures remain visible independently. The
unconditional reference applies neither gate; binary-only applies no confidence
gate. These references are not secretly treated as the selected system.

### Exact primary dev categories

B = BigVGAN-112m, G = Griffin-Lim, V = Vocos. Every context has 20 genuine inputs;
closed3 has 60 known synthetic inputs, and each withheld context has 40 known
plus 20 withheld synthetic inputs.

| Context | Genuine, no label | Genuine, wrong label | Known correct | Known wrong | Known rejected | Withheld real-like rejection | Withheld uncertainty rejection | Withheld wrong label |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| closed3 | 20 | 0 | 17 | 1 | 42 | N/A | N/A | N/A |
| Without B | 20 | 0 | 21 | 0 | 19 | 19 | 0 | 1 |
| Without G | 20 | 0 | 0 | 1 | 39 | 19 | 0 | 1 |
| Without V | 20 | 0 | 11 | 0 | 29 | 19 | 0 | 1 |

Known rejection decomposes into binary failure / incremental confidence failure:
**41/1** in closed3, **17/2** without B, **39/0** without G and **29/0** without V.
Thus the confidence stage removes very few labels at these operating points.
Fourteen of closed3's seventeen correct labels are G; three are V; none is B.
The one accepted known B is wrongly attributed. Oracle closed-set macro-F1
remains **0.8108747045**, evaluated separately from this heavily gated cascade.

**All 60 withheld dev inputs pass their selected confidence cutoff.** Of these,
57 fail the binary synthetic route and three receive wrong known-source labels.
Calling the 57 real-like rejections successful unknown-source recognition would
misdescribe the implementation and the measurements.

### Resolving withheld-count ambiguity

The selected `retain95` policy consistently gives **19/20 non-attributions and
1/20 wrong labels in each withheld dev trial**, in the saved CSVs, saved metrics,
recomputed model scores and Phase 4B report. No primary-count discrepancy was
found. The exact wrong labels are:

| Withheld context | Source utterance | Actual pipeline | Returned known source | Real margin | Confidence |
|---|---|---|---|---:|---:|
| Without B | LJ048-0129 | B | V | -1.7335531779 | MSP 0.9447325167 |
| Without G | LJ048-0129 | G | V | -2.3369384866 | MSP 0.9875362193 |
| Without V | LJ046-0046 | V | G | -1.8581382063 | Distance -73.4009483090 |

There are three distinct denominators that can otherwise be confused:

- **All returned labels** in the contexts are 22/80, 2/80 and 12/80. Of these,
  21, 1 and 11 labels respectively are on known synthetic inputs; one is on a
  withheld input. They are not 22, 2 or 12 predictions of the withheld class.
- Without V, the **unselected** MSP/energy policies reject 20/20 withheld dev
  inputs, while selected distance at retain95 rejects 19/20. Distance at retain80
  also rejects 20/20. Mixing these policies changes the count.
- Training OOF is a different population: without B, 138/140 withheld inputs are
  real-like rejections, one is an uncertainty rejection and one is wrongly labeled.
  Without G, the counts are 124/0/16; without V, 129/0/11. Dev has zero withheld
  uncertainty rejections in all three trials. An OOF uncertainty count must not
  be copied into dev.

Historical files are preserved; no retrospective prediction correction is needed.

## 2. Grouped errors, coverage and score behavior

### Fixed-policy dev tradeoffs

All-input precision counts wrong labels on genuine **and** withheld inputs in
its denominator. Known-only accepted accuracy does not. All-input coverage uses
the fixed 80-recording mixture and does not estimate deployment prevalence.

| Context / policy | Known accepted / known | Correct / accepted known | Genuine false labels / 20 | Withheld wrong labels / 20 | All labels / 80 | Correct / all labels |
|---|---:|---:|---:|---:|---:|---:|
| closed3, primary retain95 | 18/60 | 17/18 | 0 | N/A | 18 | 17/18 |
| closed3, same score retain80 | 17/60 | 16/17 | 0 | N/A | 17 | 16/17 |
| closed3, binary-only | 19/60 | 17/19 | 0 | N/A | 19 | 17/19 |
| closed3, unconditional | 60/60 | 49/60 | 20 | N/A | 80 | 49/80 |
| Without B, primary retain95 | 21/40 | 21/21 | 0 | 1 | 22 | 21/22 |
| Without B, same score retain80 | 15/40 | 15/15 | 0 | 1 | 16 | 15/16 |
| Without B, binary-only | 23/40 | 23/23 | 0 | 1 | 24 | 23/24 |
| Without G, primary or binary-only | 1/40 | 0/1 | 0 | 1 | 2 | 0/2 |
| Without G, same score retain80 | 1/40 | 0/1 | 0 | 1 | 2 | 0/2 |
| Without V, primary or binary-only | 11/40 | 11/11 | 0 | 1 | 12 | 11/12 |
| Without V, same score retain80 | 9/40 | 9/9 | 0 | 0 | 9 | 9/9 |

The reject-all reference has zero coverage and undefined accepted precision,
even though withheld non-attribution is 100%. Unconditional attribution falsely
labels all genuine/withheld inputs; its known-only source accuracy is not
end-to-end evidence. The complete 36-policy table per partition is saved in
`dev_fixed_policy_tradeoffs.csv` / `oof_fixed_policy_tradeoffs.csv`. These points
are descriptive, not a newly tuned risk/coverage curve or new operating point.

Primary training-OOF categories, retaining the original train-selection caveat:

| Context | Genuine no label / false label | Known correct / wrong / rejected | Withheld real-like / uncertain / wrong |
|---|---|---|---|
| closed3 | 134 / 6 | 101 / 16 / 303 | N/A |
| Without B | 136 / 4 | 119 / 2 / 159 | 138 / 1 / 1 |
| Without G | 134 / 6 | 16 / 10 / 254 | 124 / 0 / 16 |
| Without V | 135 / 5 | 79 / 2 / 199 | 129 / 0 / 11 |

### Distributions and gate failures

Define binary excess as `real_margin - binary_threshold`: positive means the
binary gate blocks attribution. Define support excess as confidence minus its
cutoff: negative means confidence fails. These are uncalibrated score units,
not comparable probabilities across contexts. Full min/quartile/median/max
distributions by context and actual pipeline are saved in the path summaries.

| Context / actual pipeline | Binary failures / 20 | Median binary excess | Median support excess |
|---|---:|---:|---:|
| closed3 / genuine | 20 | 2.3845 | 0.6268 |
| closed3 / B | 19 | 1.8586 | 0.6177 |
| closed3 / G | 6 | -0.4652 | 1.0450 |
| closed3 / V | 16 | 1.5887 | 0.4176 |
| Without B / withheld B | 19 | 2.1983 | 0.2543 |
| Without G / withheld G | 19 | 1.7093 | 0.4009 |
| Without V / withheld V | 19 | 2.2631 | 32.2029 |

Support-excess ranges on withheld dev inputs are entirely positive:
B **[0.0952, 0.2939]**, G **[0.1761, 0.4476]**, V **[1.8833, 64.5838]**.
Even the low-support tails pass. Genuine support also usually passes: 19/20,
18/20, 18/20 and 19/20 in context order. It is the binary gate that prevents
these genuine source labels, not a confidence score separating genuine from
known synthetic speech. Conversely, the gate also blocks most B/V known inputs.

This agrees with the saved known-versus-withheld confidence AUROCs: without G,
MSP is **0.06125** on dev, indicating reversed ranking rather than useful
unknown detection. No score orientation is flipped after inspecting withheld
outcomes. High MSP is not evidence of a supported generator identity.

### Variability across source groups

| Context | Sources returning 0 / 1 / 2 / 3 labels | Sources with 0 / 1 / 2 correct labels |
|---|---|---|
| closed3 | 6 / 11 / 2 / 1 | 6 / 11 / 3 |
| Without B | 2 / 15 / 2 / 1 | 2 / 15 / 3 |
| Without G | 19 / 0 / 1 / 0 | 20 / 0 / 0 |
| Without V | 9 / 10 / 1 / 0 | 9 / 11 / 0 |

Without G, both errors occur on **LJ048-0129**; nineteen other sources produce
no labels. Counting the two errors as independent source failures would conceal
that clustering. The same utterance also has the without-B withheld error.
Per-source category counts remain in the path summaries; conditions, policies
and transforms stay descendants of that source group.

Phase 4B's conditional source-bootstrap intervals remain applicable to its
unchanged primary results. For example, closed3 dev known coverage is
[20.0%, 41.67%], all-input precision [85.0%, 100%]. Zero genuine errors among
twenty exposed, single-speaker dev sources cannot establish a future zero error
rate or certify a 5% budget. The without-G precision interval has undefined
no-label replicates. No additional independent dev population is manufactured.

## 3. Predetermined train-only robustness experiment

### Design and controls

Use twelve source groups, four from each original training outer fold. Within
each fold, sort source IDs by SHA-256 of
`20260907:vocotrace4c:robustness:<utt_id>` and take four. Selection does not use
audio, margins, correctness, speaker content or duration. Exact groups:

| Existing outer fold | Selected training source IDs |
|---|---|
| 0 | LJ007-0043, LJ038-0204, LJ040-0217, LJ045-0067 |
| 1 | LJ021-0054, LJ023-0023, LJ028-0500, LJ041-0140 |
| 2 | LJ002-0223, LJ016-0347, LJ025-0090, LJ028-0337 |

Score each group using only its already fitted Phase 4B **outer-held** model.
Assertions verify that no selected group occurs in that model's fit or
calibration groups. All genuine/B/G/V recordings stay together. All four
contexts are evaluated; withheld-class data never changes that context's model,
normalizer, support reference or cutoff. There are zero research fits and no
new selection. The historical score-method choice used full training OOF
outcomes, so this is an exploratory, model-disjoint stress comparison, not an
unbiased nested estimate of the whole method-selection procedure.

| Cell | Predetermined analysis operation, before unchanged inference |
|---|---|
| Clean | Original paired input; exact saved clean feature replay to 1e-12 |
| Gain -6 dB | Multiply every condition by `10**(-6/20)` |
| Gain -12 dB | Multiply every condition by `10**(-12/20)` |
| Below 6 kHz | Sanctioned common zero-phase filter with 6,000-Hz stopband edge |
| Below 4 kHz | Same filter family with 4,000-Hz stopband edge |
| First 1 s | Same initial source-coordinate support, at most 22,050 samples |
| First 2 s | Same initial source-coordinate support, at most 44,100 samples |

All seven cells use the existing mandatory common 8-kHz filter, sanctioned
PCM16 write/reread, delivered-band checks and the unchanged 64-feature extractor.
Lower-band stress therefore explicitly has **lower-band filtering followed by
the existing 8-kHz inference filter**. The delivered PCM16 output is additionally
checked at its lower cutoff. No filter verification is disabled. Every condition,
including genuine, gets identical processing; above-8-kHz resampler/cliff cues
remain excluded. These tests do not identify a generalizable vocoder fingerprint
or eliminate all processing cues inside the retained band.

Gain is not renormalized or repaired. Peak/zero-fraction receipts expose
quantization effects. Positive gain and clipping repair are not introduced.
Duration tests truncate common initial support and then use the existing
repeat-to-89,027-sample wrapper. They test this wrapper's response to reduced
available speech, not a new variable-duration model or extra independent crops.
Actual support lengths are recorded; there is no content-dependent crop search.
No partial recordings or splice controls are part of this whole-audio experiment.
All selected parents have 57,785--204,729 samples, so each duration cell supplies
exactly its nominal one or two seconds before the unchanged repeat wrapper.

**All 12 groups / 48 original recordings passed all seven cells: 336 feature
rows, no exclusions or failed empirical gates.** Temporary analysis WAVs were
removed by the temporary-directory lifecycle, not used to repair failures or
alter source files. Parent hashes, temporary PCM16 hashes, feature hashes,
sample counts, fold IDs and output ordering are saved. The 336-row feature
array corresponds in order to `analysis_waveform_receipts.csv`.

### Closed3 results: 12 genuine and 36 known recordings per cell

| Cell | Oracle source macro-F1 | Accepted known | Correct known | Genuine falsely labeled / 12 | All labels / 48 | Correct / all labels |
|---|---:|---:|---:|---:|---:|---:|
| Clean | 0.8222 | 10 | 10 | 0 | 10 | 10/10 |
| Gain -6 dB | 0.5222 | 5 | 4 | 0 | 5 | 4/5 |
| Gain -12 dB | 0.3833 | 0 | 0 | 0 | 0 | Undefined |
| Below 6 kHz | 0.1667 | 36 | 12 | 12 | 48 | 12/48 |
| Below 4 kHz | 0.2857 | 0 | 0 | 0 | 0 | Undefined |
| First 1 s | 0.5820 | 9 | 7 | 2 | 11 | 7/11 |
| First 2 s | 0.6919 | 13 | 11 | 1 | 14 | 11/14 |

The 6-kHz cell returns **BigVGAN for all 48 inputs**, including all twelve
genuine recordings. Genuine median binary excess moves from **+2.0480** clean
to **-7.1551**; its entire transformed range [-9.6250, -0.3119] passes the
synthetic gate. All twelve also pass energy support, with positive support
excess [1.3963, 7.7905]. The confidence gate does not protect against this
common processing change. This is an observed response of the frozen model;
it does not establish a causal explanation for every learned coefficient.

A **post-hoc, non-selective** linear-response diagnostic checks this failure
using only saved features and frozen outer-fold heads. For each genuine source,
the sum of `(x_6k-x_clean)/binary_scale * binary_coefficient` reproduces its exact
binary-margin change to 1e-12. The median largest absolute standardized feature
moves from **2.2347** clean to **11.3435** after filtering. Large opposing mean
margin contributions include the 7,750--8,000-Hz log-power mean (**-26.3449**)
and 7,500--7,750-Hz log-power mean (**+17.4632**). This is evidence of sensitivity
to the altered below-8-kHz spectrum and extrapolation beyond clean feature
scales. It does not isolate all generation confounds or establish the response
on other channels. No coefficients, scores, crops, transforms or cutoffs are
chosen from this diagnostic. Details are in the audit's
`feature_failure_diagnostic.json`; the original sealed run is unchanged.

The 4-kHz cell and -12 dB cell return no labels. Their apparent absence of false
labels therefore comes with no useful known-source coverage. Processing
response is not monotonic: a stronger low-pass does not simply improve or
degrade a calibrated notion of uncertainty. The 6-kHz failure is not evidence
that all band-limited inputs should be called synthetic.

### Each withheld context: 12 genuine, 24 known, 12 withheld per cell

Each cell below is `known accepted / correct known; genuine false labels;
withheld wrong labels; correct / all returned labels`. Every count is explicit
so a high known-only accepted accuracy cannot hide false source labels elsewhere.

| Cell | Without B | Without G | Without V |
|---|---|---|---|
| Clean | 12/12; 0; 0; 12/12 | 0/0; 0; 0; undefined | 7/7; 0; 0; 7/7 |
| Gain -6 dB | 8/8; 0; 0; 8/8 | 0/0; 0; 0; undefined | 0/0; 0; 0; undefined |
| Gain -12 dB | 7/7; 0; 0; 7/7 | 0/0; 0; 0; undefined | 0/0; 0; 0; undefined |
| Below 6 kHz | 10/5; 5; 5; 5/20 | 23/11; 11; 12; 11/46 | 0/0; 0; 0; undefined |
| Below 4 kHz | 0/0; 0; 0; undefined | 2/1; 1; 1; 1/4 | 0/0; 0; 0; undefined |
| First 1 s | 12/11; 5; 2; 11/19 | 4/3; 2; 2; 3/8 | 0/0; 0; 0; undefined |
| First 2 s | 14/14; 1; 0; 14/15 | 6/4; 2; 3; 4/11 | 4/4; 0; 0; 4/4 |

Here `0/0` is a pair of counts, not 100% accepted accuracy. Full confusion
matrices, per-vocoder results, all decision paths, ranking statistics, per-source
counts and conditional intervals are in the saved robustness artifacts. The
below-6-kHz without-G context wrongly attributes **12/12 withheld recordings**;
its unchanged clean rejection rate is not a robust unknown-source capability.

### Paired uncertainty and limits

Two thousand fixed-seed bootstrap resamples select the **same source groups on
both sides** of every changed-minus-clean comparison, keeping all conditions
together. The twelve source groups, not 336 transformed rows, determine sample
size. No model or threshold is refitted during resampling. Score methods and
cutoffs are fixed. These are exploratory conditional intervals, not corrected
multiple-testing claims or uncertainty over speakers/generators/model selection.

Examples for closed3, in percentage points:

| Changed-minus-clean quantity | Point change | Conditional 95% interval |
|---|---:|---:|
| -6 dB known coverage | -13.89 | [-27.78, -2.78] |
| -6 dB correct-known fraction | -16.67 | [-30.56, -5.56] |
| -12 dB known coverage | -27.78 | [-44.44, -13.89] |
| 6-kHz genuine false-attribution rate | +100.00 | [+100.00, +100.00] |
| 1-s genuine false-attribution rate | +16.67 | [0.00, +41.67] |
| 2-s genuine false-attribution rate | +8.33 | [0.00, +25.00] |

The degenerate 6-kHz interval reflects twelve observed source failures, not
certainty about future failure frequency. Precision is undefined in every
replicate of an all-reject changed condition; the artifact records those
undefined replicates rather than reporting a perfect value. Genuine clean
zero-error intervals similarly cannot provide future error guarantees.

### Common resampling: feasible preprocessing, unsupported model scoring

The sanctioned **single** `soxr_hq` 22,050 -> 16,000-Hz analysis derivation was
run symmetrically on all 48 selected training recordings. PCM16 writing/reading,
finite waveform, expected `ceil(N*16000/22050)` length and duration error less
than one 16-kHz sample pass. Parent identities and output hashes are recorded
in `resampling_preflight.json`. These are temporary analysis derivatives, not
new source-tier dataset artifacts.

The supplementary `verify_resampling_loudness.py` also runs the existing
`derive_zeroshot` implementation, with measured parent full-band loudness and
its **unchanged** derived-loudness gate, on the same 48 training parents. All
pass; maximum absolute drift is **1.7064125962 LU**. All 48 PCM16 hashes are
identical to the format preflight. `resampling_loudness_validation.json` records
the per-parent measurements. This adds the canonical loudness check without
changing or resealing the original analysis run, and applies no new gain.

The existing `FeatureConfig` correctly refuses all 48 derived WAVs because its
contract requires 22,050 Hz. Therefore **zero resampled classification scores**
are reported. No header rewriting, hidden upsampling, sample-count reuse at the
wrong rate, second resample or silent feature-contract change is introduced.
This does not test classification robustness to resampling. A separately
versioned rate-aware feature/inference protocol preserving physical bands,
durations, schema identity and matched controls is required first, as Phase 4A
already specified. Resampling feasibility and detector robustness are distinct.

## 4. Supported outputs and application decision

| Output | Current support and safe meaning |
|---|---|
| Known-vocoder source evidence | A **research candidate match** to a particular included reference pipeline when both frozen gates pass, with model/protocol/band/support and scope disclosed. Clean, exposed-domain selective performance is measurable. Robust forensic identity across processing is unsupported. |
| No supported attribution | The system did not return a known-source label. Show whether the binary route failed, support failed, or the policy rejected all. This says nothing conclusive about authenticity or whether an unseen generator exists. |
| Inconclusive | Appropriate for conflicting/weak evidence, unsupported rates, too little or altered support, and input domains not independently validated. Do not use it as a disguised diagnosis of an unknown generator. |

The existing machine states remain unchanged: `known_source`,
`no_supported_synthetic_source`, `unknown_or_inconclusive`. Every existing
`genuine_claim_supported=false`, `unknown_generator_identity_supported=false`
and uncalibrated-score disclosure remains. No audio is certified genuine.
Real-like rejection is explicitly a property of a weak binary model, not
positive evidence of authenticity or successful unknown-source recognition.

**Recommendation: C first, B next; defer A as a user-facing forensic tool.**
Acquire and verify rights/provenance for independently frozen genuine,
known-source and truly withheld-generator cohorts with multiple speakers and
recording/processing conditions. Keep the current frozen eval split untouched;
do not manufacture a fresh holdout by redistributing exposed dev sources.

The next implementation milestone is a bounded **training-only protocol
comparison**, after a separate specification/review: preserve this frozen
baseline, define rate-aware processing where needed, evaluate one justified
representation/reference reproduction, and compare clean training against a
predeclared symmetric gain/filter/duration training protocol with disjoint
model-fit/calibration/assessment source groups. Keep withheld generators out of
their context's normalization, fit, calibration and selection. Evaluate genuine
false matches, known coverage/correctness and withheld rejection mechanism
separately; include reject-all and binary-only references. Domain-support
checks must face altered genuine recordings too. Recalibrating on these exposed
failures without new group-disjoint assessment would not establish robustness.

This task does not start that training, acquire data, reproduce a new model or
build a frontend. Current evidence justifies a research diagnostic/inspection
artifact, not a general-source identification or authenticity application.

## 5. Implementation, checks and preservation

New source: `src/experiments/vocotrace_abstention.py`. It validates and replays
the historical numeric artifacts, expands all decision paths, records fixed
policy comparisons/distributions/source counts, and runs the locked train-fold
stress experiment with paired uncertainty. Existing detector/training/metric
behavior is not changed. It uses the existing manifest resolver, sanctioned I/O,
filters, feature extractor, numeric bundles and source-group metric utilities.

New tests: `tests/test_vocotrace_abstention.py` (23 cases). They cover all error
categories, both-gate/counterfactual reasons, threshold ties, genuine false
labels, deterministic source selection, no fabricated fold coverage, model-fit/
calibration/dev/eval role rejection, symmetric deterministic transformations,
unchanged waveform parents, delivered PCM16 lower-band checks, undeclared and
invalid conditions, single-step resampling/rate rejection, paired bootstrap
denominators, saved-row tampering, protected output overlap and fresh-plan order.
A synthetic end-to-end robustness fixture includes a forbidden eval path,
repeats generation exactly, compares saved decisions/metrics/waveform receipts,
and rechecks source hashes. No pre-existing test is modified.

Focused tests: **55 passed** (23 new plus 32 existing verification cases),
156.40 s. The initial focused run had one test-fixture failure: an excessively
quiet deterministic tone exceeded the unchanged PCM16 out-of-band gate after
-12 dB attenuation. Raising only the synthetic fixture amplitude made it a
valid supported fixture; no production gate, expected model result or empirical
input was changed. The initial log is preserved. All real empirical stress
inputs subsequently pass the existing gates without exclusions or repair.

Source/test Ruff passes. Repository-wide Ruff retains the **11 existing
generated-notebook findings**. Full pytest: **962 passed, 4 failed, 7 skipped,
1 xfailed**, 1,381.62 s. The four failures match the established identities:

- `test_kaggle_harness.py::TestGeneratedNotebook::test_the_committed_notebook_is_exactly_what_the_harness_generates`
- `test_phase_a_run.py::TestKaggleNotebook::test_the_generated_notebook_is_the_committed_one`
- `test_phase_a_run.py::TestUpstreamImportGuardThroughADirectoryLink::test_namespace_package_through_the_link_passes`
- `test_phase_a_run.py::TestUpstreamImportGuardThroughADirectoryLink::test_namespace_package_from_elsewhere_fails`

The first two are notebook CRLF/LF byte comparisons; the other two are Windows
directory-junction `WinError 5` failures. Existing link-privilege skips and the
unimplemented-ablation expected failure remain. No new failure appears and no
pre-existing test behavior changes. These checks establish implementation
consistency, not scientific validity on a new speaker/generator/channel.

Independent replay validates all **1,344 primary robustness predictions** from
saved features and frozen fold bundles, the 336 feature hashes/parent records,
28 cell/context point results, 2,880 dev and 20,160 OOF policy rows, and all
48 resampling receipts. It confirms source-disjoint scoring and the exact
60/60 counterfactual withheld-dev confidence pass count.

Before/after protection covers **4,605 prior files across 18 roots**, including
v0, expanded/upload data, both partial pilots, Phase 3 outputs, the original
whole-audio pilots and both sealed Phase 4B runs. Paths/sizes/mtimes agree;
all protected non-WAV digests agree. All original tracked-file hashes agree,
including CLAUDE.md, invariant code and the pre-existing `.gitignore` bytes.
The 48 consumed train WAV parent hashes are rechecked before and after scoring.
Eval WAV preservation uses stat/inventory evidence only: no claim of refreshed
eval byte hashes or empirical eval validation is made.

The new sealed analysis has **16 files / 16,740,576 bytes**, including complete
CSV traces rather than a new audio dataset. The original dataset, earlier
pilots, Phase 4B artifacts, gates and frozen splits remain unchanged. Source
changes and this report remain unstaged. Scientific limitations include one
speaker/corpus, only three reference pipelines, unresolved legacy source
provenance/admission, repeatedly exposed dev, small correlated source counts,
fixed first-crop/repeat features, uncalibrated confidence and incomplete channel/
rate/unknown-generator coverage. A clean integrity receipt is not a signal-gate
rerun, generalization result or citable record-mode experiment.

The three new repository files (this report, the analysis driver and its tests)
are ready for **manual review/commit**, with the known full-suite failures
disclosed. They remain untracked/unstaged and the Git index is empty. This is
not readiness for publication or a forensic application.

## 6. Exact commands

From repository PowerShell, with a **fresh** output directory:

```powershell
& '.venv/Scripts/python.exe' -B -X utf8 -m experiments.vocotrace_abstention `
  --historical-run runs/vocotrace-phase4b-verification-v2-20261105 `
  --manifest runs/onboarding/v0-kaggle-input/manifest.csv `
  --out-dir runs/vocotrace-phase4c-abstention-20261106

& '.venv/Scripts/python.exe' -B -m pytest -o addopts='' -q -ra `
  tests/test_vocotrace_abstention.py tests/test_source_verification.py `
  --basetemp runs/vocotrace-phase4c-audit-20261106/pytest-focused-2

& '.venv/Scripts/python.exe' -B -m pytest -o addopts='' -q -ra `
  --basetemp runs/vocotrace-phase4c-audit-20261106/pytest-full-1

& '.venv/Scripts/python.exe' -B -m ruff check src tests
& '.venv/Scripts/python.exe' -B -m ruff check .
& '.venv/Scripts/python.exe' -B -X utf8 `
  runs/vocotrace-phase4c-audit-20261106/independent_verify.py
& '.venv/Scripts/python.exe' -B -X utf8 `
  runs/vocotrace-phase4c-audit-20261106/diagnose_features.py
& '.venv/Scripts/python.exe' -B -X utf8 `
  runs/vocotrace-phase4c-audit-20261106/verify_resampling_loudness.py

git --no-optional-locks rev-parse HEAD
git --no-optional-locks status --short
git --no-optional-locks diff --cached --name-only
git --no-optional-locks diff --check
```

These are the actual output/basetemp names; use new names on rerun. The driver
refuses an existing output or overlap with its historical run/source directory.
Keep its historical run trusted and unchanged. Its locked plan, exact models,
input identities, code digests, environment, output inventory and analysis
receipts define this development replay. No historical artifact is resealed.
