# VocoPatch partial v1: development training and evaluation protocol

Phase 3A follow-up (2026-11-04): [the first CPU temporal baseline](vocopatch_temporal_baseline.md)
implements train/dev sliding-window experiments under common fixed contexts,
explicit IIR-support masks and numeric model bundles. Its poor control
specificity and complete non-citable dev results are reported there. The eval
partition receives no features/predictions. The planning requirements below
remain relevant to future models and final research admission.

2026-11-04. **Development-only and non-citable.** This protocol prepares future
work; no detector is trained and no localization performance is reported here.
The 18 rules in `CLAUDE.md` remain authoritative. These edited development
examples are not Phase A outputs or newly admitted research data.

## Frozen partitions and permitted uses

Start from the original btpvf-v0 utterance assignments: 140 train, 20 dev,
40 eval before eligibility exclusions. Never split generated recordings,
windows or crops independently. All vocoders, both controls, derivatives,
previews and crops of a source utterance inherit that utterance's original
split. The real-to-real donor is from that same real utterance. For any future
multi-speaker corpus, retain speaker-disjoint partitions too.

Only train examples may fit network weights, normalization statistics,
feature selection, learned preprocessing and augmentation distributions.
Development examples may select hyperparameters, stopping epochs, operating
thresholds and proposed robustness checks. Log every such use. They cannot be
added to training while still being reported as held-out development data.
Before any eval inference, freeze the implementation, weights, preprocessing,
thresholds, metric definitions, event rules and confound comparisons in an
existing user-created commit plus a model/protocol hash receipt. No fitting,
threshold selection or redesign based on eval labels or results is permitted.

**This eval cohort is not a pristine final research holdout.** Four eval source
groups and their labels/audio were explicitly reviewed during Phase 1, and
structural/sample integrity of the expanded data is examined during generation.
That is disclosed development exposure, not model optimization in Phase 2.
Preserve its frozen assignment and isolate it from upcoming fitting/tuning;
label any future evaluation as a legacy development benchmark. A final citable
study needs independently verified provenance, applicable signal gates and a
separately frozen untouched cohort. Do not redistribute these 200 utterances
to manufacture a fresh test set. No new ratio is proposed here.

## Prediction targets and annotation coordinates

The primary task is localization in full recordings, not a five-way recording
class prediction. Sample coordinates are zero-based half-open archive indices
at 22,050 Hz. A positive sample (`1`) is in a pure paired-vocoder core of a
mixed recording. Negatives (`0`) are untouched real samples and the pure-real
cores of both control categories. Every mixed recording contains positives
and negatives; null and real-to-real controls contain no synthetic samples.
Source-vocoder identity is a supervised annotation for stratified reporting,
not an input feature. Null identity and different-region real editing are
distinct negative conditions; do not merge their robustness results.

For target `[s,e)` and fade `F`, ignore (`-1`) is `[s,s+F)` and `[e-F,e)`;
the pure core is `[s+F,e-F)`. The control recordings use the same ignore policy
although their physical speech is real everywhere. Exclude ignore samples from
losses and all scored denominators. Never map them to real or synthetic.
Labels are exact origin/manipulation annotations for the saved PCM_16 audio,
not phonetic timestamps or a statement that quantization makes every mixed
sample numerically distinguishable from the real source.

No labels, edit indices, paths, names, class metadata, source digests, group IDs,
ignore masks or donor choices may be model features. The ignore mask may mask
training loss and metric scoring; it must not reveal a prediction boundary at
inference. Inference runs on complete audio without ground-truth timestamps.
Inference windowing must be identical across conditions; define common crop
duration/repeat padding using the repository's INV-11 helpers for the tier.
Do not present crops or repeated samples as independent source utterances.
Mask artificial padding during localization scoring and map predictions back
to original coordinates. No localization-specific crop override is authorized.

## Processing and confound checks required before interpretation

Store the full-band archive unchanged. The primary comparison must apply the
existing INV-17 analysis transform to the entire comparison set, including
real and both controls. Retain INV-18 membership: Griffin-Lim and Vocos cannot
be interpreted as clean high-band vocoder comparisons. A full-band development
result must be explicitly labeled and paired with a band-limited result;
full-band success alone can exploit known bandwidth/resampling signatures.
Any filtering, resampling or temporal-context feature extraction must map labels
and conservatively widen transition/ignore support before scoring. Record the
actual transform, support calculation and effective valid sample counts;
never simply reuse raw sample masks after temporal mixing. No such transform
or label policy is implemented or certified by this dataset expansion.

Before a learned result is believed, rerun applicable signal gates with an
explicit adapted partial-audio protocol; structural validation is insufficient.
The old one-sided gate orientation, historical EER revalidation and paired
uncertainty-method limitations remain separate unresolved research tasks.
Do not reinterpret a passed local integrity receipt as an INV-12 gate pass.

Predeclare boundary-focused and boundary-excluded analyses, control false-alarm
comparisons, per-vocoder and per-split summaries, duration/silence/RMS/bandwidth
checks, and full-band versus primary-band comparisons using train/dev only.
Fix guard widths on dev, documenting effective support, before eval. Future
fade-length/hard-splice ablations need separate deterministic variants; they
are not part of this one-recipe dataset. Null controls test identity processing;
real-to-real controls test some editing discontinuities. The latter relocate
speech content and alter phase/prosody/continuity, so they do not reproduce all
real-to-vocoder seams or eliminate every splice confound.

## Proposed localization metrics, frozen before model evaluation

Use one score per original sample or a declared common frame grid mapped back
to sample coordinates. For every score threshold, remove ignored samples from
both prediction and reference scoring. Fix the deployment threshold on dev.
Report sample precision, recall, F1 and average precision/AUPRC, with the
threshold/grid and valid real/synthetic sample totals. Pool the five recordings
within each source group, calculate its confusion counts and F1, then macro
average over source utterances as the primary summary. Also report pooled
micro counts and per-vocoder summaries; different windows are not independent
replicates. A no-prediction source with positives has recall/F1 zero.
Precision with no predicted positives is undefined and must be shown as such,
not silently converted to a perfect score. Controls have no reference events:
report their false-positive samples per valid minute and predicted events per
minute separately rather than inventing negative-only recall/F1.

For segment metrics let `G` be a pure synthetic reference core, `P` a predicted
segment and `V` the valid mask. Score
`IoU = |P intersect G intersect V| / |(P union G) intersect V|`.
Predicted components lying wholly in ignore are unscored; do not split or
construct model predictions from ground-truth masks. Predeclare deterministic
one-to-one maximum-IoU matching and event precision/recall/F1 at IoU 0.50 and
0.75. Unmatched references count as misses and unmatched valid predictions as
false alarms. Report mean best-core IoU with misses assigned zero; supply
matched detection coverage alongside onset/offset absolute errors. Boundary
errors refer to pure-core bounds, not the onset of a physical crossfade.
Scores at several IoU levels are descriptive, not opportunities to tune on eval.

Any uncertainty resampling must cluster by original source utterance and keep
all its conditions/controls together; window/file-wise independent intervals
are invalid here. A validated grouped/paired uncertainty implementation is
future work, not a capability claimed in this task. Clip-level EER, if later
reported, uses the corrected repository routine and explicitly declared real
score direction/protocol; its interpolated boundary is not a fitted deployment
threshold. Never combine these matched development results with zero-shot or
quality-versus-detectability thesis conclusions without the required evidence.
