# Enforcement audit — do the Enforced-by clauses describe the code?

Audited 2026-09-08, all 17 invariants in [`CLAUDE.md`](../CLAUDE.md).

**Why this exists.** The `np.blackman` / Blackman-Harris divergence was a
docstring *and* a CLAUDE.md invariant both describing code that did something
else, and nothing caught it. The failure mode is specific: an Enforced-by clause
is prose, so it drifts silently while the tests around it keep passing. This
pass checks each clause against the code it names.

**Status (2026-09-08, second pass).** All findings triaged and actioned:
D-1 fixed (single ordering + equivalence test), D-6 fixed (fail-closed speaker
splits), D-2/D-3/D-4/D-7/D-9 corrected in prose, D-8 now has a test, D-5 left
as-is with its dormancy stated. A fourth instance of the shape-3 pattern was
found *while fixing D-3* — see the generated-constants blocker in CLAUDE.md.

The findings below are the original pass, kept as written.

## Mechanical results

- **Every symbol named in an Enforced-by clause resolves.** 17/17 invariants,
  0 unresolvable references.
- **Every named enforcement has at least one call site.** No enforcement is
  defined-but-never-invoked, with one qualification (D-5).

Reproduce: `scratchpad/audit_enforced.py` walks the Enforced-by clauses,
resolves each dotted symbol by import, and greps the call graph.

## Divergences

Ordered by how much a reader would be misled.

### D-1 — INV-07: "the only sanctioned path" is not the only path · **serious**

> **Claim.** "`data.preprocess.process_condition_output` is the only sanctioned
> path; `PIPELINE_ORDER` records it."

`build_real_condition` ([phase_a_resynthesis.py:192-202](../src/experiments/phase_a_resynthesis.py))
does **not** call it. It open-codes the pipeline: `resample_once` → slice →
`band_limit_to_ladder` → `check_band_limit` → `normalise_loudness`. So the real
condition runs a second, parallel implementation of the ordering, while every
vocoder condition goes through `process_condition_output`.

The two agree today. Nothing enforces that they keep agreeing, and INV-07 exists
precisely to stop two conditions being processed in different orders. This is
the invariant's own failure mode living inside its enforcement.

Why it happened: `process_condition_output` takes a `TrimSpan`, and the real
condition is what *derives* the span, so it cannot call the function that
consumes it without restructuring.

### D-2 — INV-05: a refusal that does not exist · moderate

> **Claim.** "`write_audio` refuses any non-`.wav` path **and any non-`PCM_16`
> subtype**."

It refuses non-`.wav` paths. There is **no subtype check**: `subtype=SUBTYPE` is
hardcoded in the `sf.write` call and `write_audio` exposes no parameter for it,
so no caller can supply a different subtype.

The guarantee is arguably *stronger* than the claim — impossible beats refused —
but the claim describes a validation that is not there. A reader auditing INV-05
would look for a check and not find one.

### D-3 — INV-11: stale rule text, and a nominal enforcement · moderate

Two problems in one invariant.

> **Claim (Rule).** "Same crop length (**64600 samples**, the ASVspoof
> convention)".

Stale. The crop became duration-based at the two-tier change:
`crop_samples_for_rate(sr)` returns 64600 at 16 kHz and **89027** at 22.05 kHz.
`aasist.py` was updated; the CLAUDE.md rule was not. As written the rule
prescribes the exact bug that change fixed — a 64600-sample crop at the archive
rate is 2.93 s, not 4.04 s.

> **Claim (Enforced by).** "`detectors.base.fixed_length_crop`."

The function exists and is correct, but is **never called** — only imported and
re-exported by `aasist.py` and `detectors/__init__.py`. Every detector `score()`
is `NotImplementedError`, so the crop policy is currently documentation. That is
expected at this stage; the point is that "Enforced by" overstates it.

### D-4 — INV-04: "on every write" is only on archive writes · moderate

> **Claim.** "`check_waveform` (verifies against `LOUDNESS_TOLERANCE_LU` **on
> every write**)".

The LUFS check is conditional: `if measured_lufs is not None`. `_emit_pair`
passes it for the archive write and **deliberately omits it for the zero-shot
write**, because the derived tier is not re-normalised and legitimately drifts
(INV-01). So it verifies on archive writes only.

The behaviour is correct. The claim is not: "every write" would be a bug if it
were true.

### D-5 — INV-08: an enforcement nothing invokes · moderate

> **Claim.** "`vocoders.checkpoints.verify_checkpoint` (digest mismatch ⇒
> raise)."

Correct in itself, and reached from `fetch_and_verify`. But **no pipeline code
calls either** — every vocoder `load()` is `NotImplementedError`, so nothing
verifies a digest on any real path. It is an available tool presented as an
active guard.

Also unenforced in the same invariant: *"Fix sampler step counts and seeds for
any stochastic vocoder"* has no mechanism at all.

### D-6 — INV-09: speaker-disjointness is opt-in, not conditional · moderate

> **Claim (Rule).** "Speaker-disjoint as well **when the corpus is
> multi-speaker**."

`assign_splits(..., speaker_disjoint=False)` and `run(..., speaker_disjoint=False)`
both default to off, and `Corpus` exposes no multi-speaker property. Nothing
connects the corpus to the flag: switching to VCTK requires remembering to pass
`speaker_disjoint=True`, and forgetting is silent.

Moot while `VCTK.utterances()` raises `NotImplementedError` — which is exactly
when it should be fixed, before it can bite.

### D-7 — INV-12: "all four probes" is not tier-true · minor

> **Claim (Rule).** "Probes on duration, silence, RMS and bandwidth must **all**
> sit near chance."

`run_all` runs the bandwidth probe on the archive tier only — correctly, since
on the derived tier every condition shares one downsampler. So three of four run
on zero-shot. The rule states four without the tier qualification it needs.

### D-8 — INV-10 and INV-13: honest about being unenforced, still untested · minor

INV-10 says "Convention and review"; INV-13 says "Lazy imports throughout". Both
accurately decline to claim mechanical enforcement.

INV-13's claim *is* mechanically checkable, though, and it currently **holds**:
importing `vocoders`, `detectors`, `data` and `metrics` pulls none of `torch`,
`torchaudio`, `transformers`, `pesq`, `pyworld`. There is no test asserting it,
so a stray top-level `import torch` would pass CI and only surface as a
dependency conflict on Kaggle.

### D-9 — INV-02: a dormant mechanism · informational

> **Claim.** "a config whose `source` still starts with `TODO` is unverified,
> and `checkpoint_audit` reports that condition as blocked."

The logic is present and correct. But no condition has a `TODO` source any more,
so the branch is never exercised. Not a divergence — flagged because it is the
kind of check that rots unnoticed once nothing trips it.

## The pattern

Nothing is fabricated: every symbol exists, and no invariant is enforced by
code that does the opposite of what it says. The failures are all **overstated
scope**, in three recurring shapes:

1. **"only" / "every" that is not.** D-1, D-4. A qualifier that was true when
   written and quietly stopped being true.
2. **Enforcement that exists but is not yet wired.** D-3, D-5. True of anything
   downstream of an unimplemented `load()` or `score()`; the risk is that
   "Enforced by" reads as active.
3. **Rule text drifting behind the code.** D-3, D-7, and the original
   `np.blackman` case. The Enforced-by clause was updated and the Rule was not,
   or vice versa.

The blackman case belongs to shape 3, and D-3 is the same shape in the same
invariant family — which suggests the recurring risk is **prose in two places
describing one behaviour**, not enforcement being absent.
