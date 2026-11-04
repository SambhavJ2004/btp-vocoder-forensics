# VocoPatch Phase 2 packaging integrity fix

2026-11-04. **Development-only, non-citable.** Starting HEAD remains
`6628c7a3f6bb95eae2f7a177060e9bca2f827d98`; Phase 2 edits remain unstaged.
This follow-up fixes the one unresolved finding in the independent review.
It changes package admission/validation only. No generator, scientific invariant,
signal gate, checkpoint pin, production audio or saved production receipt changes.

## Reproduction and root cause

Read AGENTS.md, applicable CLAUDE.md rules, the Phase 2 report, preparation and
validation code, and the external independent review/reproducer. Capture all
source-v0, both pilots, expanded generation and existing upload bytes before
edits. Rerun the review's reproduction on a copied 3-source/15-WAV fixture.
The original functions copied and accepted `demo/unrelated-review-note.txt`
after its generation inventory was refreshed. No secret or real input was
modified. `runs/vocopatch-packaging-fix-20261104/reproduction-before.json`
records the successful pre-fix reproduction.

The old preparer admitted paths using a directory prefix: any ordinary path
under `demo/` could pass. The demo validator checked expected preview contents
but did not reject extras. The root inventory then accurately hashed both
legitimate assets and the extra file. An internally consistent inventory did
not establish schema membership. No checksum was assumed valid merely because
the inventory was refreshed: actual current bytes/hashes are checked separately.

| Trust boundary | Authority after the fix |
|---|---|
| Required metadata | Fixed generation and preparation file-name sets |
| Primary audio | Manifest output paths, ultimately recomputed from original sources and deterministic recipes by the unchanged full validator |
| Demo assets | Existing deterministic `preview_records` recipe plus the five defined demo text/timeline files; not the directory prefix or caller-supplied inventory |
| File inventory | Must describe exactly that schema-derived set, excluding only its own root receipt; it cannot extend permitted membership |
| Recorded hashes | Compared with newly computed hashes of every admitted file; self-hashes establish integrity rather than authentication |
| Upload copy | Exact schema, copied generation bytes, documentation hashes, preparation claims and full original-source/sample reconstruction |

## Exact change

The existing preparation module now performs one shared recursive schema check
before generation-package copying and before upload validation. It inspects
entries before descending, rejects aliases/escaping paths, symlinks/junctions,
hardlinks and non-regular entries, requires nonempty files, and validates
canonical relative names (no absolute paths, backslashes, drives or traversal).
Unexpected files **and empty unexpected directories** fail without repair,
deletion, silent filtering or copying. All expected metadata, primary audio and
preview assets must exist. Disabled previews permit no implicit demo assets.

After exact membership is established, every inventory path must equal the
defined file set and every recorded hash is compared to actual bytes. Required
file sizes must be positive; every copied file's byte count must match its
validated source. The unchanged full validator also checks WAV format, frame
counts/duration, all reconstructed samples and labels. Prepared receipts must
contain exactly the admitted generation-file map, not merely a required subset.
Total bytes/files are derived from the checked recursive listing.

No new packaging framework, sidecar schema or regeneration algorithm is added.
Existing legitimate metadata/demo assets are accepted. Existing preparation
code hashes continue to describe their historical creation; the new validator's
code identity is recorded in the separate revalidation audit rather than by
rewriting the old receipt.

The same pre-fix fixture is rejected by both the fixed preparer and validator.
Its unrelated file remains present for diagnosis; no output/staging folder is
created by the rejected preparation attempt. Evidence is `reproduction-after.json`.

## Regression coverage

New `tests/test_partial_audio_packaging.py` adds 30 deterministic CPU cases:
valid packages/legitimate previews, root and recursively nested extra files
with and without resealing, pre-copy rejection, required metadata/primary and
preview removals despite resealing, modified required bytes, empty required
files, traversal/absolute/drive/backslash names, escaped inventory entries,
unexpected empty directories, link rejection and disabled-preview behavior.
Physical links are tested where Windows permits them; a portable branch test
also exercises rejection without requiring link-creation privileges.

Existing Phase 2, Phase 1 review and original partial-audio tests run unchanged.
The legitimate existing 906-file upload receives a separate read-only full
integration validation. All resulting logs/receipts are stored in the new
ignored audit directory, not in an existing dataset folder.

## Revalidation commands

From repository PowerShell, using the existing environment. These are initial
run paths; choose fresh pytest basetemp paths for reruns.

```powershell
& '.venv/Scripts/python.exe' -B -m pytest -o addopts='' -q -ra tests/test_partial_audio_packaging.py tests/test_partial_audio_expansion.py tests/test_partial_audio_review.py tests/test_partial_audio.py --basetemp runs/vocopatch-packaging-fix-20261104/pytest-focused-1
& '.venv/Scripts/python.exe' -B -m pytest -o addopts='' -q -ra --basetemp runs/vocopatch-packaging-fix-20261104/pytest-full-1
& '.venv/Scripts/python.exe' -B -m ruff check src tests
& '.venv/Scripts/python.exe' -B -m ruff check .
& '.venv/Scripts/python.exe' -B -m experiments.cli partial validate-upload --dataset-dir runs/btpvf-partial-v1-kaggle-upload-20261104 --source-manifest runs/onboarding/v0-kaggle-input/manifest.csv
```

Focused result: **209 passed, 3 skipped** in 146.77 s (182 existing cases plus
30 new cases, 27 of which execute on this Windows host). Physical file/directory
symlinks skip with WinError 1314; hardlink creation skips with WinError 5.
These are three new platform skips, separate from the four existing full-suite
symlink skips. Source/test Ruff passes; whole-repository Ruff retains the same
11 generated-notebook findings.

Full pytest: **751 passed, 4 failed, 7 skipped, 1 xfailed**, 346.32 s. Failure
IDs exactly match the preceding Phase 2 run: two generated-notebook CRLF/LF
comparisons and two Windows directory-junction WinError 5 failures. The seven
skips comprise the four existing file-symlink privilege skips and the three
new physical-link cases above. The existing ablation PCM_16-driver expected
failure remains. No new failure, weakened test or changed scientific gate occurs.

The stricter standalone validator passed the **existing upload folder**:
**906 files, 282,837,884 bytes**, including all **870 primary WAVs**. Complete
before/after SHA-256 maps verify every protected artifact unchanged:

| Protected root | Files unchanged |
|---|---:|
| Original btpvf-v0 | 1,606 |
| Initial partial pilot | 77 |
| Phase 1 reviewed pilot | 91 |
| Phase 2 expanded generation | 901 |
| Phase 2 prepared upload | 906 |

The three original source/pilot maps also match the saved pre-expansion
baseline. All 870 primary WAVs, manifests, labels, previews, provenance and
validation receipts remain exact. Source groups remain **174**, with frozen
eligible counts **123 train / 17 dev / 34 eval**. All 200 original source groups
retain their original assignments; 26 exclusions and development-only/non-citable
flags are unchanged. No production dataset regeneration was necessary.
Generation, CLI, sanctioned audio I/O, manifest/invariant code, the checkpoint
verification helper, CLAUDE.md and pyproject hashes match the recorded generation
code identities. Checkpoint pin descriptors are also unchanged in the working tree.

`runs/vocopatch-packaging-fix-20261104/verification.json` consolidates these
checks, code identity and test/lint outcomes. `code_identity.json` connects the
old preparer hash to the original receipt and the fixed validator's new hash;
`packaging-module.patch` shows the follow-up change against the independent
review's exact source copy. The Git index remains unstaged/unchanged and HEAD
remains the existing manually created commit.

## Ownership, licensing and publication

The intended reference is user-confirmed **`sambhavjainsj/btpvf-partial-v1`**.
The [read-only licensing/owner review](vocopatch_licensing_owner_review.md)
separates verified public-domain/MIT/ISC statements from remaining input-lineage,
jurisdiction, revision-specific terms and dataset-license questions. No audio
license or executable upload metadata is invented. Existing artifact metadata
is intentionally preserved, including its null-owner/empty-license template.
Private visibility does not settle redistribution rights. Publishing remains
subject to license/rights review, live owner/slug checks and separate explicit
authorization. No credentials, models, training, upload or Git writes are used.

## Files and review status

This follow-up changes `src/experiments/partial_audio_upload.py`, adds
`tests/test_partial_audio_packaging.py`, this report and the licensing/owner
review, and adds a follow-up pointer in the original Phase 2 report. All earlier
unstaged changes, including the pre-existing `.gitignore` edit, are preserved.
Original source-v0/production generation/pilot artifacts remain immutable.
Scientific limitations, non-citable status and all 18 invariants remain intact.

**Code is ready for manual review and commit**, with the documented existing
notebook/platform test/lint limitations. The one unresolved preparation gap is
fixed and its original reproduction is rejected. No new implementation blocker
is identified in the tested preparation paths. **Kaggle publishing is not yet
authorized or license-cleared**: intended identity is confirmed, but dataset
license/rights, lineage and applicable terms still need resolution, followed by
live owner/slug/privacy checks and explicit user authorization. The old metadata
template remains intentionally non-executable. A manual code commit will not
make this development dataset citable or establish research-gate admission.
