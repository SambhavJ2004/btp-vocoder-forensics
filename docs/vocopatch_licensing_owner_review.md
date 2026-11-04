# VocoPatch owner and licensing review

2026-11-04, Asia/Calcutta. **Development-only and non-citable.** Read-only
review of public upstream material and saved local metadata. This records
evidence and open questions; it does not grant rights or authorize publishing.
No credentials, account authentication, model downloads or remote mutation
were used. Private visibility does not remove licensing obligations.

## Owner and identity evidence

The user explicitly confirmed **`sambhavjainsj/btpvf-partial-v1`** as the intended
new private dataset reference during this task. Historical local
`runs/20260928T063211Z/code/dataset-metadata.json` also records
`sambhavjainsj/btpvf-code`, and its kernel metadata records the same owner.
Those files corroborate past configuration, not current account permissions.
No live account-control, slug-collision or dataset-privacy check was performed.
No version of `btpvf-v0` is intended.

The existing immutable upload folder still has `id=null`, `licenses=[]` in
`dataset-metadata.template.json`, no executable `dataset-metadata.json`, and
`upload_authorized=false`, `publication_ready=false` with planned private
visibility. The template's earlier unresolved-owner wording is a historical
preparation state. The confirmation above resolves intended identity only;
authenticated ownership and any future publication remain separate checks.
No template, card, receipt, checksum or audio artifact was rewritten.

## Verified upstream facts

| Component | Verified evidence and scope |
|---|---|
| LJSpeech | The original publisher describes the dataset's text, audio and annotations as public domain in the US, with no use restrictions and optional attribution. The page credits Keith Ito and Linda Johnson/LibriVox and provides a 2017 citation. This is the publisher's stated status; it is not a worldwide rights determination. [Original publisher](https://keithito.com/LJ-Speech-Dataset/). |
| LibriVox | Its policy bases the public-domain claim on US law and explicitly notes that status elsewhere can differ. [LibriVox policy](https://librivox.org/pages/public-domain/). |
| BigVGAN-112m | The official card for `nvidia/bigvgan_v2_22khz_80band_fmax8k_256x` declares MIT; the displayed card commit is `a9d8b71`. The upstream code license is MIT, copyright NVIDIA 2024. [Model card](https://huggingface.co/nvidia/bigvgan_v2_22khz_80band_fmax8k_256x/blob/main/README.md), [code license](https://raw.githubusercontent.com/NVIDIA/BigVGAN/main/LICENSE). |
| Vocos | The exact configured model-card revision `0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21` declares MIT and describes its code as MIT. The official code license credits Charactr Inc., 2023. [Pinned model card](https://huggingface.co/charactr/vocos-mel-24khz/blob/0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21/README.md), [code license](https://raw.githubusercontent.com/gemelo-ai/vocos/main/LICENSE). |
| Griffin-Lim | The local adapter uses librosa and has no learned checkpoint. The saved v0 environment records librosa 0.11.0; its official code license is ISC. [Versioned license](https://raw.githubusercontent.com/librosa/librosa/0.11.0/LICENSE.md). |

The reviewed MIT/ISC terms cover software and notices when distributing copies
of that software. The reviewed cards/licenses contain no express output-specific
or noncommercial restriction. This is a limited textual finding, **not** proof
that every waveform may be relicensed under MIT, ISC or CC0. Generated waveform
rights and any input/third-party obligations require their own review. The
upload contains audio/metadata and preview copies, not vocoder weights or
library source. If software/weights are later distributed, retain their exact
applicable license/copyright notices separately.

The current BigVGAN card was readable, but direct retrieval of its full configured
revision `a9d8b711c344f3c49cc1a05d0c8b6741cf733aa8` and its linked parent-Hub
license failed in the browser tool. The official current code license/card are
verified above; a full revision-specific license archive was not established
here. This must not be reported as authenticated historical checkpoint use.

## Provenance and mirror limits

Saved historical kernel metadata declares
`mathurinache/the-lj-speech-dataset` as an input. The public
[Kaggle mirror page](https://www.kaggle.com/datasets/mathurinache/the-lj-speech-dataset)
returned no readable license text in this review, so its version-specific
metadata/terms were not verified. Current local source hashes do not prove a
download/version chain back to the original publisher or that mirror. The v0
provenance records `git_commit=unknown`; historical execution/alignment evidence
remains unauthenticated. Source and model pins describe current configurations,
not verified historical execution of those bytes.

The original publisher also notes that LibriVox source recordings were MP3;
current PCM_16 WAV integrity does not erase inherited source-codec history.
This is a source limitation, not permission to introduce lossy processing or
change any project invariant. No source data or processing was changed here.

## License selection and outstanding decisions

No dataset license is selected automatically. Public-domain source status is
not itself a formal CC0 dedication of all new edits, annotations, documents
or possible third-party material. The repository's MIT software license and
the old code publisher's CC0 metadata do not establish an audio-package license.

Before publishing, resolve and record:

1. The local input/version lineage, mirror metadata and applicable publishing
   jurisdiction/rights; current integrity hashes alone are insufficient.
2. Revision-specific model/code terms applicable to the actual legacy generation
   and any additional conditions beyond the reviewed public cards/licenses.
3. Who can license the generated labels/curation/docs, and the agreed Kaggle
   dataset license or component-specific notices. Do not fabricate a single
   blanket license or claim that upstream software licenses transfer to audio.
4. Actual account permission and slug availability for the user-confirmed
   `sambhavjainsj/btpvf-partial-v1`; obtain separate explicit upload authorization.
5. A new private publication plan with complete directories and verified retrieved
   hashes/paths, preserving the immutable local evidence and all non-citable flags.

## Attribution and reproducibility record

Retain source credits/URLs, LJSpeech version and utterance IDs, BigVGAN/Vocos
model identifiers and recorded revisions/digests, Griffin-Lim/librosa version,
processing recipe, exclusions, split inheritance, code hashes/HEAD, input/output
inventories and development-only disclosures. LJSpeech attribution is optional
under its publisher's statement; retaining it supports research provenance.
Upstream model citations are requested for research credit and should remain
distinct from license conditions and from any claim that VocoPatch is citable.
No publication or model-validity claim follows from this licensing review.
