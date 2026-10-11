# The Kaggle harness

`experiments.kaggle_harness`, driven by `btpvf kaggle`. It snapshots the code,
runs a kernel and brings the output home. It is a **runner**: it changes nothing
about trim, loudness, alignment, the band limit, the resampler pin or the
exclusion sets, and no invariant depends on it.

## Why it exists

`notebooks/kaggle_v0.ipynb` used to `git clone` GitHub. A run launched after an
edit therefore executed the **last pushed commit**, not the edit — and the person
launching it had no way to tell from the output. The code now travels as a
private Kaggle Dataset built from the working tree.

## Two modes, and the difference is load-bearing

| mode | source | `sha` | citable |
|---|---|---|---|
| `iteration` | the working tree, uncommitted edits included | `"dirty"` | **no** |
| `record` | `git archive <sha>` | the full commit sha | yes |

**Only record-mode output may be cited in `docs/measurements.md`.**
`assert_citable(run)` refuses anything else, and the kernel prints a warning in
its own log. A citation has to name something a reader can check out; iteration
mode ships bytes no commit describes.

Both modes record a `tree_sha256`, a content hash over the staged tree. That is
the code identity a run is attributable to, and it is what makes `"dirty"` still
precise: two iteration runs with the same hash ran the same bytes.

```bash
btpvf kaggle run --mode iteration --conditions griffin_lim --n-utterances 24
btpvf kaggle run --mode record --sha 13b988d          # the v0 spec's defaults
btpvf kaggle code --mode record --sha 13b988d         # snapshot only, no kernel
btpvf kaggle notebook                                 # regenerate the v0 notebook
```

## What ships, and what cannot

The file list comes from `git ls-files --exclude-standard` (plus untracked,
not-ignored files in iteration mode), then `data/` is dropped except
`data/measurements/`.

Using git's own exclusion rather than a hand-maintained list is the security
property: **anything `.gitignore` covers is unreachable from the uploader**, so
`kaggle.json`, `.env` and `*.token` cannot enter the snapshot even by accident.
A hand-written filter would have to anticipate every name.

`data/` is anchored on the first path component, because an unanchored `data/`
also matches `src/data/` — Person A's package. `.gitignore` documents that trap
for itself; the snapshot filter avoids it the same way.

A tracked file that is **not on disk** is not in the working tree and so is not
shipped, but it is recorded in `missing_from_worktree` and warned about. Shipping
less than the index holds, silently, is the class of surprise this harness exists
to remove.

## Credentials

The token is the CLI's business. This module never opens `~/.kaggle`, never puts
a credential in an argument vector or an environment, and never prints one.
`check_auth()` reports the username and auth *method* — which is all
`kaggle config view` prints — and then calls a real endpoint, because a revoked
token looks identical on disk to a good one.

`_kaggle()` adds exactly two environment variables, both encodings (below), and
inherits the rest.

`runs/` is gitignored: it holds logs and downloaded audio, and a log is the
artifact most likely to have captured something it should not.

## The GPU guard

Kaggle assigns either a T4 (compute capability 7.5) or a **P100 (6.0)**, and
recent PyTorch wheels ship no kernels for 6.0. On a P100 a run dies inside the
first convolution with a message about a missing kernel image, hundreds of lines
after the point where it could have said "wrong GPU".

So the **first cell of every kernel** prints the device and exits if capability
< `MIN_COMPUTE_CAPABILITY = 7.0`, before any audio is generated. Restarting the
session usually gets a T4.

A CPU kernel runs the same cell and prints what it got, because "prefer CPU where
the job does not need a GPU" is only checkable if every run says which it was.
CPU kernels consume **no weekly GPU quota**, and `btpvf kaggle run` prints the
requested accelerator — and, for a GPU push, the remaining quota — before it
pushes. `pip_extras` also drops the `[resynth]` extra entirely when no requested
condition needs torch, so a Griffin-Lim-only run installs nothing heavy.

The guard is tested by **executing** it against a stub torch at 6.0 and 7.5. A
guard that is never run is a guard nobody has checked.

## What had to be discovered about the CLI

Installed: **Kaggle CLI 2.2.4**, `auth_method: ACCESS_TOKEN`. The user's note said
"needs kaggle CLI >= 1.8.0"; the shipped version numbering has moved on, and the
command surface with it. Everything below was read out of the installed package
or learned from a failed run, not assumed.

**Metadata fields.** `kernels init` writes `id, title, code_file, language,
kernel_type, is_private, enable_gpu, enable_tpu, enable_internet, machine_shape,
dataset_sources, competition_sources, kernel_sources, model_sources`. The push
path also reads three the template omits: `keywords`, `docker_image` and
`docker_image_pinning_type`. Booleans are written as *strings* in the template
and `get_bool` accepts either.

**`--accelerator` sets `machine_shape`,** and the package says outright that "the
allowed names are in an enum that is not currently included in kagglesdk". So the
accelerator is requested through `enable_gpu` and `machine_shape` is left empty:
a guessed value is a silent request for hardware nobody chose.

**`datasets create` / `version` default to `--dir-mode skip`, which drops every
subdirectory.** The snapshot is uploaded as one zip so that flag has nothing to
drop.

**No command in 2.2.4 reports a dataset's version number.** Not `files`, not
`status`, not `metadata`. The run records `tree_sha256` as the code identity,
which is content-addressed and therefore stronger, and records the version only
when a push happens to print one.

**Kaggle expands an uploaded `.zip` into a directory named after it, extension
stripped.** `btpvf_code.zip` arrives as `btpvf_code/`, with the zip's own top
folder inside it.

**An attached dataset's mount path differs between images.** This account's
kernels get `/kaggle/input/datasets/<owner>/<slug>/`, not the
`/kaggle/input/<slug>/` the docs imply — which is why `meta.json` recorded an
LJSpeech path with `/datasets/` in it. Both the code snapshot and the corpus are
therefore located **by shape** (the provenance file the snapshot always carries;
`metadata.csv` beside `wavs/`) with bounded globs, never `rglob`, which would walk
every wav of an attached corpus.

**`/kaggle/input` is read-only**, so `pip install -e` cannot write its egg-info
there. The tree is copied into `/kaggle/working` first.

**An editable install is not importable in the process that installed it.** The
`.pth` file takes effect when an interpreter *starts*, and the notebook's kernel
started first, so the bootstrap puts `<target>/src` on `sys.path` and imports the
package immediately — failing there rather than three cells later.

**`kernels status` prints the enum's repr:** `has status
"KernelWorkerStatus.COMPLETE"`, not `COMPLETE`. A regex for the bare name never
matches, and the poll loop spins to its timeout on a kernel that finished in
seconds.

**`kernels logs` returns JSON, not a log** — an array of
`{"stream_name", "time", "data"}` records. Piping it anywhere expecting text
gives quoted JSON with every newline escaped.

**On Windows the CLI cannot print its own logs.** A Kaggle log is full of tqdm
block characters, and the CLI writes stdout at the console encoding, so it fails
with `'charmap' codec can't encode character '▍'` and returns *that* where
the log should be. `_kaggle()` sets `PYTHONIOENCODING=utf-8` and `PYTHONUTF8=1`
and decodes with `errors="replace"`: a log is diagnostic output and must never be
the thing that fails.

**`splitlines()` breaks on carriage returns**, so the "keep only the last redraw"
step in `strip_progress` never saw one. It splits on `\n` alone, and matches a
bar anywhere on the line, because tqdm prefixes it with a description
(`real:  38%|###   | 9/24`).

**`kernels get -m`** is the only way to see whether a `dataset_sources` entry was
accepted or silently dropped, and it returns the pinned `docker_image` digest —
the one piece of the environment the harness does not choose. Both go into the
run record.

## The proof runs

Five pushes, each failing on one of the discoveries above, then two that reached
Phase A. All in **iteration mode**, so **none of these numbers may be cited in
`docs/measurements.md`** — they are recorded here because they are evidence about
the harness, not about the ladder.

**`griffin_lim`, 24 utterances, CPU.** Every harness stage worked: snapshot,
dataset version, attach, mount discovery, install, corpus discovery, package
import, Phase A. It then hit the INV-12 gate, which **refused** it:

```
floor -53.565 dBFS  threshold -45.565  real reads 25.8 ms against 40 ms  (36% off)
```

The output came home anyway — `runs/<ts>/output/resynth_v0.failed` — so
`btpvf silence-diag` ran on the returned audio locally, which is the loop paying
for itself. That run also confirmed the shared-implementation fix on real data:
the floors table and the calibration both read −53.565 dBFS.

**`griffin_lim`, 200 utterances, CPU.** All gates passed.

| | |
|---|---|
| derived threshold | **−38.854114 dBFS** (archive), −38.096605 (zeroshot) |
| floor / quiet-frame cut | −46.854114 / −54.272858 |
| real's pad at that threshold | **44.5 ms** against 40 ms — 11.4% off, inside tolerance |
| `silence_duration` EER | 0.470 archive, 0.455 zeroshot |
| `band_offset_db_mean` | 2.36e-09 dB |
| `trim_pad_above_real_db` | +0.8878 archive, +0.8619 zeroshot |

The threshold matched the value predicted from the committed v0 artifacts to six
decimal places, and the pad landed where the interpolation said it would.

**And the subset size matters.** The admissible band — where real reads
`2 × TRIM_PAD_MS` — sits at about −40 dBFS on *both* subsets, because speech
level does not move. The floor estimate moved 6.7 dB between them, so
`floor + 8 dB` lands inside the band at n=200 and 5.6 dB below it at n=24. The
rule holds its shape and the check caught the failure, but the margin is
calibrated against a quantity that varies with which utterances were drawn. That
is a finding for INV-12's open item, not something to tune here.

## A run directory

Output retrieval follows continuation tokens until the CLI returns a successful
page without one. A nonzero CLI exit, a repeated token (including a cycle), or
the page limit raises `KaggleHarnessError`; none is evidence of a complete
download. Already downloaded files are retained for diagnosis. Mocked CLI tests
cover these cases; this does not verify a current remote run or provide resume
and retry support.

```
runs/<UTC timestamp>/
  code/        the upload: btpvf_code.zip, btpvf_snapshot.json, dataset-metadata.json
  kernel/      the notebook and kernel-metadata.json as pushed, plus the server's view
  log.txt      the kernel log, JSON-parsed, progress bars stripped, stderr marked "!"
  output/      whatever the kernel left in /kaggle/working
  run.json     mode, sha, tree hash, dataset action, kernel metadata, status, timings,
               CLI version, docker image
```

The kernel also merges the snapshot's provenance into the dataset's own
`manifest.provenance.json` under `code_snapshot`, so a dataset can never be read
without knowing whether the code that produced it is checkoutable.

## The notebook in the repository is the notebook that runs

`notebooks/kaggle_v0.ipynb` is generated from `V0_SPEC` by `btpvf kaggle
notebook`, and a test asserts that regenerating it changes nothing. Edit the
harness, not the notebook — otherwise the repository stops holding what runs,
which is the same failure the GitHub clone caused in a different place.

A run of some *other* spec writes its notebook only into its own run directory,
so a smoke run cannot quietly become what the repository publishes.
