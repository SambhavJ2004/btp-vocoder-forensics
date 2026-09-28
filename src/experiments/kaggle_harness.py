"""Kaggle automation — snapshot the code, run a kernel, bring the output home.

This is a RUNNER, not part of the measurement pipeline. It generates and pushes
what Kaggle executes and it records what ran; it never touches trim, loudness,
alignment, the band limit, the resampler pin or the exclusion sets.

Why it exists: ``notebooks/kaggle_v0.ipynb`` used to clone GitHub, so a run
launched after an edit executed the last PUSHED commit rather than the edit. The
code now travels as a private Kaggle Dataset built from the working tree.

Two modes, and the difference is load-bearing:

  iteration   built from the working tree, including uncommitted edits. Fast to
              turn around, and marked ``dirty`` because there is no commit that
              names what ran. Debugging only.
  record      built from ``git archive <sha>`` of a real commit, marked clean.
              **Only record-mode output may be cited in docs/measurements.md**
              (:func:`assert_citable`), because a citation has to name something
              a reader can check out.

Credentials: the Kaggle token is read by the ``kaggle`` CLI itself from
``~/.kaggle``. This module never opens it, never passes it as an argument and
never puts it in the environment it builds. :func:`check_auth` reports the auth
METHOD and username, which is what ``kaggle config view`` prints -- it does not
print the key. The snapshot's file list comes from ``git ls-files
--exclude-standard``, so anything ``.gitignore`` excludes -- ``kaggle.json``,
``.env``, ``*.token`` -- cannot enter the upload even by accident.

What was verified against the installed CLI rather than assumed, because the
field set has changed between versions (see docs/kaggle_harness.md):

  * CLI 2.2.4, ``auth_method: ACCESS_TOKEN``.
  * ``kernel-metadata.json`` accepts ``id, title, code_file, language,
    kernel_type, is_private, enable_gpu, enable_tpu, enable_internet,
    machine_shape, dataset_sources, competition_sources, kernel_sources,
    model_sources`` -- and, undocumented by ``kernels init``, ``keywords``,
    ``docker_image`` and ``docker_image_pinning_type``.
  * ``--accelerator`` on push overrides ``machine_shape``, whose vocabulary the
    installed package does not ship ("the allowed names are in an enum that is
    not currently included in kagglesdk"), so this module drives the accelerator
    through ``enable_gpu`` and leaves ``machine_shape`` empty.
  * ``datasets create``/``version`` default to ``--dir-mode skip``, which
    silently DROPS every subdirectory. The snapshot is uploaded as one zip
    instead, so no directory is handed to that flag.
  * No command in 2.2.4 reports a dataset's version number, so the code
    identity recorded with a run is the snapshot's content hash, which is
    stronger; the version is recorded when the push happens to print one.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import subprocess
import tarfile
import time
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO_ROOT / "runs"

CODE_DATASET_SLUG = "btpvf-code"
CODE_ARCHIVE_NAME = "btpvf_code.zip"
SNAPSHOT_PROVENANCE_NAME = "btpvf_snapshot.json"
SNAPSHOT_ROOT_IN_ZIP = "btpvf-code"

ITERATION, RECORD = "iteration", "record"
MODES = (ITERATION, RECORD)

# The minimum CUDA compute capability recent PyTorch wheels still build kernels
# for. Kaggle can assign a P100, which is 6.0, and a run on one fails deep inside
# the first convolution rather than at the top.
MIN_COMPUTE_CAPABILITY = 7.0

# Kaggle mounts /kaggle/input read-only, so `pip install -e` cannot write its
# egg-info there. The snapshot is materialised into /kaggle/working first.
KAGGLE_INPUT = "/kaggle/input"
KAGGLE_WORKING = "/kaggle/working"

# CLI 2.2.4 prints the enum's repr, `has status "KernelWorkerStatus.COMPLETE"`,
# not the bare name. A regex for the bare name never matches and the poll
# loop spins to its timeout on a kernel that finished in seconds.
_STATUS_RE = re.compile(r'has status "(?:KernelWorkerStatus\.)?([A-Z_]+)"')
_VERSION_RE = re.compile(r"version\s+(\d+)", re.IGNORECASE)
_TERMINAL_STATUS = {"COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED"}


class KaggleHarnessError(RuntimeError):
    """A harness failure. Deliberately NOT InvariantViolation: nothing here is an
    invariant, and conflating "the upload failed" with "the science is wrong"
    would train the team to read past both."""


# --- driving the CLI ----------------------------------------------------------


def _kaggle(*args: str, check: bool = True, timeout: int = 1800) -> subprocess.CompletedProcess:
    """One place the CLI is invoked. No token on the command line, ever.

    The CLI finds its own credentials in ``~/.kaggle``; this passes none. The
    environment is inherited and only two ENCODING variables are added, so there
    is no path by which a secret reaches an argument vector, an environment or a
    log.

    Those two are not optional. ``kernels logs`` prints the log through stdout at
    the console's encoding, and a Kaggle log is full of tqdm block characters, so
    on Windows the CLI dies with ``'charmap' codec can't encode character
    '▍'`` and returns that message where the log should be. Reading is done
    with ``errors="replace"`` for the same reason: a log is diagnostic output and
    must never be the thing that fails.
    """
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    proc = subprocess.run(
        ["kaggle", *args], capture_output=True, text=True, timeout=timeout, cwd=REPO_ROOT,
        env=env, encoding="utf-8", errors="replace",
    )
    if check and proc.returncode != 0:
        raise KaggleHarnessError(
            f"kaggle {' '.join(args)} failed ({proc.returncode})\n"
            f"{proc.stdout.strip()}\n{proc.stderr.strip()}"
        )
    return proc


def _git(*args: str, binary: bool = False) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        ["git", *args], capture_output=not binary, text=not binary, cwd=REPO_ROOT,
        stdout=subprocess.PIPE if binary else None, check=False,
    )
    if proc.returncode != 0:
        detail = proc.stderr if not binary and proc.stderr else ""
        raise KaggleHarnessError(f"git {' '.join(args)} failed ({proc.returncode}) {detail}")
    return proc


def check_auth() -> dict[str, str]:
    """Username and auth METHOD, from ``kaggle config view``, which prints no key.

    Raises if the CLI cannot authenticate. Calling a real endpoint rather than
    trusting a file's presence: a stale or revoked token looks identical on disk.
    """
    config = _kaggle("config", "view").stdout
    values = dict(
        line.lstrip("- ").split(":", 1) for line in config.splitlines() if line.startswith("- ")
    )
    username = values.get("username", "").strip()
    if not username:
        raise KaggleHarnessError(
            "kaggle CLI reports no username. Place a token in ~/.kaggle and check "
            "`kaggle config view`; this harness never reads the token itself."
        )
    _kaggle("datasets", "list", "--mine", "--page-size", "1")
    return {"username": username, "auth_method": values.get("auth_method", "").strip()}


def accelerator_quota() -> str:
    """The weekly GPU/TPU quota table, printed before a GPU push."""
    return _kaggle("quota").stdout.strip()


# --- the code snapshot --------------------------------------------------------


@dataclass(frozen=True)
class Snapshot:
    """What went into the code dataset, and whether a reader can check it out."""

    mode: str
    sha: str                # a commit sha, or "dirty"
    tree_sha256: str        # content hash of the staged tree -- the real identity
    n_files: int
    n_bytes: int
    created_utc: str
    archive: str = CODE_ARCHIVE_NAME
    # Tracked files that are not on disk right now. Iteration mode ships the
    # WORKING TREE, so a file deleted but not yet staged is genuinely not in it --
    # but shipping less than the index holds, silently, is the class of surprise
    # this harness exists to remove. Recorded and warned about instead.
    missing_from_worktree: tuple[str, ...] = ()

    @property
    def citable(self) -> bool:
        return self.mode == RECORD and self.sha != "dirty"

    def as_record(self) -> dict:
        return {**asdict(self), "citable": self.citable}


def _snapshot_paths(mode: str, sha: str | None) -> list[str]:
    """Repo-relative paths to ship, as POSIX strings.

    ``--exclude-standard`` is what keeps credentials out: the snapshot can only
    contain files git would track, so everything .gitignore covers is
    unreachable from here. Iteration mode adds untracked-but-not-ignored files so
    a new module in progress is actually tested.
    """
    if mode == RECORD:
        listed = _git("ls-tree", "-r", "--name-only", sha or "HEAD").stdout
    else:
        listed = _git("ls-files", "-c", "-o", "--exclude-standard").stdout
    return sorted(p for p in (line.strip() for line in listed.splitlines()) if p and _keep(p))


def _keep(rel: str) -> bool:
    """data/ is excluded except data/measurements/, and caches never ship."""
    parts = Path(rel).parts
    if parts and parts[0] == "data":
        return len(parts) > 1 and parts[1] == "measurements"
    if any(p in {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache", "runs"}
           for p in parts):
        return False
    return not rel.endswith((".pyc", ".pyo"))


def _archive_bytes(mode: str, sha: str | None, paths: list[str]) -> dict[str, bytes]:
    """The snapshot's contents in memory, keyed by repo-relative POSIX path."""
    if mode == RECORD:
        raw = _git("archive", "--format=tar", sha or "HEAD", binary=True).stdout
        out: dict[str, bytes] = {}
        with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
            wanted = set(paths)
            for member in tar.getmembers():
                if member.isfile() and member.name in wanted:
                    handle = tar.extractfile(member)
                    if handle is not None:
                        out[member.name] = handle.read()
        missing = wanted - set(out)
        if missing:
            raise KaggleHarnessError(
                f"git archive of {sha} is missing {len(missing)} listed files "
                f"(e.g. {sorted(missing)[:3]})"
            )
        return out
    return {rel: (REPO_ROOT / rel).read_bytes() for rel in paths}


def _tree_sha256(contents: dict[str, bytes]) -> str:
    """Content hash of the snapshot: order-independent, path-sensitive.

    This is the code identity a run records. In iteration mode there is no commit
    to name, and this is what makes "dirty" still attributable to exact bytes.
    """
    digest = hashlib.sha256()
    for rel in sorted(contents):
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(contents[rel]).digest())
    return digest.hexdigest()


def stage_snapshot(upload_dir: Path, *, mode: str, sha: str | None = None) -> Snapshot:
    """Build the upload directory for the code dataset. Returns its provenance.

    The upload holds exactly three files: one deterministic zip of the tree, the
    snapshot provenance as plain JSON so it is readable on Kaggle without
    unpacking anything, and the dataset metadata. No subdirectory, because
    ``--dir-mode`` defaults to dropping those.
    """
    if mode not in MODES:
        raise KaggleHarnessError(f"mode must be one of {MODES}, got {mode!r}")
    if mode == RECORD:
        if not sha:
            raise KaggleHarnessError(
                "record mode needs a commit: it exists so a citation can name "
                "something a reader can check out."
            )
        sha = _git("rev-parse", "--verify", f"{sha}^{{commit}}").stdout.strip()
    elif sha:
        raise KaggleHarnessError("iteration mode ships the working tree; --sha is record-only.")

    paths = _snapshot_paths(mode, sha)
    missing: tuple[str, ...] = ()
    if mode == ITERATION:
        present = [p for p in paths if (REPO_ROOT / p).is_file()]
        missing = tuple(p for p in paths if p not in set(present))
        paths = present
    if not paths:
        raise KaggleHarnessError("the snapshot is empty; refusing to upload nothing.")
    contents = _archive_bytes(mode, sha, paths)

    snapshot = Snapshot(
        mode=mode,
        sha=sha if mode == RECORD else "dirty",
        tree_sha256=_tree_sha256(contents),
        n_files=len(contents),
        n_bytes=sum(len(b) for b in contents.values()),
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        missing_from_worktree=missing,
    )
    if missing:
        print(
            f"[code] WARNING: {len(missing)} tracked file(s) are not on disk and are "
            f"NOT in this snapshot: {', '.join(missing[:5])}"
            + (" ..." if len(missing) > 5 else "")
            + "\n        `git status` to see them; `git restore <path>` to bring one back."
        )

    upload_dir.mkdir(parents=True, exist_ok=True)
    # A fixed timestamp on every entry makes the zip a function of the content
    # alone, so an unchanged tree produces an identical upload.
    with zipfile.ZipFile(upload_dir / CODE_ARCHIVE_NAME, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in sorted(contents):
            info = zipfile.ZipInfo(f"{SNAPSHOT_ROOT_IN_ZIP}/{rel}", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, contents[rel])
    (upload_dir / SNAPSHOT_PROVENANCE_NAME).write_text(
        json.dumps(snapshot.as_record(), indent=2, sort_keys=True), encoding="utf-8"
    )
    return snapshot


def push_code_dataset(upload_dir: Path, snapshot: Snapshot, *, username: str) -> dict:
    """Create or version the private ``btpvf-code`` dataset. Returns what it did.

    Private on both paths: ``datasets create`` is private unless ``--public`` is
    passed, and a version inherits the dataset's visibility.
    """
    ref = f"{username}/{CODE_DATASET_SLUG}"
    (upload_dir / "dataset-metadata.json").write_text(
        json.dumps(
            {"title": "btpvf-code", "id": ref, "licenses": [{"name": "CC0-1.0"}]},
            indent=2,
        ),
        encoding="utf-8",
    )
    exists = _kaggle("datasets", "status", ref, check=False).returncode == 0
    notes = f"{snapshot.mode} {snapshot.sha[:12]} tree {snapshot.tree_sha256[:12]}"
    if exists:
        out = _kaggle("datasets", "version", "-p", str(upload_dir), "-m", notes,
                      "-r", "skip", "-q").stdout
    else:
        out = _kaggle("datasets", "create", "-p", str(upload_dir), "-r", "skip").stdout
    found = _VERSION_RE.search(out)
    return {
        "ref": ref,
        "action": "version" if exists else "create",
        "notes": notes,
        # CLI 2.2.4 exposes no command that reports a dataset's version number;
        # tree_sha256 is the identity that matters and is recorded regardless.
        "version": int(found.group(1)) if found else None,
        "version_source": "parsed from push output" if found else "not exposed by kaggle CLI 2.2.4",
        "cli_output": out.strip(),
    }


def wait_for_dataset(ref: str, *, timeout_s: int = 900, poll_s: int = 10) -> str:
    """Block until the dataset finishes processing; a kernel attached too early
    mounts the previous version."""
    deadline = time.monotonic() + timeout_s
    status = ""
    while time.monotonic() < deadline:
        status = _kaggle("datasets", "status", ref, check=False).stdout.strip()
        if status == "ready":
            return status
        if status.startswith("error"):
            raise KaggleHarnessError(f"dataset {ref} failed to process: {status}")
        time.sleep(poll_s)
    raise KaggleHarnessError(f"dataset {ref} still {status!r} after {timeout_s}s")


# --- the generated kernel -----------------------------------------------------


_GPU_GUARD = """\
# --- GPU guard (generated; see experiments.kaggle_harness) ---
import torch

EXPECT_GPU = __EXPECT_GPU__
MIN_CAPABILITY = __MIN_CAPABILITY__

print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
if EXPECT_GPU:
    if not torch.cuda.is_available():
        raise SystemExit("pushed with enable_gpu=true but no GPU is visible")
    major, minor = torch.cuda.get_device_capability(0)
    capability = major + minor / 10.0
    print("device:", torch.cuda.get_device_name(0), "| compute capability:", capability)
    if capability < MIN_CAPABILITY:
        raise SystemExit(
            f"Refusing to run on compute capability {capability} (< {MIN_CAPABILITY}). "
            "Kaggle assigned a P100; recent PyTorch builds ship no kernels for it, so "
            "this would fail deep inside the first convolution instead of here. "
            "Restart the session to be reassigned, usually a T4."
        )
else:
    print("CPU kernel: no accelerator requested, so this run costs no GPU quota.")
    if torch.cuda.is_available():
        print("note: a GPU is attached but unused; push without --gpu to free the quota.")
"""


_BOOTSTRAP = """\
# --- install the code snapshot (generated) ---
import json, shutil, subprocess, sys, zipfile
from pathlib import Path

# Nothing about where Kaggle puts things is assumed here, because every assumption
# has already cost a run. Two of them:
#   * an attached dataset mounts at /kaggle/input/datasets/<owner>/<slug>/ in this
#     account's image, not the /kaggle/input/<slug>/ the docs imply;
#   * Kaggle EXPANDS an uploaded .zip into a directory named after it with the
#     extension stripped, so btpvf_code.zip arrives as btpvf_code/.
# So: find the mount by the provenance file it always carries, then find the code
# by the pyproject.toml it must contain. Bounded globs, never rglob, which would
# walk every wav of an attached corpus.
_root = Path("__INPUT__")
_found = [h for d in (1, 2, 3, 4)
          for h in sorted(_root.glob("/".join(["*"] * d) + "/__PROVENANCE__"))]
if not _found:
    raise SystemExit("no __PROVENANCE__ under %s; attached: %s" % (
        _root, sorted(p.name for p in _root.iterdir()) if _root.is_dir() else "nothing"))
mounted = _found[0].parent
print("code dataset mounted at", mounted)

target = Path("__WORKING__/__ROOT__")
_trees = [h for d in (0, 1, 2, 3)
          for h in sorted(mounted.glob("/".join(["*"] * d + ["pyproject.toml"])))]
if _trees:
    # /kaggle/input is read-only, so `pip install -e` cannot write its egg-info
    # there: the tree is copied into /kaggle/working first.
    shutil.copytree(_trees[0].parent, target, dirs_exist_ok=True)
    print("copied expanded snapshot from", _trees[0].parent)
elif (mounted / "__ARCHIVE__").is_file():
    with zipfile.ZipFile(mounted / "__ARCHIVE__") as zf:
        zf.extractall("__WORKING__")
    print("unpacked __ARCHIVE__")
else:
    raise SystemExit("no code under %s: %s" % (
        mounted, sorted(p.name for p in mounted.iterdir())))

snapshot = json.loads((mounted / "__PROVENANCE__").read_text())
print("code snapshot:", json.dumps(snapshot, indent=2))
if not snapshot["citable"]:
    print("\\n*** ITERATION MODE -- output of this run must NOT be cited in docs. ***\\n")

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", str(target) + "__EXTRAS__"],
               check=True)

# An editable install is wired up by a .pth file, which only takes effect when an
# interpreter STARTS. This notebook's kernel started before the install, so the
# package is on disk but not importable until its source root is on sys.path.
sys.path.insert(0, str(target / "src"))
import importlib
importlib.invalidate_caches()
import experiments, data, vocoders  # noqa: F401  -- fail here, not three cells later
print("package importable from", experiments.__file__)
"""


def gpu_guard_source(*, expect_gpu: bool) -> str:
    """The first cell of every kernel: say what hardware this is, then refuse a P100.

    Kaggle assigns either a T4 (capability 7.5) or a P100 (6.0), and recent PyTorch
    wheels ship no kernels for 6.0. On a P100 the run dies inside the first
    convolution with a message about a missing kernel image, hundreds of lines
    after the point where it could have said "wrong GPU". This fails in the first
    cell instead, before any audio is generated.

    On a CPU kernel it still prints what it got, because "prefer CPU where the job
    does not need a GPU" is only checkable if every run says which it was.
    """
    return (_GPU_GUARD
            .replace("__EXPECT_GPU__", "True" if expect_gpu else "False")
            .replace("__MIN_CAPABILITY__", repr(MIN_COMPUTE_CAPABILITY)))


def pip_extras(conditions: tuple[str, ...] | list[str]) -> str:
    """``[resynth]`` only if a requested condition actually needs it.

    ``griffin_lim`` is signal processing: it needs neither torch nor vocos, so a
    Griffin-Lim-only run installs nothing heavy and finishes in a fraction of the
    time. Pairs with preferring CPU kernels -- the cheapest run that answers the
    question is the one to prefer, and INV-13 is happier the fewer trees are
    installed.
    """
    from vocoders.registry import get_spec

    if any(get_spec(c).family != "signal_processing" for c in conditions):
        return "[resynth]"
    return ""


def bootstrap_source(extras: str = "[resynth]") -> str:
    """Install the snapshot. The only logic that cannot live in the package,
    because it is what makes the package importable."""
    return (_BOOTSTRAP
            .replace("__EXTRAS__", extras)
            .replace("__INPUT__", KAGGLE_INPUT)
            .replace("__WORKING__", KAGGLE_WORKING)
            .replace("__SLUG__", CODE_DATASET_SLUG)
            .replace("__ROOT__", SNAPSHOT_ROOT_IN_ZIP)
            .replace("__ARCHIVE__", CODE_ARCHIVE_NAME)
            .replace("__PROVENANCE__", SNAPSHOT_PROVENANCE_NAME))


def find_ljspeech_root(search: str | Path = KAGGLE_INPUT, *, max_depth: int = 4) -> Path:
    """Locate an LJSpeech root under the mounted inputs, by its shape.

    Kaggle does not guarantee where an attached dataset lands: the same dataset
    appears as ``/kaggle/input/<slug>/LJSpeech-1.1`` in some images and
    ``/kaggle/input/datasets/<owner>/<slug>/LJSpeech-1.1`` in others, and
    ``data/measurements/meta.json`` records the second shape while the dataset's
    own file listing implies the first. Guessing costs a whole run, so this looks
    for the shape ``data.corpus.LJSpeech`` actually requires -- ``metadata.csv``
    beside a ``wavs/`` directory -- and says what it found when it fails.
    """
    root = Path(str(search))
    if not root.is_dir():
        raise KaggleHarnessError(f"{root} is not a directory; is this running on Kaggle?")
    seen: list[str] = []
    queue: list[tuple[Path, int]] = [(root, 0)]
    while queue:
        current, depth = queue.pop(0)
        if (current / "metadata.csv").is_file() and (current / "wavs").is_dir():
            return current
        if depth >= max_depth:
            continue
        for child in sorted(p for p in current.iterdir() if p.is_dir()):
            seen.append(str(child))
            queue.append((child, depth + 1))
    raise KaggleHarnessError(
        f"no LJSpeech root (metadata.csv beside wavs/) under {root}. Attach "
        "mathurinache/the-lj-speech-dataset, or pass --corpus explicitly. "
        f"Directories seen: {seen[:40]}"
    )


@dataclass(frozen=True)
class KernelSpec:
    """Everything that decides what Kaggle runs. Recorded with the output."""

    slug: str
    title: str
    conditions: tuple[str, ...]
    n_utterances: int
    corpus: str = "auto"
    out_dir: str = f"{KAGGLE_WORKING}/resynth_v0"
    gpu: bool = False
    dataset_sources: tuple[str, ...] = ()
    timeout_s: int = 32400


def notebook_source(spec: KernelSpec) -> dict:
    """The notebook the harness pushes. A driver: every decision is a constant
    and every step is one call into the installed package.

    Deterministic, because ``notebooks/kaggle_v0.ipynb`` is this output written
    to disk and a test asserts regenerating it changes nothing -- the repo then
    holds what actually runs rather than something that resembles it.
    """
    def md(text: str) -> dict:
        return {"cell_type": "markdown", "metadata": {}, "source": text}

    def code(text: str) -> dict:
        return {"cell_type": "code", "execution_count": None, "metadata": {},
                "outputs": [], "source": text}

    conditions = list(spec.conditions)
    cells = [
        md(
            f"# {spec.title}\n\n"
            "**Generated by `btpvf kaggle run` (`experiments.kaggle_harness`). Do not "
            "edit here** -- edit the harness and regenerate, or the repo stops "
            "holding what runs.\n\n"
            "The code comes from the private `btpvf-code` Dataset built from the "
            "working tree or from `git archive` of a commit, **not** from GitHub: a "
            "run launched after an edit would otherwise execute the last pushed "
            "commit. The snapshot's mode, sha and content hash are printed below and "
            "recorded in the manifest's provenance sidecar.\n\n"
            "**INV-13.** `griffin_lim`, `bigvgan_112m` and `vocos` share one "
            "environment here. That is the one stated exception, because BigVGAN and "
            "Vocos were *reported* by the team to coexist in testing; the repository "
            "has not measured it. Preflight refuses any other combination.\n\n"
            f"**Accelerator.** Pushed with `enable_gpu = {str(spec.gpu).lower()}`."
            + ("" if spec.gpu else " A CPU kernel costs no weekly GPU quota.")
        ),
        code(gpu_guard_source(expect_gpu=spec.gpu)),
        code(bootstrap_source(pip_extras(spec.conditions))),
        *(
            [
                md(
                    "## Optional: a Hugging Face token\n\nAnonymous downloads can be "
                    "rate limited. Add a Kaggle Secret named `HF_TOKEN` and uncomment "
                    "the two lines below. It changes nothing about *what* is "
                    "downloaded, because every checkpoint is pinned by revision "
                    "(INV-08). The package reads it from the environment only and "
                    "never prints, logs or records it (INV-15), so it ships commented "
                    "out."
                ),
                code(
                    "import os  # noqa: F401\n\n"
                    "# from kaggle_secrets import UserSecretsClient\n"
                    '# os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")'
                ),
            ]
            if pip_extras(spec.conditions)
            else []
        ),
        md(
            "## Phase A, gated\n\nEvery gate runs: preflight (INV-08/13), alignment "
            "(INV-16), manifest validation, the analysis band limit (INV-17) and the "
            "sanity probes (INV-12). A failed gate writes no output directory; "
            "`--keep-failed` leaves staging aside as `<out>.failed` for diagnosis."
        ),
        code(
            "import subprocess\n\n"
            "from experiments.kaggle_harness import find_ljspeech_root\n\n"
            f"CONDITIONS = {conditions!r}\n"
            f"CORPUS = {spec.corpus!r}\n"
            f"OUT_DIR = {spec.out_dir!r}\n"
            f"N_UTTERANCES = {spec.n_utterances}\n\n"
            "# Kaggle mounts an attached dataset at a path that differs between\n"
            "# images, so the corpus root is found by its shape unless one was given.\n"
            'CORPUS = str(find_ljspeech_root()) if CORPUS == "auto" else CORPUS\n'
            'print("corpus:", CORPUS)\n\n'
            "subprocess.run(\n"
            '    ["btpvf", "resynth", "--corpus-root", CORPUS, "--conditions", *CONDITIONS,\n'
            '     "--out-dir", OUT_DIR, "--n-utterances", str(N_UTTERANCES), "--keep-failed"],\n'
            "    check=True,\n"
            ")"
        ),
        md(
            "## Record which code produced this\n\nThe snapshot's mode, sha and content "
            "hash go into the manifest's provenance sidecar, so a dataset can never be "
            "read without knowing whether the code that made it is checkoutable."
        ),
        code(
            "from experiments.kaggle_harness import record_code_provenance\n\n"
            "# The mount is found by the file the snapshot always carries, not by a\n"
            "# path: this account's kernels do not use the documented layout.\n"
            'print(record_code_provenance(manifest=f"{OUT_DIR}/manifest.csv"))'
        ),
        md("## Reports and output"),
        code(
            "subprocess.run(\n"
            '    ["btpvf", "loudness", "--manifest", f"{OUT_DIR}/manifest.csv",\n'
            f'     "--out", "{KAGGLE_WORKING}/loudness_gain.csv"],\n'
            "    check=True,\n"
            ")\n"
            "import shutil\n"
            "shutil.make_archive(OUT_DIR, \"zip\", root_dir=OUT_DIR)"
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


V0_NOTEBOOK = REPO_ROOT / "notebooks" / "kaggle_v0.ipynb"

# The v0 run, as `notebooks/kaggle_v0.ipynb` holds it. One object so the committed
# notebook, the CLI's defaults and the regeneration test cannot drift apart.
V0_SPEC = KernelSpec(
    slug="btpvf-v0",
    title="btpvf v0 resynthesis",
    conditions=("bigvgan_112m", "vocos", "griffin_lim"),
    n_utterances=200,
    dataset_sources=("mathurinache/the-lj-speech-dataset",),
    gpu=True,
)


def write_v0_notebook(path: Path | None = None) -> Path:
    """Regenerate the committed v0 notebook from :data:`V0_SPEC`.

    Separate from :func:`run_kernel` on purpose: a run of some OTHER spec must not
    overwrite the notebook the repository publishes as its v0 driver.
    """
    return write_notebook(V0_SPEC, path or V0_NOTEBOOK)


def write_notebook(spec: KernelSpec, path: Path) -> Path:
    """Write the generated notebook, newline-stable so a diff means a real change."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(notebook_source(spec), indent=1, ensure_ascii=False) + "\n"
    path.write_bytes(text.encode("utf-8"))
    return path


_SLUG_STRIP = re.compile(r"[^a-z0-9]+")
_KERNEL_URL_RE = re.compile(r"kaggle\.com/code/([\w-]+)/([\w-]+)")


def slugify(title: str) -> str:
    """Kaggle's own slug rule, as far as it can be observed.

    Lowercase, every run of non-alphanumerics collapsed to one hyphen, trimmed.
    """
    return _SLUG_STRIP.sub("-", title.lower()).strip("-")


def check_slug(spec: KernelSpec) -> None:
    """Refuse a title that will not produce the requested slug.

    **Kaggle derives the kernel slug from the TITLE, not from the ``id`` in
    kernel-metadata.json.** Pushing title "btpvf v1 hifigan_v1" with id
    ``btpvf-v1-hifigan`` creates ``btpvf-v1-hifigan-v1`` and prints a warning in
    the middle of its normal output. Everything afterwards -- status, logs,
    output -- then addresses a kernel that does not exist, and the first symptom
    is an unrelated-looking failure several minutes later.

    Checked before the push rather than reconciled after it, because the id is
    what the run record, the output directory and the reader all use.
    """
    expected = slugify(spec.title)
    if expected != spec.slug:
        raise KaggleHarnessError(
            f"Kaggle would slug the title {spec.title!r} as {expected!r}, not "
            f"{spec.slug!r}, and would create the kernel under that name instead. "
            f"Use --title with something that slugs to {spec.slug!r} "
            f"(e.g. {spec.slug.replace('-', ' ')!r}), or --slug {expected!r}."
        )


def kernel_metadata(spec: KernelSpec, *, username: str, code_file: str) -> dict:
    """The fields CLI 2.2.4 actually reads, and nothing invented.

    ``machine_shape`` is left empty on purpose: ``--accelerator`` overrides it and
    the installed package does not ship the enum of valid names, so a guessed
    value would be a silent request for hardware nobody chose.
    """
    return {
        "id": f"{username}/{spec.slug}",
        "title": spec.title,
        "code_file": code_file,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": str(spec.gpu).lower(),
        "enable_tpu": "false",
        "enable_internet": "true",
        "machine_shape": "",
        "dataset_sources": [f"{username}/{CODE_DATASET_SLUG}", *spec.dataset_sources],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }


# --- the run loop -------------------------------------------------------------


def strip_progress(text: str) -> str:
    """Drop carriage-return progress redraws and leftover tqdm bars.

    Kaggle's log keeps every redraw of every bar, which buries the gate output
    under tens of thousands of lines. Only the last state of a redrawn line is
    kept, and a line that is nothing but a bar is dropped.
    """
    out: list[str] = []
    # split("\n"), not splitlines(): splitlines() also breaks on the carriage
    # return a progress bar redraws itself with, so the "keep only the last
    # redraw" step below would never see one and every frame would survive.
    for line in text.split("\n"):
        last = line.split("\r")[-1].rstrip()
        if not last:
            continue
        # A bar can sit anywhere on the line, because tqdm prefixes it with its
        # description: "real:  38%|###   | 9/24" is the shape Phase A produces, so
        # anchoring the match at the start of the line misses every one of them.
        if re.search(r"\d+%\|", last) or re.fullmatch(r"[\s|/\d▏▎▍▌▋▊▉█-]+", last):
            continue
        out.append(last)
    return "\n".join(out)


def read_log(ref: str) -> str:
    """The kernel's log as readable text, progress redraws removed.

    ``kaggle kernels logs`` does not return text: it returns a JSON array of
    ``{"stream_name", "time", "data"}`` records, so piping it anywhere expecting a
    log gives you quoted JSON with every newline escaped. Parsed here, with the
    stream marked so stderr is findable, and the timestamps dropped because the
    useful ordering is already the array's.
    """
    raw = _kaggle("kernels", "logs", ref, check=False).stdout.strip()
    if not raw:
        return ""
    try:
        records = json.loads(raw)
    except json.JSONDecodeError:
        return strip_progress(raw)  # a future CLI may return plain text
    lines: list[str] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        for line in str(record.get("data", "")).splitlines():
            prefix = "! " if record.get("stream_name") == "stderr" else "  "
            lines.append(prefix + line)
    return strip_progress("\n".join(lines))


_PAGE_TOKEN_RE = re.compile(r"Next Page Token\s*=\s*(\S+)")
OUTPUT_PAGE_SIZE = 500


def download_output(ref: str, dest: Path, *, max_pages: int = 200) -> int:
    """Download every output file, following the pagination. Returns pages fetched.

    ``kernels output`` is **paginated with a default page size of 20**, and a
    single call returns one page and prints a ``Next Page Token`` that a caller is
    expected to follow. A run whose output is not followed comes back silently
    PARTIAL: the first bigvgan_base run returned 200 condition files, 8 of real's
    200, and no ``manifest.csv`` at all -- a truncation that looks exactly like a
    complete download unless you count. Counting is what this function replaces.
    """
    dest.mkdir(parents=True, exist_ok=True)
    token: str | None = None
    for page in range(1, max_pages + 1):
        args = ["kernels", "output", ref, "-p", str(dest), "-q",
                "--page-size", str(OUTPUT_PAGE_SIZE)]
        if token:
            args += ["--page-token", token]
        proc = _kaggle(*args, check=False)
        found = _PAGE_TOKEN_RE.search(proc.stdout + proc.stderr)
        if not found or found.group(1) == token:
            return page
        token = found.group(1)
    raise KaggleHarnessError(
        f"{ref}: still paginating after {max_pages} pages of output; refusing to loop"
    )


def poll_kernel(ref: str, *, timeout_s: int, poll_s: int = 20) -> tuple[str, float]:
    """Block until the kernel reaches a terminal status. Returns it and the wait."""
    started = time.monotonic()
    deadline = started + timeout_s
    status = "UNKNOWN"
    while time.monotonic() < deadline:
        out = _kaggle("kernels", "status", ref, check=False).stdout
        found = _STATUS_RE.search(out)
        status = found.group(1) if found else status
        if status in _TERMINAL_STATUS:
            return status, time.monotonic() - started
        time.sleep(poll_s)
    raise KaggleHarnessError(f"{ref} still {status} after {timeout_s}s; check the kernel on Kaggle")


def _server_metadata(ref: str, into: Path) -> dict:
    """What the server recorded for this kernel, including the docker image it ran.

    ``kernels get -m`` is the only way to see that a ``dataset_sources`` entry was
    accepted rather than dropped, and it returns the pinned ``docker_image``
    digest, which is the one piece of the environment the harness cannot choose.
    """
    scratch = into / "server"
    scratch.mkdir(parents=True, exist_ok=True)
    _kaggle("kernels", "get", ref, "-p", str(scratch), "-m", check=False)
    path = scratch / "kernel-metadata.json"
    return json.loads(path.read_text("utf-8")) if path.is_file() else {}


def find_code_snapshot(search: str | Path = KAGGLE_INPUT, *, max_depth: int = 4) -> Path:
    """The directory of the mounted code dataset, found by the file it always holds.

    The same discovery the bootstrap cell does inline, in the package where every
    other caller can use it. The bootstrap has to repeat it because it runs before
    the package is importable -- that duplication is the price of bootstrapping --
    but nothing after the install should be guessing a mount path. Leaving the
    literal ``/kaggle/input/btpvf-code`` in a second place cost a complete
    200-utterance run whose gates had all passed.
    """
    root = Path(str(search))
    hits = [
        hit
        for depth in range(1, max_depth + 1)
        for hit in sorted(root.glob("/".join(["*"] * depth) + f"/{SNAPSHOT_PROVENANCE_NAME}"))
    ] if root.is_dir() else []
    if not hits:
        attached = sorted(p.name for p in root.iterdir()) if root.is_dir() else []
        raise KaggleHarnessError(
            f"no {SNAPSHOT_PROVENANCE_NAME} under {root}; is the {CODE_DATASET_SLUG} "
            f"dataset attached? Found: {attached}"
        )
    return hits[0].parent


def record_code_provenance(manifest: str | Path, mounted: str | Path | None = None) -> Path:
    """Merge the code snapshot's provenance into a run's sidecar. Runs on Kaggle.

    Kept here rather than in the notebook so it is in the reproducibility record,
    and kept out of Phase A so no pipeline code depends on how a run was launched.
    ``mounted`` defaults to :func:`find_code_snapshot` rather than to a path.
    """
    sidecar = Path(str(manifest)).with_suffix(".provenance.json")
    if not sidecar.is_file():
        raise KaggleHarnessError(f"no provenance sidecar at {sidecar}; did the run write output?")
    root = Path(str(mounted)) if mounted is not None else find_code_snapshot()
    snapshot = json.loads((root / SNAPSHOT_PROVENANCE_NAME).read_text("utf-8"))
    payload = json.loads(sidecar.read_text("utf-8"))
    payload["code_snapshot"] = snapshot
    sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return sidecar


def assert_citable(run: dict) -> None:
    """Refuse to treat iteration-mode output as evidence.

    A citation in docs/measurements.md has to name something a reader can check
    out and re-run. Iteration mode ships uncommitted bytes, so its numbers are
    for debugging: real, but not attributable to any commit.
    """
    snapshot = run.get("snapshot", {})
    if not snapshot.get("citable"):
        raise KaggleHarnessError(
            f"this run is not citable: mode={snapshot.get('mode')!r} "
            f"sha={snapshot.get('sha')!r} tree={snapshot.get('tree_sha256', '')[:12]}. "
            "Re-run with --mode record --sha <commit> before citing it in "
            "docs/measurements.md."
        )


def run_kernel(spec: KernelSpec, *, mode: str, sha: str | None = None,
               notebook: Path | None = None, runs_dir: Path | None = None) -> dict:
    """Snapshot the code, push the kernel, wait, bring the output home.

    Returns the run record, which is also written to ``runs/<timestamp>/run.json``.
    """
    auth = check_auth()
    check_slug(spec)  # before a byte is uploaded: a renamed kernel is unaddressable
    username = auth["username"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = (runs_dir or RUNS_DIR) / stamp
    (run_dir / "kernel").mkdir(parents=True, exist_ok=True)

    snapshot = stage_snapshot(run_dir / "code", mode=mode, sha=sha)
    print(f"[code] {snapshot.mode} {snapshot.sha[:12]} "
          f"tree {snapshot.tree_sha256[:12]} {snapshot.n_files} files "
          f"{snapshot.n_bytes / 1e6:.2f} MB")
    dataset = push_code_dataset(run_dir / "code", snapshot, username=username)
    print(f"[code] {dataset['action']} {dataset['ref']}: {dataset['notes']}")
    wait_for_dataset(dataset["ref"])

    # Into the run directory, never over the committed v0 notebook: that one is
    # regenerated from V0_SPEC by `btpvf kaggle notebook`, so a smoke run with
    # different conditions cannot quietly become what the repository publishes.
    code_file = f"{spec.slug}.ipynb"
    write_notebook(spec, run_dir / "kernel" / code_file)
    if notebook is not None:
        write_notebook(spec, notebook)
    metadata = kernel_metadata(spec, username=username, code_file=code_file)
    (run_dir / "kernel" / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    # Say what hardware is being asked for, before asking for it.
    print(f"[push] {metadata['id']} enable_gpu={metadata['enable_gpu']} "
          f"enable_tpu={metadata['enable_tpu']} private={metadata['is_private']}")
    print(f"[push] attaching: {metadata['dataset_sources']}")
    if spec.gpu:
        print("[push] GPU requested; weekly quota now:")
        print(accelerator_quota())
    else:
        print("[push] CPU kernel: no GPU quota consumed.")

    push = _kaggle("kernels", "push", "-p", str(run_dir / "kernel"),
                   "-t", str(spec.timeout_s)).stdout.strip()
    print(f"[push] {push}")

    # The ref Kaggle reports back, not the one that was asked for: `check_slug`
    # should have made them equal, and if it somehow did not, the created kernel
    # is the one to poll.
    ref = metadata["id"]
    found = _KERNEL_URL_RE.search(push)
    if found and f"{found.group(1)}/{found.group(2)}" != ref:
        ref = f"{found.group(1)}/{found.group(2)}"
        print(f"[push] NOTE: Kaggle created {ref}, not {metadata['id']}; following it.")

    server = _server_metadata(ref, run_dir / "kernel")
    if not server:
        raise KaggleHarnessError(
            f"{ref}: the server returned no metadata for the kernel just pushed, so "
            "there is no way to confirm what it recorded. Check the kernel on Kaggle."
        )
    dropped = set(metadata["dataset_sources"]) - set(server.get("dataset_sources", []))
    if dropped:
        raise KaggleHarnessError(
            f"{ref} was pushed with {sorted(dropped)} attached but the server did not "
            "record them; the run would start without its inputs."
        )
    status, waited = poll_kernel(ref, timeout_s=spec.timeout_s + 900)
    print(f"[run] {ref} -> {status} after {waited / 60:.1f} min")

    log = read_log(ref)
    (run_dir / "log.txt").write_text(log, encoding="utf-8")
    output = run_dir / "output"
    pages = download_output(ref, output)

    record = {
        "stamp": stamp,
        "kernel": ref,
        "kernel_requested": metadata["id"],
        "status": status,
        "waited_s": round(waited, 1),
        "snapshot": snapshot.as_record(),
        "code_dataset": dataset,
        "kernel_metadata": metadata,
        "spec": asdict(spec),
        "auth_method": auth["auth_method"],
        "cli_version": _kaggle("--version").stdout.strip(),
        "server_metadata": server,
        "outputs": sorted(p.name for p in output.iterdir()),
        "output_pages": pages,
        "output_files": sum(1 for _ in output.rglob("*") if _.is_file()),
    }
    (run_dir / "run.json").write_text(
        json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"\n[log] {run_dir / 'log.txt'}")
    print(log)
    if status != "COMPLETE":
        raise KaggleHarnessError(f"{ref} finished as {status}; see {run_dir / 'log.txt'}")
    return record
