"""The Kaggle harness: what gets uploaded, what gets generated, what gets refused.

Nothing here touches the network. The CLI is exercised through its output formats
(which is where three of the harness's bugs lived) rather than by calling it, and
the generated cells are compiled and, for the GPU guard, executed against a stub
torch -- a guard that is never run is a guard nobody has checked.
"""

from __future__ import annotations

import json
import subprocess
import sys
import types
import zipfile
from pathlib import Path

import pytest

from experiments.kaggle_harness import (
    _STATUS_RE,
    CODE_DATASET_SLUG,
    MIN_COMPUTE_CAPABILITY,
    V0_SPEC,
    KaggleHarnessError,
    KernelSpec,
    Snapshot,
    _keep,
    _snapshot_paths,
    assert_citable,
    bootstrap_source,
    download_output,
    find_code_snapshot,
    find_ljspeech_root,
    gpu_guard_source,
    kernel_metadata,
    notebook_source,
    pip_extras,
    read_log,
    record_code_provenance,
    stage_snapshot,
    strip_progress,
    write_v0_notebook,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _zip_entries(upload: Path) -> list[str]:
    with zipfile.ZipFile(upload / "btpvf_code.zip") as zf:
        return [n.split("btpvf-code/", 1)[1] for n in zf.namelist()]


class TestCodeSnapshot:
    """What travels to Kaggle, and what must never travel with it."""

    def test_iteration_ships_the_working_tree_and_is_not_citable(self, tmp_path):
        snapshot = stage_snapshot(tmp_path, mode="iteration")
        assert snapshot.mode == "iteration" and snapshot.sha == "dirty"
        assert not snapshot.citable
        assert snapshot.n_files > 40 and snapshot.n_bytes > 0
        assert "src/experiments/kaggle_harness.py" in _zip_entries(tmp_path)

    def test_record_ships_a_commit_and_is_citable(self, tmp_path):
        snapshot = stage_snapshot(tmp_path, mode="record", sha="HEAD")
        assert snapshot.mode == "record" and len(snapshot.sha) == 40
        assert snapshot.citable
        assert "pyproject.toml" in _zip_entries(tmp_path)

    def test_record_needs_a_commit_and_iteration_refuses_one(self, tmp_path):
        with pytest.raises(KaggleHarnessError, match="record mode needs a commit"):
            stage_snapshot(tmp_path, mode="record")
        with pytest.raises(KaggleHarnessError, match="record-only"):
            stage_snapshot(tmp_path, mode="iteration", sha="HEAD")
        with pytest.raises(KaggleHarnessError, match="mode must be"):
            stage_snapshot(tmp_path, mode="whatever")

    def test_data_is_excluded_except_measurements(self, tmp_path):
        """The archive is published as a Kaggle Dataset; only the small committed
        measurement artifacts belong in the code snapshot."""
        entries = _zip_entries(stage_snapshot(tmp_path, mode="iteration") and tmp_path)
        top_data = [e for e in entries if e.startswith("data/")]
        assert top_data, "the measurement artifacts must ship"
        assert all(e.startswith("data/measurements/") for e in top_data), top_data

    def test_src_data_is_not_caught_by_the_data_rule(self, tmp_path):
        """`data/` also matches `src/data/`, which is Person A's package.

        .gitignore documents this trap for itself; the snapshot filter anchors on
        the first path component for the same reason.
        """
        assert _keep("src/data/preprocess.py")
        assert not _keep("data/archive/real/LJ001-0001.wav")
        assert _keep("data/measurements/probes.json")
        entries = _zip_entries(stage_snapshot(tmp_path, mode="iteration") and tmp_path)
        assert any(e.startswith("src/data/") for e in entries)

    def test_caches_and_the_repository_itself_never_ship(self, tmp_path):
        for rel in (".git/config", ".venv/pyvenv.cfg", "src/__pycache__/x.pyc",
                    "runs/20260101T000000Z/run.json", "src/data/x.pyc"):
            assert not _keep(rel), rel
        entries = _zip_entries(stage_snapshot(tmp_path, mode="iteration") and tmp_path)
        assert not any(part in {".git", ".venv", "__pycache__", "runs"}
                       for e in entries for part in e.split("/"))

    def test_a_gitignored_credential_cannot_enter_the_snapshot(self, tmp_path):
        """INV-15, structurally.

        The file list comes from `git ls-files --exclude-standard`, so everything
        .gitignore covers is unreachable from the uploader -- not filtered out by
        a list this module maintains, which would need to anticipate every name.
        """
        planted = REPO_ROOT / "kaggle.json"
        assert not planted.exists(), "refusing to overwrite a real file"
        planted.write_text('{"username": "x", "key": "NOT-A-REAL-KEY"}', encoding="utf-8")
        try:
            assert "kaggle.json" not in _snapshot_paths("iteration", None)
            entries = _zip_entries(stage_snapshot(tmp_path, mode="iteration") and tmp_path)
            assert "kaggle.json" not in entries
        finally:
            planted.unlink()

    def test_the_upload_is_a_function_of_the_content_alone(self, tmp_path):
        """An unchanged tree must produce identical bytes, or every run looks like
        a code change."""
        a, b = tmp_path / "a", tmp_path / "b"
        first = stage_snapshot(a, mode="record", sha="HEAD")
        second = stage_snapshot(b, mode="record", sha="HEAD")
        assert first.tree_sha256 == second.tree_sha256
        assert (a / "btpvf_code.zip").read_bytes() == (b / "btpvf_code.zip").read_bytes()

    def test_the_tree_hash_moves_when_the_content_does(self, tmp_path):
        head = stage_snapshot(tmp_path / "head", mode="record", sha="HEAD")
        older = stage_snapshot(tmp_path / "prev", mode="record", sha="HEAD~1")
        assert head.tree_sha256 != older.tree_sha256
        assert head.sha != older.sha

    def test_the_upload_holds_no_subdirectory(self, tmp_path):
        """`datasets create/version` default to --dir-mode skip, which silently
        DROPS directories. One zip, so that flag has nothing to drop."""
        stage_snapshot(tmp_path, mode="iteration")
        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "btpvf_code.zip", "btpvf_snapshot.json"
        ]
        assert not any(p.is_dir() for p in tmp_path.iterdir())

    def test_a_tracked_file_absent_from_disk_is_recorded_not_hidden(self, tmp_path, monkeypatch):
        """Iteration mode ships the working tree, so a file deleted but not staged
        is genuinely not in it -- and shipping less than the index holds, quietly,
        is the class of surprise this harness exists to remove."""
        import experiments.kaggle_harness as harness

        real = harness._snapshot_paths

        def with_ghost(mode, sha):
            return [*real(mode, sha), "no/such/file.py"]

        monkeypatch.setattr(harness, "_snapshot_paths", with_ghost)
        snapshot = stage_snapshot(tmp_path, mode="iteration")
        # `in`, not equality: the real working tree may legitimately have its own
        # deleted-but-unstaged files, and this test is about the recording.
        assert "no/such/file.py" in snapshot.missing_from_worktree
        assert "no/such/file.py" in json.loads(
            (tmp_path / "btpvf_snapshot.json").read_text(encoding="utf-8")
        )["missing_from_worktree"]


class TestCitability:
    """Only record-mode output may be cited in docs/measurements.md."""

    def _run(self, snapshot: Snapshot) -> dict:
        return {"snapshot": snapshot.as_record()}

    def test_record_mode_is_citable(self, tmp_path):
        assert_citable(self._run(stage_snapshot(tmp_path, mode="record", sha="HEAD")))

    def test_iteration_mode_is_refused_with_its_hash_in_the_message(self, tmp_path):
        snapshot = stage_snapshot(tmp_path, mode="iteration")
        with pytest.raises(KaggleHarnessError, match="not citable") as excinfo:
            assert_citable(self._run(snapshot))
        assert snapshot.tree_sha256[:12] in str(excinfo.value)
        assert "--mode record" in str(excinfo.value)

    def test_a_run_with_no_snapshot_is_refused(self):
        with pytest.raises(KaggleHarnessError, match="not citable"):
            assert_citable({})


class TestGpuGuard:
    """The guard is executed, not merely generated."""

    def _torch(self, monkeypatch, *, available: bool, capability: tuple[int, int], name="stub"):
        stub = types.ModuleType("torch")
        stub.__version__ = "0.0.0+stub"
        stub.cuda = types.SimpleNamespace(
            is_available=lambda: available,
            get_device_capability=lambda _i: capability,
            get_device_name=lambda _i: name,
        )
        monkeypatch.setitem(sys.modules, "torch", stub)

    def _run(self, source: str) -> None:
        exec(compile(source, "guard", "exec"), {"__name__": "__main__"})

    def test_a_p100_is_refused_in_the_first_cell(self, monkeypatch, capsys):
        self._torch(monkeypatch, available=True, capability=(6, 0), name="Tesla P100-PCIE-16GB")
        with pytest.raises(SystemExit) as excinfo:
            self._run(gpu_guard_source(expect_gpu=True))
        assert "6.0" in str(excinfo.value) and "P100" in str(excinfo.value)
        assert "Tesla P100" in capsys.readouterr().out

    def test_a_t4_passes(self, monkeypatch, capsys):
        self._torch(monkeypatch, available=True, capability=(7, 5), name="Tesla T4")
        self._run(gpu_guard_source(expect_gpu=True))
        assert "7.5" in capsys.readouterr().out

    def test_the_threshold_is_the_constant_not_a_literal(self):
        assert MIN_COMPUTE_CAPABILITY == 7.0
        assert f"MIN_CAPABILITY = {MIN_COMPUTE_CAPABILITY}" in gpu_guard_source(expect_gpu=True)

    def test_a_gpu_kernel_with_no_gpu_fails_rather_than_running_on_cpu(self, monkeypatch):
        self._torch(monkeypatch, available=False, capability=(0, 0))
        with pytest.raises(SystemExit, match="no GPU is visible"):
            self._run(gpu_guard_source(expect_gpu=True))

    def test_a_cpu_kernel_says_it_costs_no_quota(self, monkeypatch, capsys):
        self._torch(monkeypatch, available=False, capability=(0, 0))
        self._run(gpu_guard_source(expect_gpu=False))
        assert "no GPU quota" in capsys.readouterr().out

    def test_a_cpu_kernel_handed_a_gpu_says_so(self, monkeypatch, capsys):
        """Prefer CPU where the job does not need one -- checkable only if a run
        that was handed an accelerator it did not ask for says so."""
        self._torch(monkeypatch, available=True, capability=(7, 5))
        self._run(gpu_guard_source(expect_gpu=False))
        assert "attached but unused" in capsys.readouterr().out


class TestKernelMetadata:
    def test_only_fields_the_installed_cli_reads(self):
        """Pinned against CLI 2.2.4's own parsing, because the field set has moved
        between versions and an unread field is a setting that silently does
        nothing."""
        allowed = {
            "id", "title", "code_file", "language", "kernel_type", "is_private",
            "enable_gpu", "enable_tpu", "enable_internet", "machine_shape",
            "dataset_sources", "competition_sources", "kernel_sources", "model_sources",
            "keywords", "docker_image", "docker_image_pinning_type",
        }
        meta = kernel_metadata(V0_SPEC, username="someone", code_file="x.ipynb")
        assert set(meta) <= allowed, set(meta) - allowed

    def test_private_and_the_code_dataset_always_attached(self):
        meta = kernel_metadata(V0_SPEC, username="someone", code_file="x.ipynb")
        assert meta["is_private"] == "true"
        assert meta["dataset_sources"][0] == f"someone/{CODE_DATASET_SLUG}"
        assert "mathurinache/the-lj-speech-dataset" in meta["dataset_sources"]
        assert meta["kernel_type"] == "notebook" and meta["language"] == "python"

    @pytest.mark.parametrize("gpu", [True, False])
    def test_the_accelerator_is_requested_only_through_enable_gpu(self, gpu):
        """`machine_shape` is left empty on purpose: the installed package does not
        ship the enum of valid names, so any value would be a guess at hardware."""
        meta = kernel_metadata(
            KernelSpec(slug="s", title="t", conditions=("griffin_lim",), n_utterances=24,
                       gpu=gpu),
            username="someone", code_file="x.ipynb",
        )
        assert meta["enable_gpu"] == str(gpu).lower()
        assert meta["machine_shape"] == ""
        assert meta["enable_tpu"] == "false"


class TestGeneratedNotebook:
    def test_every_code_cell_compiles(self):
        for cell in notebook_source(V0_SPEC)["cells"]:
            if cell["cell_type"] == "code":
                compile(cell["source"], "cell", "exec")

    def test_the_gpu_guard_is_the_first_code_cell(self):
        cells = notebook_source(V0_SPEC)["cells"]
        first_code = next(c for c in cells if c["cell_type"] == "code")
        assert "GPU guard" in first_code["source"]

    def test_it_installs_from_the_mounted_dataset_and_never_clones(self):
        source = "\n".join(c["source"] for c in notebook_source(V0_SPEC)["cells"])
        assert "pip" in source and "install" in source and "-e" in source
        assert "/kaggle/input" in source
        for forbidden in ("git clone", "github.com", "REPO_URL"):
            assert forbidden not in source, forbidden

    def test_the_committed_notebook_is_exactly_what_the_harness_generates(self, tmp_path):
        """The repo holds what runs, or it holds fiction. Regenerating is a no-op."""
        committed = (REPO_ROOT / "notebooks" / "kaggle_v0.ipynb").read_bytes()
        assert committed == write_v0_notebook(tmp_path / "kaggle_v0.ipynb").read_bytes()

    def test_extras_follow_the_conditions(self):
        assert pip_extras(("griffin_lim",)) == ""
        assert pip_extras(("griffin_lim", "vocos")) == "[resynth]"
        assert "[resynth]" not in bootstrap_source(pip_extras(("griffin_lim",)))

    def test_the_editable_install_is_put_on_sys_path(self):
        """A .pth file only takes effect at interpreter start, and this kernel
        started before the install."""
        source = bootstrap_source()
        assert "sys.path.insert" in source and "invalidate_caches" in source


class TestCliOutputFormats:
    """Three of this harness's bugs were in parsing the CLI, not in calling it."""

    def test_status_is_parsed_from_the_enum_repr(self):
        for printed, expected in (
            ('x has status "KernelWorkerStatus.COMPLETE"', "COMPLETE"),
            ('x has status "KernelWorkerStatus.ERROR"', "ERROR"),
            ('x has status "RUNNING"', "RUNNING"),
        ):
            assert _STATUS_RE.search(printed).group(1) == expected

    def test_logs_are_json_records_not_text(self, monkeypatch):
        import experiments.kaggle_harness as harness

        payload = json.dumps([
            {"stream_name": "stdout", "time": 1.0, "data": "real: 100%|##| 24/24\nall gates\n"},
            {"stream_name": "stderr", "time": 2.0, "data": "a warning\n"},
        ])
        monkeypatch.setattr(harness, "_kaggle",
                            lambda *a, **k: types.SimpleNamespace(stdout=payload, returncode=0))
        log = read_log("someone/something")
        assert "  all gates" in log
        assert "! a warning" in log
        assert "24/24" not in log, "the progress redraw should be gone"

    def test_plain_text_logs_still_work(self, monkeypatch):
        import experiments.kaggle_harness as harness

        monkeypatch.setattr(harness, "_kaggle",
                            lambda *a, **k: types.SimpleNamespace(stdout="just text", returncode=0))
        assert read_log("x/y") == "just text"

    def test_progress_bars_are_stripped_and_content_is_not(self):
        raw = (
            "starting\n"
            "real:   0%|          | 0/24\rreal: 100%|##########| 24/24 [00:03<00:00]\n"
            "  50%|#####     | 12/24\n"
            "manifest OK: all table-level invariants hold\n"
        )
        cleaned = strip_progress(raw)
        assert "starting" in cleaned
        assert "manifest OK: all table-level invariants hold" in cleaned
        assert "0%" not in cleaned and "12/24" not in cleaned


class TestOutputDownload:
    """A partial download must never be reported as complete. No network calls."""

    def _cli_pages(self, monkeypatch, pages):
        calls = []
        replies = iter(pages)

        def run(args, **kwargs):
            calls.append(args)
            return subprocess.CompletedProcess(args, *next(replies))

        monkeypatch.setattr("experiments.kaggle_harness.subprocess.run", run)
        return calls

    def test_all_pages_are_followed_including_tokens_on_stderr(self, monkeypatch, tmp_path):
        calls = self._cli_pages(monkeypatch, [
            (0, "Next Page Token = page-a\n", ""),
            (0, "", "Next Page Token = page-b\n"),
            (0, "Downloaded final page\n", ""),
        ])
        assert download_output("someone/run", tmp_path / "output") == 3
        assert "--page-token" not in calls[0]
        assert calls[1][-2:] == ["--page-token", "page-a"]
        assert calls[2][-2:] == ["--page-token", "page-b"]
        assert all(call[call.index("--page-size") + 1] == "500" for call in calls)

    @pytest.mark.parametrize("failed_page", [1, 2])
    def test_a_cli_failure_is_not_a_successful_last_page(
        self, monkeypatch, tmp_path, failed_page
    ):
        pages = [(0, "Next Page Token = next\n", "")] * (failed_page - 1)
        pages.append((1, "", "Download failed"))
        calls = self._cli_pages(monkeypatch, pages)
        with pytest.raises(KaggleHarnessError, match=r"failed \(1\)"):
            download_output("someone/run", tmp_path)
        assert len(calls) == failed_page

    @pytest.mark.parametrize("tokens", [["a", "a"], ["a", "b", "a"]])
    def test_repeated_tokens_and_cycles_refuse_completeness(
        self, monkeypatch, tmp_path, tokens
    ):
        calls = self._cli_pages(monkeypatch, [
            (0, f"Next Page Token = {token}\n", "") for token in tokens
        ])
        with pytest.raises(KaggleHarnessError, match="repeated output page token"):
            download_output("someone/run", tmp_path)
        assert len(calls) == len(tokens)

    def test_the_page_limit_is_a_failure(self, monkeypatch, tmp_path):
        calls = self._cli_pages(monkeypatch, [
            (0, "Next Page Token = a\n", ""),
            (0, "Next Page Token = b\n", ""),
        ])
        with pytest.raises(KaggleHarnessError, match="still paginating after 2 pages"):
            download_output("someone/run", tmp_path, max_pages=2)
        assert len(calls) == 2

    def test_an_empty_successful_page_terminates(self, monkeypatch, tmp_path):
        calls = self._cli_pages(monkeypatch, [(0, "", "")])
        assert download_output("someone/run", tmp_path / "output") == 1
        assert len(calls) == 1
        assert (tmp_path / "output").is_dir()


class TestCorpusDiscovery:
    """Kaggle's mount path differs between images, so it is found by shape."""

    @pytest.mark.parametrize("prefix", ["the-lj-speech-dataset",
                                        "datasets/mathurinache/the-lj-speech-dataset"])
    def test_found_at_either_mount_depth(self, tmp_path, prefix):
        root = tmp_path / prefix / "LJSpeech-1.1"
        (root / "wavs").mkdir(parents=True)
        (root / "metadata.csv").write_text("x|y|z", encoding="utf-8")
        assert find_ljspeech_root(tmp_path) == root

    def test_it_names_what_it_saw_when_it_fails(self, tmp_path):
        (tmp_path / "something-else").mkdir()
        with pytest.raises(KaggleHarnessError, match="no LJSpeech root") as excinfo:
            find_ljspeech_root(tmp_path)
        assert "something-else" in str(excinfo.value)

    def test_a_directory_without_wavs_is_not_a_corpus(self, tmp_path):
        (tmp_path / "decoy").mkdir()
        (tmp_path / "decoy" / "metadata.csv").write_text("x", encoding="utf-8")
        with pytest.raises(KaggleHarnessError, match="no LJSpeech root"):
            find_ljspeech_root(tmp_path)


class TestSnapshotDiscovery:
    """The literal `/kaggle/input/btpvf-code` cost a complete 200-utterance run
    whose gates had all passed, because the bootstrap's mount discovery was not
    reused by the provenance step."""

    @pytest.mark.parametrize("prefix", ["btpvf-code", "datasets/someone/btpvf-code"])
    def test_found_at_either_mount_depth(self, tmp_path, prefix):
        mount = tmp_path / prefix
        mount.mkdir(parents=True)
        (mount / "btpvf_snapshot.json").write_text("{}", encoding="utf-8")
        assert find_code_snapshot(tmp_path) == mount

    def test_it_names_what_is_attached_when_it_fails(self, tmp_path):
        (tmp_path / "the-lj-speech-dataset").mkdir()
        with pytest.raises(KaggleHarnessError, match="btpvf-code") as excinfo:
            find_code_snapshot(tmp_path)
        assert "the-lj-speech-dataset" in str(excinfo.value)

    def test_the_notebook_does_not_hardcode_a_mount_path(self):
        """Every post-install mount reference must go through the discovery."""
        cells = [c["source"] for c in notebook_source(V0_SPEC)["cells"]
                 if c["cell_type"] == "code"]
        after_bootstrap = "\n".join(cells[2:])
        assert f"/kaggle/input/{CODE_DATASET_SLUG}" not in after_bootstrap


class TestProvenanceRecording:
    def test_the_mount_is_discovered_when_none_is_given(self, tmp_path, monkeypatch):
        """Nothing after the install may name a mount path."""
        import experiments.kaggle_harness as harness

        mount = tmp_path / "datasets" / "someone" / "btpvf-code"
        mount.mkdir(parents=True)
        (mount / "btpvf_snapshot.json").write_text('{"citable": false}', encoding="utf-8")
        monkeypatch.setattr(harness, "find_code_snapshot", lambda: mount)

        manifest = tmp_path / "manifest.csv"
        manifest.with_suffix(".provenance.json").write_text("{}", encoding="utf-8")
        record_code_provenance(manifest)
        payload = json.loads(manifest.with_suffix(".provenance.json").read_text(encoding="utf-8"))
        assert payload["code_snapshot"] == {"citable": False}

    def test_the_snapshot_is_merged_into_the_run_sidecar(self, tmp_path):
        mounted = tmp_path / "mounted"
        mounted.mkdir()
        snapshot = {"mode": "record", "sha": "a" * 40, "tree_sha256": "b" * 64, "citable": True}
        (mounted / "btpvf_snapshot.json").write_text(json.dumps(snapshot), encoding="utf-8")
        manifest = tmp_path / "manifest.csv"
        sidecar = manifest.with_suffix(".provenance.json")
        sidecar.write_text(json.dumps({"git_commit": "unknown"}), encoding="utf-8")

        record_code_provenance(manifest, mounted)
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        assert payload["code_snapshot"] == snapshot
        assert payload["git_commit"] == "unknown", "existing provenance is preserved"

    def test_it_refuses_when_the_run_wrote_no_sidecar(self, tmp_path):
        with pytest.raises(KaggleHarnessError, match="no provenance sidecar"):
            record_code_provenance(tmp_path / "manifest.csv", tmp_path)


class TestTokenHandling:
    """The token is the CLI's business and never this module's."""

    SOURCE = (REPO_ROOT / "src" / "experiments" / "kaggle_harness.py").read_text(encoding="utf-8")

    def test_the_harness_never_reads_or_prints_the_token(self):
        for forbidden in ("access_token", "kaggle.json", "print-access-token",
                          "KAGGLE_KEY", "KAGGLE_USERNAME"):
            assert forbidden not in self.SOURCE.replace(
                "``kaggle.json``", ""
            ).replace("``access_token``", ""), forbidden

    def test_gitignore_covers_every_shape_of_the_credential(self):
        patterns = set((REPO_ROOT / ".gitignore").read_text(encoding="utf-8").split())
        for pattern in ("kaggle.json", ".kaggle/", "access_token", "**/access_token",
                        "**/.kaggle/", "*.token"):
            assert pattern in patterns, pattern

    def test_run_output_is_not_committable(self):
        """runs/ holds logs and downloaded audio, and a log is the one artifact most
        likely to have captured something it should not."""
        assert "runs/" in (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").split()
