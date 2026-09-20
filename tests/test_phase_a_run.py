"""The first real Phase A run: adapters, the gated runner, the CLI, the notebook.

The neural adapters cannot execute here -- torch, torchaudio and vocos are
deliberately not dependencies of this package (INV-13) -- so what is tested is
everything that decides whether a load is ALLOWED and whether its output is
ACCEPTED: pins, digests, config cross-checks, module provenance, device
resolution. The gated runner is driven end to end with stand-in vocoders, which
exercises every gate for real: alignment, manifest, band, sanity, and the atomic
move of output into place.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from data.invariants import (
    ARCHIVE_SR,
    ARCHIVE_TIER,
    LADDER_FMAX,
    SANITY_EER_FLOOR,
    SILENCE_FLOOR_MARGIN_DB,
    SILENCE_FLOOR_PEAK_PERCENTILE,
    SILENCE_PAD_AGREEMENT,
    InvariantViolation,
)
from experiments.sanity_checks import SILENCE_DIAGNOSTIC_THRESHOLDS

REPO_ROOT = Path(__file__).resolve().parents[1]
N_UTTS = 22  # >= ALIGNMENT_PROBE_FILES


ROOM_TONE_DBFS = -70.0  # a recorded noise floor, which is what a corpus has


def _word_gate(n: int, word_s: float = 0.17, gap_s: float = 0.08) -> np.ndarray:
    """An envelope of "words" separated by pauses, with abrupt 5 ms edges.

    Two properties of real speech that fixtures here have to reproduce, because
    the silence machinery reads exactly them:

    - **Edges are abrupt.** Under a slow taper the signal takes hundreds of ms to
      cross any threshold above the floor, so the probe legitimately reads
      hundreds of ms of "silence" and INV-03's pad agreement cannot hold.
    - **There are interior pauses.** The silence threshold is derived from the
      quietest ``SILENCE_FLOOR_FRAME_PERCENTILE`` of frames, so a file that is
      speech end to end has no floor in its quietest decile and the derivation
      returns a level from inside speech. LJSpeech is 17% silent by INV-03's own
      criterion; this gate is 32%.
    """
    gate = np.zeros(n)
    word, gap, edge = (int(s * ARCHIVE_SR) for s in (word_s, gap_s, 0.005))
    start = 0
    while start + word <= n:
        seg = np.ones(word)
        seg[:edge] *= np.linspace(0.0, 1.0, edge)
        seg[-edge:] *= np.linspace(1.0, 0.0, edge)
        gate[start:start + word] = seg
        start += word + gap
    return gate


def _room(x: np.ndarray, rng: np.random.Generator, tau_s: float = 0.008) -> np.ndarray:
    """A short decaying tail, as a recording in a small room has.

    Load-bearing for the silence machinery, not atmosphere. ``librosa.effects.trim``
    returns frame-quantised boundaries and its END overshoots the last non-silent
    frame, so the trim span reaches ~15 ms past where speech stops. Real speech
    keeps decaying through that window; a hard-gated synthetic signal drops
    straight to the floor, which leaves ~20 ms of extra silence inside the span
    and makes the pad-agreement check fail on audio no corpus produces.
    """
    n = int(6 * tau_s * ARCHIVE_SR)
    ir = 0.05 * np.exp(-np.arange(n) / (tau_s * ARCHIVE_SR)) * rng.standard_normal(n)
    ir[0] = 1.0
    return np.convolve(x, ir)[: len(x)]


def _speechlike(seconds: float = 1.5, seed: int = 0) -> np.ndarray:
    """Speech-shaped enough for the silence machinery to be exercised honestly.

    Words with pauses between them (:func:`_word_gate`), a decaying room
    (:func:`_room`), and a ROOM TONE at a realistic distance below speech -- the
    silence threshold is derived from that floor (INV-03/INV-12), and a fixture
    without one is not testing the derivation, it is dodging it.
    """
    rng = np.random.default_rng(seed)
    n = int(ARCHIVE_SR * seconds)
    t = np.arange(n) / ARCHIVE_SR
    f0 = 120 + 5 * (seed % 11)
    x = sum((0.3 / k) * np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 6.28))
            for k in range(1, 60))
    x = _room(x * (1 + 0.4 * np.sin(2 * np.pi * 3.1 * t)) * _word_gate(n), rng)
    x = 0.35 * x / np.max(np.abs(x))
    pad = int(0.15 * ARCHIVE_SR)
    body = np.concatenate([np.zeros(pad), x, np.zeros(pad)])
    room = 10 ** (ROOM_TONE_DBFS / 20.0) * rng.standard_normal(len(body))
    return (body + room).astype(np.float32)


@pytest.fixture
def corpus(tmp_path):
    import soundfile as sf

    root = tmp_path / "LJSpeech-1.1"
    (root / "wavs").mkdir(parents=True)
    lines = []
    for i in range(N_UTTS):
        uid = f"LJ001-{i:04d}"
        sf.write(root / "wavs" / f"{uid}.wav", _speechlike(seed=i), ARCHIVE_SR, subtype="PCM_16")
        lines.append(f"{uid}|x|x")
    (root / "metadata.csv").write_text("\n".join(lines), encoding="utf-8")
    return root


def _register(monkeypatch, transform):
    """Register a stand-in vocoder under 'hifigan_v1': real audio -> transform."""
    import vocoders.registry as registry
    from data.mel import HIFIGAN_V1_MEL
    from vocoders.base import Vocoder, VocoderSpec

    spec = VocoderSpec(
        key="hifigan_v1", display_name="stand-in", family="gan", year=2020,
        params_m=1.0, mel=HIFIGAN_V1_MEL, checkpoint="stand-in",
        checkpoint_revision="0" * 40,  # passes the audit; nothing is downloaded
    )

    class StandIn(Vocoder):
        def load(self):
            return None

        def synthesize(self, mel):
            raise AssertionError("resynthesize is overridden")

        def resynthesize(self, wav, sr):
            return transform(wav.astype(np.float32))

    StandIn.spec = spec
    monkeypatch.setitem(registry.SPECS, "hifigan_v1", spec)
    monkeypatch.setitem(registry._BUILDERS, "hifigan_v1", StandIn)


# ============================================================ the gated run


class TestGatedRun:
    def test_cli_writes_a_gated_dataset_only_after_every_gate(self, corpus, tmp_path, monkeypatch):
        import pandas as pd

        from data.manifest import read_manifest, validate_manifest
        from experiments.cli import main

        _register(monkeypatch, lambda w: w)  # identity: must pass everything
        out = tmp_path / "resynth_v0"
        code = main([
            "resynth", "--corpus-root", str(corpus), "--conditions", "hifigan_v1",
            "--out-dir", str(out), "--n-utterances", str(N_UTTS), "--device", "cpu",
        ])
        assert code == 0
        assert not (tmp_path / "resynth_v0.staging").exists()

        manifest = read_manifest(out / "manifest.csv")
        validate_manifest(manifest, require_zeroshot=True)
        assert set(manifest["condition"]) == {"real", "hifigan_v1"}
        assert set(manifest["tier"]) == {"archive", "zeroshot"}
        assert all(Path(p).is_file() and Path(p).is_relative_to(out) for p in manifest["path"])

        gates = json.loads((out / "gates.json").read_text())
        assert gates["preflight"] == gates["manifest"] == "passed"
        assert gates["alignment"]["hifigan_v1"]["passed"]
        assert gates["band"]["utterances"] == 5
        assert gates["sanity"]["leaked"] == 0 and gates["sanity"]["split"] == "all"

        sanity = pd.read_csv(out / "sanity.csv")
        assert not sanity["leaked"].any()
        loudness = pd.read_csv(out / "loudness_gain.csv")
        assert loudness.set_index("condition").loc["hifigan_v1", "gain_delta_db_mean"] == 0.0

        prov = json.loads((out / "manifest.provenance.json").read_text())
        assert prov["environment_shared_by"] == ["real", "hifigan_v1"]
        assert prov["environment"]["packages"]["librosa"]

    def test_alignment_failure_leaves_no_output(self, corpus, tmp_path, monkeypatch):
        from experiments.phase_a_resynthesis import run_gated

        _register(monkeypatch, lambda w: np.concatenate([np.zeros(40, np.float32), w[:-40]]))
        out = tmp_path / "out"
        with pytest.raises(InvariantViolation, match="INV-16"):
            run_gated(corpus, ["hifigan_v1"], out, N_UTTS, device="cpu")
        assert not out.exists()
        assert not (tmp_path / "out.staging").exists()
        assert not (tmp_path / "out.failed").exists()

    def test_sanity_leak_fails_loudly_and_keep_failed_keeps_it_aside(
        self, corpus, tmp_path, monkeypatch
    ):
        from scipy.signal import butter, sosfiltfilt

        from experiments.phase_a_resynthesis import run_gated

        sos = butter(8, 2000 / (ARCHIVE_SR / 2), output="sos")
        _register(monkeypatch, lambda w: sosfiltfilt(sos, w).astype(np.float32))
        out = tmp_path / "out"
        with pytest.raises(InvariantViolation, match="INV-12"):
            run_gated(corpus, ["hifigan_v1"], out, N_UTTS, device="cpu", keep_failed=True)
        assert not out.exists()
        failed = tmp_path / "out.failed"
        assert (failed / "sanity.csv").is_file(), "the gate's evidence is kept for inspection"


class TestPreflight:
    def _pf(self, corpus, conditions, out, n=N_UTTS):
        from experiments.phase_a_resynthesis import preflight

        preflight(corpus, conditions, out, n)

    def test_griffin_lim_alone_passes(self, corpus, tmp_path):
        self._pf(corpus, ["griffin_lim"], tmp_path / "out")

    @pytest.mark.parametrize(
        "conditions, match",
        [
            ([], "no conditions"),
            (["griffin_lim", "griffin_lim"], "duplicate"),
            (["real"], "not generable"),
            (["specdiff_gan"], "not generable"),
            (["hifigan_v1"], "INV-08.*blocked"),
            (["melgan"], "INV-08.*blocked"),
            (["bigvgan_base"], "INV-08.*checkpoint_sha256"),
            (["bigvgan_112m", "bigvgan_v2_22khz_fullband"], "INV-08"),
        ],
    )
    def test_refuses_before_writing_anything(self, corpus, tmp_path, conditions, match):
        with pytest.raises(InvariantViolation, match=match):
            self._pf(corpus, conditions, tmp_path / "out")
        assert not any(tmp_path.glob("out*"))

    def test_shared_environment_is_limited_to_the_reported_set(
        self, corpus, tmp_path, monkeypatch
    ):
        """INV-13: bigvgan_112m and vocos may share a process; nothing else may."""
        import vocoders.registry as registry

        monkeypatch.setitem(
            registry.SPECS, "bigvgan_base",
            replace(registry.SPECS["bigvgan_base"], checkpoint_sha256="a" * 64),
        )
        with pytest.raises(InvariantViolation, match="INV-13.*share"):
            self._pf(corpus, ["bigvgan_112m", "bigvgan_base"], tmp_path / "out")

    def test_the_v0_subset_is_the_reported_shared_set(self):
        from vocoders.registry import SHARED_ENV_REPORTED, V0_CONDITIONS

        assert set(V0_CONDITIONS) == SHARED_ENV_REPORTED == {
            "griffin_lim", "bigvgan_112m", "vocos"
        }

    def test_missing_runtime_modules_fail_before_generation(self, corpus, tmp_path, monkeypatch):
        import importlib.util

        real_find = importlib.util.find_spec
        monkeypatch.setattr(
            importlib.util, "find_spec",
            lambda name, *a, **k: None if name in ("vocos", "torch") else real_find(name, *a, **k),
        )
        with pytest.raises(InvariantViolation, match="INV-13.*not importable"):
            self._pf(corpus, ["vocos"], tmp_path / "out")

    def test_refuses_an_existing_output(self, corpus, tmp_path):
        (tmp_path / "out").mkdir()
        with pytest.raises(InvariantViolation, match="already exists"):
            self._pf(corpus, ["griffin_lim"], tmp_path / "out")

    def test_refuses_a_leftover_staging_directory(self, corpus, tmp_path):
        (tmp_path / "out.staging").mkdir()
        with pytest.raises(InvariantViolation, match="already exists"):
            self._pf(corpus, ["griffin_lim"], tmp_path / "out")

    def test_too_few_utterances_for_the_alignment_gate(self, corpus, tmp_path):
        with pytest.raises(InvariantViolation, match="INV-16"):
            self._pf(corpus, ["griffin_lim"], tmp_path / "out", n=10)

    def test_corpus_must_be_ljspeech_shaped(self, tmp_path):
        with pytest.raises(InvariantViolation, match="LJSpeech"):
            self._pf(tmp_path, ["griffin_lim"], tmp_path / "out")


# ============================================================ adapters


class TestGriffinLimAdapter:
    def test_deterministic_full_band_zero_and_length(self):
        from data.invariants import out_of_band_energy
        from vocoders.registry import get_vocoder

        gl = get_vocoder("griffin_lim")
        gl.load()
        x = _speechlike(0.6, seed=3)
        a, b = gl.resynthesize(x, ARCHIVE_SR), gl.resynthesize(x, ARCHIVE_SR)
        assert np.array_equal(a, b), "INV-08: the phase init must be seeded"
        assert abs(len(a) - len(x)) < gl.spec.mel.hop_length
        assert out_of_band_energy(a, ARCHIVE_SR, LADDER_FMAX) < 1e-6
        assert a.dtype == np.float32

    def test_seed_is_passed_under_the_installed_librosa_name(self):
        import inspect

        import librosa

        from vocoders import griffin_lim

        assert griffin_lim._SEED_KWARG in inspect.signature(librosa.griffinlim).parameters

    def test_needs_nothing_at_runtime(self):
        from vocoders.griffin_lim import GriffinLim

        assert GriffinLim.runtime_modules == () and GriffinLim.required_pins == ()


class TestBigVGANAdapter:
    # config.json at nvidia/bigvgan_v2_22khz_80band_fmax8k_256x@a9d8b711, mel keys
    CONFIG = {"sampling_rate": 22050, "n_fft": 1024, "num_mels": 80, "hop_size": 256,
              "win_size": 1024, "fmin": 0, "fmax": 8000}

    def test_pinned_config_matches_the_audit(self):
        from vocoders.bigvgan import LARGE_SPEC, check_hparams

        check_hparams(self.CONFIG, LARGE_SPEC.mel, condition="bigvgan_112m")

    def test_null_fmax_resolves_to_nyquist(self):
        from vocoders.bigvgan import FULLBAND_SPEC, check_hparams

        check_hparams({**self.CONFIG, "fmax": None}, FULLBAND_SPEC.mel, condition="fullband")

    @pytest.mark.parametrize("key, value", [("num_mels", 100), ("fmax", 11025), ("hop_size", 300)])
    def test_config_disagreeing_with_the_audit_is_refused(self, key, value):
        from vocoders.bigvgan import LARGE_SPEC, check_hparams

        with pytest.raises(InvariantViolation, match="INV-02"):
            check_hparams({**self.CONFIG, key: value}, LARGE_SPEC.mel, condition="bigvgan_112m")

    def _downloader(self, root, calls, weights=b"not really weights"):
        def download(*, repo_id, filename, revision, token=None):
            calls.append((repo_id, filename, revision))
            path = root / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(weights if filename.endswith(".pt") else b"# upstream file\n")
            return str(path)
        return download

    def test_everything_comes_from_one_pinned_revision_and_is_verified(self, tmp_path):
        import hashlib

        from vocoders.bigvgan import (
            LARGE_SPEC,
            UPSTREAM_CODE_FILES,
            UPSTREAM_CONFIG,
            UPSTREAM_WEIGHTS,
            fetch_pinned_snapshot,
        )

        calls = []
        spec = replace(LARGE_SPEC, checkpoint_sha256=hashlib.sha256(b"w").hexdigest())
        root = fetch_pinned_snapshot(spec, self._downloader(tmp_path / "snap", calls, b"w"))
        assert root == tmp_path / "snap"
        assert {f for _, f, _ in calls} == {*UPSTREAM_CODE_FILES, UPSTREAM_CONFIG, UPSTREAM_WEIGHTS}
        assert {(r, rev) for r, _, rev in calls} == {
            (LARGE_SPEC.checkpoint, LARGE_SPEC.checkpoint_revision)
        }

    def test_weight_digest_mismatch_fails_before_any_import(self, tmp_path):
        from vocoders.bigvgan import LARGE_SPEC, fetch_pinned_snapshot

        with pytest.raises(InvariantViolation, match="INV-08: checkpoint digest mismatch"):
            fetch_pinned_snapshot(LARGE_SPEC, self._downloader(tmp_path / "snap", []))
        assert "bigvgan" not in sys.modules

    def test_no_revision_or_no_digest_downloads_nothing(self, tmp_path):
        from vocoders.bigvgan import BASE_SPEC, LARGE_SPEC, fetch_pinned_snapshot

        for spec in (replace(LARGE_SPEC, checkpoint_revision=""), BASE_SPEC):
            calls = []
            with pytest.raises(InvariantViolation, match="INV-08"):
                fetch_pinned_snapshot(spec, self._downloader(tmp_path / "snap", calls))
            assert calls == []

    def test_upstream_modules_must_come_from_the_snapshot(self, tmp_path, monkeypatch):
        from vocoders.bigvgan import import_upstream

        root = tmp_path / "snap"
        root.mkdir()
        for name in ("bigvgan", "meldataset", "env"):
            (root / f"{name}.py").write_text(f"WHO = {name!r}\n")
        stray = SimpleNamespace(__file__=str(tmp_path / "elsewhere" / "utils.py"))
        monkeypatch.setitem(sys.modules, "utils", stray)
        with pytest.raises(InvariantViolation, match="INV-13.*'utils'"):
            import_upstream(root)

    def test_clean_import_is_proven_to_come_from_the_snapshot(self, tmp_path, monkeypatch):
        from vocoders.bigvgan import import_upstream

        root = tmp_path / "snap"
        root.mkdir()
        for name in ("bigvgan", "meldataset", "env"):
            (root / f"{name}.py").write_text(f"WHO = {name!r}\n")
        monkeypatch.setattr(sys, "path", list(sys.path))
        for name in ("bigvgan", "meldataset", "env"):
            monkeypatch.delitem(sys.modules, name, raising=False)
        try:
            modules = import_upstream(root)
            assert {n: m.WHO for n, m in modules.items()} == {
                "bigvgan": "bigvgan", "meldataset": "meldataset", "env": "env"
            }
        finally:
            for name in ("bigvgan", "meldataset", "env"):
                sys.modules.pop(name, None)

    def test_declares_what_it_needs(self):
        from vocoders.bigvgan import BigVGAN

        assert {"torch", "huggingface_hub", "matplotlib"} <= set(BigVGAN.runtime_modules)
        assert BigVGAN.required_pins == ("checkpoint_revision", "checkpoint_sha256")


class TestVocosAdapter:
    def _fe(self, **over):
        mel_spec = dict(sample_rate=24000, n_fft=1024, hop_length=256, win_length=1024,
                        n_mels=100, f_min=0.0, f_max=None)
        mel_spec.update(over)
        return SimpleNamespace(mel_spec=SimpleNamespace(**mel_spec), padding="center")

    def test_audited_front_end_passes(self):
        from vocoders.vocos import SPEC, check_feature_extractor

        check_feature_extractor(self._fe(), SPEC.mel)

    @pytest.mark.parametrize("field, value", [("n_mels", 80), ("sample_rate", 22050),
                                              ("f_max", 8000.0), ("hop_length", 300)])
    def test_front_end_disagreeing_with_the_audit_is_refused(self, field, value):
        from vocoders.vocos import SPEC, check_feature_extractor

        with pytest.raises(InvariantViolation, match="INV-02"):
            check_feature_extractor(self._fe(**{field: value}), SPEC.mel)

    def test_padding_and_extractor_type_are_checked(self):
        from vocoders.vocos import SPEC, check_feature_extractor

        fe = self._fe()
        fe.padding = "same"
        with pytest.raises(InvariantViolation, match="padding"):
            check_feature_extractor(fe, SPEC.mel)
        with pytest.raises(InvariantViolation, match="not the audited"):
            check_feature_extractor(SimpleNamespace(), SPEC.mel)

    def test_load_refuses_an_unpinned_revision_before_importing(self):
        from vocoders.vocos import SPEC, Vocos

        v = Vocos()
        v.spec = replace(SPEC, checkpoint_revision="")
        with pytest.raises(InvariantViolation, match="INV-08"):
            v.load()

    def test_declares_what_it_needs(self):
        from vocoders.vocos import Vocos

        assert {"torch", "torchaudio", "vocos"} <= set(Vocos.runtime_modules)
        assert Vocos.required_pins == ("checkpoint_revision",)


class TestDeviceResolution:
    def _fake_torch(self, monkeypatch, cuda: bool):
        cudnn = SimpleNamespace(benchmark=True, deterministic=False)
        fake = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: cuda),
                               backends=SimpleNamespace(cudnn=cudnn))
        monkeypatch.setitem(sys.modules, "torch", fake)
        return cudnn

    def test_auto_picks_cpu_without_a_gpu_and_pins_cudnn(self, monkeypatch):
        from vocoders.base import resolve_torch_device

        cudnn = self._fake_torch(monkeypatch, cuda=False)
        assert resolve_torch_device(None) == resolve_torch_device("auto") == "cpu"
        assert cudnn.benchmark is False and cudnn.deterministic is True

    def test_explicit_cuda_without_a_gpu_raises(self, monkeypatch):
        from vocoders.base import resolve_torch_device

        self._fake_torch(monkeypatch, cuda=False)
        with pytest.raises(RuntimeError, match="no CUDA"):
            resolve_torch_device("cuda")


# ============================================================ the notebook


class TestKaggleNotebook:
    PATH = REPO_ROOT / "notebooks" / "kaggle_v0.ipynb"

    def _nb(self):
        return json.loads(self.PATH.read_text(encoding="utf-8"))

    def _code(self):
        return ["".join(c["source"]) for c in self._nb()["cells"] if c["cell_type"] == "code"]

    def test_is_a_valid_v4_notebook(self):
        nb = self._nb()
        assert nb["nbformat"] == 4 and nb["cells"]
        for c in nb["cells"]:
            assert c["cell_type"] in ("markdown", "code")
            if c["cell_type"] == "code":
                assert c["outputs"] == [] and c["execution_count"] is None

    def test_holds_no_logic(self):
        """A driver: constants and CLI calls. Logic in a notebook is outside the
        reproducibility record (working rules)."""
        import re

        banned = re.compile(r"^\s*(def |class |for |while |if |elif |else:|try:|with |lambda)",
                            re.MULTILINE)
        for src in self._code():
            assert not banned.search(src), f"logic in notebook cell:\n{src}"

    def test_runs_the_gated_cli_for_the_v0_subset(self):
        from vocoders.registry import V0_CONDITIONS

        code = "\n".join(self._code())
        assert "btpvf" in code and '"resynth"' in code and '"loudness"' in code
        assert "N_UTTERANCES = 200" in code
        for cond in V0_CONDITIONS:
            assert f'"{cond}"' in code
        assert "make_archive" in code
        assert "check=True" in code, "a failed gate must stop the notebook"

    def test_states_the_shared_environment_rather_than_assuming_it(self):
        md = "\n".join("".join(c["source"]) for c in self._nb()["cells"]
                       if c["cell_type"] == "markdown")
        assert "INV-13" in md and "same environment" in md
        assert "hifigan_v1" in md and "melgan" in md

    def test_carries_no_credentials(self):
        """INV-15. The HF_TOKEN variable name may appear; a token value may not."""
        import re

        raw = self.PATH.read_text(encoding="utf-8")
        for needle in ("kaggle.json", "api_key", "password"):
            assert needle not in raw.lower()
        assert not re.search(r"\b(hf|ghp|gho|github_pat)_[A-Za-z0-9_]{20,}", raw)

    def test_hf_token_is_optional_and_never_printed(self):
        """Rate-limit relief is opt-in: the secret lines ship commented out, and no
        line that touches the token prints anything."""
        lines = [ln for src in self._code() for ln in src.splitlines() if "HF_TOKEN" in ln]
        assert lines, "the notebook should show how to supply HF_TOKEN"
        for ln in lines:
            assert "print" not in ln and "display" not in ln, ln
            if "get_secret" in ln or "os.environ[" in ln:
                assert ln.lstrip().startswith("#"), f"must ship commented out: {ln}"


# ============================================================ the HF cache layout


def _file_symlink(target: Path, link: Path) -> None:
    """A real relative file symlink, as huggingface_hub makes them. Skips where the OS
    refuses (Windows without Developer Mode or elevation); never mocked."""
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(os.path.relpath(target, link.parent), link)
    except (OSError, NotImplementedError) as err:
        pytest.skip(f"this OS will not create file symlinks here: {err}")


def _dir_link(target: Path, link: Path) -> None:
    """A real directory link: a symlink, or on Windows a junction (needs no privilege)."""
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(target, link, target_is_directory=True)
        return
    except (OSError, NotImplementedError):
        pass
    if sys.platform == "win32":
        import _winapi

        _winapi.CreateJunction(str(target), str(link))
        return
    pytest.skip("this OS will create neither a directory symlink nor a junction")


_UPSTREAM_STUBS = {
    "bigvgan.py": (
        "WHO = 'bigvgan'\nimport utils, activations\nimport alias_free_activation.torch.act\n"
    ),
    "meldataset.py": "WHO = 'meldataset'\n",
    "env.py": "WHO = 'env'\n",
    "utils.py": "WHO = 'utils'\n",
    "activations.py": "WHO = 'activations'\n",
    "alias_free_activation/torch/__init__.py": "",
    "alias_free_activation/torch/act.py": "WHO = 'act'\n",
}
_IMPORTED = ("bigvgan", "meldataset", "env", "utils", "activations", "alias_free_activation",
             "alias_free_activation.torch", "alias_free_activation.torch.act")


@pytest.fixture
def isolated_imports(monkeypatch):
    """Upstream's generic module names must not leak between tests."""
    import importlib

    monkeypatch.setattr(sys, "path", list(sys.path))
    for name in _IMPORTED:
        monkeypatch.delitem(sys.modules, name, raising=False)
    yield
    for name in _IMPORTED:
        sys.modules.pop(name, None)
    importlib.invalidate_caches()


@pytest.fixture
def hf_snapshot(tmp_path):
    """models--x/blobs/<sha256> real files; snapshots/<rev>/ a tree of relative symlinks
    into them -- the layout huggingface_hub builds, and the thing that broke on Kaggle."""
    import hashlib

    repo = tmp_path / "hub" / "models--nvidia--bigvgan_v2_22khz_80band_fmax8k_256x"
    blobs = repo / "blobs"
    snap = repo / "snapshots" / "a9d8b711c344f3c49cc1a05d0c8b6741cf733aa8"
    blobs.mkdir(parents=True)
    for rel, text in _UPSTREAM_STUBS.items():
        blob = blobs / hashlib.sha256((rel + text).encode()).hexdigest()
        blob.write_text(text)
        _file_symlink(blob, snap / rel)
    return snap


class TestUpstreamImportGuardOnTheHFCache:
    """INV-13's import guard against the real HuggingFace cache layout.

    The first Kaggle run failed here: Python loads through snapshots/<rev>/, whose
    files are symlinks into blobs/, and the guard compared the resolved module file
    against the snapshot DIRECTORY. It now compares resolved file with resolved
    snapshot entry. These tests build the symlinks; they do not mock them.
    """

    def test_symlinked_snapshot_passes(self, hf_snapshot, isolated_imports):
        from vocoders.bigvgan import import_upstream

        modules = import_upstream(hf_snapshot)
        loaded = Path(modules["bigvgan"].__file__).resolve()
        # The layout really reproduces the failure: what Python loaded resolves into
        # blobs/, outside the snapshot directory the old guard compared against.
        assert "blobs" in loaded.parts
        assert not loaded.is_relative_to(hf_snapshot.resolve())
        assert {n: m.WHO for n, m in modules.items()} == {
            "bigvgan": "bigvgan", "meldataset": "meldataset", "env": "env"
        }

    def test_a_second_load_in_the_same_process_passes(self, hf_snapshot, isolated_imports):
        """The pre-import check sees upstream's own modules, symlinked, from the first load."""
        from vocoders.bigvgan import import_upstream

        import_upstream(hf_snapshot)
        assert "utils" in sys.modules and "alias_free_activation" in sys.modules
        import_upstream(hf_snapshot)

    def test_module_resolving_outside_the_snapshot_fails(
        self, hf_snapshot, isolated_imports, tmp_path
    ):
        """A same-named file earlier on sys.path wins the import; the guard must catch it."""
        from vocoders.bigvgan import import_upstream

        impostor = tmp_path / "site-packages"
        impostor.mkdir()
        (impostor / "bigvgan.py").write_text("WHO = 'impostor'\n")
        sys.path[:0] = [str(impostor), str(hf_snapshot)]  # snapshot present, but behind it
        with pytest.raises(InvariantViolation, match="INV-13: imported 'bigvgan'"):
            import_upstream(hf_snapshot)

    def test_same_named_module_already_imported_from_elsewhere_fails(
        self, hf_snapshot, isolated_imports, tmp_path
    ):
        from vocoders.bigvgan import import_upstream

        stray = type(sys)("utils")
        stray.__file__ = str(tmp_path / "site-packages" / "utils.py")
        sys.modules["utils"] = stray
        with pytest.raises(InvariantViolation, match="INV-13: a module named 'utils'"):
            import_upstream(hf_snapshot)


class TestUpstreamImportGuardThroughADirectoryLink:
    """The namespace-package branch (alias_free_activation has no __init__) through a
    real directory link. Junctions need no privilege on Windows, so unlike the file
    symlinks above this runs everywhere."""

    def _snapshot(self, tmp_path) -> Path:
        store = tmp_path / "store" / "alias_free_activation"
        (store / "torch").mkdir(parents=True)
        (store / "torch" / "__init__.py").write_text("")
        snap = tmp_path / "snapshot"
        snap.mkdir()
        for name in ("bigvgan", "meldataset", "env"):
            (snap / f"{name}.py").write_text(f"WHO = {name!r}\n")
        _dir_link(store, snap / "alias_free_activation")
        return snap

    def test_namespace_package_through_the_link_passes(self, tmp_path, isolated_imports):
        import importlib

        from vocoders.bigvgan import import_upstream

        snap = self._snapshot(tmp_path)
        sys.path.insert(0, str(snap))
        namespace = importlib.import_module("alias_free_activation")
        (entry,) = list(namespace.__path__)
        assert Path(entry).resolve() != Path(entry).absolute(), "the link was not followed"
        import_upstream(snap)

    def test_namespace_package_from_elsewhere_fails(self, tmp_path, isolated_imports):
        from vocoders.bigvgan import import_upstream

        snap = self._snapshot(tmp_path)
        other = tmp_path / "elsewhere" / "alias_free_activation"
        other.mkdir(parents=True)
        stray = type(sys)("alias_free_activation")
        stray.__path__ = [str(other)]
        sys.modules["alias_free_activation"] = stray
        with pytest.raises(InvariantViolation, match="INV-13: a module named 'alias_free"):
            import_upstream(snap)


# ============================================================ audit agrees with the CLI


class TestAuditAgreesWithTheCLI:
    def test_blocked_column_is_exactly_what_preflight_refuses_for_pins(self, corpus, tmp_path):
        """For every generable condition: the audit says blocked if and only if
        preflight refuses it on INV-08 grounds. Other refusals (missing torch here)
        are environmental, not properties of the condition."""
        from experiments.phase_a_resynthesis import preflight
        from vocoders.registry import ALL_CONDITIONS, checkpoint_audit

        rows = {r["condition"]: r for r in checkpoint_audit(include_unavailable=False)}
        for cond in ALL_CONDITIONS:
            try:
                preflight(corpus, [cond], tmp_path / f"out-{cond}", N_UTTS)
                refused_on_pins = False
            except InvariantViolation as err:
                refused_on_pins = str(err).startswith("INV-08")
            assert refused_on_pins == rows[cond]["blocked"], cond

    def test_missing_pins_are_named(self):
        from vocoders.registry import checkpoint_audit

        rows = {r["condition"]: r for r in checkpoint_audit()}
        for cond in ("bigvgan_base", "bigvgan_v2_22khz_fullband"):
            assert rows[cond]["missing_pins"] == "checkpoint_sha256", cond
            assert rows[cond]["blocked"] is True, cond
        for cond in ("bigvgan_112m", "vocos", "griffin_lim"):
            assert rows[cond]["missing_pins"] == "" and rows[cond]["blocked"] is False, cond


# ============================================================ HF_TOKEN

_SENTINEL = "hf_SENTINELtokenVALUEthatMUSTneverAPPEAR0123"


class TestHFToken:
    def test_optional_and_read_from_the_environment(self, monkeypatch):
        from vocoders.checkpoints import hf_token

        monkeypatch.delenv("HF_TOKEN", raising=False)
        assert hf_token() is None
        monkeypatch.setenv("HF_TOKEN", "   ")
        assert hf_token() is None
        monkeypatch.setenv("HF_TOKEN", _SENTINEL)
        assert hf_token() == _SENTINEL

    def _recording_downloader(self, root, calls):
        def download(*, repo_id, filename, revision, token):
            calls.append(token)
            path = root / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"w")
            return str(path)
        return download

    def test_bigvgan_downloads_carry_the_token_when_set(self, tmp_path, monkeypatch):
        import hashlib

        from vocoders.bigvgan import LARGE_SPEC, fetch_pinned_snapshot

        spec = replace(LARGE_SPEC, checkpoint_sha256=hashlib.sha256(b"w").hexdigest())
        for value, expected in ((_SENTINEL, _SENTINEL), (None, None)):
            if value is None:
                monkeypatch.delenv("HF_TOKEN", raising=False)
            else:
                monkeypatch.setenv("HF_TOKEN", value)
            calls = []
            fetch_pinned_snapshot(spec, self._recording_downloader(tmp_path / "s", calls))
            assert calls and set(calls) == {expected}

    def test_token_never_appears_in_a_failure(self, tmp_path, monkeypatch):
        from vocoders.bigvgan import LARGE_SPEC, fetch_pinned_snapshot

        monkeypatch.setenv("HF_TOKEN", _SENTINEL)
        with pytest.raises(InvariantViolation) as err:  # digest mismatch
            fetch_pinned_snapshot(LARGE_SPEC, self._recording_downloader(tmp_path / "s", []))
        assert _SENTINEL not in str(err.value)

    def test_token_never_appears_in_run_output(self, corpus, tmp_path, monkeypatch, capsys):
        from experiments.cli import main

        monkeypatch.setenv("HF_TOKEN", _SENTINEL)
        _register(monkeypatch, lambda w: w)
        out = tmp_path / "run"
        assert main(["resynth", "--corpus-root", str(corpus), "--conditions", "hifigan_v1",
                     "--out-dir", str(out), "--n-utterances", str(N_UTTS),
                     "--device", "cpu"]) == 0
        captured = capsys.readouterr()
        assert _SENTINEL not in captured.out + captured.err
        texts = [p for p in out.rglob("*") if p.suffix in (".json", ".csv")]
        assert texts
        for p in texts:
            assert _SENTINEL not in p.read_text(encoding="utf-8"), p.name


class TestSilenceDiagnostic:
    """INV-12 diagnosis: WHY a condition separates on silence duration.

    The first full v0 run tripped the probe on griffin_lim (EER 0.350 archive,
    0.355 zeroshot). Trim spans are reference-derived, so the difference cannot
    be leading/trailing silence LENGTH; it has to be what sits inside those
    regions. This reports the level there, per condition, and sweeps the probe's
    own threshold. It decides nothing: the floor and the verdicts are untouched.
    """

    FLOOR_DBFS = -70.0
    NOISE_DBFS = -45.0

    def _corpus(self, tmp_path, noise_dbfs=NOISE_DBFS):
        """real: speech between quiet pads. 'griffin_lim': the same, with low-level
        noise added everywhere -- the shape of a phase-reconstruction floor.

        Both go through the REAL pipeline -- one reference-derived trim span,
        applied to both, then the shared post-trim ordering -- so the written
        files carry exactly the TRIM_PAD_MS of context INV-03 promises. A fixture
        that wrote 200 ms pads instead would make the silence threshold's
        pad-agreement check fail on audio that is not what Phase A produces.
        """
        from data.audio_io import write_audio
        from data.preprocess import (
            derive_trim_span,
            finalise_reference,
            process_condition_output,
        )

        rows = []
        for i in range(12):
            rng = np.random.default_rng(i)
            n = int(1.0 * ARCHIVE_SR)
            t = np.arange(n) / ARCHIVE_SR
            # Abrupt edges, interior pauses and a decaying room: see _word_gate
            # and _room for why each is required rather than decorative.
            speech = _room(0.3 * np.sin(2 * np.pi * (120 + i) * t) * _word_gate(n), rng)
            pad = np.zeros(int(0.2 * ARCHIVE_SR))
            src = np.concatenate([pad, speech, pad]).astype(np.float32)
            src = (src + rng.normal(0, 10 ** (self.FLOOR_DBFS / 20), len(src))).astype(
                np.float32
            )
            noisy_src = (
                src + rng.normal(0, 10 ** (noise_dbfs / 20), len(src))
            ).astype(np.float32)

            utt = f"u{i:02d}"
            span = derive_trim_span(src, utt)
            real = finalise_reference(src, span).wav
            noisy = process_condition_output(
                noisy_src, ARCHIVE_SR, span, condition="griffin_lim"
            ).wav
            for cond, wav in (("real", real), ("griffin_lim", noisy)):
                path = tmp_path / cond / f"{utt}.wav"
                write_audio(path, wav, ARCHIVE_SR)
                rows.append({"utt_id": utt, "condition": cond, "tier": ARCHIVE_TIER,
                             "path": str(path), "split": "eval",
                             "duration_s": len(wav) / ARCHIVE_SR})
        return pd.DataFrame(rows)

    def test_reference_silent_mask_is_derived_from_real_alone(self):
        from experiments.sanity_checks import reference_silent_mask

        n = int(0.5 * ARCHIVE_SR)
        loud = 0.3 * np.sin(2 * np.pi * 200 * np.arange(n) / ARCHIVE_SR)
        wav = np.concatenate([np.zeros(n), loud, np.zeros(n)]).astype(np.float32)
        mask = reference_silent_mask(wav, ARCHIVE_SR)
        assert len(mask) == len(wav)
        assert mask[: n // 2].all() and mask[-n // 2:].all()
        assert not mask[n + n // 4: n + n // 2].any()

    def test_it_finds_the_elevated_floor_and_where_the_separation_dies(self, tmp_path):
        from experiments.sanity_checks import silence_diagnostic, silence_probe

        df = self._corpus(tmp_path)
        # The probe fires, exactly as it did on the real run.
        assert silence_probe(df, "griffin_lim", split=None)["leaked"]

        floors, sweep, ref = silence_diagnostic(df, split=None)
        gl = floors.set_index("condition").loc["griffin_lim"]
        # The injected floor is well above real's, in the regions real calls silent
        # and in the trim pad the probe actually reads.
        assert gl["above_real_db"] > 6.0
        assert gl["trim_pad_above_real_db"] > 6.0
        assert floors.set_index("condition").loc["real", "above_real_db"] == 0.0

        at = sweep.set_index("threshold_dbfs")
        assert at.loc[-60.0, "eer"] < SANITY_EER_FLOOR, "separable at a floor-level threshold"
        # What distinguishes a noise floor from a duration difference: once the
        # threshold is well above the added noise, the two conditions measure the
        # SAME silence. The EER is read on the feature, not the other way round --
        # with the durations agreeing to microseconds it is unstable on near-ties.
        assert at.loc[-60.0, "condition_silence_s_mean"] < (
            0.5 * at.loc[-60.0, "real_silence_s_mean"]
        )
        assert abs(
            at.loc[-30.0, "condition_silence_s_mean"] - at.loc[-30.0, "real_silence_s_mean"]
        ) < 0.005

    def test_a_condition_with_no_added_noise_separates_nowhere(self, tmp_path):
        from experiments.sanity_checks import silence_diagnostic

        df = self._corpus(tmp_path, noise_dbfs=-200.0)
        floors, sweep, ref = silence_diagnostic(df, split=None)
        assert abs(floors.set_index("condition").loc["griffin_lim", "above_real_db"]) < 1.0
        assert (
            (sweep["condition_silence_s_mean"] - sweep["real_silence_s_mean"]).abs() < 0.005
        ).all()

    def test_it_is_diagnosis_and_changes_no_verdict(self, tmp_path):
        """The gate keeps its floor, and silence is never expected_by_design."""
        from experiments.sanity_checks import run_all, silence_diagnostic

        df = self._corpus(tmp_path)
        before = run_all(df, split=None)
        silence_diagnostic(df, split=None)
        after = run_all(df, split=None)
        pd.testing.assert_frame_equal(before, after)
        silence = before[before["probe"] == "silence_duration"]
        assert silence["leaked"].all()
        assert not silence["expected_by_design"].any()

    def test_cli_writes_both_tables(self, tmp_path):
        from experiments.cli import main

        manifest = tmp_path / "m.csv"
        self._corpus(tmp_path).to_csv(manifest, index=False)
        out = tmp_path / "diag"
        assert main(["silence-diag", "--manifest", str(manifest), "--out-dir", str(out)]) == 0
        floors = pd.read_csv(out / "silence_floors.csv")
        # The estimator's own inputs, so the derivation is reproducible from this
        # output rather than from a separate calculation.
        assert {"quiet_frame_rms_dbfs", "quiet_peak_dbfs"} <= set(floors.columns)
        assert floors["quiet_peak_dbfs"].notna().all()

        sweep = pd.read_csv(out / "silence_threshold_sweep.csv")
        assert set(SILENCE_DIAGNOSTIC_THRESHOLDS) < set(sweep["threshold_dbfs"])
        gate = sweep[sweep["gate_threshold"]]
        assert len(gate) == 1, "the derived threshold is swept alongside the fixed points"

        calibration = pd.read_csv(out / "silence_calibration.csv").iloc[0]
        assert calibration["margin_db"] == SILENCE_FLOOR_MARGIN_DB
        assert calibration["threshold_dbfs"] == pytest.approx(
            calibration["floor_dbfs"] + SILENCE_FLOOR_MARGIN_DB, abs=1e-3
        )
        assert calibration["threshold_dbfs"] == pytest.approx(
            gate["threshold_dbfs"].iloc[0], abs=1e-3
        )


# ================================================ manifest paths and relocation


@pytest.fixture(scope="class")
def gated_dataset(tmp_path_factory):
    """One complete gated run, built once. Each test copies it before moving it.

    Class-scoped because a gated run generates two tiers of two conditions and
    every test below wants a real dataset rather than a hand-built frame -- the
    defect this class covers lived in what Phase A WROTE, not in what a fixture
    could assert about it.
    """
    import soundfile as sf

    from experiments.phase_a_resynthesis import run_gated

    root = tmp_path_factory.mktemp("relocation")
    corpus_root = root / "LJSpeech-1.1"
    (corpus_root / "wavs").mkdir(parents=True)
    lines = []
    for i in range(N_UTTS):
        uid = f"LJ001-{i:04d}"
        sf.write(
            corpus_root / "wavs" / f"{uid}.wav", _speechlike(seed=i), ARCHIVE_SR,
            subtype="PCM_16",
        )
        lines.append(f"{uid}|x|x")
    (corpus_root / "metadata.csv").write_text("\n".join(lines), encoding="utf-8")

    out = root / "resynth_v0"
    with pytest.MonkeyPatch.context() as mp:
        _register(mp, lambda w: w)
        run_gated(corpus_root, ["hifigan_v1"], out, N_UTTS, device="cpu")
    return out


class TestManifestPathsSurviveRelocation:
    """A manifest describes the tree it sits in, not the machine that wrote it.

    `btpvf silence-diag` crashed on the first v0 run because the manifest
    recorded absolute `<out>.staging` paths and run_gated renames staging to
    `<out>.failed` on a failed gate -- so every recorded path pointed into a
    directory that no longer existed. The same defect breaks the handoff: a
    teammate mounting the published output at /kaggle/input/<slug>/ would get a
    manifest of paths into someone else's /kaggle/working.

    It went unnoticed because `btpvf loudness` reads columns and never opens a
    file. Consumers that open audio are the ones that break, which is why every
    one of them is exercised here after a move.
    """

    def _copy(self, dataset: Path, dest: Path) -> Path:
        shutil.copytree(dataset, dest)
        return dest

    def test_stored_paths_are_relative_to_the_manifest(self, gated_dataset):
        raw = pd.read_csv(gated_dataset / "manifest.csv")
        assert not any(Path(p).is_absolute() for p in raw["path"])
        # And relative to the manifest itself, in the layout Phase A writes.
        assert set(raw["path"]) == {
            f"{r.tier}/{r.condition}/{r.utt_id}.wav" for r in raw.itertuples()
        }
        assert all((gated_dataset / p).is_file() for p in raw["path"])

    def test_read_manifest_resolves_against_the_manifest_it_came_from(self, gated_dataset):
        from data.manifest import read_manifest

        df = read_manifest(gated_dataset / "manifest.csv")
        assert df.attrs["manifest_root"] == str(gated_dataset.resolve())
        assert all(Path(p).is_file() for p in df["path"])

    def test_a_renamed_directory_still_resolves(self, gated_dataset, tmp_path):
        """The reported crash: staging renamed to `.failed` under the manifest."""
        from data.manifest import read_manifest

        staging = self._copy(gated_dataset, tmp_path / "resynth_v0.staging")
        failed = staging.parent / "resynth_v0.failed"
        staging.rename(failed)

        df = read_manifest(failed / "manifest.csv")
        assert all(Path(p).is_file() for p in df["path"])
        assert all(Path(p).is_relative_to(failed.resolve()) for p in df["path"])

    def test_the_whole_tree_moved_elsewhere_still_resolves(self, gated_dataset, tmp_path):
        """The handoff: the same dataset mounted read-only under another root."""
        from data.manifest import read_manifest, validate_manifest

        mounted = self._copy(gated_dataset, tmp_path / "kaggle" / "input" / "btpvf-v0")
        df = read_manifest(mounted / "manifest.csv")
        validate_manifest(df, require_zeroshot=True)
        assert all(Path(p).is_file() for p in df["path"])
        assert all(Path(p).is_relative_to(mounted.resolve()) for p in df["path"])

    def test_every_consumer_reads_the_relocated_files(self, gated_dataset, tmp_path):
        """Each consumer named in the handoff, driven on a moved tree."""
        from data.invariants import ZEROSHOT_TIER
        from data.manifest import read_manifest
        from detectors.protocols import Protocol, score_condition
        from experiments.sanity_checks import run_all, silence_diagnostic
        from metrics.runner import evaluate_archive

        mounted = self._copy(gated_dataset, tmp_path / "elsewhere" / "v0")
        df = read_manifest(mounted / "manifest.csv")

        sanity = run_all(df, split=None)          # INV-12 probes: open every file
        assert not sanity["leaked"].any()

        floors, sweep, _ = silence_diagnostic(df, split=None, limit=5)
        assert set(floors["condition"]) == {"real", "hifigan_v1"}
        assert not sweep["eer"].isna().any()

        quality = evaluate_archive(df, with_f0=False)   # Person B
        assert len(quality) == N_UTTS and quality["mcd"].notna().all()

        class Dummy:                                    # Person C
            spec = SimpleNamespace(key="dummy")

            def score_batch(self, wavs):
                return np.asarray([float(np.std(w)) for w in wavs])

        result = score_condition(
            Dummy(), df, "hifigan_v1", protocol=Protocol.MISMATCHED,
            tier=ZEROSHOT_TIER, split="eval", with_ci=False,
        )
        assert result.n_bonafide == result.n_spoof > 0

    def test_a_failed_run_is_readable_where_it_was_left(self, corpus, tmp_path, monkeypatch):
        """End to end: the exact sequence that produced the LibsndfileError."""
        from scipy.signal import butter, sosfiltfilt

        from experiments.cli import main
        from experiments.phase_a_resynthesis import run_gated

        sos = butter(8, 2000 / (ARCHIVE_SR / 2), output="sos")
        _register(monkeypatch, lambda w: sosfiltfilt(sos, w).astype(np.float32))
        out = tmp_path / "resynth_v0"
        with pytest.raises(InvariantViolation, match="INV-12"):
            run_gated(corpus, ["hifigan_v1"], out, N_UTTS, device="cpu", keep_failed=True)

        failed = tmp_path / "resynth_v0.failed"
        assert main([
            "silence-diag", "--manifest", str(failed / "manifest.csv"),
            "--out-dir", str(tmp_path / "diag"), "--limit", "5",
        ]) == 0
        assert (tmp_path / "diag" / "silence_floors.csv").is_file()

    def test_a_legacy_absolute_manifest_is_read_from_its_own_directory(
        self, gated_dataset, tmp_path
    ):
        """Manifests written before this change point at another machine entirely.

        They are recovered from the layout their own columns record, not guessed
        at: <manifest dir>/<tier>/<condition>/<utt_id>.wav. Recovered, and said
        so -- `paths_relocated` is what the CLI prints rather than staying quiet.
        """
        from data.manifest import read_manifest

        mounted = self._copy(gated_dataset, tmp_path / "input" / "v0")
        raw = pd.read_csv(mounted / "manifest.csv")
        raw["path"] = [f"/kaggle/working/resynth_v0.staging/{p}" for p in raw["path"]]
        raw.to_csv(mounted / "manifest.csv", index=False)

        df = read_manifest(mounted / "manifest.csv")
        assert df.attrs["paths_relocated"] == len(df)
        assert all(Path(p).is_file() for p in df["path"])

    def test_an_unrecoverable_path_keeps_the_one_that_was_recorded(self, gated_dataset, tmp_path):
        """No invention: if the file is nowhere, the error names what was written."""
        from data.manifest import read_manifest

        mounted = self._copy(gated_dataset, tmp_path / "input" / "v0")
        raw = pd.read_csv(mounted / "manifest.csv")
        recorded = "/kaggle/working/resynth_v0.staging/archive/real/LJ001-0000.wav"
        raw.loc[0, "path"] = recorded
        raw.loc[0, "utt_id"] = "not-a-real-utterance"
        raw.to_csv(mounted / "manifest.csv", index=False)

        df = read_manifest(mounted / "manifest.csv")
        assert df.iloc[0]["path"] == recorded
        assert df.attrs["paths_relocated"] == 0

    def test_write_manifest_refuses_a_path_outside_its_own_directory(self, tmp_path):
        from data.manifest import ManifestRow, write_manifest

        row = ManifestRow(
            utt_id="u0", condition="real", path=str(tmp_path / "elsewhere" / "u0.wav"),
            speaker="LJ", split="train", duration_s=1.0, lufs_out=-27.0,
            trim_start=0, trim_end=10, trim_ref_length=10,
        )
        with pytest.raises(InvariantViolation, match="outside its own directory"):
            write_manifest([row], tmp_path / "dataset" / "manifest.csv")

    def test_the_generating_directory_is_provenance_not_a_path_column(self, gated_dataset):
        """The one absolute path a run records, kept where staleness is the point.

        It names the session that generated the data -- for a gated run, the
        staging directory the bytes were written into, which no longer exists by
        the time anyone reads this. That is what provenance is for, and it is
        why it is not the column consumers open files with.
        """
        prov = json.loads((gated_dataset / "manifest.provenance.json").read_text())
        assert Path(prov["generated_in"]).is_absolute()
        assert Path(prov["generated_in"]).name.startswith(gated_dataset.name)
        raw = pd.read_csv(gated_dataset / "manifest.csv")
        assert not any(Path(p).is_absolute() for p in raw["path"])


# ==================================== the derived silence threshold (INV-03/INV-12)


def _pipeline_corpus(
    tmp_path: Path,
    transform,
    *,
    n: int = 12,
    floor_dbfs: float = ROOM_TONE_DBFS,
    condition: str = "griffin_lim",
) -> pd.DataFrame:
    """A two-condition archive frame built through the REAL pipeline.

    One reference-derived trim span per utterance, applied to both members, then
    the shared post-trim ordering -- so the written files carry exactly the
    TRIM_PAD_MS of context INV-03 promises and the pad-agreement check is being
    tested against audio Phase A could actually produce.
    """
    from data.audio_io import write_audio
    from data.preprocess import (
        derive_trim_span,
        finalise_reference,
        process_condition_output,
    )

    rows = []
    for i in range(n):
        rng = np.random.default_rng(i)
        m = int(1.0 * ARCHIVE_SR)
        t = np.arange(m) / ARCHIVE_SR
        speech = _room(0.3 * np.sin(2 * np.pi * (120 + i) * t) * _word_gate(m), rng)
        pad = np.zeros(int(0.2 * ARCHIVE_SR))
        src = np.concatenate([pad, speech, pad]).astype(np.float32)
        src = (src + rng.normal(0, 10 ** (floor_dbfs / 20), len(src))).astype(np.float32)

        utt = f"u{i:02d}"
        span = derive_trim_span(src, utt)
        real = finalise_reference(src, span).wav
        other = process_condition_output(
            transform(src, rng), ARCHIVE_SR, span, condition=condition
        ).wav
        for cond, wav in (("real", real), (condition, other)):
            path = tmp_path / cond / f"{utt}.wav"
            write_audio(path, wav, ARCHIVE_SR)
            rows.append({"utt_id": utt, "condition": cond, "tier": ARCHIVE_TIER,
                         "path": str(path), "split": "eval",
                         "duration_s": len(wav) / ARCHIVE_SR})
    return pd.DataFrame(rows)


class TestDerivedSilenceThreshold:
    """INV-03/INV-12: the silence probe's threshold comes from the corpus.

    INV-03 defines silence RELATIVELY -- TRIM_TOP_DB below the loudest frame --
    and this probe compares |sample| to an absolute level. The two only agree if
    the level is placed against the corpus's own floor. The previous -50.0 dBFS
    default sat ~3 dB inside LJSpeech's floor, so the feature measured how often
    the signal's noise crossed a level near its own noise. Nobody noticed for
    three weeks, partly because the value was in no constants block.
    """

    def test_the_four_constants_live_in_invariants(self):
        from data import invariants

        assert invariants.SILENCE_FLOOR_FRAME_PERCENTILE == 10.0
        assert invariants.SILENCE_FLOOR_PEAK_PERCENTILE == 99.9
        assert invariants.SILENCE_FLOOR_MARGIN_DB == 8.0
        assert invariants.SILENCE_PAD_AGREEMENT == 0.25

    def test_no_caller_can_get_a_threshold_without_choosing_one(self):
        """The mechanism that hid the old value: a default nobody had to supply."""
        import inspect

        from experiments.sanity_checks import _leading_trailing_silence

        threshold = inspect.signature(_leading_trailing_silence).parameters["threshold_db"]
        assert threshold.default is inspect.Parameter.empty
        assert threshold.kind is inspect.Parameter.KEYWORD_ONLY
        with pytest.raises(TypeError):
            _leading_trailing_silence(np.zeros(100, np.float32), ARCHIVE_SR)

    def test_the_floor_is_a_sample_peak_not_an_rms(self):
        """Peak against peak: the probe thresholds |sample|, so the floor must too.

        For a noise-like floor the p99.9 of |x| sits ~10.4 dB above its RMS. A
        floor given as an RMS would leave that term inside the margin, where it
        would scale with region length and sample rate.
        """
        from experiments.sanity_checks import silence_floor_dbfs

        rng = np.random.default_rng(0)
        sigma = 10 ** (-60.0 / 20.0)
        noise = [(sigma * rng.standard_normal(ARCHIVE_SR)).astype(np.float32) for _ in range(6)]
        floor, cut = silence_floor_dbfs(noise, ARCHIVE_SR)
        assert floor == pytest.approx(-60.0 + 10.35, abs=1.0)
        assert cut < floor, "the RMS cut sits below the peak floor, by the crest factor"
        # Homogeneous files, so pooling would have agreed here -- which is why the
        # pooled statistic's error was invisible until a real corpus ran.
        assert silence_floor_dbfs(noise * 8, ARCHIVE_SR)[0] == pytest.approx(floor, abs=1e-9)

    def test_the_threshold_tracks_the_corpus_floor(self, tmp_path):
        """The VCTK argument: a corpus with a lower floor gets a lower threshold.

        This is the whole reason the value is derived rather than written down.
        VCTK is recorded in a hemi-anechoic chamber and will not match LJSpeech;
        a constant tuned on one is wrong on the other, in a way that reads as a
        clean EER either way. Measured on floors that are unambiguous, because
        the claim is about the derivation and not about what happens to set a
        particular fixture's quietest frames.
        """
        from experiments.sanity_checks import silence_floor_dbfs

        rng = np.random.default_rng(3)
        floors = {}
        for level in (-75.0, -55.0):
            noise = [(10 ** (level / 20) * rng.standard_normal(ARCHIVE_SR)).astype(np.float32)
                     for _ in range(6)]
            floors[level], _ = silence_floor_dbfs(noise, ARCHIVE_SR)
        assert floors[-55.0] - floors[-75.0] == pytest.approx(20.0, abs=0.5)

    def test_a_pipeline_corpus_derives_a_threshold_that_reads_its_pad(self, tmp_path):
        """End to end: threshold = floor + margin, and it lands on the trim pad."""
        from experiments.sanity_checks import silence_reference

        df = _pipeline_corpus(tmp_path, lambda s, r: s)
        ref = silence_reference(df, split=None)
        assert ref.threshold_dbfs == pytest.approx(
            ref.floor_dbfs + SILENCE_FLOOR_MARGIN_DB, abs=1e-6
        )
        assert ref.ok and ref.pad_error <= SILENCE_PAD_AGREEMENT
        assert ref.pad_measured_s == pytest.approx(0.04, abs=0.04 * SILENCE_PAD_AGREEMENT)

    def test_a_floor_too_close_to_speech_is_refused_not_reported(self, tmp_path):
        """The rule has a range requirement, and it fails closed when it is unmet.

        ``floor + SILENCE_FLOOR_MARGIN_DB`` only reads the pad while the floor is
        far enough below speech. Raise the corpus floor toward speech and the
        threshold climbs into the onset, so the probe reads more than the pad and
        the run stops. That is the designed outcome: the alternative is an EER
        measured at a threshold that is not reading silence, which is what -50
        dBFS was doing on LJSpeech.
        """
        from experiments.sanity_checks import silence_reference

        df = _pipeline_corpus(tmp_path, lambda s, r: s, floor_dbfs=-55.0)
        with pytest.raises(InvariantViolation, match="does not read silence"):
            silence_reference(df, split=None)
        assert not silence_reference(df, split=None, strict=False).ok

    def test_the_floor_does_not_drift_with_the_number_of_files(self):
        """The estimate of "the floor" must not depend on how much was measured.

        The first version concatenated every file's quiet samples and took one
        percentile of the pool. On a corpus whose per-file floors differ that is
        a different quantity -- the level 0.1% of all quiet samples ANYWHERE
        exceed, whose tail is supplied by the noisiest files -- and it climbs as
        files are added. The suite passed at n=12 and the first 200-file run
        derived a floor 7.0 dB too high, which put the threshold inside speech.
        Nested subsets, so this measures the estimator and not resampling noise.
        """
        from experiments.sanity_checks import _dbfs_amp, _quiet_mask, silence_floor_dbfs

        rng = np.random.default_rng(0)
        corpus = []
        for _ in range(240):  # per-file floors spread by 5 dB, as a real corpus's are
            level = -62.0 + rng.normal(0, 5.0)
            corpus.append(
                (10 ** (level / 20) * rng.standard_normal(int(0.6 * ARCHIVE_SR))
                 ).astype(np.float32)
            )

        def pooled(wavs):  # the statistic that was wrong, kept as a mutation check
            pool = np.concatenate(
                [np.abs(w[_quiet_mask(w, ARCHIVE_SR)].astype(np.float64)) for w in wavs]
            )
            return _dbfs_amp(float(np.percentile(pool, SILENCE_FLOOR_PEAK_PERCENTILE)))

        small, large = corpus[:8], corpus
        derived_drift = (silence_floor_dbfs(large, ARCHIVE_SR)[0]
                         - silence_floor_dbfs(small, ARCHIVE_SR)[0])
        pooled_drift = pooled(large) - pooled(small)

        assert abs(derived_drift) < 2.5, "the median across files is stable in n"
        assert pooled_drift > 3.0, (
            "the pooled percentile climbs with n -- if this stops being true the "
            "mutation check has lost its teeth, not the estimator its bug"
        )
        assert abs(derived_drift) < pooled_drift / 2

    def test_the_diagnostic_and_the_gate_agree_on_the_floor(self, tmp_path):
        """One implementation, so the two paths cannot report different floors.

        They disagreed on the first real run -- the diagnostic's floors table said
        -46.854 dBFS for real and the calibration said -39.808 -- because one took
        a median of per-file percentiles and the other a percentile of the pool.
        Two numbers under one name is the defect, whichever is right.
        """
        from experiments.sanity_checks import silence_diagnostic

        df = _pipeline_corpus(tmp_path, lambda s, r: s)
        floors, _, reference = silence_diagnostic(df, split=None)
        real = floors.set_index("condition").loc["real"]

        assert real["quiet_peak_dbfs"] == pytest.approx(reference.floor_dbfs, abs=1e-9)
        assert real["quiet_frame_rms_dbfs"] == pytest.approx(
            reference.frame_rms_percentile_dbfs, abs=1e-9
        )
        # And the threshold is that same floor plus the margin, not a third number.
        assert reference.threshold_dbfs == pytest.approx(
            real["quiet_peak_dbfs"] + SILENCE_FLOOR_MARGIN_DB, abs=1e-9
        )

    def test_digital_silence_is_not_a_noise_floor(self, tmp_path):
        """Exact zeros carry no floor information and must not become the floor.

        A run of PCM_16 zeros is the ABSENCE of a recorded floor. Letting it into
        the estimate would drag the threshold to the quantiser and the probe would
        call the whole file voiced.
        """
        from experiments.sanity_checks import silence_floor_dbfs

        rng = np.random.default_rng(1)
        sigma = 10 ** (-55.0 / 20.0)
        wav = np.concatenate([
            np.zeros(ARCHIVE_SR // 2, np.float32),
            (sigma * rng.standard_normal(ARCHIVE_SR)).astype(np.float32),
        ])
        floor, _ = silence_floor_dbfs([wav], ARCHIVE_SR)
        assert floor == pytest.approx(-55.0 + 10.35, abs=1.5)

    def test_pad_disagreement_raises_rather_than_reporting(self, tmp_path):
        """A number nobody can interpret is worse than none, so the probe stops."""
        from data.audio_io import write_audio
        from experiments.sanity_checks import run_all, silence_probe

        rows = []
        for i in range(12):
            rng = np.random.default_rng(i)
            m = int(1.0 * ARCHIVE_SR)
            t = np.arange(m) / ARCHIVE_SR
            speech = _room(0.3 * np.sin(2 * np.pi * (120 + i) * t) * _word_gate(m), rng)
            # 200 ms of pad instead of TRIM_PAD_MS: untrimmed audio, which is not
            # what INV-03 produces and must not be reported on as if it were.
            pad = np.zeros(int(0.2 * ARCHIVE_SR))
            wav = np.concatenate([pad, speech, pad]).astype(np.float32)
            wav = (wav + rng.normal(0, 10 ** (ROOM_TONE_DBFS / 20), len(wav))).astype(np.float32)
            for cond in ("real", "griffin_lim"):
                path = tmp_path / cond / f"u{i:02d}.wav"
                write_audio(path, wav, ARCHIVE_SR)
                rows.append({"utt_id": f"u{i:02d}", "condition": cond, "tier": ARCHIVE_TIER,
                             "path": str(path), "split": "eval",
                             "duration_s": len(wav) / ARCHIVE_SR})
        df = pd.DataFrame(rows)

        with pytest.raises(InvariantViolation, match="does not read silence"):
            silence_probe(df, "griffin_lim", split=None)
        with pytest.raises(InvariantViolation, match="TRIM_PAD_MS"):
            run_all(df, split=None)

    def test_the_diagnostic_reports_what_the_gate_raises_on(self, tmp_path):
        """Diagnosis is the one place a disagreeing threshold must still report."""
        from data.audio_io import write_audio
        from experiments.sanity_checks import silence_diagnostic

        rows = []
        for i in range(8):
            rng = np.random.default_rng(i)
            m = int(1.0 * ARCHIVE_SR)
            t = np.arange(m) / ARCHIVE_SR
            speech = _room(0.3 * np.sin(2 * np.pi * (120 + i) * t) * _word_gate(m), rng)
            pad = np.zeros(int(0.2 * ARCHIVE_SR))
            wav = np.concatenate([pad, speech, pad]).astype(np.float32)
            wav = (wav + rng.normal(0, 10 ** (ROOM_TONE_DBFS / 20), len(wav))).astype(np.float32)
            for cond in ("real", "griffin_lim"):
                path = tmp_path / cond / f"u{i:02d}.wav"
                write_audio(path, wav, ARCHIVE_SR)
                rows.append({"utt_id": f"u{i:02d}", "condition": cond, "tier": ARCHIVE_TIER,
                             "path": str(path), "split": "eval",
                             "duration_s": len(wav) / ARCHIVE_SR})
        floors, sweep, ref = silence_diagnostic(pd.DataFrame(rows), split=None)
        assert not ref.ok and ref.pad_error > SILENCE_PAD_AGREEMENT
        assert not floors.empty and not sweep.empty

    def test_the_trim_pad_level_is_recorded_and_ungated(self, tmp_path):
        """Task 3: the ~1 dB pad difference stays visible without becoming a gate.

        The derived threshold sits above the pad's noise by design, so the silence
        probe can no longer see a pad-level difference, and `rms_level` cannot
        either -- whole-file RMS is dominated by speech. Adding noise at the
        floor's own level raises the condition's pad by ~3 dB, which is inside
        the margin, so the EER stays at chance while the column shows the effect.
        """
        from experiments.sanity_checks import run_all

        sigma = 10 ** (ROOM_TONE_DBFS / 20)
        df = _pipeline_corpus(
            tmp_path,
            lambda s, r: (s + r.normal(0, sigma, len(s))).astype(np.float32),
        )
        sanity = run_all(df, split=None)
        silence = sanity[sanity["probe"] == "silence_duration"].iloc[0]

        # Around 1 dB -- the same order as the +0.887 dB the real run recorded for
        # griffin_lim, and well inside the margin, so it cannot move the threshold.
        assert 0.2 < silence["trim_pad_above_real_db"] < SILENCE_FLOOR_MARGIN_DB
        assert not bool(silence["leaked"]), "recorded, not gated"
        assert silence["eer"] >= SANITY_EER_FLOOR
        assert not sanity["leaked"].any()
        # And it is a column, not a probe: no row's verdict is derived from it.
        assert "trim_pad_above_real_db" not in {"probe", "eer", "leaked"}
        assert sanity.loc[sanity["probe"] != "silence_duration",
                          "trim_pad_above_real_db"].isna().all()

    def test_every_sanity_row_carries_the_threshold_it_was_measured_at(self, tmp_path):
        from experiments.sanity_checks import run_all, silence_reference

        df = _pipeline_corpus(tmp_path, lambda s, r: s)
        sanity = run_all(df, split=None)
        ref = silence_reference(df, split=None)
        for column, value in (
            ("silence_threshold_dbfs", ref.threshold_dbfs),
            ("silence_floor_dbfs", ref.floor_dbfs),
            ("silence_frame_rms_cut_dbfs", ref.frame_rms_percentile_dbfs),
            ("real_trim_pad_dbfs", ref.pad_level_dbfs),
        ):
            assert np.allclose(sanity[column].to_numpy(float), value), column

    def test_the_gated_run_records_the_calibration_in_the_sidecar(self, gated_dataset):
        """It is a property OF the dataset, so it travels with the dataset."""
        prov = json.loads((gated_dataset / "manifest.provenance.json").read_text())
        calibration = prov["silence_threshold"]
        assert set(calibration) == {"archive", "zeroshot"}
        for tier, record in calibration.items():
            assert record["tier"] == tier
            assert record["pad_agreement_ok"] is True
            assert record["margin_db"] == SILENCE_FLOOR_MARGIN_DB
            assert record["threshold_dbfs"] == pytest.approx(
                record["floor_dbfs"] + SILENCE_FLOOR_MARGIN_DB, abs=1e-3
            )
            assert record["pad_expected_s"] == pytest.approx(0.04)

        sanity = pd.read_csv(gated_dataset / "sanity.csv")
        for tier, record in calibration.items():
            rows = sanity[sanity["tier"] == tier]
            assert np.allclose(
                rows["silence_threshold_dbfs"].to_numpy(float),
                record["threshold_dbfs"], atol=1e-3,
            )


class TestSilenceCalibrationArtifacts:
    """The committed `btpvf silence-diag` output from the run whose gate raised.

    `data/measurements/silence_{calibration,floors,threshold_sweep}.csv`, 200
    LJSpeech utterances. Committed because they record two things at once: the
    refutation of the reconstruction-noise hypothesis, and the floor estimator's
    pooled-percentile defect that the pad-agreement check caught.
    """

    def _artifacts(self):
        from data.measurements import (
            SILENCE_CALIBRATION,
            SILENCE_FLOORS,
            SILENCE_SWEEP,
            load_csv,
        )

        return (
            load_csv(SILENCE_CALIBRATION).iloc[0],
            load_csv(SILENCE_FLOORS).set_index("condition"),
            load_csv(SILENCE_SWEEP),
        )

    def test_the_run_raised_and_the_record_says_so(self):
        cal, _, _ = self._artifacts()
        assert cal["n_real_files"] == 200
        assert not bool(cal["pad_agreement_ok"])
        assert cal["pad_measured_s"] == pytest.approx(0.079233, abs=1e-6)
        assert cal["pad_error"] > SILENCE_PAD_AGREEMENT
        # The gate would have PASSED at that threshold: every condition reads
        # ~chance. It was refused for not measuring silence, not for its EER.
        _, _, sweep = self._artifacts()
        at_gate = sweep[sweep["gate_threshold"]]
        assert len(at_gate) == 3 and (at_gate["eer"] > SANITY_EER_FLOOR).all()

    def test_the_two_paths_disagreed_on_the_floor_and_agreed_on_the_cut(self):
        """The defect, pinned in the data that exposed it."""
        cal, floors, _ = self._artifacts()
        real = floors.loc["real"]
        assert cal["quiet_frame_rms_cut_dbfs"] == pytest.approx(
            real["quiet_frame_rms_dbfs"], abs=5e-4
        ), "the frame-RMS cut always agreed -- both paths took a median of per-file cuts"
        gap = float(cal["floor_dbfs"]) - float(real["quiet_peak_dbfs"])
        assert gap == pytest.approx(7.05, abs=0.05), (
            "the peak floor did not: pooled percentile against median of per-file "
            "percentiles"
        )

    def test_the_corrected_floor_lands_inside_the_band_the_sweep_admits(self):
        """Derived from the committed files, not from a re-run: the audio is not here.

        Real's own sweep column gives what the probe reads at each threshold;
        `2 * TRIM_PAD_MS` is what INV-03 leaves. Interpolating between them says
        whether the corrected threshold reads the pad.
        """
        from data.invariants import TRIM_PAD_MS

        _, floors, sweep = self._artifacts()
        threshold = float(floors.loc["real", "quiet_peak_dbfs"]) + SILENCE_FLOOR_MARGIN_DB
        assert -43.0 < threshold < -37.0, "inside the band real's sweep admits"

        real = (sweep.drop_duplicates("threshold_dbfs")
                .set_index("threshold_dbfs")["real_silence_s_mean"].sort_index())
        xs, ys = real.index.to_numpy(float), real.to_numpy(float)
        expected = 2 * TRIM_PAD_MS * 1e-3
        for predicted in (
            float(np.interp(threshold, xs, ys)),                       # linear
            float(np.exp(np.interp(threshold, xs, np.log(ys)))),       # log-linear
        ):
            assert abs(predicted - expected) / expected < SILENCE_PAD_AGREEMENT

    def test_the_refutation_is_in_the_floors_table(self):
        """No elevated noise floor: the synthetic +13.2 dB was ~100x the truth."""
        _, floors, _ = self._artifacts()
        assert floors.loc["griffin_lim", "above_real_db"] == pytest.approx(0.108, abs=0.01)
        assert floors["above_real_db"].abs().max() < 0.2
        # The one effect that survives, and the reason the pad column is recorded.
        assert floors.loc["griffin_lim", "trim_pad_above_real_db"] == pytest.approx(
            0.887, abs=0.01
        )
