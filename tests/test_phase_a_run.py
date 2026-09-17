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
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from data.invariants import ARCHIVE_SR, LADDER_FMAX, InvariantViolation

REPO_ROOT = Path(__file__).resolve().parents[1]
N_UTTS = 22  # >= ALIGNMENT_PROBE_FILES


def _speechlike(seconds: float = 1.5, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(ARCHIVE_SR * seconds)
    t = np.arange(n) / ARCHIVE_SR
    f0 = 120 + 5 * (seed % 11)
    x = sum((0.3 / k) * np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 6.28))
            for k in range(1, 60))
    x = x * np.hanning(n) + 0.01 * rng.standard_normal(n)
    pad = int(0.15 * ARCHIVE_SR)
    x = np.concatenate([np.zeros(pad), x, np.zeros(pad)])
    return (0.35 * x / np.max(np.abs(x))).astype(np.float32)


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
        def download(*, repo_id, filename, revision):
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
        text = self.PATH.read_text(encoding="utf-8").lower()
        for needle in ("kaggle.json", "hf_token", "api_key", "password", "ghp_"):
            assert needle not in text
