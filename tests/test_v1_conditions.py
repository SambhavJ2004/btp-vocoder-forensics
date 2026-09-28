"""The v1 additions: two new adapters, the compatibility shim, and the merge.

The adapters themselves cannot execute here -- torch is deliberately not a
dependency (INV-13) -- so what is tested is everything that decides whether a
load is ALLOWED and whether what it produced may be combined: pins, the mel
checks, the shim's exactness, and the merge's refusals.

The shim is tested against upstream's REAL text, fetched and cached, because a
substitution that matches a paraphrase of upstream is worthless.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.invariants import ARCHIVE_SR, ARCHIVE_TIER, InvariantViolation
from vocoders.compat import SHIMS, Rule, apply_shims, describe

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestPins:
    """INV-08. Every condition in tonight's run is pinned before it can load."""

    def test_bigvgan_base_carries_revision_and_digest(self):
        from vocoders.bigvgan import BASE_SPEC

        assert len(BASE_SPEC.checkpoint_revision) == 40
        assert len(BASE_SPEC.checkpoint_sha256) == 64

    def test_hifigan_pins_code_and_weights_separately(self):
        """Its two hosts are the weakest provenance on the ladder, so both are
        pinned and the fact is visible in the spec."""
        from vocoders.hifigan import CODE_COMMIT, SPEC, WEIGHTS_REVISION

        assert SPEC.checkpoint_revision == WEIGHTS_REVISION
        assert SPEC.upstream_commit == CODE_COMMIT
        assert len(SPEC.checkpoint_sha256) == 64
        assert "jaketae" in SPEC.checkpoint and "jik876" in SPEC.checkpoint

    def test_melgan_pins_one_commit_for_code_and_weights(self):
        from vocoders.melgan import CODE_COMMIT, SPEC

        assert SPEC.upstream_commit == CODE_COMMIT
        assert len(SPEC.checkpoint_sha256) == 64
        assert "linda_johnson" in SPEC.checkpoint

    @pytest.mark.parametrize("condition", ["hifigan_v1", "melgan", "bigvgan_base"])
    def test_the_audit_agrees_the_condition_is_runnable(self, condition):
        from vocoders.registry import checkpoint_audit

        row = {r["condition"]: r for r in checkpoint_audit()}[condition]
        assert row["blocked"] is False, row
        assert row["missing_pins"] == ""

    @pytest.mark.parametrize("condition", ["hifigan_v1", "melgan"])
    def test_load_refuses_without_its_pins(self, condition):
        from dataclasses import replace

        from vocoders.registry import get_spec, get_vocoder

        adapter = type(get_vocoder(condition))
        for field in adapter.required_pins:
            spec = replace(get_spec(condition), **{field: ""})
            module = __import__(f"vocoders.{condition.split('_')[0]}", fromlist=["x"])
            with pytest.raises(InvariantViolation, match="INV-08"):
                module.fetch_pinned_snapshot(spec)


class TestCompatibilityShim:
    """Upstream's 2019-2020 code does not run on current libraries.

    The shim is a recorded modification, so what matters is that it matches
    upstream EXACTLY, refuses to guess, and says what it changed.
    """

    def _upstream(self, tmp_path, url: str, name: str) -> Path:
        import urllib.request

        cache = REPO_ROOT / ".pytest_cache" / "upstream"
        cache.mkdir(parents=True, exist_ok=True)
        cached = cache / name
        if not cached.is_file():
            try:
                urllib.request.urlretrieve(url, cached)  # noqa: S310 -- pinned raw URL
            except OSError as err:
                pytest.skip(f"cannot reach upstream to verify the shim: {err}")
        target = tmp_path / name
        target.write_bytes(cached.read_bytes())
        return target

    HIFIGAN = (
        "https://raw.githubusercontent.com/jik876/hifi-gan/"
        "4769534d45265d52a904b850da5a622601885777/meldataset.py"
    )
    MELGAN = (
        "https://raw.githubusercontent.com/descriptinc/melgan-neurips/"
        "6488045bfba1975602288de07a58570c7b4d66ea/mel2wav/modules.py"
    )

    @pytest.mark.parametrize("name,url", [("meldataset.py", HIFIGAN), ("modules.py", MELGAN)])
    def test_every_rule_matches_the_real_upstream_file(self, tmp_path, name, url):
        path = self._upstream(tmp_path, url, name)
        record = apply_shims(path, condition="test")
        assert record["shimmed"] is True
        assert len(record["rules"]) == len(SHIMS[name]), "a rule silently did not apply"
        assert record["upstream_sha256"] != record["shimmed_sha256"]

    @pytest.mark.parametrize("name,url", [("meldataset.py", HIFIGAN), ("modules.py", MELGAN)])
    def test_the_shimmed_file_still_compiles(self, tmp_path, name, url):
        path = self._upstream(tmp_path, url, name)
        apply_shims(path, condition="test")
        compile(path.read_text(encoding="utf-8"), name, "exec")

    @pytest.mark.parametrize("name,url", [("meldataset.py", HIFIGAN), ("modules.py", MELGAN)])
    def test_it_changes_only_the_lines_it_claims(self, tmp_path, name, url):
        """A shim that touched anything else would be a fork wearing a patch."""
        path = self._upstream(tmp_path, url, name)
        before = path.read_text(encoding="utf-8").splitlines()
        apply_shims(path, condition="test")
        after = path.read_text(encoding="utf-8").splitlines()
        changed = [
            i for i in range(min(len(before), len(after)))
            if before[i] != after[i]
        ]
        # Two call sites; the stft rules add a line, so allow the tail to shift.
        assert 1 <= len(SHIMS[name]) <= 2
        assert changed, "nothing changed"
        # No line outside the neighbourhood of a rule may differ.
        joined_before, joined_after = "\n".join(before), "\n".join(after)
        for rule in SHIMS[name]:
            assert rule.old in joined_before
            assert rule.new in joined_after

    def test_it_is_idempotent(self, tmp_path):
        path = self._upstream(tmp_path, self.HIFIGAN, "meldataset.py")
        first = apply_shims(path, condition="test")
        second = apply_shims(path, condition="test")
        assert second["already_shimmed"] is True
        assert second["shimmed_sha256"] == first["shimmed_sha256"]

    def test_a_rule_that_does_not_match_raises_rather_than_guessing(self, tmp_path):
        """If upstream's text moved, the answer is to look, not to patch fuzzily.

        A silent no-op would leave the condition running unshimmed code, which
        fails later and further from the cause.
        """
        path = tmp_path / "meldataset.py"
        path.write_text("# upstream, but rewritten since\n", encoding="utf-8")
        with pytest.raises(InvariantViolation, match="INV-02"):
            apply_shims(path, condition="melgan")

    def test_a_file_with_no_rules_is_left_alone(self, tmp_path):
        path = tmp_path / "models.py"
        path.write_text("x = 1\n", encoding="utf-8")
        record = apply_shims(path, condition="hifigan_v1")
        assert record["shimmed"] is False
        assert path.read_text(encoding="utf-8") == "x = 1\n"

    def test_every_rule_states_why_it_preserves_behaviour(self):
        for name, rules in SHIMS.items():
            for rule in rules:
                assert isinstance(rule, Rule)
                assert len(rule.why) > 40, (name, rule.why)

    def test_describe_names_both_digests(self, tmp_path):
        path = self._upstream(tmp_path, self.MELGAN, "modules.py")
        text = describe([apply_shims(path, condition="melgan")])
        assert "modules.py" in text and "->" in text


class TestMelGANMelAudit:
    """INV-02 for a vocoder that ships no config file."""

    def _audio2mel(self, **overrides):
        defaults = dict(n_fft=1024, hop_length=256, win_length=1024, sampling_rate=22050,
                        n_mel_channels=80, mel_fmin=0.0, mel_fmax=None)
        defaults.update(overrides)

        class Audio2Mel:
            def __init__(
                self,
                n_fft=defaults["n_fft"],
                hop_length=defaults["hop_length"],
                win_length=defaults["win_length"],
                sampling_rate=defaults["sampling_rate"],
                n_mel_channels=defaults["n_mel_channels"],
                mel_fmin=defaults["mel_fmin"],
                mel_fmax=defaults["mel_fmax"],
            ):
                pass

        return Audio2Mel

    def test_upstream_defaults_matching_the_audit_pass(self):
        from data.mel import MELGAN_MEL
        from vocoders.melgan import check_audio2mel_defaults

        check_audio2mel_defaults(self._audio2mel(), MELGAN_MEL)

    @pytest.mark.parametrize("field,value", [
        ("n_fft", 2048), ("hop_length", 300), ("sampling_rate", 24000),
        ("n_mel_channels", 100), ("mel_fmin", 50.0),
    ])
    def test_a_changed_default_is_refused(self, field, value):
        from data.mel import MELGAN_MEL
        from vocoders.melgan import check_audio2mel_defaults

        with pytest.raises(InvariantViolation, match="INV-02"):
            check_audio2mel_defaults(self._audio2mel(**{field: value}), MELGAN_MEL)

    def test_a_bounded_fmax_is_refused_because_it_would_move_ladder_fmax(self):
        """The audit records 11025 for a default of None. A release that started
        bounding fmax would change this condition's band, and LADDER_FMAX with it."""
        from data.mel import MELGAN_MEL
        from vocoders.melgan import check_audio2mel_defaults

        with pytest.raises(InvariantViolation, match="LADDER_FMAX"):
            check_audio2mel_defaults(self._audio2mel(mel_fmax=7600.0), MELGAN_MEL)


def _run(tmp_path: Path, name: str, condition: str, *, real_seed: int = 0,
         n: int = 3) -> Path:
    """A per-condition Phase A run, shaped like the real thing: two tiers, a
    manifest with relative paths, a provenance sidecar."""
    from data.audio_io import write_audio
    from data.invariants import (
        LOUDNESS_TARGET_LUFS,
        ZEROSHOT_SR,
        ZEROSHOT_TIER,
    )

    root = tmp_path / name
    rows = []
    for tier, sr, derived, src, deriv in (
        (ARCHIVE_TIER, ARCHIVE_SR, False, ARCHIVE_SR, f"resample_once:{ARCHIVE_SR}->{ARCHIVE_SR}"),
        (ZEROSHOT_TIER, ZEROSHOT_SR, True, ARCHIVE_SR,
         f"downsample_once:{ARCHIVE_SR}->{ZEROSHOT_SR}"),
    ):
        for cond in ("real", condition):
            for i in range(n):
                rng = np.random.default_rng(real_seed if cond == "real" else 100 + i)
                wav = (0.1 * rng.standard_normal(sr // 2)).astype(np.float32)
                rel = f"{tier}/{cond}/u{i}.wav"
                write_audio(root / rel, wav, sr)
                rows.append({
                    "utt_id": f"u{i}", "condition": cond, "tier": tier, "path": rel,
                    "speaker": "LJ", "split": "train", "derived": derived,
                    "source_rate": src, "derivation": deriv, "resample_steps": 1,
                    "sr_out": sr, "subtype": "PCM_16", "duration_s": 0.5,
                    "lufs_out": LOUDNESS_TARGET_LUFS,
                    "lufs_in": -30.0 if not derived else np.nan,
                    "loudness_gain_db": 3.2 if not derived else np.nan,
                    "loudness_target_lufs": LOUDNESS_TARGET_LUFS if not derived else np.nan,
                    "lufs_in_band": -30.2 if not derived else np.nan,
                    "lufs_out_band": LOUDNESS_TARGET_LUFS if not derived else np.nan,
                    "trim_start": 10, "trim_end": 900, "trim_ref_length": 1000,
                    "mel_n_fft": 1024, "mel_hop": 256, "mel_win": 1024, "mel_n_mels": 80,
                    "mel_fmin": 0.0, "mel_fmax": 8000.0,
                    "alignment_checked": cond != "real",
                    "alignment_peak_lag": None if cond == "real" else 0,
                    "archive_band_hz": ARCHIVE_SR / 2.0, "archive_band": "full_band",
                    "high_band_fraction": 0.015,
                    "vocoder_checkpoint": "", "vocoder_commit": "", "seed": 1,
                })
    manifest = root / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    manifest.with_suffix(".provenance.json").write_text(
        json.dumps({"code_snapshot": {"mode": "record", "sha": "a" * 40,
                                      "tree_sha256": "b" * 64, "citable": True}}),
        encoding="utf-8",
    )
    return manifest


class TestMerge:
    """INV-13's other half, and the check that makes it evidence."""

    def test_two_runs_become_one_validated_dataset(self, tmp_path):
        from data.manifest import read_manifest
        from data.merge import merge_runs

        a = _run(tmp_path, "run_a", "hifigan_v1")
        b = _run(tmp_path, "run_b", "melgan")
        record = merge_runs([a, b], tmp_path / "v1")

        merged = read_manifest(tmp_path / "v1" / "manifest.csv")
        assert set(merged["condition"]) == {"real", "hifigan_v1", "melgan"}
        assert all(Path(p).is_file() for p in merged["path"])
        assert record["real_identical_across_runs"]["identical"] is True
        assert record["conditions"] == {"real": "run_a", "hifigan_v1": "run_a",
                                        "melgan": "run_b"}

    def test_real_is_taken_from_one_run_only(self, tmp_path):
        """Two copies of real under one label would break pairing and double the
        bona fide set."""
        from data.manifest import read_manifest
        from data.merge import merge_runs

        a = _run(tmp_path, "run_a", "hifigan_v1")
        b = _run(tmp_path, "run_b", "melgan")
        merge_runs([a, b], tmp_path / "v1")
        merged = read_manifest(tmp_path / "v1" / "manifest.csv")
        real = merged[(merged["condition"] == "real") & (merged["tier"] == ARCHIVE_TIER)]
        assert len(real) == 3 and real["utt_id"].is_unique

    def test_a_real_that_differs_between_runs_refuses_the_merge(self, tmp_path):
        """The failure INV-13 exists to catch: one environment's pipeline drifted.

        This is the merge earning its place -- one shared environment could never
        have detected it.
        """
        from data.merge import merge_runs

        a = _run(tmp_path, "run_a", "hifigan_v1", real_seed=0)
        b = _run(tmp_path, "run_b", "melgan", real_seed=1)
        with pytest.raises(InvariantViolation, match="not byte-identical"):
            merge_runs([a, b], tmp_path / "v1")

    def test_the_mismatch_can_be_inspected_without_being_shipped(self, tmp_path):
        from data.manifest import read_manifest
        from data.merge import compare_real

        a = _run(tmp_path, "run_a", "hifigan_v1", real_seed=0)
        b = _run(tmp_path, "run_b", "melgan", real_seed=1)
        check = compare_real({"a": read_manifest(a), "b": read_manifest(b)})
        assert check["identical"] is False
        assert check["n_differing"] > 0
        assert {d["utt_id"] for d in check["differing"]} <= {"u0", "u1", "u2"}

    def test_the_same_condition_twice_is_refused(self, tmp_path):
        from data.merge import merge_runs

        a = _run(tmp_path, "run_a", "hifigan_v1")
        b = _run(tmp_path, "run_b", "hifigan_v1")
        with pytest.raises(InvariantViolation, match="appears in both"):
            merge_runs([a, b], tmp_path / "v1")

    def test_runs_over_different_utterances_are_refused(self, tmp_path):
        from data.merge import merge_runs

        a = _run(tmp_path, "run_a", "hifigan_v1", n=3)
        b = _run(tmp_path, "run_b", "melgan", n=2)
        with pytest.raises(InvariantViolation, match="INV-06"):
            merge_runs([a, b], tmp_path / "v1")

    def test_it_refuses_an_existing_output_directory(self, tmp_path):
        from data.merge import merge_runs

        a = _run(tmp_path, "run_a", "hifigan_v1")
        (tmp_path / "v1").mkdir()
        with pytest.raises(InvariantViolation, match="already exists"):
            merge_runs([a], tmp_path / "v1")

    def test_a_merge_is_only_as_citable_as_its_least_citable_run(self, tmp_path):
        from data.merge import merge_is_citable, merge_runs

        a = _run(tmp_path, "run_a", "hifigan_v1")
        b = _run(tmp_path, "run_b", "melgan")
        # run_b was built in iteration mode: no commit names what ran.
        sidecar = b.with_suffix(".provenance.json")
        sidecar.write_text(json.dumps({"code_snapshot": {
            "mode": "iteration", "sha": "dirty", "tree_sha256": "c" * 64, "citable": False,
        }}), encoding="utf-8")
        record = merge_runs([a, b], tmp_path / "v1")
        assert merge_is_citable(record) is False
        assert record["code_snapshots"]["run_b"]["mode"] == "iteration"

    def test_the_merged_manifest_passes_the_ordinary_gate(self, tmp_path):
        """No relaxation: the union goes through validate_manifest unchanged."""
        from data.manifest import read_manifest, validate_manifest
        from data.merge import merge_runs

        a = _run(tmp_path, "run_a", "hifigan_v1")
        b = _run(tmp_path, "run_b", "melgan")
        merge_runs([a, b], tmp_path / "v1")
        validate_manifest(read_manifest(tmp_path / "v1" / "manifest.csv"),
                          require_zeroshot=True)

    def test_paths_in_the_merged_manifest_are_relative_to_it(self, tmp_path):
        from data.merge import merge_runs

        a = _run(tmp_path, "run_a", "hifigan_v1")
        merge_runs([a], tmp_path / "v1")
        raw = pd.read_csv(tmp_path / "v1" / "manifest.csv")
        assert not any(Path(p).is_absolute() for p in raw["path"])
        assert all((tmp_path / "v1" / p).is_file() for p in raw["path"])

    def test_the_cli_merges(self, tmp_path):
        from experiments.cli import main

        a = _run(tmp_path, "run_a", "hifigan_v1")
        b = _run(tmp_path, "run_b", "melgan")
        code = main(["merge", "--manifest", str(a), "--manifest", str(b),
                     "--out-dir", str(tmp_path / "v1")])
        assert code == 0
        assert (tmp_path / "v1" / "manifest.merge.json").is_file()
