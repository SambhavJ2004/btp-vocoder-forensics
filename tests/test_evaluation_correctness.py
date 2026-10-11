"""Directional EER and separate confound diagnostics, using synthetic data only."""

from __future__ import annotations

from itertools import permutations
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_curve

from data.invariants import ARCHIVE_SR, SANITY_EER_FLOOR, TIER_RATES
from detectors.eer import bootstrap_eer_ci, compute_eer
from detectors.protocols import Protocol, score_condition
from experiments.sanity_checks import _verdict, probe_orientation_diagnostic


class TestDirectionalEER:
    @pytest.mark.parametrize("bona,spoof,expected", [
        ([0] * 4, [0] * 4, 0.5),
        ([7] * 2, [7] * 5, 0.5),
        ([0, 1, 2], [0, 1, 2], 0.5),
        ([5] * 4, [-5] * 4, 0.0),
        ([-5] * 4, [5] * 4, 1.0),
        ([0, 1, 1], [0, 0, 1], 1 / 3),
        ([0, 1, 1, 2], [0, 0, 1], 3 / 10),
        ([1, 2, 3], [0, 1], 1 / 5),
        ([0, 3, 5], [1, 2], 1 / 3),
    ])
    def test_hand_derived_error_rates(self, bona, spoof, expected):
        # Mixed ties: rejecting 0 gives (FRR,FAR)=(1/4,1/3); rejecting
        # 1 gives (3/4,0). Their intersection is at weight 1/10, EER 3/10.
        # Unequal classes: (0,1/2)->(1/3,0) intersects at weight 3/5, EER 1/5.
        # With unique scores, (1/3,1/2)->(1/3,0) intersects at EER 1/3.
        eer, threshold = compute_eer(bona, spoof)
        assert eer == pytest.approx(expected, abs=1e-12)
        assert threshold in set(bona) | set(spoof)
        assert np.isfinite(threshold)

    def test_interpolated_eer_is_not_a_deterministic_split_of_constant_scores(self):
        bona, spoof = np.zeros(4), np.zeros(4)
        eer, threshold = compute_eer(bona, spoof)
        assert (eer, threshold) == (0.5, 0.0)
        # Under the documented > rule, all ties reject together. There is no
        # attainable deterministic threshold with FRR == FAR == 1/2 here.
        assert np.mean(bona <= threshold) == 1.0
        assert np.mean(spoof > threshold) == 0.0

    def test_exact_crossing_threshold_has_the_documented_decision_rule(self):
        bona, spoof = np.array([0, 1, 1]), np.array([0, 0, 1])
        eer, threshold = compute_eer(bona, spoof)
        assert threshold == 0.0
        assert np.mean(bona <= threshold) == pytest.approx(eer)
        assert np.mean(spoof > threshold) == pytest.approx(eer)

    def test_permuting_either_class_changes_neither_eer_nor_boundary(self):
        for bona in set(permutations([0, 1, 1, 2])):
            for spoof in set(permutations([0, 0, 1])):
                assert compute_eer(bona, spoof) == pytest.approx((0.3, 1.0))

    @pytest.mark.parametrize("bona,spoof,expected", [
        ([2, 3], [-2, -3], 0.0),
        ([-2, -3], [2, 3], 1.0),
        ([0, 1, 1, 2], [0, 0, 1], 0.3),
        ([0, 1, 1], [0, 0, 1], 1 / 3),
        ([0, 0], [0, 0], 0.5),
    ])
    def test_score_sign_reversal_preserves_directional_meaning(self, bona, spoof, expected):
        forward, _ = compute_eer(bona, spoof)
        reverse, _ = compute_eer(-np.asarray(bona), -np.asarray(spoof))
        assert forward == pytest.approx(expected)
        assert reverse == pytest.approx(1 - expected)

    @pytest.mark.parametrize("seed", range(12))
    def test_agrees_with_independent_sklearn_roc(self, seed):
        rng = np.random.default_rng(seed)
        bona = rng.integers(-3, 4, size=11 + seed)
        spoof = rng.integers(-3, 4, size=5 + seed)
        labels = np.r_[np.ones(len(bona)), np.zeros(len(spoof))]
        far, tpr, _ = roc_curve(labels, np.r_[bona, spoof], drop_intermediate=False)
        # sklearn supplies the attainable, grouped ROC. Interpolate FAR at
        # FRR-FAR=0, independently of our threshold/group construction.
        difference = (1 - tpr) - far
        reference = np.interp(0.0, difference[::-1], far[::-1])
        assert compute_eer(bona, spoof)[0] == pytest.approx(reference, abs=1e-12)

    def test_extreme_finite_scores_need_no_epsilon_or_overflowing_midpoint(self):
        maximum = np.finfo(np.float64).max
        assert compute_eer([maximum], [-maximum]) == (0.0, -maximum)
        assert compute_eer([-maximum], [maximum]) == (1.0, -maximum)
        assert compute_eer([maximum], [maximum]) == (0.5, maximum)

    def test_inputs_are_not_mutated(self):
        bona, spoof = np.array([2.0, 0.0, 1.0]), np.array([1.0, 0.0])
        before_bona, before_spoof = bona.copy(), spoof.copy()
        compute_eer(bona, spoof)
        np.testing.assert_array_equal(bona, before_bona)
        np.testing.assert_array_equal(spoof, before_spoof)


class TestInvalidScores:
    @pytest.mark.parametrize("class_index", [0, 1])
    @pytest.mark.parametrize("invalid,message", [
        ([], "both bona fide and spoof"),
        (0.0, "one-dimensional"),
        (None, "one-dimensional"),
        ([[0.0, 1.0]], "one-dimensional"),
        ([[0.0], [1.0]], "one-dimensional"),
        ([[0.0], [1.0, 2.0]], "one-dimensional"),
        ([np.nan], "finite"),
        ([np.inf], "finite"),
        ([-np.inf], "finite"),
        ([1 + 2j], "real scores"),
        (["not a score"], "numeric"),
    ])
    def test_invalid_input_in_either_class_is_explicit(self, invalid, message, class_index):
        inputs = [[0.0, 1.0], [0.0, 1.0]]
        inputs[class_index] = invalid
        with pytest.raises(ValueError, match=message):
            compute_eer(*inputs)

    def test_both_classes_missing(self):
        with pytest.raises(ValueError, match="both bona fide and spoof"):
            compute_eer([], [])


class TestConsumers:
    @pytest.mark.parametrize("bona,spoof,expected", [
        ([0] * 3, [0] * 7, (0.5, 0.5)),
        ([1] * 3, [0] * 7, (0.0, 0.0)),
        ([0] * 3, [1] * 7, (1.0, 1.0)),
    ])
    def test_classwise_bootstrap_uses_the_corrected_metric(self, bona, spoof, expected):
        assert bootstrap_eer_ci(bona, spoof, n_boot=32, seed=7) == expected

    def test_bootstrap_does_not_flatten_invalid_input_before_metric_validation(self):
        with pytest.raises(ValueError, match="one-dimensional"):
            bootstrap_eer_ci([[0, 1]], [0, 1], n_boot=2)

    def test_bootstrap_seed_remains_reproducible(self):
        bona, spoof = np.array([0, 1, 1, 2]), np.array([0, 0, 1])
        a = bootstrap_eer_ci(bona, spoof, n_boot=64, seed=21)
        assert a == bootstrap_eer_ci(bona, spoof, n_boot=64, seed=21)
        assert 0.0 <= a[0] <= a[1] <= 1.0

    @pytest.mark.parametrize("protocol,tier", [
        (Protocol.MATCHED, "archive"), (Protocol.MISMATCHED, "zeroshot"),
    ])
    @pytest.mark.parametrize("spoof_score,expected", [(0.0, 0.5), (1.0, 1.0)])
    def test_protocol_scoring_keeps_direction_labels_tier_and_ci(
        self, monkeypatch, protocol, tier, spoof_score, expected
    ):
        frame = pd.DataFrame([
            {"utt_id": f"u{i}", "condition": cond, "tier": tier,
             "split": "eval", "path": f"{cond}/{i}"}
            for cond in ("real", "hifigan_v1") for i in range(4)
        ])

        def read(path, expected_sr):
            assert expected_sr == TIER_RATES[tier]
            return np.array([0.0 if path.startswith("real/") else spoof_score])

        monkeypatch.setattr("data.audio_io.read_processed", read)
        detector = SimpleNamespace(
            spec=SimpleNamespace(key="synthetic-fixture"),
            score_batch=lambda wavs: np.array([w[0] for w in wavs]),
        )
        trained = ("real", "hifigan_v1") if protocol is Protocol.MATCHED else ()
        result = score_condition(
            detector, frame, "hifigan_v1", protocol=protocol, tier=tier,
            train_conditions=trained,
        )
        assert result.eer == expected
        assert result.eer_ci_low == result.eer_ci_high == expected
        assert result.protocol is protocol and result.tier == tier
        assert result.train_conditions == trained
        assert result.n_bonafide == result.n_spoof == 4


class TestProbeOrientation:
    @pytest.mark.parametrize("real,condition,expected", [
        ([0] * 4, [1] * 4, (1.0, 0.0, 0.0)),
        ([1] * 4, [0] * 4, (0.0, 1.0, 0.0)),
        ([0] * 4, [0] * 4, (0.5, 0.5, 0.5)),
        ([0, 1, 1, 2], [0, 0, 1], (0.3, 0.7, 0.3)),
    ])
    def test_diagnostic_reports_both_correct_orientations(self, real, condition, expected):
        result = probe_orientation_diagnostic(real, condition)
        assert (result["eer_forward"], result["eer_reverse"], result["eer_separability"]) == (
            pytest.approx(expected)
        )

    @pytest.mark.parametrize("probe", [
        "duration", "silence_duration", "rms_level", "bandwidth_rolloff",
    ])
    def test_reversed_separability_does_not_silently_change_existing_verdict(self, probe):
        real, condition = np.zeros(4), np.ones(4)
        before = _verdict(probe, "hifigan_v1", compute_eer(real, condition)[0], "archive")
        diagnostic = probe_orientation_diagnostic(real, condition)
        after = _verdict(probe, "hifigan_v1", compute_eer(real, condition)[0], "archive")
        assert before == after
        assert diagnostic["eer_separability"] == 0.0
        assert not before["leaked"]  # Existing one-sided issue, awaiting review.
        assert not before["expected_by_design"]

    @pytest.mark.parametrize("eer,leaked", [(0.0, True), (0.4, False), (0.6, False), (1.0, False)])
    def test_current_floor_and_one_sided_rule_are_unchanged(self, eer, leaked):
        result = _verdict("rms_level", "hifigan_v1", eer, "archive")
        assert result["floor"] == SANITY_EER_FLOOR == 0.40
        assert result["leaked"] is leaked

    @pytest.mark.parametrize("condition", [
        "griffin_lim", "vocos", "bigvgan_v2_22khz_fullband",
    ])
    def test_existing_bandwidth_design_exceptions_are_unchanged(self, condition):
        result = _verdict("bandwidth_rolloff", condition, 0.0, "archive")
        assert result["expected_by_design"] and not result["leaked"]

    @pytest.mark.parametrize("probe_name", [
        "duration", "silence_duration", "rms_level", "bandwidth_rolloff",
    ])
    def test_each_raw_feature_probe_can_have_the_reversed_orientation(
        self, monkeypatch, probe_name
    ):
        import experiments.sanity_checks as sanity

        # These are isolated feature-probe fixtures, not admissible Phase A runs.
        # In particular, the duration fixture deliberately breaks equal lengths.
        n = ARCHIVE_SR
        t = np.arange(n) / ARCHIVE_SR
        real = 0.1 * np.sin(2 * np.pi * 120 * t)
        condition = real.copy()
        if probe_name == "rms_level":
            condition *= 2
        elif probe_name == "bandwidth_rolloff":
            condition = 0.1 * np.sin(2 * np.pi * 6000 * t)
        elif probe_name == "silence_duration":
            for wav, pad_s in ((real, 0.05), (condition, 0.1)):
                pad = round(pad_s * ARCHIVE_SR)
                wav[:pad] = wav[-pad:] = 0.0
        elif probe_name == "duration":
            condition = np.tile(real, 2)
        waves = {"real": real.astype(np.float32), "hifigan_v1": condition.astype(np.float32)}
        frame = pd.DataFrame([
            {"utt_id": f"u{i}", "condition": cond, "tier": "archive", "split": "eval",
             "path": cond, "duration_s": len(wav) / ARCHIVE_SR}
            for cond, wav in waves.items() for i in range(4)
        ])
        monkeypatch.setattr(sanity, "read_processed", lambda p, sr: waves[p])
        # The extra diagnostic must remain separate from every current probe.
        def forbidden(*args, **kwargs):
            raise AssertionError("the current gate must not call the new orientation diagnostic")

        monkeypatch.setattr(sanity, "probe_orientation_diagnostic", forbidden)
        if probe_name == "silence_duration":
            feature = sum(sanity._leading_trailing_silence(real, ARCHIVE_SR, threshold_db=-40))
            reference = SimpleNamespace(
                real_feature={f"u{i}": feature for i in range(4)},
                threshold_dbfs=-40, pad_level_dbfs=-240,
            )
            result = sanity.silence_probe(frame, "hifigan_v1", reference=reference)
        else:
            probe = {"duration": sanity.duration_probe, "rms_level": sanity.loudness_probe,
                     "bandwidth_rolloff": sanity.bandwidth_probe}[probe_name]
            result = probe(frame, "hifigan_v1")
        assert result["eer"] == 1.0
        assert not result["leaked"] and not result["expected_by_design"]
