import csv
import tempfile
import unittest
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = next(parent for parent in CURRENT_FILE.parents if (parent / "src").is_dir())
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from failure_and_attribution_analysis.agent_failure_evaluator import build_default_failure_evaluator
from iterative_testing.iterative_failure_simulation import ClosedLoopFailureSimulation


NO_ATTACK_SCENARIO = {
    "StateObservationAttack_level": 0,
    "ActionAttack_level": 0,
    "StateTransferAttack_level": 0,
    "RewardAttack_level": 0,
    "ExperiencePoolAttack_level": 0,
    "ModelTampAttack_level": 0,
}

CONSTELLATION_0_NO_ATTACK_SCENARIO = {**NO_ATTACK_SCENARIO, "ConstellationConfig": 0}
CONSTELLATION_2_NO_ATTACK_SCENARIO = {**NO_ATTACK_SCENARIO, "ConstellationConfig": 2}


class InitialBaselineEvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.evaluator = build_default_failure_evaluator()

    def test_healthy_no_attack_baseline(self):
        result = self.evaluator._evaluate_baseline_status(
            scenario=dict(NO_ATTACK_SCENARIO),
            terminal_metrics={"AverageEndingReward": 0.70, "PacketLossRate": 0.10},
            terminal_hard_failure=False,
            true_failure_v2=False,
        )

        self.assertEqual(result["baseline_status"], "healthy")
        self.assertTrue(result["baseline_valid"])
        self.assertFalse(result["baseline_warning"])
        self.assertEqual(result["baseline_reason_codes"], [])

    def test_operable_no_attack_baseline(self):
        result = self.evaluator._evaluate_baseline_status(
            scenario=dict(NO_ATTACK_SCENARIO),
            terminal_metrics={"AverageEndingReward": 0.50, "PacketLossRate": 0.20},
            terminal_hard_failure=False,
            true_failure_v2=False,
        )

        self.assertEqual(result["baseline_status"], "operable")
        self.assertTrue(result["baseline_valid"])
        self.assertFalse(result["baseline_warning"])
        self.assertIn("reward_below_healthy", result["baseline_reason_codes"])
        self.assertIn("packet_loss_above_healthy", result["baseline_reason_codes"])

    def test_invalid_no_attack_baseline(self):
        result = self.evaluator._evaluate_baseline_status(
            scenario=dict(NO_ATTACK_SCENARIO),
            terminal_metrics={"AverageEndingReward": 0.40, "PacketLossRate": 0.10},
            terminal_hard_failure=False,
            true_failure_v2=False,
        )

        self.assertEqual(result["baseline_status"], "invalid")
        self.assertFalse(result["baseline_valid"])
        self.assertIn("reward_below_operable", result["baseline_reason_codes"])

    def test_warning_on_true_failure_v2_for_valid_baseline(self):
        result = self.evaluator._evaluate_baseline_status(
            scenario=dict(NO_ATTACK_SCENARIO),
            terminal_metrics={"AverageEndingReward": 0.70, "PacketLossRate": 0.10},
            terminal_hard_failure=False,
            true_failure_v2=True,
        )

        self.assertEqual(result["baseline_status"], "healthy")
        self.assertTrue(result["baseline_warning"])
        self.assertIn("true_failure_v2_warning", result["baseline_reason_codes"])

    def test_attack_scenario_is_not_applicable(self):
        scenario = dict(NO_ATTACK_SCENARIO)
        scenario["RewardAttack_level"] = 1
        result = self.evaluator._evaluate_baseline_status(
            scenario=scenario,
            terminal_metrics={"AverageEndingReward": 0.20, "PacketLossRate": 0.80},
            terminal_hard_failure=True,
            true_failure_v2=True,
        )

        self.assertEqual(result["baseline_status"], "not_applicable")
        self.assertTrue(result["baseline_valid"])
        self.assertFalse(result["baseline_warning"])

    def test_non_constellation_two_hard_failure_logic_is_unchanged(self):
        metrics = {
            "AverageEndingReward": 0.49,
            "PacketLossRate": 0.10,
            "AverageE2eDelay": 2.0,
            "NetworkThroughput": 100.0,
        }

        result = self.evaluator._terminal_hard_failure(
            metrics,
            reward_threshold=0.5,
            scenario=dict(CONSTELLATION_0_NO_ATTACK_SCENARIO),
        )

        self.assertTrue(result)

    def test_constellation_two_fragile_baseline_is_not_direct_hard_fail(self):
        metrics = {
            "AverageEndingReward": 0.33,
            "PacketLossRate": 0.18,
            "AverageE2eDelay": 3.75,
            "NetworkThroughput": 5305.0,
        }

        result = self.evaluator._terminal_hard_failure(
            metrics,
            reward_threshold=0.5,
            scenario=dict(CONSTELLATION_2_NO_ATTACK_SCENARIO),
        )

        self.assertFalse(result)

    def test_constellation_two_non_hard_fail_baseline_uses_fragile_profile(self):
        result = self.evaluator._evaluate_baseline_status(
            scenario=dict(CONSTELLATION_2_NO_ATTACK_SCENARIO),
            terminal_metrics={"AverageEndingReward": 0.40, "PacketLossRate": 0.10},
            terminal_hard_failure=False,
            true_failure_v2=False,
        )

        self.assertEqual(result["baseline_status"], "fragile")
        self.assertTrue(result["baseline_valid"])
        self.assertEqual(result["baseline_profile"], "constellation_2")
        self.assertIn("constellation2_fragile_baseline", result["baseline_reason_codes"])
        self.assertIn("constellation2_reward_drop_vs_anchor", result["baseline_reason_codes"])
        self.assertNotIn("constellation2_terminal_hard_failure", result["baseline_reason_codes"])

    def test_constellation_two_hard_fail_baseline_remains_invalid(self):
        result = self.evaluator._evaluate_baseline_status(
            scenario=dict(CONSTELLATION_2_NO_ATTACK_SCENARIO),
            terminal_metrics={"AverageEndingReward": 0.18, "PacketLossRate": 0.40},
            terminal_hard_failure=True,
            true_failure_v2=True,
        )

        self.assertEqual(result["baseline_status"], "invalid")
        self.assertFalse(result["baseline_valid"])
        self.assertEqual(result["baseline_profile"], "constellation_2")
        self.assertIn("constellation2_terminal_hard_failure", result["baseline_reason_codes"])


class InitialBaselineEarlyFailTests(unittest.TestCase):
    def setUp(self):
        self.simulation = ClosedLoopFailureSimulation.__new__(ClosedLoopFailureSimulation)
        self.simulation.base_config = {"environment": {}}
        self.simulation.args = SimpleNamespace(
            constellation2_anchor_ending_reward=None,
            constellation2_anchor_packet_loss=None,
            constellation2_anchor_e2e_delay=None,
            constellation2_anchor_throughput=None,
        )
        self.simulation.traffic_profile = "low"

    def test_round_zero_invalid_no_attack_baseline_raises(self):
        invalid_record = {
            "test_id": 3,
            "scenario": dict(NO_ATTACK_SCENARIO),
            "baseline_status": "invalid",
            "terminal_average_ending_reward": 0.40,
            "terminal_packet_loss_rate": 0.35,
            "terminal_hard_failure": True,
            "baseline_reason_codes": ["terminal_hard_failure", "packet_loss_above_operable"],
        }

        with self.assertRaises(RuntimeError) as exc_info:
            self.simulation._validate_initial_baseline_gate(0, [invalid_record])

        message = str(exc_info.exception)
        self.assertIn("round_000", message)
        self.assertIn("\"test_id\": 3", message)
        self.assertIn("\"AverageEndingReward\": 0.4", message)

    def test_round_zero_operable_baseline_passes(self):
        operable_record = {
            "test_id": 1,
            "scenario": dict(NO_ATTACK_SCENARIO),
            "baseline_status": "operable",
            "terminal_average_ending_reward": 0.50,
            "terminal_packet_loss_rate": 0.20,
            "terminal_hard_failure": False,
            "baseline_reason_codes": ["reward_below_healthy"],
        }

        self.simulation._validate_initial_baseline_gate(0, [operable_record])

    def test_round_zero_constellation_two_fragile_baseline_passes(self):
        fragile_record = {
            "test_id": 7,
            "scenario": dict(CONSTELLATION_2_NO_ATTACK_SCENARIO),
            "baseline_status": "fragile",
            "terminal_average_ending_reward": 0.46,
            "terminal_packet_loss_rate": 0.17,
            "terminal_hard_failure": False,
            "baseline_reason_codes": ["constellation2_fragile_baseline"],
        }

        self.simulation._validate_initial_baseline_gate(0, [fragile_record])

    def test_non_initial_round_does_not_raise(self):
        invalid_record = {
            "test_id": 5,
            "scenario": dict(NO_ATTACK_SCENARIO),
            "baseline_status": "invalid",
            "terminal_average_ending_reward": 0.30,
            "terminal_packet_loss_rate": 0.40,
            "terminal_hard_failure": True,
            "baseline_reason_codes": ["terminal_hard_failure"],
        }

        self.simulation._validate_initial_baseline_gate(1, [invalid_record])

    def test_attack_scenario_is_ignored_by_round_zero_gate(self):
        attack_record = {
            "test_id": 6,
            "scenario": {**NO_ATTACK_SCENARIO, "RewardAttack_level": 2},
            "baseline_status": "invalid",
            "terminal_average_ending_reward": 0.10,
            "terminal_packet_loss_rate": 0.90,
            "terminal_hard_failure": True,
            "baseline_reason_codes": ["terminal_hard_failure"],
        }

        self.simulation._validate_initial_baseline_gate(0, [attack_record])


class FusedEffectiveEntryTests(unittest.TestCase):
    def setUp(self):
        self.simulation = ClosedLoopFailureSimulation.__new__(ClosedLoopFailureSimulation)
        self.simulation.args = SimpleNamespace(threshold_calibration_scope="terminal_only")
        self.simulation.true_failure_v2_policy = "strict"
        self.simulation.failure_decision_mode = "single_fused_score"
        self.simulation.evaluator = build_default_failure_evaluator()
        self.simulation.low_failure_regime_config = {
            "enabled": "on",
            "fallback_policy": "dual_threshold_v2",
            "trigger": {
                "min_effective_support": 30,
                "require_both_classes_in_train": "on",
                "min_fused_holdout_auc": 0.55,
                "enable_zero_prediction_guard": "on",
            },
            "allow_small_sample_fused_experiment": "off",
            "small_sample_threshold_min_support": 12,
        }
        self.simulation.last_failure_model_info = {}
        self.simulation.last_threshold_stats = {}

    def test_fused_effective_record_keeps_true_failure_without_terminal_hard_failure(self):
        record = {
            "terminal_hard_failure": False,
            "true_failure_v2": True,
            "true_failure_v2_strict": True,
        }

        self.assertTrue(self.simulation._is_fused_effective_record(record))

    def test_fused_effective_record_keeps_terminal_hard_failure_without_true_failure(self):
        record = {
            "terminal_hard_failure": True,
            "true_failure_v2": False,
            "true_failure_v2_strict": False,
        }

        self.assertTrue(self.simulation._is_fused_effective_record(record))

    def test_fused_effective_record_excludes_non_terminal_negative(self):
        record = {
            "terminal_hard_failure": False,
            "true_failure_v2": False,
            "true_failure_v2_strict": False,
        }

        self.assertFalse(self.simulation._is_fused_effective_record(record))

    def test_fused_effective_record_uses_strict_policy_resolution(self):
        record = {
            "terminal_hard_failure": False,
            "true_failure_v2": True,
            "true_failure_v2_strict": False,
        }

        self.assertFalse(self.simulation._is_fused_effective_record(record))

    def test_build_fused_training_matrix_uses_scheme_b_entry_rule(self):
        positive_without_terminal = {
            "terminal_hard_failure": False,
            "true_failure_v2": True,
            "true_failure_v2_strict": True,
            "terminal_risk_score": 0.7,
            "decision_score_v2": 0.6,
            "converged_mean_v2": 0.5,
            "converged_std_v2": 0.1,
            "converged_slope_v2": 0.2,
            "converged_high_ratio_v2": 0.4,
            "terminal_score_gap_v2": 0.2,
        }
        negative_with_terminal = {
            "terminal_hard_failure": True,
            "true_failure_v2": False,
            "true_failure_v2_strict": False,
            "terminal_risk_score": 0.8,
            "decision_score_v2": 0.3,
            "converged_mean_v2": 0.2,
            "converged_std_v2": 0.1,
            "converged_slope_v2": 0.0,
            "converged_high_ratio_v2": 0.1,
            "terminal_score_gap_v2": 0.6,
        }
        excluded_negative = {
            "terminal_hard_failure": False,
            "true_failure_v2": False,
            "true_failure_v2_strict": False,
            "terminal_risk_score": 0.1,
            "decision_score_v2": 0.1,
            "converged_mean_v2": 0.1,
            "converged_std_v2": 0.1,
            "converged_slope_v2": 0.0,
            "converged_high_ratio_v2": 0.0,
            "terminal_score_gap_v2": 0.0,
        }

        features, labels, filtered_records = self.simulation._build_fused_training_matrix(
            [positive_without_terminal, negative_with_terminal, excluded_negative]
        )

        self.assertEqual(features.shape, (2, 8))
        self.assertEqual(labels.tolist(), [1.0, 0.0])
        self.assertEqual(filtered_records, [positive_without_terminal, negative_with_terminal])

    def test_direct_failure_training_matrix_keeps_original_terminal_only_filter(self):
        positive_without_terminal = {
            "terminal_hard_failure": False,
            "true_failure_v2": True,
            "true_failure_v2_strict": True,
            "converged_mean_v2": 0.5,
            "converged_p75_v2": 0.6,
            "converged_max_v2": 0.7,
            "converged_slope_v2": 0.2,
            "converged_std_v2": 0.1,
            "converged_high_ratio_v2": 0.4,
            "terminal_risk_score": 0.7,
            "terminal_score_gap_v2": 0.2,
        }
        negative_with_terminal = {
            "terminal_hard_failure": True,
            "true_failure_v2": False,
            "true_failure_v2_strict": False,
            "converged_mean_v2": 0.2,
            "converged_p75_v2": 0.3,
            "converged_max_v2": 0.4,
            "converged_slope_v2": 0.0,
            "converged_std_v2": 0.1,
            "converged_high_ratio_v2": 0.1,
            "terminal_risk_score": 0.8,
            "terminal_score_gap_v2": 0.6,
        }

        features, labels, filtered_records = self.simulation._build_direct_failure_training_matrix(
            [positive_without_terminal, negative_with_terminal]
        )

        self.assertEqual(features.shape, (1, 8))
        self.assertEqual(labels.tolist(), [0.0])
        self.assertEqual(filtered_records, [negative_with_terminal])


class LowFailureFallbackTests(unittest.TestCase):
    def setUp(self):
        self.simulation = ClosedLoopFailureSimulation.__new__(ClosedLoopFailureSimulation)
        self.simulation.args = SimpleNamespace(
            threshold_calibration_scope="terminal_only",
            threshold_min_support=30,
        )
        self.simulation.true_failure_v2_policy = "strict"
        self.simulation.failure_decision_mode = "single_fused_score"
        self.simulation.evaluator = build_default_failure_evaluator()
        self.simulation.low_failure_regime_config = {
            "enabled": "on",
            "fallback_policy": "dual_threshold_v2",
            "trigger": {
                "min_effective_support": 30,
                "require_both_classes_in_train": "on",
                "min_fused_holdout_auc": 0.55,
                "enable_zero_prediction_guard": "on",
            },
            "allow_small_sample_fused_experiment": "off",
            "small_sample_threshold_min_support": 12,
        }
        self.simulation.last_failure_model_info = {
            "fused_model_status": "fitted",
            "fused_model_holdout_record_count": 12,
            "primary_score_holdout_auc": 0.80,
            "fused_threshold": 0.5,
            "final_threshold": 0.5,
        }
        self.simulation.last_threshold_stats = {
            "effective_support": 40,
            "train_support": 32,
            "positive_count": 8,
            "negative_count": 24,
        }
        self.simulation.summary_records = []
        self.simulation.low_failure_regime_state = {
            "enabled": "on",
            "fallback_applied": False,
            "fallback_reason": "",
            "effective_decision_mode": "single_fused_score",
        }

    def test_no_fallback_when_support_is_sufficient(self):
        applied, reason = self.simulation._should_fallback_from_fused(
            self.simulation.last_threshold_stats,
            self.simulation.last_failure_model_info,
        )

        self.assertFalse(applied)
        self.assertEqual(reason, "")

    def test_fallback_on_insufficient_effective_support(self):
        applied, reason = self.simulation._should_fallback_from_fused(
            {"effective_support": 14, "train_support": 14, "positive_count": 14, "negative_count": 0},
            self.simulation.last_failure_model_info,
        )

        self.assertTrue(applied)
        self.assertEqual(reason, "insufficient_effective_support")

    def test_fallback_on_single_class_labels(self):
        applied, reason = self.simulation._should_fallback_from_fused(
            {"effective_support": 40, "train_support": 32, "positive_count": 8, "negative_count": 0},
            self.simulation.last_failure_model_info,
        )

        self.assertTrue(applied)
        self.assertEqual(reason, "single_class_labels")

    def test_fallback_on_unavailable_fused_model(self):
        applied, reason = self.simulation._should_fallback_from_fused(
            self.simulation.last_threshold_stats,
            {
                "fused_model_status": "frozen",
                "fused_model_holdout_record_count": 0,
                "primary_score_holdout_auc": 0.80,
            },
        )

        self.assertTrue(applied)
        self.assertEqual(reason, "fused_model_unavailable")

    def test_fallback_on_low_holdout_auc(self):
        applied, reason = self.simulation._should_fallback_from_fused(
            self.simulation.last_threshold_stats,
            {
                "fused_model_status": "fitted",
                "fused_model_holdout_record_count": 12,
                "primary_score_holdout_auc": 0.20,
            },
        )

        self.assertTrue(applied)
        self.assertEqual(reason, "low_fused_holdout_auc")

    def test_fallback_on_zero_prediction_guard(self):
        applied, reason = self.simulation._should_fallback_from_fused(
            self.simulation.last_threshold_stats,
            self.simulation.last_failure_model_info,
            predicted_failure_count=0,
            true_failure_count=3,
        )

        self.assertTrue(applied)
        self.assertEqual(reason, "zero_prediction_guard")

    def test_direct_mode_does_not_use_low_failure_fallback(self):
        self.simulation.failure_decision_mode = "direct_failure_model"

        applied, reason = self.simulation._should_fallback_from_fused(
            self.simulation.last_threshold_stats,
            self.simulation.last_failure_model_info,
        )

        self.assertFalse(applied)
        self.assertEqual(reason, "")

    def test_recompute_predictions_uses_dual_threshold_when_fallback_applies(self):
        self.simulation.last_threshold_stats = {
            "effective_support": 14,
            "train_support": 14,
            "positive_count": 14,
            "negative_count": 0,
        }
        self.simulation.summary_records = [
            {
                "decision_score_v2": 0.45,
                "terminal_risk_score": 0.70,
                "terminal_hard_failure": False,
                "true_failure_v2": True,
                "true_failure_v2_strict": True,
                "fused_score": 0.0,
            }
        ]
        self.simulation._refresh_decision_scores_on_records = lambda: None
        self.simulation._compute_fused_score_and_logit_for_record = lambda record: (0.10, -2.0)
        self.simulation.evaluator.set_v2_failure_threshold(0.35)
        self.simulation.evaluator.set_terminal_threshold_v2(0.55)

        self.simulation._recompute_predictions_from_thresholds()

        record = self.simulation.summary_records[0]
        self.assertTrue(record["system_failure_v2"])
        self.assertEqual(record["effective_decision_mode"], "dual_threshold_v2")
        self.assertTrue(record["low_failure_fallback_applied"])
        self.assertEqual(record["low_failure_fallback_reason"], "insufficient_effective_support")


class PressureRouterAndLowPressureTests(unittest.TestCase):
    def setUp(self):
        self.simulation = ClosedLoopFailureSimulation.__new__(ClosedLoopFailureSimulation)
        self.simulation.args = SimpleNamespace(
            threshold_calibration_scope="terminal_only",
            threshold_calibration_mode="offline_holdout",
            threshold_calibration_holdout_ratio=0.2,
            threshold_min_support=30,
            threshold_objective="balanced_accuracy",
            threshold_min_precision=0.7,
            fused_mlp_hidden_dim=8,
            fused_model_type="mlp",
        )
        self.simulation.true_failure_v2_policy = "strict"
        self.simulation.failure_decision_mode = "single_fused_score"
        self.simulation.evaluator = build_default_failure_evaluator()
        self.simulation.device = torch.device("cpu")
        self.simulation.round_index = 1
        self.simulation.pressure_router_config = {
            "enabled": "on",
            "high_pressure_threshold": 0.45,
            "bandwidth_std_norm_max": 0.20,
            "score_formula": {
                "degraded_edge_ratio_weight": 0.40,
                "edge_disconnect_ratio_weight": 0.35,
                "edge_bandwidth_mean_decrease_ratio_weight": 0.20,
                "edge_bandwidth_decrease_std_norm_weight": 0.05,
            },
            "override": {
                "enabled": "on",
                "apply_scope": "next_round_only",
                "scope": "session",
                "high_risk_decision_threshold": 0.30,
                "high_risk_terminal_threshold": 0.50,
                "upgrade_if_batch_failure_ratio_ge": 0.20,
                "upgrade_if_batch_high_risk_ratio_ge": 0.35,
            },
        }
        self.simulation.pressure_router_state = {
            "router_enabled": "on",
            "override_enabled": "on",
            "pending_override_signal": "keep",
            "pending_override_apply_round": None,
            "last_round_batch_failure_ratio": 0.0,
            "last_round_batch_high_risk_ratio": 0.0,
            "last_round_override_signal": "keep",
            "last_round_override_applied_to_next_round": False,
        }
        self.simulation.low_pressure_classifier_config = {
            "enabled": "on",
            "model_type": "mlp",
            "hidden_dim": 16,
            "dropout": 0.10,
            "learning_rate": 5e-4,
            "weight_decay": 1e-4,
            "epochs": 20,
            "batch_size": 8,
            "patience": 5,
            "pos_weight": 20.0,
            "holdout_ratio": 0.20,
            "feature_set": "summary_v1",
            "threshold": {"objective": "recall_at_precision", "min_precision": 0.50},
            "fallback": {
                "min_effective_support": 2,
                "require_both_classes_in_train": "on",
                "policy": "dual_threshold_v2",
            },
        }
        self.simulation.low_failure_regime_config = {
            "enabled": "on",
            "fallback_policy": "dual_threshold_v2",
            "trigger": {
                "min_effective_support": 30,
                "require_both_classes_in_train": "on",
                "min_fused_holdout_auc": 0.55,
                "enable_zero_prediction_guard": "on",
            },
            "allow_small_sample_fused_experiment": "off",
            "small_sample_threshold_min_support": 12,
        }
        self.simulation.last_failure_model_info = {
            "fused_model_status": "fitted",
            "fused_model_holdout_record_count": 12,
            "primary_score_holdout_auc": 0.80,
            "fused_threshold": 0.5,
            "final_threshold": 0.5,
        }
        self.simulation.last_threshold_stats = {
            "effective_support": 40,
            "train_support": 32,
            "positive_count": 8,
            "negative_count": 24,
        }
        self.simulation.last_low_pressure_model_info = {
            "low_pressure_model_status": "fitted",
            "low_pressure_model_holdout_record_count": 8,
            "low_pressure_model_holdout_auc": 0.72,
            "low_pressure_threshold": 0.40,
        }
        self.simulation.last_low_pressure_threshold_stats = {
            "effective_support": 20,
            "train_support": 16,
            "positive_count": 6,
            "negative_count": 10,
        }
        self.simulation.low_failure_regime_state = {
            "enabled": "on",
            "fallback_applied": False,
            "fallback_reason": "",
            "effective_decision_mode": "single_fused_score",
        }
        self.simulation.last_distribution_balance_guard_info = {}
        self.simulation.latest_coverage_metrics = {
            "coverage_lower_bound": 0.0,
            "coverage_upper_bound": 0.0,
            "coverage_decomposition": {},
        }
        self.simulation.last_decision_model_info = {}
        self.simulation.threshold_update_status = "unknown"
        self.simulation.threshold_support_guard_config = {
            "enabled": "off",
            "support_metric": "train_positive_count",
        }
        self.simulation.distribution_balance_guard_config = {"enabled": "off"}
        self.simulation.threshold_split_config = {
            "mode": "chronological",
            "holdout_ratio": 0.2,
            "late_window_ratio": 0.25,
            "holdout_late_fraction": 0.70,
        }
        self.simulation.threshold_split_seed = 0
        self.simulation.initial_baseline_gate_failure_details = []
        self.simulation.stop_reason = ""
        self.simulation.session_dir = Path(tempfile.mkdtemp())
        self.simulation.step_records = []
        self.simulation.summary_records = []

    def test_pressure_score_formula_and_threshold(self):
        record = {
            "DegradedEdgeRatio": 0.5,
            "EdgeDisconnectRatio": 0.4,
            "EdgeBandwidthMeanDecreaseRatio": 0.3,
            "EdgeBandwidthDecreaseStd": 0.1,
        }
        score = self.simulation._compute_pressure_score(record)
        self.assertAlmostEqual(score, 0.40 * 0.5 + 0.35 * 0.4 + 0.20 * 0.3 + 0.05 * 0.5, places=6)
        self.assertEqual(self.simulation._compute_initial_pressure_regime(record), "low_pressure")

    def test_pressure_override_applies_to_next_round_only(self):
        record = {
            "DegradedEdgeRatio": 0.2,
            "EdgeDisconnectRatio": 0.2,
            "EdgeBandwidthMeanDecreaseRatio": 0.2,
            "EdgeBandwidthDecreaseStd": 0.0,
        }
        record["initial_pressure_regime"] = self.simulation._compute_initial_pressure_regime(record)
        self.assertEqual(
            self.simulation._resolve_effective_pressure_regime(record, round_index=1),
            "low_pressure",
        )
        self.simulation._set_pending_pressure_override("upgrade_to_high", 2)
        self.assertEqual(
            self.simulation._resolve_effective_pressure_regime(record, round_index=1),
            "low_pressure",
        )
        self.assertEqual(
            self.simulation._resolve_effective_pressure_regime(record, round_index=2),
            "high_pressure",
        )

    def test_compute_round_pressure_override_signal(self):
        records = [
            {"decision_score_v2": 0.10, "terminal_risk_score": 0.10, "system_failure_v2": False},
            {"decision_score_v2": 0.35, "terminal_risk_score": 0.20, "system_failure_v2": False},
            {"decision_score_v2": 0.20, "terminal_risk_score": 0.60, "system_failure_v2": True},
        ]
        result = self.simulation._compute_round_pressure_override_signal(records)
        self.assertEqual(result["signal"], "upgrade_to_high")
        self.assertAlmostEqual(result["batch_failure_ratio"], 1.0 / 3.0, places=6)
        self.assertAlmostEqual(result["batch_high_risk_ratio"], 2.0 / 3.0, places=6)

    def test_low_pressure_training_matrix_uses_pressure_regime(self):
        low_record = {
            "effective_pressure_regime": "low_pressure",
            "true_failure_v2": True,
            "true_failure_v2_strict": True,
            "converged_mean_v2": 0.1,
            "converged_p75_v2": 0.2,
            "converged_max_v2": 0.3,
            "converged_std_v2": 0.1,
            "converged_slope_v2": 0.0,
            "converged_high_ratio_v2": 0.2,
            "terminal_risk_score": 0.6,
            "terminal_score_gap_v2": 0.5,
            "decision_score_v2": 0.4,
            "pressure_score": 0.2,
        }
        high_record = dict(low_record)
        high_record["effective_pressure_regime"] = "high_pressure"
        features, labels, filtered = self.simulation._build_low_pressure_training_matrix([low_record, high_record])
        self.assertEqual(features.shape, (1, 10))
        self.assertEqual(labels.tolist(), [1.0])
        self.assertEqual(filtered, [low_record])

    def test_low_pressure_fallback_on_single_class_labels(self):
        applied, reason = self.simulation._should_fallback_from_low_pressure(
            {"effective_support": 20, "train_support": 16, "positive_count": 6, "negative_count": 0},
            self.simulation.last_low_pressure_model_info,
        )
        self.assertTrue(applied)
        self.assertEqual(reason, "single_class_labels")

    def test_recompute_predictions_uses_low_pressure_classifier(self):
        self.simulation.summary_records = [
            {
                "test_id": 11,
                "round_index": 1,
                "DegradedEdgeRatio": 0.2,
                "EdgeDisconnectRatio": 0.2,
                "EdgeBandwidthMeanDecreaseRatio": 0.2,
                "EdgeBandwidthDecreaseStd": 0.0,
                "decision_score_v2": 0.2,
                "terminal_risk_score": 0.2,
                "terminal_hard_failure": False,
                "true_failure_v2": True,
                "true_failure_v2_strict": True,
                "initial_pressure_regime": "low_pressure",
                "effective_pressure_regime": "low_pressure",
            }
        ]
        self.simulation._refresh_decision_scores_on_records = lambda: None
        self.simulation._compute_low_pressure_score_and_logit_for_record = lambda record: (0.8, 1.2)
        self.simulation._should_fallback_from_low_pressure = lambda *args, **kwargs: (False, "")
        self.simulation._compute_fused_score_and_logit_for_record = lambda record: (0.1, -2.0)
        self.simulation._recompute_predictions_from_thresholds()
        record = self.simulation.summary_records[0]
        self.assertTrue(record["system_failure_v2"])
        self.assertEqual(record["effective_decision_mode"], "low_pressure_classifier")
        self.assertEqual(record["decision_source"], "low_pressure_classifier")
        self.assertAlmostEqual(record["low_pressure_score"], 0.8, places=6)

    def test_summary_metrics_keep_global_mode_but_report_record_counts(self):
        self.simulation.low_failure_regime_state["effective_decision_mode"] = "dual_threshold_v2"
        self.simulation.summary_records = [
            {
                "test_id": 1,
                "effective_decision_mode": "low_pressure_classifier",
                "decision_source": "low_pressure_classifier",
                "effective_pressure_regime": "low_pressure",
                "initial_pressure_regime": "low_pressure",
                "pressure_score": 0.2,
                "system_failure_v2": True,
                "true_failure_v2": True,
                "true_failure_v2_strict": True,
            },
            {
                "test_id": 2,
                "effective_decision_mode": "dual_threshold_v2",
                "decision_source": "dual_threshold_v2",
                "effective_pressure_regime": "high_pressure",
                "initial_pressure_regime": "high_pressure",
                "pressure_score": 0.8,
                "system_failure_v2": False,
                "true_failure_v2": False,
                "true_failure_v2_strict": False,
            },
        ]
        metrics = self.simulation._collect_current_summary_metrics()
        self.assertEqual(metrics["effective_decision_mode"], "dual_threshold_v2")
        self.assertEqual(metrics["effective_decision_mode_scope"], "global_summary_state")
        self.assertEqual(
            metrics["effective_decision_mode_counts"],
            {"dual_threshold_v2": 1, "low_pressure_classifier": 1},
        )
        self.assertEqual(metrics["low_pressure_classifier_record_count"], 1)
        self.assertEqual(metrics["high_pressure_dual_threshold_fallback_record_count"], 1)

    def test_low_pressure_threshold_uses_holdout_scores(self):
        self.simulation.last_low_pressure_model_info["threshold_split"] = {
            "train_positions": [0, 1, 2, 3],
            "holdout_positions": [4, 5, 6],
        }
        records = [
            {"effective_pressure_regime": "low_pressure", "true_failure_v2": True, "true_failure_v2_strict": True, "mock_score": 0.90},
            {"effective_pressure_regime": "low_pressure", "true_failure_v2": True, "true_failure_v2_strict": True, "mock_score": 0.80},
            {"effective_pressure_regime": "low_pressure", "true_failure_v2": False, "true_failure_v2_strict": False, "mock_score": 0.70},
            {"effective_pressure_regime": "low_pressure", "true_failure_v2": False, "true_failure_v2_strict": False, "mock_score": 0.10},
            {"effective_pressure_regime": "low_pressure", "true_failure_v2": True, "true_failure_v2_strict": True, "mock_score": 0.60},
            {"effective_pressure_regime": "low_pressure", "true_failure_v2": False, "true_failure_v2_strict": False, "mock_score": 0.55},
            {"effective_pressure_regime": "low_pressure", "true_failure_v2": False, "true_failure_v2_strict": False, "mock_score": 0.20},
        ]
        self.simulation._compute_low_pressure_score_and_logit_for_record = lambda record: (record["mock_score"], 0.0)
        stats = self.simulation._calibrate_low_pressure_threshold(records)
        self.assertEqual(stats["status"], "updated")
        self.assertAlmostEqual(stats["low_pressure_threshold"], 0.60, places=6)
        self.assertEqual(stats["low_pressure_threshold_selected_from"], "holdout")
        self.assertEqual(stats["low_pressure_threshold_constraint_status"], "satisfied")

    def test_low_pressure_threshold_freezes_when_holdout_support_is_insufficient(self):
        self.simulation.last_low_pressure_model_info["threshold_split"] = {
            "train_positions": [0, 1, 2, 3],
            "holdout_positions": [4],
        }
        records = [
            {"effective_pressure_regime": "low_pressure", "true_failure_v2": True, "true_failure_v2_strict": True, "mock_score": 0.90},
            {"effective_pressure_regime": "low_pressure", "true_failure_v2": True, "true_failure_v2_strict": True, "mock_score": 0.80},
            {"effective_pressure_regime": "low_pressure", "true_failure_v2": False, "true_failure_v2_strict": False, "mock_score": 0.40},
            {"effective_pressure_regime": "low_pressure", "true_failure_v2": False, "true_failure_v2_strict": False, "mock_score": 0.20},
            {"effective_pressure_regime": "low_pressure", "true_failure_v2": True, "true_failure_v2_strict": True, "mock_score": 0.60},
        ]
        self.simulation._compute_low_pressure_score_and_logit_for_record = lambda record: (record["mock_score"], 0.0)
        stats = self.simulation._calibrate_low_pressure_threshold(records)
        self.assertEqual(stats["status"], "frozen")
        self.assertEqual(stats["reason"], "insufficient_holdout_support")
        self.assertEqual(stats["low_pressure_threshold_reason"], "insufficient_holdout_support")

    def test_low_pressure_threshold_uses_fallback_pool_when_precision_not_met(self):
        self.simulation.low_pressure_classifier_config["threshold"]["min_precision"] = 0.95
        self.simulation.last_low_pressure_model_info["threshold_split"] = {
            "train_positions": [0, 1, 2, 3],
            "holdout_positions": [4, 5, 6],
        }
        records = [
            {"effective_pressure_regime": "low_pressure", "true_failure_v2": True, "true_failure_v2_strict": True, "mock_score": 0.90},
            {"effective_pressure_regime": "low_pressure", "true_failure_v2": True, "true_failure_v2_strict": True, "mock_score": 0.80},
            {"effective_pressure_regime": "low_pressure", "true_failure_v2": False, "true_failure_v2_strict": False, "mock_score": 0.70},
            {"effective_pressure_regime": "low_pressure", "true_failure_v2": False, "true_failure_v2_strict": False, "mock_score": 0.10},
            {"effective_pressure_regime": "low_pressure", "true_failure_v2": True, "true_failure_v2_strict": True, "mock_score": 0.50},
            {"effective_pressure_regime": "low_pressure", "true_failure_v2": False, "true_failure_v2_strict": False, "mock_score": 0.60},
            {"effective_pressure_regime": "low_pressure", "true_failure_v2": False, "true_failure_v2_strict": False, "mock_score": 0.40},
        ]
        self.simulation._compute_low_pressure_score_and_logit_for_record = lambda record: (record["mock_score"], 0.0)
        stats = self.simulation._calibrate_low_pressure_threshold(records)
        self.assertEqual(stats["status"], "updated")
        self.assertEqual(stats["low_pressure_threshold_constraint_status"], "all_candidates_below_min_precision")
        self.assertAlmostEqual(stats["low_pressure_threshold"], 0.50, places=6)

    def test_low_pressure_threshold_rebuilds_split_when_positions_are_missing(self):
        self.simulation.low_pressure_classifier_config["holdout_ratio"] = 0.4
        self.simulation.threshold_split_config = {
            "mode": "chronological",
            "holdout_ratio": 0.4,
            "late_window_ratio": 0.25,
            "holdout_late_fraction": 0.70,
        }
        self.simulation.last_low_pressure_model_info["threshold_split"] = {
            "threshold_split_mode": "chronological",
            "threshold_split_holdout_ratio": 0.4,
            "threshold_split_train_support": 4,
            "threshold_split_holdout_support": 3,
        }
        records = [
            {"effective_pressure_regime": "low_pressure", "true_failure_v2": False, "true_failure_v2_strict": False, "mock_score": 0.05},
            {"effective_pressure_regime": "low_pressure", "true_failure_v2": False, "true_failure_v2_strict": False, "mock_score": 0.10},
            {"effective_pressure_regime": "low_pressure", "true_failure_v2": True, "true_failure_v2_strict": True, "mock_score": 0.65},
            {"effective_pressure_regime": "low_pressure", "true_failure_v2": False, "true_failure_v2_strict": False, "mock_score": 0.15},
            {"effective_pressure_regime": "low_pressure", "true_failure_v2": True, "true_failure_v2_strict": True, "mock_score": 0.90},
            {"effective_pressure_regime": "low_pressure", "true_failure_v2": False, "true_failure_v2_strict": False, "mock_score": 0.20},
            {"effective_pressure_regime": "low_pressure", "true_failure_v2": True, "true_failure_v2_strict": True, "mock_score": 0.80},
        ]
        self.simulation._compute_low_pressure_score_and_logit_for_record = lambda record: (record["mock_score"], 0.0)
        stats = self.simulation._calibrate_low_pressure_threshold(records)
        self.assertEqual(stats["status"], "updated")
        self.assertGreater(stats["train_support"], 0)
        self.assertGreater(stats["holdout_support"], 0)
        self.assertTrue(self.simulation.last_low_pressure_model_info["threshold_split"]["train_positions"])
        self.assertTrue(self.simulation.last_low_pressure_model_info["threshold_split"]["holdout_positions"])

    def test_write_low_pressure_score_debug_csv(self):
        self.simulation.summary_records = [
            {
                "test_id": 101,
                "effective_pressure_regime": "low_pressure",
                "initial_pressure_regime": "low_pressure",
                "true_failure_v2": True,
                "true_failure_v2_strict": True,
                "low_pressure_score": 0.8,
                "low_pressure_threshold": 0.4,
                "low_pressure_pred": True,
                "system_failure_v2": False,
                "effective_decision_mode": "low_pressure_dual_threshold_fallback",
                "decision_source": "low_pressure_dual_threshold_fallback",
            },
            {
                "test_id": 102,
                "effective_pressure_regime": "high_pressure",
                "initial_pressure_regime": "high_pressure",
                "true_failure_v2": False,
                "true_failure_v2_strict": False,
                "system_failure_v2": True,
                "effective_decision_mode": "single_fused_score",
                "decision_source": "single_fused_score",
            },
        ]
        self.simulation._write_low_pressure_score_debug_csv("after_offline_recompute")
        csv_path = self.simulation.session_dir / "low_pressure_score_debug.csv"
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]["sample_id"], "101")
        self.assertEqual(rows[0]["pressure_group"], "low_pressure")
        self.assertEqual(rows[0]["decision_source"], "low_pressure_dual_threshold_fallback")
        self.assertEqual(rows[0]["low_pressure_pred"], "True")
        self.assertEqual(rows[0]["final_pred"], "False")
        self.assertEqual(rows[1]["sample_id"], "102")
        self.assertEqual(rows[1]["low_pressure_score"], "")
        self.assertEqual(rows[1]["decision_source"], "single_fused_score")

    def test_post_run_offline_recompute_restores_low_pressure_state(self):
        original_model_info = {
            "low_pressure_model_status": "fitted",
            "low_pressure_model_holdout_record_count": 8,
            "low_pressure_model_holdout_auc": 0.72,
            "low_pressure_threshold": 0.40,
        }
        original_threshold_stats = {
            "status": "updated",
            "low_pressure_threshold": 0.40,
            "train_support": 16,
            "holdout_support": 4,
        }
        self.simulation.last_low_pressure_model_info = dict(original_model_info)
        self.simulation.last_low_pressure_threshold_stats = dict(original_threshold_stats)
        self.simulation.summary_records = [
            {
                "test_id": 1,
                "effective_pressure_regime": "low_pressure",
                "initial_pressure_regime": "low_pressure",
                "true_failure_v2": True,
                "true_failure_v2_strict": True,
                "system_failure_v2": True,
            }
        ]
        self.simulation._fit_failure_decision_models = lambda: (
            self.simulation.last_low_pressure_model_info.update({"low_pressure_threshold": 0.12}) or {}
        )
        self.simulation._calibrate_failure_threshold_v2 = lambda: {
            "status": "updated",
            "threshold": 0.2,
        }
        self.simulation._recompute_predictions_from_thresholds = lambda: (
            self.simulation.last_low_pressure_threshold_stats.update({"low_pressure_threshold": 0.12}) or None
        )
        summary = self.simulation._simulate_post_run_offline_recompute_summary()
        self.assertEqual(summary["status"], "ok")
        self.assertEqual(self.simulation.last_low_pressure_model_info, original_model_info)
        self.assertEqual(self.simulation.last_low_pressure_threshold_stats, original_threshold_stats)


if __name__ == "__main__":
    unittest.main()
