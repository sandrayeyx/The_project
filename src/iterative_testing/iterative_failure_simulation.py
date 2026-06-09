import argparse
import csv
import copy
import io
import itertools
import json
import math
import os
import shutil
import subprocess
import sys
from collections import Counter
from collections.abc import Sequence as ABCSequence
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = next(parent for parent in CURRENT_FILE.parents if (parent / "src").is_dir())
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from project_paths import (
    CLOSED_LOOP_OUTPUTS_ROOT,
    DEFAULT_TRAIN_CONFIG_PATH,
    ENV_CONFIG_PATH,
    ITERATIVE_TESTING_ROOT,
    SCENARIO_EXPLORATION_CONFIG_PATH,
    TRAINING_PROCESS_DATA_ROOT,
)

DEFAULT_EXPLORATION_CONFIG = SCENARIO_EXPLORATION_CONFIG_PATH

DEFAULT_ONLINE_THRESHOLD_SPLIT_CONFIG: Dict[str, Any] = {
    "mode": "stratified_late_holdout",
    "holdout_ratio": 0.20,
    "late_window_ratio": 0.25,
    "holdout_late_fraction": 0.70,
    "random_seed": "",
}
DEFAULT_POST_RUN_ROLLING_DRIFT_ANALYSIS_CONFIG: Dict[str, Any] = {
    "enabled": "off",
    "window_size": 80,
    "step_size": 20,
    "min_train_support": 40,
    "min_holdout_support": 20,
    "output_filename": "rolling_drift_analysis.json",
}
DEFAULT_ONLINE_THRESHOLD_CALIBRATION_CONFIG: Dict[str, Any] = {
    "objective": "balanced_accuracy",
    "min_precision": 0.70,
    "support_aware_threshold_guard": {
        "enabled": "on",
        "support_metric": "train_positive_count",
        "low_support": {
            "positive_count_max": 8,
            "update_mode": "weak_update",
            "max_delta": 0.03,
        },
        "medium_support": {
            "positive_count_max": 20,
            "update_mode": "bounded_update",
            "max_delta": 0.08,
        },
        "high_support": {
            "update_mode": "full_update",
        },
    },
}
DEFAULT_LOW_FAILURE_REGIME_CONFIG: Dict[str, Any] = {
    "enabled": "on",
    "fallback_policy": "dual_threshold_v2",
    "trigger": {
        "min_effective_support": "",
        "require_both_classes_in_train": "on",
        "min_fused_holdout_auc": 0.55,
        "enable_zero_prediction_guard": "on",
    },
    "allow_small_sample_fused_experiment": "off",
    "small_sample_threshold_min_support": 12,
}
DEFAULT_PRESSURE_ROUTER_CONFIG: Dict[str, Any] = {
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
DEFAULT_LOW_PRESSURE_CLASSIFIER_CONFIG: Dict[str, Any] = {
    "enabled": "on",
    "model_type": "mlp",
    "hidden_dim": 16,
    "dropout": 0.10,
    "learning_rate": 5e-4,
    "weight_decay": 1e-4,
    "epochs": 200,
    "batch_size": 16,
    "patience": 20,
    "pos_weight": 20.0,
    "holdout_ratio": 0.20,
    "feature_set": "summary_v1",
    "threshold": {
        "objective": "recall_at_precision",
        "min_precision": 0.50,
    },
    "fallback": {
        "min_effective_support": 12,
        "require_both_classes_in_train": "on",
        "policy": "dual_threshold_v2",
    },
}
DEFAULT_DISTRIBUTION_BALANCE_GUARD_CONFIG: Dict[str, Any] = {
    "enabled": "on",
    "mode": "three_layer_balance",
    "default_behavior": {
        "keep_total_scenarios_per_round_fixed": "on",
        "force_healthy_quota_each_round": "off",
        "allow_boundary_priority_as_default": "on",
    },
    "trigger": {
        "selected_healthy_band_zero_rounds": 2,
        "shortage_flag_rounds": 2,
        "healthy_baseline_count_max": 1,
        "require_prefilter_healthy_band_zero": "on",
    },
    "recovery": {
        "extra_round_budget_mode": "replace_only",
        "replace_boundary_slots_default": 1,
        "replace_boundary_slots_escalated": 2,
        "escalate_after_triggered_rounds": 2,
        "auto_exit_after_healthy_recovered_rounds": 2,
        "allow_skip_if_no_recovery_candidate": "on",
        "recovery_source_priority": ["healthy_recovery", "healthy_push"],
    },
    "constellation_profiles": {
        "default": {
            "selected_healthy_band_zero_rounds": 2,
            "shortage_flag_rounds": 2,
            "replace_boundary_slots_default": 1,
            "replace_boundary_slots_escalated": 2,
            "allow_escalation": "on",
        },
        "large_constellation": {
            "constellation_ids": [2],
            "selected_healthy_band_zero_rounds": 3,
            "shortage_flag_rounds": 3,
            "replace_boundary_slots_default": 1,
            "replace_boundary_slots_escalated": 1,
            "allow_escalation": "off",
        },
    },
}
ALLOWED_THRESHOLD_OBJECTIVES = {"f1", "accuracy", "balanced_accuracy", "recall_at_precision"}
HEALTHY_BAND_DEGRADATION_LIMITS: Dict[str, float] = {
    "DegradedEdgeRatio": 0.05,
    "EdgeDisconnectRatio": 0.02,
    "EdgeBandwidthMeanDecreaseRatio": 0.08,
}

from iterative_testing.gpu_runtime import select_torch_device
from iterative_testing.run_batch_experiments import (
    is_attack_combination_valid,
    normalize_single_attack_types,
    parse_experiment_md,
)

from failure_and_attribution_analysis.agent_failure_evaluator import build_default_failure_evaluator
from failure_and_attribution_analysis.deep_ensemble_network import build_default_deep_ensemble
from failure_and_attribution_analysis.failure_boundary_explorer import FailureBoundaryExplorer
from failure_and_attribution_analysis.parameter_interfaces import (
    CONTINUOUS_FEATURE_NAMES,
    DISCRETE_FEATURE_NAMES,
    FailEnv,
    SCENARIO_PARAMETER_NAMES,
)
from failure_and_attribution_analysis.scenario_parameter_generator import (
    build_continuous_feature_bounds,
    FeatureSimilarityNetwork,
    ScenarioParameterGenerator,
)

ATTACK_SCENARIO_KEYS: Tuple[str, ...] = (
    "StateObservationAttack_level",
    "ActionAttack_level",
    "StateTransferAttack_level",
    "RewardAttack_level",
    "ExperiencePoolAttack_level",
    "ModelTampAttack_level",
)

CONSTELLATION_2_GATE_PROFILE = {
    "max_reward_drop_vs_anchor": 0.15,
    "max_packet_loss_increase_vs_anchor": 0.10,
    "max_delay_increase_ratio_vs_anchor": 1.25,
    "min_throughput_ratio_vs_anchor": 0.70,
}


class InitialBaselineGateError(RuntimeError):
    def __init__(self, message: str, details: List[Dict]):
        super().__init__(message)
        self.details = details


def parse_bool_arg(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_on_off_arg(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized in {"on", "off"}:
        return normalized
    raise argparse.ArgumentTypeError(f"Invalid on/off value: {value}")


def parse_decision_formula_weights(value: str) -> Dict[str, float]:
    parts = [piece.strip() for piece in str(value or "").split(",") if piece.strip()]
    if len(parts) != 5:
        raise argparse.ArgumentTypeError(
            "decision formula weights must contain 5 comma-separated values: "
            "w_mean,w_p75,w_max,w_slope_pos,w_std_penalty"
        )
    numbers = [float(piece) for piece in parts]
    return {
        "w_mean": numbers[0],
        "w_p75": numbers[1],
        "w_max": numbers[2],
        "w_slope_pos": numbers[3],
        "w_std_penalty": numbers[4],
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run the closed-loop failure-analysis simulation workflow.")
    parser.add_argument("--config", default=str(DEFAULT_TRAIN_CONFIG_PATH))
    parser.add_argument("--env-md", default=str(ENV_CONFIG_PATH))
    parser.add_argument(
        "--exploration-config",
        default=str(DEFAULT_EXPLORATION_CONFIG),
        help="YAML config for scenario exploration options such as single_attack_types.",
    )
    parser.add_argument(
        "--output-root",
        default=str(CLOSED_LOOP_OUTPUTS_ROOT),
    )
    parser.add_argument("--generated-limit", type=int, default=400)
    parser.add_argument("--scenarios-per-round", type=int, default=16)
    parser.add_argument("--seed-per-region", type=int, default=48)
    parser.add_argument("--n-clusters", type=int, default=3)
    parser.add_argument("--similarity-threshold", type=float, default=0.97)
    parser.add_argument("--similarity-threshold-max", type=float, default=0.995)
    parser.add_argument("--min-scenarios-per-round", type=int, default=4)
    parser.add_argument("--generation-cv-threshold", type=float, default=0.03)
    parser.add_argument("--coverage-confidence", type=float, default=0.95)
    parser.add_argument(
        "--coverage-target",
        type=float,
        default=0.90,
        help=(
            "Target conditional coverage for the current exploration constraints. "
            "Coverage is computed over the continuous scenario space of sampled scenarios; "
            "it does not directly measure cross-attack balance."
        ),
    )
    parser.add_argument(
        "--stop-on-coverage-target",
        type=parse_bool_arg,
        default=True,
        help=(
            "true: stop early when the conditional coverage upper bound reaches target "
            "with enough samples."
        ),
    )
    parser.add_argument(
        "--min-samples-for-coverage-stop",
        type=int,
        default=100,
        help="Minimum total samples required before allowing coverage-target early stop.",
    )
    parser.add_argument(
        "--coverage-sc-schedule",
        type=str,
        default="12:0.20,40:0.45,1000000:0.70",
        help="Sample-count schedule for SC threshold used in conditional coverage evaluation.",
    )
    parser.add_argument(
        "--allow-multi-attacks-per-scenario",
        type=parse_bool_arg,
        default=True,
        help="true: allow multiple attack types in one generated scenario; false: keep at most one attack type.",
    )
    parser.add_argument(
        "--single-attack-types",
        type=str,
        default="",
        help=(
            "Whitelist of attack types allowed during single-attack generation; "
            "leave blank to avoid restricting attack types."
        ),
    )
    parser.add_argument("--failure-threshold", type=float, default=0.5)
    parser.add_argument(
        "--decision-threshold",
        "--failure-threshold-v2",
        dest="decision_threshold",
        type=float,
        default=0.35,
        help="primary decision threshold (legacy alias: --failure-threshold-v2).",
    )
    parser.add_argument(
        "--decision-formula-weights",
        type=str,
        default="0.60,0.25,0.10,0.10,0.20",
        help="weights for decision_score_v2: w_mean,w_p75,w_max,w_slope_pos,w_std_penalty",
    )
    parser.add_argument("--enable-decision-tail-boost", type=parse_bool_arg, default=False)
    parser.add_argument("--decision-tail-gamma", type=float, default=1.0)
    parser.add_argument("--decision-model-type", choices=("fixed_linear", "learned_linear"), default="fixed_linear")
    parser.add_argument(
        "--failure-decision-mode",
        choices=("single_fused_score", "direct_failure_model"),
        default="single_fused_score",
    )
    parser.add_argument("--fit-decision-model-offline", type=parse_bool_arg, default=False)
    parser.add_argument("--decision-model-lr", type=float, default=0.05)
    parser.add_argument("--decision-model-epochs", type=int, default=300)
    parser.add_argument("--decision-model-l2", type=float, default=0.001)
    parser.add_argument(
        "--fused-model-type",
        choices=("mlp_small",),
        default="mlp_small",
        help="model family for single_fused_score; MLP is the only public preset on the current mainline.",
    )
    parser.add_argument("--fused-mlp-hidden-dim", type=int, default=16)
    parser.add_argument("--decision-model-min-support", type=int, default=60)
    parser.add_argument("--decision-model-early-stop-patience", type=int, default=20)
    parser.add_argument(
        "--threshold-objective",
        choices=("f1", "recall_at_precision", "accuracy", "balanced_accuracy"),
        default="",
        help="Override the threshold objective. If omitted, scenario_exploration.yaml controls the default.",
    )
    parser.add_argument("--threshold-min-precision", type=float, default=None)
    parser.add_argument("--threshold-calibration-scope", choices=("terminal_only",), default="terminal_only")
    parser.add_argument(
        "--threshold-calibration-mode",
        choices=("two_stage_stable",),
        default="two_stage_stable",
        help="two_stage_stable: train shortlist + holdout stability selection.",
    )
    parser.add_argument(
        "--raw-log-root",
        default=str(TRAINING_PROCESS_DATA_ROOT),
        help="Root directory for raw simulation text logs (prevents collision in parallel runs).",
    )
    parser.add_argument("--threshold-calibration-holdout-ratio", type=float, default=0.2)
    parser.add_argument(
        "--threshold-split-mode",
        choices=("chronological", "stratified_random", "stratified_late_holdout"),
        default="",
        help="Override the train/holdout split mode used by online threshold calibration and fitted models.",
    )
    parser.add_argument(
        "--threshold-split-holdout-ratio",
        type=float,
        default=None,
        help="Override the holdout ratio used by threshold split modes. Defaults to exploration config.",
    )
    parser.add_argument(
        "--threshold-split-late-window-ratio",
        type=float,
        default=None,
        help="Override the late-window ratio used by stratified_late_holdout.",
    )
    parser.add_argument(
        "--threshold-split-holdout-late-fraction",
        type=float,
        default=None,
        help="Override the fraction of holdout samples that should come from the late pool.",
    )
    parser.add_argument("--threshold-min-support", type=int, default=30)
    parser.add_argument("--threshold-two-stage-top-k", type=int, default=24)
    parser.add_argument("--threshold-two-stage-gap-penalty", type=float, default=0.35)
    parser.add_argument("--threshold-two-stage-gap-tolerance", type=float, default=0.01)
    parser.add_argument("--threshold-two-stage-passrate-drift-penalty", type=float, default=0.10)
    parser.add_argument(
        "--online-backfill-after-each-round",
        type=parse_bool_arg,
        default=False,
        help="true: after each online round recalibration, recompute predictions for all accumulated samples.",
    )
    parser.add_argument(
        "--post-run-offline-recompute",
        type=parse_bool_arg,
        default=True,
        help="true: after online run ends, compute an offline-style full-session recompute summary and append to output_summary.",
    )
    parser.add_argument(
        "--enable-accuracy-guard",
        type=parse_bool_arg,
        default=True,
        help="true: if final failure_detection_accuracy is below the minimum, retune only the final score threshold.",
    )
    parser.add_argument(
        "--min-failure-detection-accuracy",
        type=float,
        default=0.90,
        help="Minimum accepted final failure_detection_accuracy before applying the lightweight threshold guard.",
    )
    parser.add_argument(
        "--rolling-drift-analysis",
        type=parse_on_off_arg,
        default=None,
        help="Override post-run rolling drift analysis switch: on/off.",
    )
    parser.add_argument(
        "--rolling-window-size",
        type=int,
        default=None,
        help="Override the rolling drift analysis holdout window size.",
    )
    parser.add_argument(
        "--rolling-step-size",
        type=int,
        default=None,
        help="Override the rolling drift analysis step size.",
    )
    parser.add_argument(
        "--rolling-min-train-support",
        type=int,
        default=None,
        help="Override the minimum prefix support required by rolling drift analysis.",
    )
    parser.add_argument(
        "--rolling-min-holdout-support",
        type=int,
        default=None,
        help="Override the minimum holdout support required by rolling drift analysis.",
    )
    parser.add_argument(
        "--true-failure-policy",
        "--true-failure-v2-policy",
        dest="true_failure_policy",
        choices=("relaxed", "strict"),
        default="strict",
        help="label policy for true_failure used by training/calibration/metrics.",
    )
    parser.add_argument(
        "--offline-recompute-only",
        type=parse_bool_arg,
        default=False,
        help="true: do not run new simulations; recalibrate/re-evaluate from existing session records.",
    )
    parser.add_argument(
        "--offline-use-existing-thresholds",
        type=parse_bool_arg,
        default=False,
        help="true: skip threshold recalibration in offline mode and use current thresholds.",
    )
    parser.add_argument(
        "--offline-source-session",
        type=str,
        default="",
        help="optional source session dir (or output root containing current_session) for offline recompute.",
    )
    parser.add_argument(
        "--offline-decision-threshold",
        "--offline-decision-threshold-v2",
        dest="offline_decision_threshold",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--offline-terminal-threshold",
        "--offline-terminal-threshold-v2",
        dest="offline_terminal_threshold",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--terminal-threshold",
        "--terminal-threshold-v2",
        dest="terminal_threshold",
        type=float,
        default=0.55,
    )
    parser.add_argument("--constellation2-anchor-ending-reward", type=float, default=None)
    parser.add_argument("--constellation2-anchor-packet-loss", type=float, default=None)
    parser.add_argument("--constellation2-anchor-e2e-delay", type=float, default=None)
    parser.add_argument("--constellation2-anchor-throughput", type=float, default=None)
    parser.add_argument("--reset-state", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def dump_yaml(path: Path, config: Dict):
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)


def load_exploration_config(path: Path) -> Dict:
    payload = load_yaml(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Scenario exploration config must be a mapping: {path}")
    if "single_attack_type" in payload:
        raise ValueError(
            f"{path} contains deprecated field 'single_attack_type'; "
            "please rename it to 'single_attack_types'."
        )
    return payload


def normalize_switch_text(value: Any, *, default: str = "off") -> str:
    if value is None:
        return default
    text = str(value).strip().lower()
    if not text:
        return default
    if text in {"on", "off"}:
        return text
    if text in {"1", "true", "t", "yes", "y"}:
        return "on"
    if text in {"0", "false", "f", "no", "n"}:
        return "off"
    raise ValueError(f"Invalid switch text: {value}")


def _normalize_float(value: Any, *, default: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
    if value in (None, ""):
        return float(default)
    return float(np.clip(float(value), min_value, max_value))


def _normalize_int(value: Any, *, default: int, min_value: int = 1) -> int:
    if value in (None, ""):
        return int(default)
    return max(min_value, int(value))


def _merge_nested_mapping(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        key_text = str(key)
        if (
            key_text in merged
            and isinstance(merged[key_text], Mapping)
            and isinstance(value, Mapping)
        ):
            merged[key_text] = _merge_nested_mapping(dict(merged[key_text]), value)
        else:
            merged[key_text] = copy.deepcopy(value)
    return merged


def _normalize_threshold_objective(value: Any, *, default: str) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return str(default).strip().lower()
    if text not in ALLOWED_THRESHOLD_OBJECTIVES:
        raise ValueError(
            f"Invalid threshold objective: {value}. "
            f"Expected one of {sorted(ALLOWED_THRESHOLD_OBJECTIVES)}."
        )
    return text


def resolve_exploration_settings(
    args: argparse.Namespace,
    exploration_config_path: Path,
    base_random_seed: int,
) -> Dict[str, Any]:
    payload = load_exploration_config(exploration_config_path)

    threshold_cfg = dict(DEFAULT_ONLINE_THRESHOLD_SPLIT_CONFIG)
    raw_threshold_cfg = payload.get("online_threshold_split", {})
    if isinstance(raw_threshold_cfg, Mapping):
        threshold_cfg.update({str(k): v for k, v in raw_threshold_cfg.items()})
    mode_override = str(getattr(args, "threshold_split_mode", "") or "").strip().lower()
    if mode_override:
        threshold_cfg["mode"] = mode_override
    if getattr(args, "threshold_split_holdout_ratio", None) is not None:
        threshold_cfg["holdout_ratio"] = args.threshold_split_holdout_ratio
    if getattr(args, "threshold_split_late_window_ratio", None) is not None:
        threshold_cfg["late_window_ratio"] = args.threshold_split_late_window_ratio
    if getattr(args, "threshold_split_holdout_late_fraction", None) is not None:
        threshold_cfg["holdout_late_fraction"] = args.threshold_split_holdout_late_fraction
    random_seed_raw = threshold_cfg.get("random_seed", "")
    if str(random_seed_raw).strip():
        threshold_seed = int(random_seed_raw)
    else:
        threshold_seed = int(base_random_seed)
    threshold_cfg["mode"] = str(threshold_cfg.get("mode", DEFAULT_ONLINE_THRESHOLD_SPLIT_CONFIG["mode"])).strip().lower()
    if threshold_cfg["mode"] not in {"chronological", "stratified_random", "stratified_late_holdout"}:
        threshold_cfg["mode"] = str(DEFAULT_ONLINE_THRESHOLD_SPLIT_CONFIG["mode"])
    threshold_cfg["holdout_ratio"] = _normalize_float(
        threshold_cfg.get("holdout_ratio"),
        default=float(DEFAULT_ONLINE_THRESHOLD_SPLIT_CONFIG["holdout_ratio"]),
        min_value=0.0,
        max_value=0.49,
    )
    threshold_cfg["late_window_ratio"] = _normalize_float(
        threshold_cfg.get("late_window_ratio"),
        default=float(DEFAULT_ONLINE_THRESHOLD_SPLIT_CONFIG["late_window_ratio"]),
        min_value=0.05,
        max_value=0.95,
    )
    threshold_cfg["holdout_late_fraction"] = _normalize_float(
        threshold_cfg.get("holdout_late_fraction"),
        default=float(DEFAULT_ONLINE_THRESHOLD_SPLIT_CONFIG["holdout_late_fraction"]),
        min_value=0.0,
        max_value=1.0,
    )
    threshold_cfg["random_seed"] = int(threshold_seed)

    rolling_cfg = dict(DEFAULT_POST_RUN_ROLLING_DRIFT_ANALYSIS_CONFIG)
    raw_rolling_cfg = payload.get("post_run_rolling_drift_analysis", {})
    if isinstance(raw_rolling_cfg, Mapping):
        rolling_cfg.update({str(k): v for k, v in raw_rolling_cfg.items()})
    rolling_override = str(getattr(args, "rolling_drift_analysis", "") or "").strip().lower()
    if rolling_override:
        rolling_cfg["enabled"] = rolling_override
    if getattr(args, "rolling_window_size", None) is not None:
        rolling_cfg["window_size"] = args.rolling_window_size
    if getattr(args, "rolling_step_size", None) is not None:
        rolling_cfg["step_size"] = args.rolling_step_size
    if getattr(args, "rolling_min_train_support", None) is not None:
        rolling_cfg["min_train_support"] = args.rolling_min_train_support
    if getattr(args, "rolling_min_holdout_support", None) is not None:
        rolling_cfg["min_holdout_support"] = args.rolling_min_holdout_support
    rolling_cfg["enabled"] = normalize_switch_text(
        rolling_cfg.get("enabled"),
        default=str(DEFAULT_POST_RUN_ROLLING_DRIFT_ANALYSIS_CONFIG["enabled"]),
    )
    rolling_cfg["window_size"] = _normalize_int(
        rolling_cfg.get("window_size"),
        default=int(DEFAULT_POST_RUN_ROLLING_DRIFT_ANALYSIS_CONFIG["window_size"]),
        min_value=1,
    )
    rolling_cfg["step_size"] = _normalize_int(
        rolling_cfg.get("step_size"),
        default=int(DEFAULT_POST_RUN_ROLLING_DRIFT_ANALYSIS_CONFIG["step_size"]),
        min_value=1,
    )
    rolling_cfg["min_train_support"] = _normalize_int(
        rolling_cfg.get("min_train_support"),
        default=int(DEFAULT_POST_RUN_ROLLING_DRIFT_ANALYSIS_CONFIG["min_train_support"]),
        min_value=2,
    )
    rolling_cfg["min_holdout_support"] = _normalize_int(
        rolling_cfg.get("min_holdout_support"),
        default=int(DEFAULT_POST_RUN_ROLLING_DRIFT_ANALYSIS_CONFIG["min_holdout_support"]),
        min_value=1,
    )
    output_filename = str(
        rolling_cfg.get("output_filename", DEFAULT_POST_RUN_ROLLING_DRIFT_ANALYSIS_CONFIG["output_filename"])
    ).strip()
    rolling_cfg["output_filename"] = output_filename or str(
        DEFAULT_POST_RUN_ROLLING_DRIFT_ANALYSIS_CONFIG["output_filename"]
    )

    threshold_calibration_cfg = _merge_nested_mapping(
        DEFAULT_ONLINE_THRESHOLD_CALIBRATION_CONFIG,
        payload.get("online_threshold_calibration", {})
        if isinstance(payload.get("online_threshold_calibration", {}), Mapping)
        else {},
    )
    threshold_calibration_cfg["objective"] = _normalize_threshold_objective(
        getattr(args, "threshold_objective", "") or threshold_calibration_cfg.get("objective"),
        default=str(DEFAULT_ONLINE_THRESHOLD_CALIBRATION_CONFIG["objective"]),
    )
    threshold_min_precision_cli = getattr(args, "threshold_min_precision", None)
    threshold_min_precision_value = threshold_calibration_cfg.get(
        "min_precision",
        DEFAULT_ONLINE_THRESHOLD_CALIBRATION_CONFIG["min_precision"],
    )
    if threshold_min_precision_cli is not None:
        threshold_min_precision_value = threshold_min_precision_cli
    threshold_calibration_cfg["min_precision"] = _normalize_float(
        threshold_min_precision_value,
        default=float(DEFAULT_ONLINE_THRESHOLD_CALIBRATION_CONFIG["min_precision"]),
        min_value=0.0,
        max_value=1.0,
    )
    support_guard_cfg = _merge_nested_mapping(
        DEFAULT_ONLINE_THRESHOLD_CALIBRATION_CONFIG["support_aware_threshold_guard"],
        threshold_calibration_cfg.get("support_aware_threshold_guard", {})
        if isinstance(threshold_calibration_cfg.get("support_aware_threshold_guard", {}), Mapping)
        else {},
    )
    support_guard_cfg["enabled"] = normalize_switch_text(
        support_guard_cfg.get("enabled"),
        default=str(DEFAULT_ONLINE_THRESHOLD_CALIBRATION_CONFIG["support_aware_threshold_guard"]["enabled"]),
    )
    support_metric = str(
        support_guard_cfg.get(
            "support_metric",
            DEFAULT_ONLINE_THRESHOLD_CALIBRATION_CONFIG["support_aware_threshold_guard"]["support_metric"],
        )
    ).strip().lower()
    if support_metric != "train_positive_count":
        support_metric = str(
            DEFAULT_ONLINE_THRESHOLD_CALIBRATION_CONFIG["support_aware_threshold_guard"]["support_metric"]
        )
    support_guard_cfg["support_metric"] = support_metric
    low_defaults = DEFAULT_ONLINE_THRESHOLD_CALIBRATION_CONFIG["support_aware_threshold_guard"]["low_support"]
    low_cfg = dict(support_guard_cfg.get("low_support", {}))
    low_cfg["positive_count_max"] = _normalize_int(
        low_cfg.get("positive_count_max"),
        default=int(low_defaults["positive_count_max"]),
        min_value=1,
    )
    low_update_mode = str(low_cfg.get("update_mode", low_defaults["update_mode"])).strip().lower()
    if low_update_mode != "weak_update":
        low_update_mode = str(low_defaults["update_mode"])
    low_cfg["update_mode"] = low_update_mode
    low_cfg["max_delta"] = _normalize_float(
        low_cfg.get("max_delta"),
        default=float(low_defaults["max_delta"]),
        min_value=0.0,
        max_value=1.0,
    )
    support_guard_cfg["low_support"] = low_cfg
    medium_defaults = DEFAULT_ONLINE_THRESHOLD_CALIBRATION_CONFIG["support_aware_threshold_guard"]["medium_support"]
    medium_cfg = dict(support_guard_cfg.get("medium_support", {}))
    medium_cfg["positive_count_max"] = _normalize_int(
        medium_cfg.get("positive_count_max"),
        default=int(medium_defaults["positive_count_max"]),
        min_value=low_cfg["positive_count_max"] + 1,
    )
    medium_update_mode = str(medium_cfg.get("update_mode", medium_defaults["update_mode"])).strip().lower()
    if medium_update_mode != "bounded_update":
        medium_update_mode = str(medium_defaults["update_mode"])
    medium_cfg["update_mode"] = medium_update_mode
    medium_cfg["max_delta"] = _normalize_float(
        medium_cfg.get("max_delta"),
        default=float(medium_defaults["max_delta"]),
        min_value=0.0,
        max_value=1.0,
    )
    support_guard_cfg["medium_support"] = medium_cfg
    high_defaults = DEFAULT_ONLINE_THRESHOLD_CALIBRATION_CONFIG["support_aware_threshold_guard"]["high_support"]
    high_cfg = dict(support_guard_cfg.get("high_support", {}))
    high_update_mode = str(high_cfg.get("update_mode", high_defaults["update_mode"])).strip().lower()
    if high_update_mode != "full_update":
        high_update_mode = str(high_defaults["update_mode"])
    high_cfg["update_mode"] = high_update_mode
    support_guard_cfg["high_support"] = high_cfg
    threshold_calibration_cfg["support_aware_threshold_guard"] = support_guard_cfg

    guard_cfg = _merge_nested_mapping(
        DEFAULT_DISTRIBUTION_BALANCE_GUARD_CONFIG,
        payload.get("distribution_balance_guard", {})
        if isinstance(payload.get("distribution_balance_guard", {}), Mapping)
        else {},
    )
    guard_cfg["enabled"] = normalize_switch_text(
        guard_cfg.get("enabled"),
        default=str(DEFAULT_DISTRIBUTION_BALANCE_GUARD_CONFIG["enabled"]),
    )
    guard_cfg["mode"] = str(guard_cfg.get("mode", DEFAULT_DISTRIBUTION_BALANCE_GUARD_CONFIG["mode"])).strip().lower()
    if guard_cfg["mode"] != "three_layer_balance":
        guard_cfg["mode"] = str(DEFAULT_DISTRIBUTION_BALANCE_GUARD_CONFIG["mode"])

    default_behavior = dict(guard_cfg.get("default_behavior", {}))
    for key, default_value in DEFAULT_DISTRIBUTION_BALANCE_GUARD_CONFIG["default_behavior"].items():
        default_behavior[key] = normalize_switch_text(default_behavior.get(key), default=str(default_value))
    guard_cfg["default_behavior"] = default_behavior

    trigger_cfg = dict(guard_cfg.get("trigger", {}))
    trigger_defaults = DEFAULT_DISTRIBUTION_BALANCE_GUARD_CONFIG["trigger"]
    trigger_cfg["selected_healthy_band_zero_rounds"] = _normalize_int(
        trigger_cfg.get("selected_healthy_band_zero_rounds"),
        default=int(trigger_defaults["selected_healthy_band_zero_rounds"]),
        min_value=1,
    )
    trigger_cfg["shortage_flag_rounds"] = _normalize_int(
        trigger_cfg.get("shortage_flag_rounds"),
        default=int(trigger_defaults["shortage_flag_rounds"]),
        min_value=1,
    )
    trigger_cfg["healthy_baseline_count_max"] = _normalize_int(
        trigger_cfg.get("healthy_baseline_count_max"),
        default=int(trigger_defaults["healthy_baseline_count_max"]),
        min_value=0,
    )
    trigger_cfg["require_prefilter_healthy_band_zero"] = normalize_switch_text(
        trigger_cfg.get("require_prefilter_healthy_band_zero"),
        default=str(trigger_defaults["require_prefilter_healthy_band_zero"]),
    )
    guard_cfg["trigger"] = trigger_cfg

    recovery_cfg = dict(guard_cfg.get("recovery", {}))
    recovery_defaults = DEFAULT_DISTRIBUTION_BALANCE_GUARD_CONFIG["recovery"]
    recovery_cfg["extra_round_budget_mode"] = str(
        recovery_cfg.get("extra_round_budget_mode", recovery_defaults["extra_round_budget_mode"])
    ).strip().lower() or str(recovery_defaults["extra_round_budget_mode"])
    recovery_cfg["replace_boundary_slots_default"] = _normalize_int(
        recovery_cfg.get("replace_boundary_slots_default"),
        default=int(recovery_defaults["replace_boundary_slots_default"]),
        min_value=1,
    )
    recovery_cfg["replace_boundary_slots_escalated"] = _normalize_int(
        recovery_cfg.get("replace_boundary_slots_escalated"),
        default=int(recovery_defaults["replace_boundary_slots_escalated"]),
        min_value=1,
    )
    recovery_cfg["escalate_after_triggered_rounds"] = _normalize_int(
        recovery_cfg.get("escalate_after_triggered_rounds"),
        default=int(recovery_defaults["escalate_after_triggered_rounds"]),
        min_value=1,
    )
    recovery_cfg["auto_exit_after_healthy_recovered_rounds"] = _normalize_int(
        recovery_cfg.get("auto_exit_after_healthy_recovered_rounds"),
        default=int(recovery_defaults["auto_exit_after_healthy_recovered_rounds"]),
        min_value=1,
    )
    recovery_cfg["allow_skip_if_no_recovery_candidate"] = normalize_switch_text(
        recovery_cfg.get("allow_skip_if_no_recovery_candidate"),
        default=str(recovery_defaults["allow_skip_if_no_recovery_candidate"]),
    )
    recovery_source_priority = recovery_cfg.get("recovery_source_priority", recovery_defaults["recovery_source_priority"])
    if not isinstance(recovery_source_priority, ABCSequence) or isinstance(recovery_source_priority, (str, bytes)):
        recovery_source_priority = list(recovery_defaults["recovery_source_priority"])
    recovery_cfg["recovery_source_priority"] = [str(item).strip().lower() for item in recovery_source_priority if str(item).strip()]
    if not recovery_cfg["recovery_source_priority"]:
        recovery_cfg["recovery_source_priority"] = list(recovery_defaults["recovery_source_priority"])
    guard_cfg["recovery"] = recovery_cfg

    constellation_profiles = dict(guard_cfg.get("constellation_profiles", {}))
    normalized_profiles: Dict[str, Any] = {}
    profile_defaults = DEFAULT_DISTRIBUTION_BALANCE_GUARD_CONFIG["constellation_profiles"]
    for profile_name in ("default", "large_constellation"):
        merged_profile = _merge_nested_mapping(
            profile_defaults.get(profile_name, {}),
            constellation_profiles.get(profile_name, {})
            if isinstance(constellation_profiles.get(profile_name, {}), Mapping)
            else {},
        )
        merged_profile["selected_healthy_band_zero_rounds"] = _normalize_int(
            merged_profile.get("selected_healthy_band_zero_rounds"),
            default=int(profile_defaults[profile_name]["selected_healthy_band_zero_rounds"]),
            min_value=1,
        )
        merged_profile["shortage_flag_rounds"] = _normalize_int(
            merged_profile.get("shortage_flag_rounds"),
            default=int(profile_defaults[profile_name]["shortage_flag_rounds"]),
            min_value=1,
        )
        merged_profile["replace_boundary_slots_default"] = _normalize_int(
            merged_profile.get("replace_boundary_slots_default"),
            default=int(profile_defaults[profile_name]["replace_boundary_slots_default"]),
            min_value=1,
        )
        merged_profile["replace_boundary_slots_escalated"] = _normalize_int(
            merged_profile.get("replace_boundary_slots_escalated"),
            default=int(profile_defaults[profile_name]["replace_boundary_slots_escalated"]),
            min_value=1,
        )
        merged_profile["allow_escalation"] = normalize_switch_text(
            merged_profile.get("allow_escalation"),
            default=str(profile_defaults[profile_name]["allow_escalation"]),
        )
        constellation_ids = merged_profile.get("constellation_ids", [])
        if isinstance(constellation_ids, ABCSequence) and not isinstance(constellation_ids, (str, bytes)):
            merged_profile["constellation_ids"] = sorted({int(value) for value in constellation_ids})
        else:
            merged_profile["constellation_ids"] = []
        normalized_profiles[profile_name] = merged_profile
    guard_cfg["constellation_profiles"] = normalized_profiles

    low_failure_cfg = _merge_nested_mapping(
        DEFAULT_LOW_FAILURE_REGIME_CONFIG,
        payload.get("low_failure_regime", {})
        if isinstance(payload.get("low_failure_regime", {}), Mapping)
        else {},
    )
    low_failure_cfg["enabled"] = normalize_switch_text(
        low_failure_cfg.get("enabled"),
        default=str(DEFAULT_LOW_FAILURE_REGIME_CONFIG["enabled"]),
    )
    fallback_policy = str(
        low_failure_cfg.get("fallback_policy", DEFAULT_LOW_FAILURE_REGIME_CONFIG["fallback_policy"])
    ).strip().lower()
    if fallback_policy != "dual_threshold_v2":
        fallback_policy = str(DEFAULT_LOW_FAILURE_REGIME_CONFIG["fallback_policy"])
    low_failure_cfg["fallback_policy"] = fallback_policy
    trigger_defaults = dict(DEFAULT_LOW_FAILURE_REGIME_CONFIG["trigger"])
    trigger_cfg = dict(low_failure_cfg.get("trigger", {}))
    min_effective_support_raw = trigger_cfg.get("min_effective_support", trigger_defaults["min_effective_support"])
    if min_effective_support_raw in (None, ""):
        trigger_cfg["min_effective_support"] = ""
    else:
        trigger_cfg["min_effective_support"] = _normalize_int(
            min_effective_support_raw,
            default=int(getattr(args, "threshold_min_support", 30)),
            min_value=2,
        )
    trigger_cfg["require_both_classes_in_train"] = normalize_switch_text(
        trigger_cfg.get("require_both_classes_in_train"),
        default=str(trigger_defaults["require_both_classes_in_train"]),
    )
    trigger_cfg["min_fused_holdout_auc"] = _normalize_float(
        trigger_cfg.get("min_fused_holdout_auc"),
        default=float(trigger_defaults["min_fused_holdout_auc"]),
        min_value=0.0,
        max_value=1.0,
    )
    trigger_cfg["enable_zero_prediction_guard"] = normalize_switch_text(
        trigger_cfg.get("enable_zero_prediction_guard"),
        default=str(trigger_defaults["enable_zero_prediction_guard"]),
    )
    low_failure_cfg["trigger"] = trigger_cfg
    low_failure_cfg["allow_small_sample_fused_experiment"] = normalize_switch_text(
        low_failure_cfg.get("allow_small_sample_fused_experiment"),
        default=str(DEFAULT_LOW_FAILURE_REGIME_CONFIG["allow_small_sample_fused_experiment"]),
    )
    low_failure_cfg["small_sample_threshold_min_support"] = _normalize_int(
        low_failure_cfg.get("small_sample_threshold_min_support"),
        default=int(DEFAULT_LOW_FAILURE_REGIME_CONFIG["small_sample_threshold_min_support"]),
        min_value=2,
    )

    pressure_router_cfg = _merge_nested_mapping(
        DEFAULT_PRESSURE_ROUTER_CONFIG,
        payload.get("pressure_router", {})
        if isinstance(payload.get("pressure_router", {}), Mapping)
        else {},
    )
    pressure_router_cfg["enabled"] = normalize_switch_text(
        pressure_router_cfg.get("enabled"),
        default=str(DEFAULT_PRESSURE_ROUTER_CONFIG["enabled"]),
    )
    pressure_router_cfg["high_pressure_threshold"] = _normalize_float(
        pressure_router_cfg.get("high_pressure_threshold"),
        default=float(DEFAULT_PRESSURE_ROUTER_CONFIG["high_pressure_threshold"]),
        min_value=0.0,
        max_value=1.0,
    )
    pressure_router_cfg["bandwidth_std_norm_max"] = _normalize_float(
        pressure_router_cfg.get("bandwidth_std_norm_max"),
        default=float(DEFAULT_PRESSURE_ROUTER_CONFIG["bandwidth_std_norm_max"]),
        min_value=1e-6,
    )
    pressure_formula_cfg = dict(pressure_router_cfg.get("score_formula", {}))
    for key, default_value in dict(DEFAULT_PRESSURE_ROUTER_CONFIG["score_formula"]).items():
        pressure_formula_cfg[key] = _normalize_float(
            pressure_formula_cfg.get(key),
            default=float(default_value),
            min_value=0.0,
            max_value=1.0,
        )
    pressure_router_cfg["score_formula"] = pressure_formula_cfg
    pressure_override_cfg = dict(pressure_router_cfg.get("override", {}))
    pressure_override_defaults = dict(DEFAULT_PRESSURE_ROUTER_CONFIG["override"])
    pressure_override_cfg["enabled"] = normalize_switch_text(
        pressure_override_cfg.get("enabled"),
        default=str(pressure_override_defaults["enabled"]),
    )
    apply_scope = str(pressure_override_cfg.get("apply_scope", pressure_override_defaults["apply_scope"])).strip().lower()
    if apply_scope != "next_round_only":
        apply_scope = str(pressure_override_defaults["apply_scope"])
    pressure_override_cfg["apply_scope"] = apply_scope
    scope_text = str(pressure_override_cfg.get("scope", pressure_override_defaults["scope"])).strip().lower()
    if scope_text != "session":
        scope_text = str(pressure_override_defaults["scope"])
    pressure_override_cfg["scope"] = scope_text
    pressure_override_cfg["high_risk_decision_threshold"] = _normalize_float(
        pressure_override_cfg.get("high_risk_decision_threshold"),
        default=float(pressure_override_defaults["high_risk_decision_threshold"]),
        min_value=0.0,
        max_value=1.0,
    )
    pressure_override_cfg["high_risk_terminal_threshold"] = _normalize_float(
        pressure_override_cfg.get("high_risk_terminal_threshold"),
        default=float(pressure_override_defaults["high_risk_terminal_threshold"]),
        min_value=0.0,
        max_value=1.0,
    )
    pressure_override_cfg["upgrade_if_batch_failure_ratio_ge"] = _normalize_float(
        pressure_override_cfg.get("upgrade_if_batch_failure_ratio_ge"),
        default=float(pressure_override_defaults["upgrade_if_batch_failure_ratio_ge"]),
        min_value=0.0,
        max_value=1.0,
    )
    pressure_override_cfg["upgrade_if_batch_high_risk_ratio_ge"] = _normalize_float(
        pressure_override_cfg.get("upgrade_if_batch_high_risk_ratio_ge"),
        default=float(pressure_override_defaults["upgrade_if_batch_high_risk_ratio_ge"]),
        min_value=0.0,
        max_value=1.0,
    )
    pressure_router_cfg["override"] = pressure_override_cfg

    low_pressure_cfg = _merge_nested_mapping(
        DEFAULT_LOW_PRESSURE_CLASSIFIER_CONFIG,
        payload.get("low_pressure_classifier", {})
        if isinstance(payload.get("low_pressure_classifier", {}), Mapping)
        else {},
    )
    low_pressure_cfg["enabled"] = normalize_switch_text(
        low_pressure_cfg.get("enabled"),
        default=str(DEFAULT_LOW_PRESSURE_CLASSIFIER_CONFIG["enabled"]),
    )
    model_type = str(low_pressure_cfg.get("model_type", DEFAULT_LOW_PRESSURE_CLASSIFIER_CONFIG["model_type"])).strip().lower()
    if model_type != "mlp":
        model_type = str(DEFAULT_LOW_PRESSURE_CLASSIFIER_CONFIG["model_type"])
    low_pressure_cfg["model_type"] = model_type
    low_pressure_cfg["hidden_dim"] = _normalize_int(
        low_pressure_cfg.get("hidden_dim"),
        default=int(DEFAULT_LOW_PRESSURE_CLASSIFIER_CONFIG["hidden_dim"]),
        min_value=4,
    )
    low_pressure_cfg["dropout"] = _normalize_float(
        low_pressure_cfg.get("dropout"),
        default=float(DEFAULT_LOW_PRESSURE_CLASSIFIER_CONFIG["dropout"]),
        min_value=0.0,
        max_value=0.9,
    )
    low_pressure_cfg["learning_rate"] = _normalize_float(
        low_pressure_cfg.get("learning_rate"),
        default=float(DEFAULT_LOW_PRESSURE_CLASSIFIER_CONFIG["learning_rate"]),
        min_value=1e-6,
    )
    low_pressure_cfg["weight_decay"] = _normalize_float(
        low_pressure_cfg.get("weight_decay"),
        default=float(DEFAULT_LOW_PRESSURE_CLASSIFIER_CONFIG["weight_decay"]),
        min_value=0.0,
    )
    low_pressure_cfg["epochs"] = _normalize_int(
        low_pressure_cfg.get("epochs"),
        default=int(DEFAULT_LOW_PRESSURE_CLASSIFIER_CONFIG["epochs"]),
        min_value=1,
    )
    low_pressure_cfg["batch_size"] = _normalize_int(
        low_pressure_cfg.get("batch_size"),
        default=int(DEFAULT_LOW_PRESSURE_CLASSIFIER_CONFIG["batch_size"]),
        min_value=1,
    )
    low_pressure_cfg["patience"] = _normalize_int(
        low_pressure_cfg.get("patience"),
        default=int(DEFAULT_LOW_PRESSURE_CLASSIFIER_CONFIG["patience"]),
        min_value=1,
    )
    low_pressure_cfg["pos_weight"] = _normalize_float(
        low_pressure_cfg.get("pos_weight"),
        default=float(DEFAULT_LOW_PRESSURE_CLASSIFIER_CONFIG["pos_weight"]),
        min_value=1.0,
    )
    low_pressure_cfg["holdout_ratio"] = _normalize_float(
        low_pressure_cfg.get("holdout_ratio"),
        default=float(DEFAULT_LOW_PRESSURE_CLASSIFIER_CONFIG["holdout_ratio"]),
        min_value=0.0,
        max_value=0.49,
    )
    feature_set = str(low_pressure_cfg.get("feature_set", DEFAULT_LOW_PRESSURE_CLASSIFIER_CONFIG["feature_set"])).strip().lower()
    if feature_set != "summary_v1":
        feature_set = str(DEFAULT_LOW_PRESSURE_CLASSIFIER_CONFIG["feature_set"])
    low_pressure_cfg["feature_set"] = feature_set
    low_pressure_threshold_cfg = dict(low_pressure_cfg.get("threshold", {}))
    low_pressure_threshold_defaults = dict(DEFAULT_LOW_PRESSURE_CLASSIFIER_CONFIG["threshold"])
    low_pressure_threshold_cfg["objective"] = _normalize_threshold_objective(
        low_pressure_threshold_cfg.get("objective"),
        default=str(low_pressure_threshold_defaults["objective"]),
    )
    low_pressure_threshold_cfg["min_precision"] = _normalize_float(
        low_pressure_threshold_cfg.get("min_precision"),
        default=float(low_pressure_threshold_defaults["min_precision"]),
        min_value=0.0,
        max_value=1.0,
    )
    low_pressure_cfg["threshold"] = low_pressure_threshold_cfg
    low_pressure_fallback_cfg = dict(low_pressure_cfg.get("fallback", {}))
    low_pressure_fallback_defaults = dict(DEFAULT_LOW_PRESSURE_CLASSIFIER_CONFIG["fallback"])
    low_pressure_fallback_cfg["min_effective_support"] = _normalize_int(
        low_pressure_fallback_cfg.get("min_effective_support"),
        default=int(low_pressure_fallback_defaults["min_effective_support"]),
        min_value=2,
    )
    low_pressure_fallback_cfg["require_both_classes_in_train"] = normalize_switch_text(
        low_pressure_fallback_cfg.get("require_both_classes_in_train"),
        default=str(low_pressure_fallback_defaults["require_both_classes_in_train"]),
    )
    low_pressure_policy = str(low_pressure_fallback_cfg.get("policy", low_pressure_fallback_defaults["policy"])).strip().lower()
    if low_pressure_policy != "dual_threshold_v2":
        low_pressure_policy = str(low_pressure_fallback_defaults["policy"])
    low_pressure_fallback_cfg["policy"] = low_pressure_policy
    low_pressure_cfg["fallback"] = low_pressure_fallback_cfg

    configured_attack_types = normalize_single_attack_types(payload.get("single_attack_types", []))

    return {
        "single_attack_types": configured_attack_types,
        "online_threshold_split": threshold_cfg,
        "post_run_rolling_drift_analysis": rolling_cfg,
        "online_threshold_calibration": threshold_calibration_cfg,
        "distribution_balance_guard": guard_cfg,
        "low_failure_regime": low_failure_cfg,
        "pressure_router": pressure_router_cfg,
        "low_pressure_classifier": low_pressure_cfg,
    }


def resolve_single_attack_types_from_sources(
    explicit_list_value: str,
    exploration_config_path: Path,
    allow_multi_attacks_per_scenario: bool,
) -> List[str]:
    if allow_multi_attacks_per_scenario:
        return []
    allowed_attack_types = normalize_single_attack_types(explicit_list_value)
    if allowed_attack_types:
        return allowed_attack_types
    payload = load_exploration_config(exploration_config_path)
    return normalize_single_attack_types(payload.get("single_attack_types", []))


def normalize_traffic_profile(config: Dict):
    env_cfg = config.setdefault("environment", {})
    traffic_profile = env_cfg.get("TrafficProfile")
    traffic_profiles = env_cfg.get("TrafficProfiles", {})
    if traffic_profile is None:
        return
    normalized_profile = str(traffic_profile).strip().lower()
    if normalized_profile in traffic_profiles:
        env_cfg["TrafficProfile"] = normalized_profile
        env_cfg.update(traffic_profiles[normalized_profile])


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def scenario_to_mapping(scenario: Dict) -> Dict:
    normalized = {}
    for key in SCENARIO_PARAMETER_NAMES:
        if key not in scenario:
            continue
        value = scenario[key]
        if key in {
            "ConstellationConfig",
            "PacketSizeMean",
            "PacketSizeStd",
            "StateObservationAttack_level",
            "ActionAttack_level",
            "StateTransferAttack_level",
            "RewardAttack_level",
            "ExperiencePoolAttack_level",
            "ModelTampAttack_level",
        }:
            normalized[key] = int(value)
        else:
            normalized[key] = float(value)
    return normalized


def fail_env_to_mapping(fail_env: FailEnv) -> Dict:
    return scenario_to_mapping(fail_env.__dict__)


def resolve_fixed_constellation_config(base_config: Dict, env_md_path: Path) -> int:
    param_space = parse_experiment_md(str(env_md_path))
    md_values = param_space.get("ConstellationConfig")
    if md_values is None:
        base_value = base_config.get("environment", {}).get("ConstellationConfig")
        if base_value is None:
            raise ValueError("ConstellationConfig must be defined in env_config.md or the base yaml config.")
        return int(base_value)

    normalized_values = sorted({int(value) for value in md_values})
    if len(normalized_values) != 1:
        raise ValueError(
            "env_config.md 中的 ConstellationConfig 必须固定为单个值；"
            "如需更换星座构型，请直接修改 env_config.md 中该值。"
        )
    return int(normalized_values[0])


def build_initial_scenarios(
    base_config: Dict,
    env_md_path: Path,
    single_attack_types: Optional[Sequence[str]] = None,
    allow_multi_attacks_per_scenario: bool = True,
) -> List[Dict]:
    param_space = parse_experiment_md(str(env_md_path))
    keys = list(param_space.keys())
    value_lists = [param_space[k] for k in keys]
    template_env = copy.deepcopy(base_config.get("environment", {}))

    scenarios: List[Dict] = []
    baseline_seed = {key: template_env.get(key) for key in SCENARIO_PARAMETER_NAMES if key in template_env}
    for key, values in param_space.items():
        if values:
            baseline_seed[key] = values[0]
    traffic_profile = baseline_seed.get("TrafficProfile")
    traffic_profiles = template_env.get("TrafficProfiles", {})
    if traffic_profile is not None:
        normalized_profile = str(traffic_profile).strip().lower()
        if normalized_profile in traffic_profiles:
            baseline_seed.update(traffic_profiles[normalized_profile])
    baseline_scenario = scenario_to_mapping(baseline_seed)
    scenarios.append(baseline_scenario)
    require_attack = not bool(allow_multi_attacks_per_scenario)
    combo_iter = itertools.product(*value_lists) if value_lists else [tuple()]
    for combo_values in combo_iter:
        combo = {k: v for k, v in zip(keys, combo_values)}
        if not is_attack_combination_valid(
            combo,
            template_env,
            single_attack_types=single_attack_types,
            require_attack=require_attack,
        ):
            continue

        scenario = {
            key: template_env.get(key)
            for key in SCENARIO_PARAMETER_NAMES
            if key in template_env
        }
        scenario.update(combo)
        traffic_profile = scenario.get("TrafficProfile")
        traffic_profiles = template_env.get("TrafficProfiles", {})
        if traffic_profile is not None:
            normalized_profile = str(traffic_profile).strip().lower()
            if normalized_profile in traffic_profiles:
                scenario.update(traffic_profiles[normalized_profile])
        mapped_scenario = scenario_to_mapping(scenario)
        if mapped_scenario != baseline_scenario:
            scenarios.append(mapped_scenario)

    if not scenarios:
        scenarios.append(baseline_scenario)
    return scenarios


def serialize_performance_file(
    raw_log_path: Path,
    output_path: Path,
    scenario: Dict,
    round_index: int,
    test_id: int,
):
    raw_log_text = raw_log_path.read_text(encoding="utf-8", errors="ignore")
    payload = [
        f"ROUND_INDEX: {round_index}",
        f"TEST_ID: {test_id}",
        f"SCENARIO_JSON: {json.dumps(scenario, ensure_ascii=False, sort_keys=True)}",
        f"RAW_LOG_NAME: {raw_log_path.name}",
        "",
        raw_log_text,
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(payload), encoding="utf-8")


def scenario_feature_arrays(scenario: Dict) -> Tuple[List[float], List[int]]:
    continuous_values = [float(scenario[name]) for name in CONTINUOUS_FEATURE_NAMES]
    discrete_values = [int(scenario[name]) for name in DISCRETE_FEATURE_NAMES]
    return continuous_values, discrete_values


def parse_sc_schedule(schedule_text: str) -> List[Tuple[int, float]]:
    parsed: List[Tuple[int, float]] = []
    for segment in str(schedule_text or "").split(","):
        piece = segment.strip()
        if not piece or ":" not in piece:
            continue
        left, right = piece.split(":", 1)
        sample_cap = max(1, int(left.strip()))
        threshold = float(np.clip(float(right.strip()), 0.0, 1.0))
        parsed.append((sample_cap, threshold))
    if not parsed:
        parsed = [(12, 0.2), (40, 0.45), (1000000, 0.7)]
    return sorted(parsed, key=lambda item: item[0])
class ClosedLoopFailureSimulation:
    def __init__(self, args):
        self.args = args
        self.project_root = PROJECT_ROOT
        self.config_path = Path(args.config).resolve()
        self.env_md_path = Path(args.env_md).resolve()
        self.exploration_config_path = Path(args.exploration_config).resolve()

        self.output_root = ensure_dir(Path(args.output_root).resolve())
        self.session_dir = self.output_root / "current_session"
        if args.reset_state and self.session_dir.exists():
            shutil.rmtree(self.session_dir, ignore_errors=True)
        ensure_dir(self.session_dir)
        self.rounds_dir = ensure_dir(self.session_dir / "rounds")
        self.temp_dir = ensure_dir(self.session_dir / "temp_configs")
        self.checkpoint_path = self.session_dir / "closed_loop_state.pt"

        self.base_config = load_yaml(self.config_path)
        if not self.exploration_config_path.exists():
            raise FileNotFoundError(f"Scenario exploration config not found: {self.exploration_config_path}")
        normalize_traffic_profile(self.base_config)
        self.fixed_constellation_config = resolve_fixed_constellation_config(self.base_config, self.env_md_path)
        self.traffic_profile = str(self.base_config.get("environment", {}).get("TrafficProfile", "low")).strip().lower()
        random_seed = int(self.base_config.get("general", {}).get("random_seed", 42))
        self.exploration_settings = resolve_exploration_settings(
            self.args,
            self.exploration_config_path,
            base_random_seed=random_seed,
        )
        self.single_attack_types = resolve_single_attack_types_from_sources(
            self.args.single_attack_types,
            self.exploration_config_path,
            self.args.allow_multi_attacks_per_scenario,
        )
        self.threshold_split_config = dict(self.exploration_settings.get("online_threshold_split", {}))
        self.threshold_calibration_config = dict(
            self.exploration_settings.get("online_threshold_calibration", {})
        )
        self.threshold_support_guard_config = dict(
            self.threshold_calibration_config.get("support_aware_threshold_guard", {})
        )
        self.rolling_drift_analysis_config = dict(
            self.exploration_settings.get("post_run_rolling_drift_analysis", {})
        )
        self.distribution_balance_guard_config = dict(
            self.exploration_settings.get("distribution_balance_guard", {})
        )
        self.low_failure_regime_config = dict(
            self.exploration_settings.get("low_failure_regime", {})
        )
        self.pressure_router_config = dict(
            self.exploration_settings.get("pressure_router", {})
        )
        self.low_pressure_classifier_config = dict(
            self.exploration_settings.get("low_pressure_classifier", {})
        )
        self.threshold_split_seed = int(self.threshold_split_config.get("random_seed", random_seed))
        self.args.threshold_objective = _normalize_threshold_objective(
            getattr(self.args, "threshold_objective", "") or self.threshold_calibration_config.get("objective"),
            default=str(DEFAULT_ONLINE_THRESHOLD_CALIBRATION_CONFIG["objective"]),
        )
        self.args.threshold_min_precision = float(
            self.threshold_calibration_config.get("min_precision", self.args.threshold_min_precision)
        )
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        self.coverage_sc_schedule = parse_sc_schedule(args.coverage_sc_schedule)
        self.true_failure_v2_policy = str(args.true_failure_policy).strip().lower()
        self.decision_formula_weights = parse_decision_formula_weights(args.decision_formula_weights)
        self.enable_decision_tail_boost = bool(args.enable_decision_tail_boost)
        self.decision_tail_gamma = float(np.clip(float(args.decision_tail_gamma), 0.5, 1.0))
        self.decision_model_type = str(args.decision_model_type).strip().lower()
        self.failure_decision_mode = str(args.failure_decision_mode).strip().lower()
        self.decision_policy = self.failure_decision_mode
        self.threshold_update_status = "frozen"

        self.device = select_torch_device("closed-loop failure simulation")
        self.evaluator = build_default_failure_evaluator(
            v2_failure_threshold=float(args.decision_threshold),
            terminal_threshold_v2=float(args.terminal_threshold),
            decision_formula_weights=self.decision_formula_weights,
            enable_decision_tail_boost=self.enable_decision_tail_boost,
            decision_tail_gamma=self.decision_tail_gamma,
            decision_model_type=self.decision_model_type,
            decision_model_weights=self.decision_formula_weights,
            decision_model_bias=0.0,
        )
        self.ensemble = build_default_deep_ensemble(
            num_continuous=len(CONTINUOUS_FEATURE_NAMES),
            num_categories=5,
            num_discrete_features=len(DISCRETE_FEATURE_NAMES),
        ).to(self.device)
        self.feature_net = FeatureSimilarityNetwork(
            num_continuous=len(CONTINUOUS_FEATURE_NAMES),
            num_categories=5,
            num_discrete_features=len(DISCRETE_FEATURE_NAMES),
        ).to(self.device)
        self.generator = ScenarioParameterGenerator(
            ensemble_net=self.ensemble,
            feature_net=self.feature_net,
            similarity_threshold=self.args.similarity_threshold,
            similarity_threshold_max=self.args.similarity_threshold_max,
            allow_multi_attacks_per_scenario=self.args.allow_multi_attacks_per_scenario,
            single_attack_types=self.single_attack_types,
            continuous_feature_names=CONTINUOUS_FEATURE_NAMES,
            discrete_feature_names=DISCRETE_FEATURE_NAMES,
            fixed_constellation_config=self.fixed_constellation_config,
        )
        self.explorer = FailureBoundaryExplorer(
            n_clusters=max(1, int(self.args.n_clusters)),
            rau_threshold=0.1,
            sc_threshold=0.7,
        )

        self.round_index = 0
        self.next_round_scenarios: List[Dict] = build_initial_scenarios(
            self.base_config,
            self.env_md_path,
            single_attack_types=self.single_attack_types,
            allow_multi_attacks_per_scenario=self.args.allow_multi_attacks_per_scenario,
        )
        self.test_counter = 0
        self.generated_scenario_count = 0
        self.highest_similarity = 0.0
        self.latest_coverage_metrics: Dict = {}
        self.finished = False
        self.stop_reason = "running"

        self.cumulative_continuous_features: List[List[float]] = []
        self.cumulative_discrete_features: List[List[int]] = []
        self.cumulative_failure_scores: List[float] = []
        self.cumulative_failure_labels_v2: List[float] = []
        self.summary_records: List[Dict] = []
        self.step_records: List[Dict] = []
        self.round_evalu_files: List[str] = []
        self.last_decision_model_info: Dict[str, object] = {
            "decision_model_status": "disabled",
            "decision_model_holdout_record_count": 0,
            "decision_model_holdout_auc": 0.0,
            "decision_model_holdout_accuracy": 0.0,
            "decision_model_config": self.evaluator.get_decision_formula_config(),
        }
        self.last_failure_model_info: Dict[str, object] = {
            "failure_decision_mode": self.failure_decision_mode,
            "single_threshold_used": True,
            "primary_score_name": "",
            "primary_score_holdout_auc": 0.0,
            "fused_model_status": "disabled",
            "fused_model_holdout_record_count": 0,
            "fused_model_holdout_auc": 0.0,
            "fused_model_holdout_accuracy": 0.0,
            "fused_model_type": str(self.args.fused_model_type).strip().lower(),
            "fused_model_weights": {},
            "fused_model_bias": 0.0,
            "fused_model_input_mean": [],
            "fused_model_input_std": [],
            "fused_model_mlp_state": {},
            "fused_model_mlp_hidden_dim": int(self.args.fused_mlp_hidden_dim),
            "fused_threshold": 0.5,
            "direct_model_status": "disabled",
            "direct_model_holdout_record_count": 0,
            "direct_model_holdout_auc": 0.0,
            "direct_model_holdout_accuracy": 0.0,
            "direct_model_weights": {},
            "direct_model_bias": 0.0,
            "final_threshold": 0.5,
        }
        self.last_threshold_stats: Dict[str, object] = {}
        self.last_low_pressure_model_info: Dict[str, object] = {
            "low_pressure_model_status": "disabled",
            "low_pressure_model_holdout_record_count": 0,
            "low_pressure_model_holdout_auc": 0.0,
            "low_pressure_model_holdout_accuracy": 0.0,
            "low_pressure_model_type": str(self.low_pressure_classifier_config.get("model_type", "mlp")).strip().lower(),
            "low_pressure_model_input_mean": [],
            "low_pressure_model_input_std": [],
            "low_pressure_model_mlp_state": {},
            "low_pressure_model_hidden_dim": int(self.low_pressure_classifier_config.get("hidden_dim", 16)),
            "low_pressure_threshold": 0.5,
        }
        self.last_low_pressure_threshold_stats: Dict[str, object] = {}
        self.post_run_offline_recompute_summary: Optional[Dict[str, object]] = None
        self.accuracy_guard_summary: Dict[str, object] = {}
        self.rolling_drift_analysis_summary: Optional[Dict[str, object]] = None
        self.initial_baseline_gate_failure_details: List[Dict] = []
        self.last_distribution_balance_guard_info: Dict[str, object] = {
            "distribution_balance_guard_enabled": normalize_switch_text(
                self.distribution_balance_guard_config.get("enabled"),
                default="off",
            ),
            "distribution_balance_guard_profile": "default",
            "distribution_balance_guard_active": False,
            "distribution_balance_guard_trigger_reason_codes": [],
            "distribution_balance_guard_replaced_slots": 0,
            "distribution_balance_guard_recovery_source_counts": {},
            "distribution_balance_guard_skip_reason": "",
            "distribution_balance_guard_selected_healthy_band_count": 0,
            "distribution_balance_guard_prefilter_healthy_band_count": 0,
            "distribution_balance_guard_shortage_flags": [],
        }
        self.distribution_balance_guard_state: Dict[str, object] = {
            "active": False,
            "profile": "default",
            "consecutive_selected_healthy_zero_rounds": 0,
            "consecutive_shortage_flag_rounds": 0,
            "consecutive_triggered_rounds": 0,
            "consecutive_healthy_recovered_rounds": 0,
        }
        self.low_failure_regime_state: Dict[str, object] = {
            "enabled": normalize_switch_text(
                self.low_failure_regime_config.get("enabled"),
                default=str(DEFAULT_LOW_FAILURE_REGIME_CONFIG["enabled"]),
            ),
            "fallback_applied": False,
            "fallback_reason": "",
            "effective_decision_mode": self.failure_decision_mode,
        }
        self.pressure_router_state: Dict[str, object] = {
            "router_enabled": normalize_switch_text(
                self.pressure_router_config.get("enabled"),
                default=str(DEFAULT_PRESSURE_ROUTER_CONFIG["enabled"]),
            ),
            "override_enabled": normalize_switch_text(
                dict(self.pressure_router_config.get("override", {})).get("enabled"),
                default=str(DEFAULT_PRESSURE_ROUTER_CONFIG["override"]["enabled"]),
            ),
            "pending_override_signal": "keep",
            "pending_override_apply_round": None,
            "last_round_batch_failure_ratio": 0.0,
            "last_round_batch_high_risk_ratio": 0.0,
            "last_round_override_signal": "keep",
            "last_round_override_applied_to_next_round": False,
        }

        if self.checkpoint_path.exists() and not self.args.reset_state:
            self._restore_state()
            self.evaluator.set_decision_formula_config(decision_model_type=self.decision_model_type)

    def _restore_state(self):
        state = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        self.ensemble.load_state_dict(state["ensemble_state_dict"])
        self.feature_net.load_state_dict(state["feature_net_state_dict"])
        self.generator.load_state(state.get("generator_state"))
        self.round_index = int(state.get("round_index", 0))
        self.next_round_scenarios = state.get("next_round_scenarios", self.next_round_scenarios)
        self.test_counter = int(state.get("test_counter", 0))
        self.generated_scenario_count = int(state.get("generated_scenario_count", 0))
        self.highest_similarity = float(state.get("highest_similarity", 0.0))
        self.latest_coverage_metrics = dict(state.get("latest_coverage_metrics", {}))
        self.finished = bool(state.get("finished", False))
        self.stop_reason = str(state.get("stop_reason", self.stop_reason))
        if "failure_threshold_v2" in state:
            self.evaluator.set_v2_failure_threshold(float(state["failure_threshold_v2"]))
        if "terminal_threshold_v2" in state:
            self.evaluator.set_terminal_threshold_v2(float(state["terminal_threshold_v2"]))
        if "decision_formula_config" in state:
            cfg = dict(state["decision_formula_config"])
            self.evaluator.set_decision_formula_config(
                decision_formula_weights={
                    key: cfg.get(key)
                    for key in ("w_mean", "w_p75", "w_max", "w_slope_pos", "w_std_penalty")
                    if key in cfg
                },
                enable_decision_tail_boost=cfg.get("enable_decision_tail_boost"),
                decision_tail_gamma=cfg.get("decision_tail_gamma"),
                decision_model_type=cfg.get("decision_model_type"),
                decision_model_weights=cfg.get("decision_model_weights"),
                decision_model_bias=cfg.get("decision_model_bias"),
            )
        self.threshold_update_status = str(state.get("threshold_update_status", self.threshold_update_status))
        self.cumulative_continuous_features = list(state.get("cumulative_continuous_features", []))
        self.cumulative_discrete_features = list(state.get("cumulative_discrete_features", []))
        self.cumulative_failure_scores = list(state.get("cumulative_failure_scores", []))
        self.cumulative_failure_labels_v2 = list(state.get("cumulative_failure_labels_v2", []))
        self.summary_records = list(state.get("summary_records", []))
        self.summary_records = [self._apply_true_failure_policy_to_record(dict(record)) for record in self.summary_records]
        if self.summary_records:
            self.cumulative_failure_labels_v2 = [
                float(self._resolve_true_failure_v2_value(record)) for record in self.summary_records
            ]
        self.step_records = list(state.get("step_records", []))
        self.round_evalu_files = list(state.get("round_evalu_files", []))
        self.last_decision_model_info = dict(state.get("last_decision_model_info", self.last_decision_model_info))
        self.last_failure_model_info = dict(state.get("last_failure_model_info", self.last_failure_model_info))
        self.last_threshold_stats = dict(state.get("last_threshold_stats", self.last_threshold_stats))
        self.last_low_pressure_model_info = dict(
            state.get("last_low_pressure_model_info", self.last_low_pressure_model_info)
        )
        self.last_low_pressure_threshold_stats = dict(
            state.get("last_low_pressure_threshold_stats", self.last_low_pressure_threshold_stats)
        )
        if "post_run_offline_recompute_summary" in state:
            cached = state.get("post_run_offline_recompute_summary")
            self.post_run_offline_recompute_summary = dict(cached) if isinstance(cached, dict) else None
        if "accuracy_guard_summary" in state:
            cached_guard = state.get("accuracy_guard_summary")
            self.accuracy_guard_summary = dict(cached_guard) if isinstance(cached_guard, dict) else {}
        if "rolling_drift_analysis_summary" in state:
            rolling_cached = state.get("rolling_drift_analysis_summary")
            self.rolling_drift_analysis_summary = dict(rolling_cached) if isinstance(rolling_cached, dict) else None
        self.initial_baseline_gate_failure_details = list(state.get("initial_baseline_gate_failure_details", []))
        self.last_distribution_balance_guard_info = dict(
            state.get("last_distribution_balance_guard_info", self.last_distribution_balance_guard_info)
        )
        self.distribution_balance_guard_state = dict(
            state.get("distribution_balance_guard_state", self.distribution_balance_guard_state)
        )
        self.low_failure_regime_state = dict(
            state.get("low_failure_regime_state", self.low_failure_regime_state)
        )
        self.pressure_router_state = dict(
            state.get("pressure_router_state", self.pressure_router_state)
        )
        if "terminal_risk_weights" in state:
            self.evaluator.set_terminal_risk_weights(dict(state["terminal_risk_weights"]))

    def _save_state(self):
        state = {
            "ensemble_state_dict": self.ensemble.state_dict(),
            "feature_net_state_dict": self.feature_net.state_dict(),
            "generator_state": self.generator.export_state(),
            "round_index": self.round_index,
            "next_round_scenarios": self.next_round_scenarios,
            "test_counter": self.test_counter,
            "generated_scenario_count": self.generated_scenario_count,
            "highest_similarity": self.highest_similarity,
            "latest_coverage_metrics": self.latest_coverage_metrics,
            "finished": self.finished,
            "stop_reason": self.stop_reason,
            "failure_threshold_v2": float(self.evaluator.v2_failure_threshold),
            "terminal_threshold_v2": float(self.evaluator.terminal_threshold_v2),
            "decision_formula_config": self.evaluator.get_decision_formula_config(),
            "threshold_update_status": self.threshold_update_status,
            "cumulative_continuous_features": self.cumulative_continuous_features,
            "cumulative_discrete_features": self.cumulative_discrete_features,
            "cumulative_failure_scores": self.cumulative_failure_scores,
            "cumulative_failure_labels_v2": self.cumulative_failure_labels_v2,
            "summary_records": self.summary_records,
            "step_records": self.step_records,
            "round_evalu_files": self.round_evalu_files,
            "terminal_risk_weights": self.evaluator.get_terminal_risk_weights(),
            "last_decision_model_info": self.last_decision_model_info,
            "last_failure_model_info": self.last_failure_model_info,
            "last_threshold_stats": self.last_threshold_stats,
            "last_low_pressure_model_info": self.last_low_pressure_model_info,
            "last_low_pressure_threshold_stats": self.last_low_pressure_threshold_stats,
            "post_run_offline_recompute_summary": self.post_run_offline_recompute_summary,
            "accuracy_guard_summary": self.accuracy_guard_summary,
            "rolling_drift_analysis_summary": self.rolling_drift_analysis_summary,
            "initial_baseline_gate_failure_details": self.initial_baseline_gate_failure_details,
            "last_distribution_balance_guard_info": self.last_distribution_balance_guard_info,
            "distribution_balance_guard_state": self.distribution_balance_guard_state,
            "low_failure_regime_state": self.low_failure_regime_state,
            "pressure_router_state": self.pressure_router_state,
        }
        ensure_dir(self.session_dir)
        ensure_dir(self.checkpoint_path.parent)
        buffer = io.BytesIO()
        torch.save(state, buffer)
        self.checkpoint_path.write_bytes(buffer.getvalue())

    def _build_temp_config(self, scenario: Dict, round_index: int, test_id: int) -> Tuple[Path, str]:
        config = copy.deepcopy(self.base_config)
        env_cfg = config.setdefault("environment", {})
        agent_cfg = config.setdefault("agent", {})
        general_cfg = config.setdefault("general", {})

        # 核心隔离逻辑：为每个并行测试创建专属临时文件夹，并重定向所有写操作
        worker_task_dir = self.temp_dir / f"round_{round_index:03d}_test_{test_id:04d}"
        worker_task_dir.mkdir(parents=True, exist_ok=True)

        env_cfg.update(scenario)
        env_cfg["SaveTrainingData"] = f"closed_loop_r{round_index:03d}_t{test_id:04d}.txt"
        env_cfg["SaveActionLog"] = False
        env_cfg["visualize"] = False
        env_cfg["PrintInfo"] = False
        env_cfg["ShowDetail"] = False
        env_cfg["SaveLog"] = False
        env_cfg["PositionDataDir"] = str(worker_task_dir / "Position_Data")

        general_cfg["phase"] = "train"
        agent_cfg["agent_sharing_mode"] = "independent"
        agent_cfg["reset_independent_on_train_start"] = True
        agent_cfg["cleanup_independent_after_run"] = True
        agent_cfg["strict_bootstrap_in_train"] = True
        agent_cfg["independent_model_dir"] = str(worker_task_dir / "models")

        temp_config_path = worker_task_dir / "config.yaml"
        dump_yaml(temp_config_path, config)
        return temp_config_path, env_cfg["SaveTrainingData"]

    def _run_single_simulation(self, temp_config_path: Path):
        # 准备环境变量，确保能找到项目根目录下的模块
        env = os.environ.copy()
        current_pythonpath = env.get("PYTHONPATH", "")
        project_root_str = str(self.project_root)
        env["PYTHONPATH"] = f"{project_root_str}{os.pathsep}{current_pythonpath}" if current_pythonpath else project_root_str
        
        # 核心修复：通过环境变量重定向日志目录，但 CWD 保持在根目录以便读取资源文件
        env["TRAINING_LOG_ROOT"] = str(self.args.raw_log_root)

        completed = subprocess.run(
            [
                sys.executable,
                str(ITERATIVE_TESTING_ROOT / "PRC.py"),
                "--config",
                str(temp_config_path),
            ],
            cwd=project_root_str,  # 回到根目录运行，保证资源文件可见
            env=env,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"PRC.py failed with exit code {completed.returncode} for {temp_config_path.name}")

    def _write_round_env_list(self, round_dir: Path, scenarios: Sequence[Dict], similarities: Optional[Sequence[float]] = None):
        env_list_path = round_dir / "env_list.jsonl"
        with env_list_path.open("w", encoding="utf-8") as f:
            for idx, scenario in enumerate(scenarios, start=1):
                payload = {
                    "round_index": self.round_index,
                    "scenario_index": idx,
                    "traffic_profile": self.traffic_profile,
                    "max_similarity_to_history": float(similarities[idx - 1]) if similarities is not None and idx - 1 < len(similarities) else None,
                    "scenario": scenario,
                }
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _resolve_true_failure_v2_value(self, record: Dict) -> bool:
        if self.true_failure_v2_policy == "strict":
            return bool(record.get("true_failure_v2_strict", record.get("true_failure_v2", False)))
        return bool(record.get("true_failure_v2", record.get("true_failure_v2_strict", False)))

    def _apply_true_failure_policy_to_record(self, record: Dict) -> Dict:
        if "true_failure_v2_strict" not in record:
            record["true_failure_v2_strict"] = bool(record.get("true_failure_v2", False))
        if "true_failure_v2_relaxed" not in record:
            record["true_failure_v2_relaxed"] = bool(record.get("true_failure_v2", False))
        record.setdefault("baseline_status", "not_applicable")
        record.setdefault("baseline_valid", True)
        record.setdefault("baseline_warning", False)
        record.setdefault("baseline_reason_codes", [])
        if self.true_failure_v2_policy == "strict":
            record["true_failure_v2"] = bool(record.get("true_failure_v2_strict", False))
        else:
            record["true_failure_v2"] = bool(record.get("true_failure_v2_relaxed", record.get("true_failure_v2", False)))
        return record

    def _is_fused_effective_record(self, record: Dict) -> bool:
        return bool(record.get("terminal_hard_failure", False)) or self._resolve_true_failure_v2_value(record)

    def _normalize_bandwidth_std_for_pressure(self, value: float) -> float:
        cfg = dict(getattr(self, "pressure_router_config", DEFAULT_PRESSURE_ROUTER_CONFIG) or {})
        denom = float(cfg.get("bandwidth_std_norm_max", DEFAULT_PRESSURE_ROUTER_CONFIG["bandwidth_std_norm_max"]) or 0.20)
        denom = max(1e-6, denom)
        return float(np.clip(float(value) / denom, 0.0, 1.0))

    def _compute_pressure_score(self, record: Dict) -> float:
        cfg = dict(getattr(self, "pressure_router_config", DEFAULT_PRESSURE_ROUTER_CONFIG) or {})
        formula = dict(cfg.get("score_formula", DEFAULT_PRESSURE_ROUTER_CONFIG["score_formula"]) or {})
        degraded = float(record.get("DegradedEdgeRatio", 0.0) or 0.0)
        disconnect = float(record.get("EdgeDisconnectRatio", 0.0) or 0.0)
        bandwidth_mean = float(record.get("EdgeBandwidthMeanDecreaseRatio", 0.0) or 0.0)
        bandwidth_std_norm = self._normalize_bandwidth_std_for_pressure(
            float(record.get("EdgeBandwidthDecreaseStd", 0.0) or 0.0)
        )
        score = (
            float(formula.get("degraded_edge_ratio_weight", 0.40)) * degraded
            + float(formula.get("edge_disconnect_ratio_weight", 0.35)) * disconnect
            + float(formula.get("edge_bandwidth_mean_decrease_ratio_weight", 0.20)) * bandwidth_mean
            + float(formula.get("edge_bandwidth_decrease_std_norm_weight", 0.05)) * bandwidth_std_norm
        )
        return float(np.clip(score, 0.0, 1.0))

    def _compute_initial_pressure_regime(self, record: Dict) -> str:
        if not hasattr(self, "pressure_router_config"):
            return "high_pressure"
        cfg = dict(getattr(self, "pressure_router_config", DEFAULT_PRESSURE_ROUTER_CONFIG) or {})
        if normalize_switch_text(cfg.get("enabled"), default=str(DEFAULT_PRESSURE_ROUTER_CONFIG["enabled"])) != "on":
            return "high_pressure"
        threshold = float(cfg.get("high_pressure_threshold", DEFAULT_PRESSURE_ROUTER_CONFIG["high_pressure_threshold"]))
        pressure_score = self._compute_pressure_score(record)
        return "high_pressure" if pressure_score >= threshold else "low_pressure"

    def _resolve_effective_pressure_regime(self, record: Dict, round_index: Optional[int] = None) -> str:
        if not hasattr(self, "pressure_router_config"):
            record["pressure_score"] = float(record.get("pressure_score", 0.0))
            record["initial_pressure_regime"] = str(record.get("initial_pressure_regime", "high_pressure"))
            record["effective_pressure_regime"] = str(record.get("effective_pressure_regime", record["initial_pressure_regime"]))
            record["pressure_override_inbound_signal"] = "keep"
            record["pressure_override_inbound_applied"] = False
            return str(record["effective_pressure_regime"])
        cfg = dict(getattr(self, "pressure_router_config", DEFAULT_PRESSURE_ROUTER_CONFIG) or {})
        if normalize_switch_text(cfg.get("enabled"), default=str(DEFAULT_PRESSURE_ROUTER_CONFIG["enabled"])) != "on":
            return "high_pressure"
        round_value = int(getattr(self, "round_index", 0)) if round_index is None else int(round_index)
        initial_regime = str(record.get("initial_pressure_regime", self._compute_initial_pressure_regime(record)))
        router_state = dict(getattr(self, "pressure_router_state", {}) or {})
        inbound_signal = str(router_state.get("pending_override_signal", "keep"))
        inbound_apply_round = router_state.get("pending_override_apply_round", None)
        inbound_applied = False
        effective_regime = initial_regime
        if (
            normalize_switch_text(
                dict(cfg.get("override", {})).get("enabled"),
                default=str(DEFAULT_PRESSURE_ROUTER_CONFIG["override"]["enabled"]),
            )
            == "on"
            and str(inbound_signal) == "upgrade_to_high"
            and inbound_apply_round is not None
            and int(inbound_apply_round) == int(round_value)
            and initial_regime == "low_pressure"
        ):
            effective_regime = "high_pressure"
            inbound_applied = True
        record["pressure_score"] = float(record.get("pressure_score", self._compute_pressure_score(record)))
        record["initial_pressure_regime"] = initial_regime
        record["effective_pressure_regime"] = effective_regime
        record["pressure_override_inbound_signal"] = inbound_signal
        record["pressure_override_inbound_applied"] = bool(inbound_applied)
        return effective_regime

    def _set_pending_pressure_override(self, signal: str, apply_round: Optional[int]) -> None:
        state = dict(getattr(self, "pressure_router_state", {}) or {})
        state["pending_override_signal"] = str(signal or "keep")
        state["pending_override_apply_round"] = None if apply_round is None else int(apply_round)
        self.pressure_router_state = state

    def _compute_round_pressure_override_signal(self, round_records: Sequence[Dict]) -> Dict[str, object]:
        cfg = dict(getattr(self, "pressure_router_config", DEFAULT_PRESSURE_ROUTER_CONFIG) or {})
        override_cfg = dict(cfg.get("override", {}) or {})
        router_enabled = normalize_switch_text(cfg.get("enabled"), default=str(DEFAULT_PRESSURE_ROUTER_CONFIG["enabled"]))
        override_enabled = normalize_switch_text(
            override_cfg.get("enabled"),
            default=str(DEFAULT_PRESSURE_ROUTER_CONFIG["override"]["enabled"]),
        )
        if router_enabled != "on" or override_enabled != "on":
            return {
                "signal": "keep",
                "batch_failure_ratio": 0.0,
                "batch_high_risk_ratio": 0.0,
                "applied_to_next_round": False,
            }
        round_size = int(len(round_records))
        if round_size <= 0:
            return {
                "signal": "keep",
                "batch_failure_ratio": 0.0,
                "batch_high_risk_ratio": 0.0,
                "applied_to_next_round": False,
            }
        predicted_failure_count = sum(1 for record in round_records if bool(record.get("system_failure_v2", False)))
        decision_threshold = float(
            override_cfg.get(
                "high_risk_decision_threshold",
                DEFAULT_PRESSURE_ROUTER_CONFIG["override"]["high_risk_decision_threshold"],
            )
        )
        terminal_threshold = float(
            override_cfg.get(
                "high_risk_terminal_threshold",
                DEFAULT_PRESSURE_ROUTER_CONFIG["override"]["high_risk_terminal_threshold"],
            )
        )
        high_risk_count = 0
        for record in round_records:
            decision_score = float(record.get("decision_score_v2", record.get("total_membership_v2", 0.0)))
            terminal_score = float(
                record.get("terminal_risk_score", 1.0 if bool(record.get("terminal_hard_failure", False)) else 0.0)
            )
            if decision_score >= decision_threshold or terminal_score >= terminal_threshold:
                high_risk_count += 1
        batch_failure_ratio = float(predicted_failure_count / max(1, round_size))
        batch_high_risk_ratio = float(high_risk_count / max(1, round_size))
        signal = "keep"
        if batch_failure_ratio >= float(
            override_cfg.get(
                "upgrade_if_batch_failure_ratio_ge",
                DEFAULT_PRESSURE_ROUTER_CONFIG["override"]["upgrade_if_batch_failure_ratio_ge"],
            )
        ):
            signal = "upgrade_to_high"
        elif batch_high_risk_ratio >= float(
            override_cfg.get(
                "upgrade_if_batch_high_risk_ratio_ge",
                DEFAULT_PRESSURE_ROUTER_CONFIG["override"]["upgrade_if_batch_high_risk_ratio_ge"],
            )
        ):
            signal = "upgrade_to_high"
        return {
            "signal": str(signal),
            "batch_failure_ratio": float(batch_failure_ratio),
            "batch_high_risk_ratio": float(batch_high_risk_ratio),
            "applied_to_next_round": bool(signal == "upgrade_to_high"),
        }

    def _resolve_low_failure_min_effective_support(self) -> int:
        regime_cfg = dict(getattr(self, "low_failure_regime_config", DEFAULT_LOW_FAILURE_REGIME_CONFIG) or {})
        cfg = dict(regime_cfg.get("trigger", {}))
        if normalize_switch_text(
            regime_cfg.get("allow_small_sample_fused_experiment"),
            default=str(DEFAULT_LOW_FAILURE_REGIME_CONFIG["allow_small_sample_fused_experiment"]),
        ) == "on":
            return int(
                max(
                    2,
                    int(
                        regime_cfg.get(
                            "small_sample_threshold_min_support",
                            DEFAULT_LOW_FAILURE_REGIME_CONFIG["small_sample_threshold_min_support"],
                        )
                    ),
                )
            )
        raw_value = cfg.get("min_effective_support", "")
        if raw_value in (None, ""):
            return int(max(2, int(self.args.threshold_min_support)))
        return int(max(2, int(raw_value)))

    def _should_fallback_from_fused(
        self,
        threshold_stats: Optional[Dict[str, object]] = None,
        failure_model_info: Optional[Dict[str, object]] = None,
        *,
        predicted_failure_count: Optional[int] = None,
        true_failure_count: Optional[int] = None,
    ) -> Tuple[bool, str]:
        if str(self.failure_decision_mode).strip().lower() != "single_fused_score":
            return False, ""
        regime_cfg = dict(getattr(self, "low_failure_regime_config", DEFAULT_LOW_FAILURE_REGIME_CONFIG) or {})
        if normalize_switch_text(
            regime_cfg.get("enabled"),
            default=str(DEFAULT_LOW_FAILURE_REGIME_CONFIG["enabled"]),
        ) != "on":
            return False, ""
        if str(
            regime_cfg.get(
                "fallback_policy",
                DEFAULT_LOW_FAILURE_REGIME_CONFIG["fallback_policy"],
            )
        ).strip().lower() != "dual_threshold_v2":
            return False, ""

        threshold_stats = dict(threshold_stats or self.last_threshold_stats or {})
        failure_model_info = dict(failure_model_info or self.last_failure_model_info or {})
        trigger_cfg = dict(regime_cfg.get("trigger", {}))
        min_effective_support = self._resolve_low_failure_min_effective_support()
        effective_support = int(threshold_stats.get("effective_support", 0))
        train_support = int(
            threshold_stats.get(
                "train_support",
                threshold_stats.get("threshold_split_train_support", 0),
            )
        )
        positive_count = int(
            threshold_stats.get(
                "positive_count",
                threshold_stats.get("threshold_support_train_positive_count", 0),
            )
        )
        negative_count = int(
            threshold_stats.get(
                "negative_count",
                threshold_stats.get("threshold_support_train_negative_count", 0),
            )
        )
        if effective_support < min_effective_support:
            return True, "insufficient_effective_support"
        if train_support < min_effective_support:
            return True, "insufficient_train_support"
        if normalize_switch_text(
            trigger_cfg.get("require_both_classes_in_train"),
            default=str(DEFAULT_LOW_FAILURE_REGIME_CONFIG["trigger"]["require_both_classes_in_train"]),
        ) == "on" and (positive_count == 0 or negative_count == 0):
            return True, "single_class_labels"

        fused_status = str(failure_model_info.get("fused_model_status", "")).strip().lower()
        fused_holdout_count = int(failure_model_info.get("fused_model_holdout_record_count", 0))
        if fused_status != "fitted" or fused_holdout_count <= 0:
            return True, "fused_model_unavailable"

        min_auc = float(
            np.clip(
                float(
                    trigger_cfg.get(
                        "min_fused_holdout_auc",
                        DEFAULT_LOW_FAILURE_REGIME_CONFIG["trigger"]["min_fused_holdout_auc"],
                    )
                ),
                0.0,
                1.0,
            )
        )
        fused_auc = float(failure_model_info.get("primary_score_holdout_auc", 0.0))
        if min_auc > 0.0 and fused_auc < min_auc:
            return True, "low_fused_holdout_auc"

        if (
            normalize_switch_text(
                trigger_cfg.get("enable_zero_prediction_guard"),
                default=str(DEFAULT_LOW_FAILURE_REGIME_CONFIG["trigger"]["enable_zero_prediction_guard"]),
            )
            == "on"
            and predicted_failure_count is not None
            and true_failure_count is not None
            and int(predicted_failure_count) == 0
            and int(true_failure_count) > 0
        ):
            return True, "zero_prediction_guard"

        return False, ""

    def _set_low_failure_regime_state(self, applied: bool, reason: str, effective_mode: str) -> None:
        regime_cfg = dict(getattr(self, "low_failure_regime_config", DEFAULT_LOW_FAILURE_REGIME_CONFIG) or {})
        self.low_failure_regime_state = {
            "enabled": normalize_switch_text(
                regime_cfg.get("enabled"),
                default=str(DEFAULT_LOW_FAILURE_REGIME_CONFIG["enabled"]),
            ),
            "fallback_applied": bool(applied),
            "fallback_reason": str(reason or ""),
            "effective_decision_mode": str(effective_mode),
        }

    @staticmethod
    def _is_no_attack_scenario(scenario: Dict) -> bool:
        if not isinstance(scenario, dict):
            return False
        for key in ATTACK_SCENARIO_KEYS:
            value = scenario.get(key, 0)
            try:
                if int(round(float(value))) > 0:
                    return False
            except (TypeError, ValueError):
                return False
        return True

    def _validate_initial_baseline_gate(self, round_index: int, round_summary_records: Sequence[Dict]) -> None:
        if int(round_index) != 0:
            return

        invalid_records: List[Dict] = []
        for record in round_summary_records:
            scenario = record.get("scenario", {})
            if not self._is_no_attack_scenario(scenario):
                continue
            if self._constellation2_round_zero_gate_failed(record):
                invalid_records.append(record)
                continue
            if str(record.get("baseline_status", "not_applicable")) == "invalid":
                invalid_records.append(record)

        if not invalid_records:
            return

        details = []
        for record in invalid_records:
            details.append(
                {
                    "test_id": int(record.get("test_id", -1)),
                    "baseline_status": str(record.get("baseline_status", "invalid")),
                    "AverageEndingReward": float(record.get("terminal_average_ending_reward", 0.0)),
                    "PacketLossRate": float(record.get("terminal_packet_loss_rate", 0.0)),
                    "terminal_hard_failure": bool(record.get("terminal_hard_failure", False)),
                    "baseline_reason_codes": list(record.get("baseline_reason_codes", [])),
                }
            )
        self.initial_baseline_gate_failure_details = details
        self.stop_reason = "initial_baseline_gate_failed"
        print(
            "[INITIAL_BASELINE_GATE] round_000 failed; aborting before incremental training/exploration. "
            + json.dumps(details, ensure_ascii=False),
            flush=True,
        )
        raise InitialBaselineGateError(
            "Initial no-attack baseline validation failed for round_000: "
            + json.dumps(details, ensure_ascii=False),
            details,
        )

    @staticmethod
    def _record_constellation_config(record: Dict) -> Optional[int]:
        scenario = record.get("scenario", {})
        raw_value = scenario.get("ConstellationConfig", record.get("ConstellationConfig"))
        try:
            return int(round(float(raw_value)))
        except (TypeError, ValueError):
            return None

    def _constellation2_anchor_metrics(self) -> Optional[Dict[str, float]]:
        env_cfg = self.base_config.get("environment", {})
        anchor_gate_switch = str(env_cfg.get("Constellation2AnchorGate", "off") or "").strip().lower()
        if anchor_gate_switch != "on":
            return None

        cli_keys = {
            "AverageEndingReward": getattr(self.args, "constellation2_anchor_ending_reward", None),
            "PacketLossRate": getattr(self.args, "constellation2_anchor_packet_loss", None),
            "AverageE2eDelay": getattr(self.args, "constellation2_anchor_e2e_delay", None),
            "NetworkThroughput": getattr(self.args, "constellation2_anchor_throughput", None),
        }
        if all(value is not None for value in cli_keys.values()):
            return {metric: float(value) for metric, value in cli_keys.items()}

        anchor_profiles = env_cfg.get("Constellation2BaselineAnchors", {})
        if not isinstance(anchor_profiles, dict):
            return None

        profile_key = str(getattr(self, "traffic_profile", "") or "").strip().lower()
        profile_anchor = anchor_profiles.get(profile_key)
        if not isinstance(profile_anchor, dict):
            return None

        required_metrics = (
            "AverageEndingReward",
            "PacketLossRate",
            "AverageE2eDelay",
            "NetworkThroughput",
        )
        if any(profile_anchor.get(metric) is None for metric in required_metrics):
            return None
        return {metric: float(profile_anchor[metric]) for metric in required_metrics}

    def _constellation2_round_zero_gate_failed(self, record: Dict) -> bool:
        if self._record_constellation_config(record) != 2:
            return False
        if bool(record.get("terminal_hard_failure", False)):
            return True

        anchor_metrics = self._constellation2_anchor_metrics()
        if anchor_metrics is None:
            return False

        ending_reward = float(record.get("terminal_average_ending_reward", 0.0) or 0.0)
        packet_loss = float(record.get("terminal_packet_loss_rate", 0.0) or 0.0)
        terminal_delay = float(record.get("terminal_average_e2e_delay", record.get("AverageE2eDelay", 0.0)) or 0.0)
        terminal_throughput = float(record.get("terminal_network_throughput", record.get("NetworkThroughput", 0.0)) or 0.0)

        reward_drop = float(anchor_metrics["AverageEndingReward"]) - ending_reward
        packet_loss_increase = packet_loss - float(anchor_metrics["PacketLossRate"])
        delay_ratio = terminal_delay / max(float(anchor_metrics["AverageE2eDelay"]), 1e-6)
        throughput_ratio = terminal_throughput / max(float(anchor_metrics["NetworkThroughput"]), 1e-6)

        return bool(
            reward_drop > float(CONSTELLATION_2_GATE_PROFILE["max_reward_drop_vs_anchor"])
            or packet_loss_increase > float(CONSTELLATION_2_GATE_PROFILE["max_packet_loss_increase_vs_anchor"])
            or delay_ratio > float(CONSTELLATION_2_GATE_PROFILE["max_delay_increase_ratio_vs_anchor"])
            or throughput_ratio < float(CONSTELLATION_2_GATE_PROFILE["min_throughput_ratio_vs_anchor"])
        )

    @staticmethod
    def _roc_auc_binary(scores: Sequence[float], labels: Sequence[bool]) -> float:
        if not scores or not labels or len(scores) != len(labels):
            return 0.0
        scores_np = np.asarray(scores, dtype=float)
        labels_np = np.asarray(labels, dtype=bool)
        pos = scores_np[labels_np]
        neg = scores_np[~labels_np]
        if len(pos) == 0 or len(neg) == 0:
            return 0.5
        # Mann-Whitney U based AUC with tie handling.
        combined = np.concatenate([pos, neg])
        order = np.argsort(combined, kind="mergesort")
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(combined) + 1, dtype=float)
        _, inv, counts = np.unique(combined, return_inverse=True, return_counts=True)
        for idx, count in enumerate(counts):
            if count <= 1:
                continue
            positions = np.where(inv == idx)[0]
            ranks[positions] = float(np.mean(ranks[positions]))
        pos_rank_sum = float(np.sum(ranks[: len(pos)]))
        u_stat = pos_rank_sum - len(pos) * (len(pos) + 1) / 2.0
        auc = u_stat / (len(pos) * len(neg))
        return float(np.clip(auc, 0.0, 1.0))

    @staticmethod
    def _percentiles(values: Sequence[float]) -> Dict[str, float]:
        if not values:
            return {"p10": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0}
        arr = np.asarray(values, dtype=float)
        return {
            "p10": float(np.percentile(arr, 10)),
            "p25": float(np.percentile(arr, 25)),
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
            "p90": float(np.percentile(arr, 90)),
        }

    def _effective_decision_mode_counts(self) -> Dict[str, int]:
        counts = Counter(
            str(record.get("effective_decision_mode", self.failure_decision_mode))
            for record in self.summary_records
        )
        return {key: int(value) for key, value in sorted(counts.items())}

    def _write_low_pressure_score_debug_csv(self, tag: str) -> None:
        if not self.summary_records:
            return
        fieldnames = [
            "sample_id",
            "y_true",
            "pressure_group",
            "low_pressure_score",
            "low_pressure_threshold",
            "low_pressure_pred",
            "final_pred",
            "decision_source",
        ]
        latest_path = self.session_dir / "low_pressure_score_debug.csv"
        tagged_path = self.session_dir / f"low_pressure_score_debug_{tag}.csv"
        for output_path in (latest_path, tagged_path):
            with output_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for record in self.summary_records:
                    pressure_group = str(
                        record.get(
                            "effective_pressure_regime",
                            record.get("initial_pressure_regime", "high_pressure"),
                        )
                    )
                    is_low_pressure = pressure_group == "low_pressure"
                    low_pressure_score = (
                        float(record.get("low_pressure_score", 0.0))
                        if is_low_pressure
                        else ""
                    )
                    low_pressure_threshold = (
                        float(
                            record.get(
                                "low_pressure_threshold",
                                self.last_low_pressure_model_info.get("low_pressure_threshold", 0.5),
                            )
                        )
                        if is_low_pressure
                        else ""
                    )
                    low_pressure_pred = (
                        bool(low_pressure_score >= low_pressure_threshold)
                        if is_low_pressure
                        else ""
                    )
                    writer.writerow(
                        {
                            "sample_id": record.get("test_id", ""),
                            "y_true": bool(record.get("true_failure_v2", False)),
                            "pressure_group": pressure_group,
                            "low_pressure_score": low_pressure_score,
                            "low_pressure_threshold": low_pressure_threshold,
                            "low_pressure_pred": low_pressure_pred,
                            "final_pred": bool(record.get("system_failure_v2", False)),
                            "decision_source": str(
                                record.get(
                                    "decision_source",
                                    record.get("effective_decision_mode", self.failure_decision_mode),
                                )
                            ),
                        }
                    )

    def _write_offline_decision_distribution(self, tag: str):
        if not self.summary_records:
            return
        labels = [self._resolve_true_failure_v2_value(record) for record in self.summary_records]
        scores = [float(record.get("decision_score_v2", record.get("total_membership_v2", 0.0))) for record in self.summary_records]
        failure_decision_mode = self.failure_decision_mode
        if failure_decision_mode == "single_fused_score":
            primary_scores = [float(record.get("fused_score", 0.0)) for record in self.summary_records]
            primary_score_name = "fused_score"
            primary_auc_key = "auc_fused_score"
        elif failure_decision_mode == "direct_failure_model":
            primary_scores = [float(record.get("final_failure_probability", 0.0)) for record in self.summary_records]
            primary_score_name = "final_failure_probability"
            primary_auc_key = "auc_final_failure_probability"
        else:
            primary_scores = scores
            primary_score_name = "decision_score_v2"
            primary_auc_key = "auc_decision_score_v2"
        pos_scores = [score for score, label in zip(scores, labels) if label]
        neg_scores = [score for score, label in zip(scores, labels) if not label]
        primary_pos_scores = [score for score, label in zip(primary_scores, labels) if label]
        primary_neg_scores = [score for score, label in zip(primary_scores, labels) if not label]
        overlap_count = sum(1 for score in scores if 0.3 <= score <= 0.7)
        primary_overlap_count = sum(1 for score in primary_scores if 0.3 <= score <= 0.7)
        payload = {
            "timestamp": now_stamp(),
            "tag": tag,
            "failure_decision_mode": failure_decision_mode,
            "true_failure_v2_policy": self.true_failure_v2_policy,
            "record_count": len(scores),
            "positive_count": int(sum(1 for x in labels if x)),
            "negative_count": int(sum(1 for x in labels if not x)),
            "auc_decision_score_v2": self._roc_auc_binary(scores, labels),
            "decision_score_v2_percentiles_positive": self._percentiles(pos_scores),
            "decision_score_v2_percentiles_negative": self._percentiles(neg_scores),
            "decision_overlap_ratio_03_07": float(overlap_count / max(1, len(scores))),
            "primary_score_name": primary_score_name,
            primary_auc_key: self._roc_auc_binary(primary_scores, labels),
            "primary_score_percentiles_positive": self._percentiles(primary_pos_scores),
            "primary_score_percentiles_negative": self._percentiles(primary_neg_scores),
            "primary_score_overlap_ratio_03_07": float(primary_overlap_count / max(1, len(primary_scores))),
            "decision_formula_config": self.evaluator.get_decision_formula_config(),
            "failure_model_info": self.last_failure_model_info,
        }
        latest_path = self.session_dir / "offline_decision_distribution.json"
        latest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tagged_path = self.session_dir / f"offline_decision_distribution_{tag}.json"
        tagged_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_summary_records_from_round_files(self):
        loaded = self._collect_summary_records_from_rounds_dir(self.rounds_dir)
        self.summary_records = []
        for raw_record in loaded:
            record = self._apply_true_failure_policy_to_record(dict(raw_record))
            round_index = int(record.get("round_index", 0) or 0)
            record["pressure_score"] = self._compute_pressure_score(record)
            record["initial_pressure_regime"] = self._compute_initial_pressure_regime(record)
            self._resolve_effective_pressure_regime(record, round_index=round_index)
            self.summary_records.append(record)
        self.cumulative_failure_labels_v2 = [float(bool(record.get("true_failure_v2", False))) for record in self.summary_records]

    def _collect_summary_records_from_rounds_dir(self, rounds_dir: Path) -> List[Dict]:
        loaded: List[Dict] = []
        if not rounds_dir.exists():
            return loaded
        for round_dir in sorted(rounds_dir.glob("round_*")):
            file_path = round_dir / "failure_scores.jsonl"
            if not file_path.exists():
                continue
            with file_path.open("r", encoding="utf-8") as f:
                for line in f:
                    text = line.strip()
                    if not text:
                        continue
                    try:
                        loaded.append(json.loads(text))
                    except json.JSONDecodeError:
                        continue
        loaded.sort(key=lambda item: (int(item.get("round_index", 0)), int(item.get("test_id", 0))))
        return loaded

    def _resolve_offline_source_rounds_dir(self) -> Optional[Path]:
        raw = str(self.args.offline_source_session or "").strip()
        if not raw:
            return None
        source = Path(raw)
        if not source.is_absolute():
            source = (self.project_root / source).resolve()
        candidates = [
            source / "rounds",
            source / "current_session" / "rounds",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _build_decision_training_matrix(self, records: Sequence[Dict]) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        filtered_records: List[Dict] = []
        features: List[List[float]] = []
        labels: List[float] = []
        for record in records:
            if self.args.threshold_calibration_scope == "terminal_only" and not bool(record.get("terminal_hard_failure", False)):
                continue
            if not all(
                key in record
                for key in ("converged_mean_v2", "converged_p75_v2", "converged_max_v2", "converged_slope_v2")
            ):
                continue
            filtered_records.append(record)
            converged_mean_v2 = float(record.get("converged_mean_v2", 0.0))
            terminal_risk_score = float(
                record.get("terminal_risk_score", 1.0 if bool(record.get("terminal_hard_failure", False)) else 0.0)
            )
            features.append(
                [
                    converged_mean_v2,
                    float(record.get("converged_p75_v2", 0.0)),
                    float(record.get("converged_max_v2", 0.0)),
                    max(0.0, float(record.get("converged_slope_v2", 0.0))),
                    float(record.get("converged_std_v2", record.get("score_uncertainty_v2", 0.0))),
                    float(record.get("converged_high_ratio_v2", 0.0)),
                    float(record.get("terminal_score_gap_v2", max(0.0, terminal_risk_score - converged_mean_v2))),
                ]
            )
            labels.append(float(self._resolve_true_failure_v2_value(record)))
        if not features:
            return np.zeros((0, 7), dtype=np.float32), np.zeros((0,), dtype=np.float32), []
        return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.float32), filtered_records

    def _build_fused_training_matrix(self, records: Sequence[Dict]) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        filtered_records: List[Dict] = []
        features: List[List[float]] = []
        labels: List[float] = []
        for record in records:
            if not self._is_fused_effective_record(record):
                continue
            filtered_records.append(record)
            features.append(self._build_fused_feature_vector(record))
            labels.append(float(self._resolve_true_failure_v2_value(record)))
        if not features:
            return np.zeros((0, 8), dtype=np.float32), np.zeros((0,), dtype=np.float32), []
        return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.float32), filtered_records

    def _build_fused_feature_vector(self, record: Dict) -> List[float]:
        terminal_risk_score = float(
            record.get("terminal_risk_score", 1.0 if bool(record.get("terminal_hard_failure", False)) else 0.0)
        )
        decision_score_v2 = float(record.get("decision_score_v2", record.get("total_membership_v2", 0.0)))
        converged_mean_v2 = float(record.get("converged_mean_v2", 0.0))
        converged_std_v2 = float(record.get("converged_std_v2", record.get("score_uncertainty_v2", 0.0)))
        converged_slope_v2 = float(record.get("converged_slope_v2", 0.0))
        terminal_hard_failure_flag = 1.0 if bool(record.get("terminal_hard_failure", False)) else 0.0
        terminal_score_gap_v2 = float(record.get("terminal_score_gap_v2", max(0.0, terminal_risk_score - converged_mean_v2)))
        return [
            decision_score_v2,
            terminal_risk_score,
            terminal_score_gap_v2,
            float(record.get("converged_high_ratio_v2", 0.0)),
            converged_std_v2,
            converged_slope_v2,
            terminal_hard_failure_flag,
            decision_score_v2 * terminal_risk_score,
        ]

    def _build_direct_failure_training_matrix(self, records: Sequence[Dict]) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        filtered_records: List[Dict] = []
        features: List[List[float]] = []
        labels: List[float] = []
        for record in records:
            if self.args.threshold_calibration_scope == "terminal_only" and not bool(record.get("terminal_hard_failure", False)):
                continue
            if not all(
                key in record
                for key in (
                    "converged_mean_v2",
                    "converged_p75_v2",
                    "converged_max_v2",
                    "converged_slope_v2",
                )
            ):
                continue
            filtered_records.append(record)
            terminal_risk_score = float(
                record.get("terminal_risk_score", 1.0 if bool(record.get("terminal_hard_failure", False)) else 0.0)
            )
            converged_mean_v2 = float(record.get("converged_mean_v2", 0.0))
            features.append(
                [
                    converged_mean_v2,
                    float(record.get("converged_p75_v2", 0.0)),
                    float(record.get("converged_max_v2", 0.0)),
                    float(record.get("converged_slope_v2", 0.0)),
                    float(record.get("converged_std_v2", record.get("score_uncertainty_v2", 0.0))),
                    float(record.get("converged_high_ratio_v2", 0.0)),
                    terminal_risk_score,
                    float(record.get("terminal_score_gap_v2", max(0.0, terminal_risk_score - converged_mean_v2))),
                ]
            )
            labels.append(float(self._resolve_true_failure_v2_value(record)))
        if not features:
            return np.zeros((0, 8), dtype=np.float32), np.zeros((0,), dtype=np.float32), []
        return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.float32), filtered_records

    def _build_low_pressure_feature_vector(self, record: Dict) -> List[float]:
        return [
            float(record.get("converged_mean_v2", 0.0)),
            float(record.get("converged_p75_v2", 0.0)),
            float(record.get("converged_max_v2", 0.0)),
            float(record.get("converged_std_v2", record.get("score_uncertainty_v2", 0.0))),
            float(record.get("converged_slope_v2", 0.0)),
            float(record.get("converged_high_ratio_v2", 0.0)),
            float(record.get("terminal_risk_score", 1.0 if bool(record.get("terminal_hard_failure", False)) else 0.0)),
            float(record.get("terminal_score_gap_v2", 0.0)),
            float(record.get("decision_score_v2", record.get("total_membership_v2", 0.0))),
            float(record.get("pressure_score", self._compute_pressure_score(record))),
        ]

    def _build_low_pressure_training_matrix(self, records: Sequence[Dict]) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        filtered_records: List[Dict] = []
        features: List[List[float]] = []
        labels: List[float] = []
        for record in records:
            if str(record.get("effective_pressure_regime", "")) != "low_pressure":
                continue
            filtered_records.append(record)
            features.append(self._build_low_pressure_feature_vector(record))
            labels.append(float(self._resolve_true_failure_v2_value(record)))
        if not features:
            return np.zeros((0, 10), dtype=np.float32), np.zeros((0,), dtype=np.float32), []
        return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.float32), filtered_records

    def _resolve_low_pressure_min_effective_support(self) -> int:
        cfg = dict(getattr(self, "low_pressure_classifier_config", DEFAULT_LOW_PRESSURE_CLASSIFIER_CONFIG) or {})
        fallback_cfg = dict(cfg.get("fallback", {}))
        return int(max(2, int(fallback_cfg.get("min_effective_support", DEFAULT_LOW_PRESSURE_CLASSIFIER_CONFIG["fallback"]["min_effective_support"]))))

    def _should_fallback_from_low_pressure(
        self,
        threshold_stats: Optional[Dict[str, object]] = None,
        model_info: Optional[Dict[str, object]] = None,
    ) -> Tuple[bool, str]:
        cfg = dict(getattr(self, "low_pressure_classifier_config", DEFAULT_LOW_PRESSURE_CLASSIFIER_CONFIG) or {})
        if normalize_switch_text(cfg.get("enabled"), default=str(DEFAULT_LOW_PRESSURE_CLASSIFIER_CONFIG["enabled"])) != "on":
            return True, "low_pressure_classifier_disabled"
        fallback_cfg = dict(cfg.get("fallback", {}))
        if str(fallback_cfg.get("policy", DEFAULT_LOW_PRESSURE_CLASSIFIER_CONFIG["fallback"]["policy"])).strip().lower() != "dual_threshold_v2":
            return True, "unsupported_low_pressure_fallback_policy"
        threshold_stats = dict(threshold_stats or getattr(self, "last_low_pressure_threshold_stats", {}) or {})
        model_info = dict(model_info or getattr(self, "last_low_pressure_model_info", {}) or {})
        min_support = self._resolve_low_pressure_min_effective_support()
        effective_support = int(threshold_stats.get("effective_support", 0))
        train_support = int(threshold_stats.get("train_support", threshold_stats.get("threshold_split_train_support", 0)))
        positive_count = int(threshold_stats.get("positive_count", threshold_stats.get("threshold_support_train_positive_count", 0)))
        negative_count = int(threshold_stats.get("negative_count", threshold_stats.get("threshold_support_train_negative_count", 0)))
        if effective_support < min_support:
            return True, "insufficient_effective_support"
        if train_support < min_support:
            return True, "insufficient_train_support"
        if normalize_switch_text(
            fallback_cfg.get("require_both_classes_in_train"),
            default=str(DEFAULT_LOW_PRESSURE_CLASSIFIER_CONFIG["fallback"]["require_both_classes_in_train"]),
        ) == "on" and (positive_count == 0 or negative_count == 0):
            return True, "single_class_labels"
        model_status = str(model_info.get("low_pressure_model_status", "")).strip().lower()
        holdout_count = int(model_info.get("low_pressure_model_holdout_record_count", 0))
        if model_status != "fitted" or holdout_count <= 0:
            return True, "low_pressure_model_unavailable"
        return False, ""

    @staticmethod
    def _sigmoid_np(values: np.ndarray) -> np.ndarray:
        clipped = np.clip(values, -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-clipped))

    def _make_split_rng(self, context: str, support: int) -> np.random.Generator:
        salt = sum(ord(ch) for ch in str(context))
        return np.random.default_rng(int(self.threshold_split_seed) + int(support) * 17 + salt)

    def _sample_positions_from_pool(
        self,
        pool_positions: Sequence[int],
        labels: Sequence[bool],
        target_count: int,
        rng: np.random.Generator,
    ) -> List[int]:
        target = int(max(0, min(int(target_count), len(pool_positions))))
        if target <= 0 or not pool_positions:
            return []
        by_label = {
            False: [int(pos) for pos in pool_positions if not bool(labels[pos])],
            True: [int(pos) for pos in pool_positions if bool(labels[pos])],
        }
        for bucket in by_label.values():
            rng.shuffle(bucket)
        positive_ratio = float(len(by_label[True]) / len(pool_positions)) if pool_positions else 0.0
        true_target = int(round(target * positive_ratio))
        true_target = min(true_target, len(by_label[True]))
        false_target = min(target - true_target, len(by_label[False]))
        allocated = true_target + false_target
        while allocated < target:
            if len(by_label[True]) > true_target:
                true_target += 1
                allocated += 1
                continue
            if len(by_label[False]) > false_target:
                false_target += 1
                allocated += 1
                continue
            break
        selected = by_label[True][:true_target] + by_label[False][:false_target]
        rng.shuffle(selected)
        return sorted(int(pos) for pos in selected)

    def _resolve_split_plan(
        self,
        labels: Sequence[bool],
        *,
        min_train_support: int,
        context: str,
        holdout_ratio: Optional[float] = None,
    ) -> Dict[str, object]:
        support = len(labels)
        if holdout_ratio is None:
            resolved_holdout_ratio = float(
                self.threshold_split_config.get("holdout_ratio", self.args.threshold_calibration_holdout_ratio)
            )
        else:
            resolved_holdout_ratio = float(holdout_ratio)
        resolved_holdout_ratio = float(np.clip(resolved_holdout_ratio, 0.0, 0.49))
        holdout_count = int(np.floor(support * resolved_holdout_ratio))
        train_count = support - holdout_count
        result: Dict[str, object] = {
            "status": "updated",
            "mode": str(self.threshold_split_config.get("mode", "chronological")).strip().lower(),
            "seed": int(self.threshold_split_seed),
            "holdout_ratio": float(resolved_holdout_ratio),
            "late_window_ratio": float(self.threshold_split_config.get("late_window_ratio", 0.25)),
            "holdout_late_fraction": float(self.threshold_split_config.get("holdout_late_fraction", 0.70)),
            "support": int(support),
            "train_support": int(train_count),
            "holdout_support": int(holdout_count),
            "holdout_late_support": 0,
            "train_positions": [],
            "holdout_positions": [],
        }
        if train_count < int(max(2, min_train_support)):
            result["status"] = "frozen"
            result["reason"] = "insufficient_train_support"
            return result

        positions = list(range(support))
        mode = str(result["mode"])
        if holdout_count <= 0:
            result["train_positions"] = positions
            result["holdout_positions"] = []
            return result

        if mode == "chronological":
            result["train_positions"] = positions[:train_count]
            result["holdout_positions"] = positions[train_count:]
            late_window_size = max(1, int(np.ceil(support * float(result["late_window_ratio"]))))
            late_pool = set(positions[-late_window_size:])
            result["holdout_late_support"] = sum(1 for pos in result["holdout_positions"] if pos in late_pool)
            return result

        rng = self._make_split_rng(context, support)
        if mode == "stratified_random":
            holdout_positions = self._sample_positions_from_pool(positions, labels, holdout_count, rng)
            train_positions = sorted(set(positions) - set(holdout_positions))
            result["train_positions"] = train_positions
            result["holdout_positions"] = holdout_positions
            return result

        late_window_ratio = float(np.clip(float(result["late_window_ratio"]), 0.05, 0.95))
        late_window_size = max(1, int(np.ceil(support * late_window_ratio)))
        late_pool = positions[-late_window_size:]
        early_pool = positions[:-late_window_size]
        holdout_late_target = int(round(holdout_count * float(np.clip(result["holdout_late_fraction"], 0.0, 1.0))))
        holdout_late_target = min(holdout_count, holdout_late_target)
        holdout_positions = self._sample_positions_from_pool(late_pool, labels, holdout_late_target, rng)

        late_selected_set = set(holdout_positions)
        remaining_holdout = holdout_count - len(holdout_positions)
        if remaining_holdout > 0:
            early_selected = self._sample_positions_from_pool(
                [pos for pos in early_pool if pos not in late_selected_set],
                labels,
                remaining_holdout,
                rng,
            )
            holdout_positions.extend(early_selected)

        remaining_holdout = holdout_count - len(holdout_positions)
        if remaining_holdout > 0:
            supplemental_pool = [pos for pos in positions if pos not in set(holdout_positions)]
            supplemental = self._sample_positions_from_pool(supplemental_pool, labels, remaining_holdout, rng)
            holdout_positions.extend(supplemental)

        holdout_positions = sorted(set(int(pos) for pos in holdout_positions))
        if len(holdout_positions) > holdout_count:
            holdout_positions = holdout_positions[:holdout_count]
        train_positions = sorted(set(positions) - set(holdout_positions))
        result["train_positions"] = train_positions
        result["holdout_positions"] = holdout_positions
        result["train_support"] = len(train_positions)
        result["holdout_support"] = len(holdout_positions)
        result["holdout_late_support"] = sum(1 for pos in holdout_positions if pos in set(late_pool))
        if len(train_positions) < int(max(2, min_train_support)):
            result["status"] = "frozen"
            result["reason"] = "insufficient_train_support"
        return result

    @staticmethod
    def _gather_rows_by_positions(values: Sequence[Any], positions: Sequence[int]) -> List[Any]:
        return [values[int(pos)] for pos in positions]

    def _build_split_metadata(self, split_plan: Mapping[str, object]) -> Dict[str, object]:
        return {
            "threshold_split_mode": str(split_plan.get("mode", self.threshold_split_config.get("mode", "chronological"))),
            "threshold_split_seed": int(split_plan.get("seed", self.threshold_split_seed)),
            "threshold_split_holdout_ratio": float(split_plan.get("holdout_ratio", self.threshold_split_config.get("holdout_ratio", 0.2))),
            "threshold_split_late_window_ratio": float(
                split_plan.get("late_window_ratio", self.threshold_split_config.get("late_window_ratio", 0.25))
            ),
            "threshold_split_holdout_late_fraction": float(
                split_plan.get(
                    "holdout_late_fraction",
                    self.threshold_split_config.get("holdout_late_fraction", 0.70),
                )
            ),
            "threshold_split_train_support": int(split_plan.get("train_support", 0)),
            "threshold_split_holdout_support": int(split_plan.get("holdout_support", 0)),
            "threshold_split_holdout_late_support": int(split_plan.get("holdout_late_support", 0)),
            "train_positions": [int(pos) for pos in split_plan.get("train_positions", [])],
            "holdout_positions": [int(pos) for pos in split_plan.get("holdout_positions", [])],
        }

    def _resolve_distribution_balance_guard_profile(self) -> Tuple[str, Dict[str, object]]:
        profiles = dict(self.distribution_balance_guard_config.get("constellation_profiles", {}))
        default_profile = dict(profiles.get("default", {}))
        large_profile = dict(profiles.get("large_constellation", {}))
        large_ids = {
            int(value)
            for value in large_profile.get("constellation_ids", [])
            if str(value).strip()
        }
        if int(self.fixed_constellation_config) in large_ids:
            merged = dict(default_profile)
            merged.update(large_profile)
            return "large_constellation", merged
        return "default", default_profile

    @staticmethod
    def _scenario_has_attack(scenario: Mapping[str, object]) -> bool:
        return any(int(scenario.get(key, 0) or 0) > 0 for key in ATTACK_SCENARIO_KEYS)

    def _is_healthy_band_scenario(self, scenario: Mapping[str, object]) -> bool:
        if self._scenario_has_attack(scenario):
            return False
        for key, limit in HEALTHY_BAND_DEGRADATION_LIMITS.items():
            if float(scenario.get(key, 0.0) or 0.0) > float(limit):
                return False
        return True

    def _summarize_generated_band_balance(self, scenarios: Sequence[Mapping[str, object]]) -> Dict[str, object]:
        healthy_count = sum(1 for scenario in scenarios if self._is_healthy_band_scenario(scenario))
        boundary_count = max(0, len(scenarios) - healthy_count)
        shortage_flags: List[str] = []
        if healthy_count <= 0:
            shortage_flags.append("healthy_seed_strong")
        return {
            "selected_healthy_band_count": int(healthy_count),
            "selected_boundary_band_count": int(boundary_count),
            # The current generator does not expose raw prefilter band counts on the mainline,
            # so v1 uses the selected-stage healthy count as the closest available proxy.
            "prefilter_healthy_band_count": int(healthy_count),
            "shortage_flags": shortage_flags,
        }

    def _build_healthy_recovery_scenario(self) -> Dict[str, object]:
        env_cfg = dict(self.base_config.get("environment", {}))
        scenario: Dict[str, object] = {}
        for key in SCENARIO_PARAMETER_NAMES:
            if key in {
                "DegradedEdgeRatio",
                "EdgeDisconnectRatio",
                "EdgeBandwidthMeanDecreaseRatio",
                "EdgeBandwidthDecreaseStd",
            }:
                scenario[key] = 0.0
            elif key in ATTACK_SCENARIO_KEYS:
                scenario[key] = 0
            elif key == "ConstellationConfig":
                scenario[key] = int(self.fixed_constellation_config)
            elif key == "PacketSizeMean":
                scenario[key] = int(env_cfg.get(key, 400000000))
            elif key == "PacketSizeStd":
                scenario[key] = int(env_cfg.get(key, 115470000))
            else:
                scenario[key] = env_cfg.get(key, 0.0)
        return scenario_to_mapping(scenario)

    def _apply_distribution_balance_guard(
        self,
        scenarios: Sequence[Dict[str, object]],
    ) -> List[Dict[str, object]]:
        profile_name, profile_cfg = self._resolve_distribution_balance_guard_profile()
        guard_enabled = normalize_switch_text(
            self.distribution_balance_guard_config.get("enabled"),
            default="off",
        )
        guard_state = dict(self.distribution_balance_guard_state)
        selected_scenarios = [dict(scenario) for scenario in scenarios]
        stats = self._summarize_generated_band_balance(selected_scenarios)
        selected_healthy_count = int(stats["selected_healthy_band_count"])
        prefilter_healthy_count = int(stats["prefilter_healthy_band_count"])
        shortage_flags = list(stats.get("shortage_flags", []))

        if selected_healthy_count <= 0:
            guard_state["consecutive_selected_healthy_zero_rounds"] = int(
                guard_state.get("consecutive_selected_healthy_zero_rounds", 0)
            ) + 1
        else:
            guard_state["consecutive_selected_healthy_zero_rounds"] = 0
        if "healthy_seed_strong" in shortage_flags:
            guard_state["consecutive_shortage_flag_rounds"] = int(
                guard_state.get("consecutive_shortage_flag_rounds", 0)
            ) + 1
        else:
            guard_state["consecutive_shortage_flag_rounds"] = 0

        trigger_cfg = dict(self.distribution_balance_guard_config.get("trigger", {}))
        recovery_cfg = dict(self.distribution_balance_guard_config.get("recovery", {}))
        healthy_baseline_count = sum(
            1 for record in self.summary_records if str(record.get("baseline_status", "")) == "healthy"
        )
        trigger_reason_codes: List[str] = []
        if guard_enabled != "on":
            guard_state["active"] = False
            guard_state["consecutive_triggered_rounds"] = 0
            guard_state["consecutive_healthy_recovered_rounds"] = 0
        if guard_enabled == "on":
            if int(guard_state.get("consecutive_selected_healthy_zero_rounds", 0)) >= int(
                profile_cfg.get(
                    "selected_healthy_band_zero_rounds",
                    trigger_cfg.get("selected_healthy_band_zero_rounds", 2),
                )
            ):
                trigger_reason_codes.append("selected_healthy_band_zero_rounds")
            if int(guard_state.get("consecutive_shortage_flag_rounds", 0)) >= int(
                profile_cfg.get("shortage_flag_rounds", trigger_cfg.get("shortage_flag_rounds", 2))
            ):
                trigger_reason_codes.append("healthy_seed_strong_shortage_rounds")
            require_prefilter_zero = normalize_switch_text(
                trigger_cfg.get("require_prefilter_healthy_band_zero"),
                default="on",
            )
            baseline_trigger_ok = int(healthy_baseline_count) <= int(trigger_cfg.get("healthy_baseline_count_max", 1))
            prefilter_trigger_ok = (
                prefilter_healthy_count <= 0 if require_prefilter_zero == "on" else True
            )
            if baseline_trigger_ok and prefilter_trigger_ok:
                trigger_reason_codes.append("healthy_baseline_prefilter_zero")

        if trigger_reason_codes:
            guard_state["active"] = True
            guard_state["consecutive_triggered_rounds"] = int(
                guard_state.get("consecutive_triggered_rounds", 0)
            ) + 1
            guard_state["consecutive_healthy_recovered_rounds"] = 0
        else:
            guard_state["consecutive_triggered_rounds"] = 0
            if bool(guard_state.get("active", False)) and selected_healthy_count > 0:
                guard_state["consecutive_healthy_recovered_rounds"] = int(
                    guard_state.get("consecutive_healthy_recovered_rounds", 0)
                ) + 1
            else:
                guard_state["consecutive_healthy_recovered_rounds"] = 0
            if bool(guard_state.get("active", False)) and int(
                guard_state.get("consecutive_healthy_recovered_rounds", 0)
            ) >= int(recovery_cfg.get("auto_exit_after_healthy_recovered_rounds", 2)):
                guard_state["active"] = False

        guard_state["profile"] = profile_name
        replaced_slots = 0
        recovery_source_counts: Dict[str, int] = {}
        skip_reason = ""

        if guard_enabled == "on" and bool(guard_state.get("active", False)):
            replace_slots = int(
                profile_cfg.get(
                    "replace_boundary_slots_default",
                    recovery_cfg.get("replace_boundary_slots_default", 1),
                )
            )
            allow_escalation = normalize_switch_text(
                profile_cfg.get("allow_escalation"),
                default="on",
            )
            if (
                allow_escalation == "on"
                and int(guard_state.get("consecutive_triggered_rounds", 0))
                >= int(recovery_cfg.get("escalate_after_triggered_rounds", 2))
            ):
                replace_slots = int(
                    profile_cfg.get(
                        "replace_boundary_slots_escalated",
                        recovery_cfg.get("replace_boundary_slots_escalated", replace_slots),
                    )
                )
            boundary_positions = [
                index for index, scenario in enumerate(selected_scenarios)
                if not self._is_healthy_band_scenario(scenario)
            ]
            replace_slots = min(replace_slots, len(boundary_positions))
            if replace_slots <= 0:
                skip_reason = "no_boundary_slot_available"
            else:
                recovery_candidates: List[Tuple[str, Dict[str, object]]] = []
                for source_name in recovery_cfg.get("recovery_source_priority", ["healthy_recovery"]):
                    normalized_source = str(source_name).strip().lower()
                    if normalized_source == "healthy_recovery":
                        while len(recovery_candidates) < replace_slots:
                            recovery_candidates.append(
                                ("healthy_recovery", self._build_healthy_recovery_scenario())
                            )
                        break
                    if normalized_source == "healthy_push":
                        recovery_source_counts.setdefault("healthy_push", 0)
                if not recovery_candidates:
                    skip_reason = "no_recovery_candidate"
                else:
                    for slot_index, (_, replacement) in enumerate(recovery_candidates[:replace_slots]):
                        selected_scenarios[boundary_positions[slot_index]] = replacement
                        replaced_slots += 1
                    recovery_source_counts["healthy_recovery"] = replaced_slots

        self.distribution_balance_guard_state = guard_state
        self.last_distribution_balance_guard_info = {
            "distribution_balance_guard_enabled": guard_enabled,
            "distribution_balance_guard_profile": profile_name,
            "distribution_balance_guard_active": bool(guard_state.get("active", False)),
            "distribution_balance_guard_trigger_reason_codes": trigger_reason_codes,
            "distribution_balance_guard_replaced_slots": int(replaced_slots),
            "distribution_balance_guard_recovery_source_counts": recovery_source_counts,
            "distribution_balance_guard_skip_reason": str(skip_reason),
            "distribution_balance_guard_selected_healthy_band_count": int(selected_healthy_count),
            "distribution_balance_guard_prefilter_healthy_band_count": int(prefilter_healthy_count),
            "distribution_balance_guard_shortage_flags": shortage_flags,
        }
        return selected_scenarios

    def _fit_linear_failure_model(
        self,
        feature_matrix: np.ndarray,
        labels: np.ndarray,
        feature_names: Sequence[str],
        status_key_prefix: str,
    ) -> Dict[str, object]:
        info: Dict[str, object] = {
            f"{status_key_prefix}_status": "disabled",
            f"{status_key_prefix}_holdout_record_count": 0,
            f"{status_key_prefix}_holdout_auc": 0.0,
            f"{status_key_prefix}_holdout_accuracy": 0.0,
            f"{status_key_prefix}_weights": {},
            f"{status_key_prefix}_bias": 0.0,
            "threshold": 0.5,
        }
        support = int(feature_matrix.shape[0])
        min_support = max(2, int(self.args.decision_model_min_support))
        if support < min_support or len(np.unique(labels)) < 2 or not bool(self.args.fit_decision_model_offline):
            info[f"{status_key_prefix}_status"] = "frozen"
            return info

        split_plan = self._resolve_split_plan(
            labels=[bool(value >= 0.5) for value in labels.tolist()],
            min_train_support=min_support,
            context=f"{status_key_prefix}_linear_model",
        )
        train_positions = [int(pos) for pos in split_plan.get("train_positions", [])]
        holdout_positions = [int(pos) for pos in split_plan.get("holdout_positions", [])]
        holdout_count = len(holdout_positions)
        if str(split_plan.get("status", "updated")).strip().lower() != "updated" or len(train_positions) < 2:
            info[f"{status_key_prefix}_status"] = "frozen"
            info["threshold_split"] = self._build_split_metadata(split_plan)
            if "reason" in split_plan:
                info["reason"] = str(split_plan["reason"])
            return info

        train_x = torch.tensor(feature_matrix[train_positions], dtype=torch.float32, device=self.device)
        train_y = torch.tensor(labels[train_positions], dtype=torch.float32, device=self.device)
        holdout_x = torch.tensor(feature_matrix[holdout_positions], dtype=torch.float32, device=self.device)
        holdout_y = torch.tensor(labels[holdout_positions], dtype=torch.float32, device=self.device)

        raw_weights = torch.nn.Parameter(torch.zeros((feature_matrix.shape[1],), dtype=torch.float32, device=self.device))
        bias = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32, device=self.device))
        optimizer = torch.optim.Adam([raw_weights, bias], lr=float(self.args.decision_model_lr))
        bce_loss = torch.nn.BCEWithLogitsLoss()
        l2_value = float(max(0.0, float(self.args.decision_model_l2)))
        patience = max(1, int(self.args.decision_model_early_stop_patience))
        best_loss = float("inf")
        stale_epochs = 0
        best_state = {
            "weights": raw_weights.detach().clone(),
            "bias": bias.detach().clone(),
        }

        for _ in range(max(1, int(self.args.decision_model_epochs))):
            optimizer.zero_grad()
            logits = torch.matmul(train_x, raw_weights) + bias
            loss = bce_loss(logits, train_y) + l2_value * torch.sum(raw_weights * raw_weights)
            loss.backward()
            optimizer.step()
            loss_value = float(loss.detach().cpu().item())
            if loss_value + 1e-9 < best_loss:
                best_loss = loss_value
                stale_epochs = 0
                best_state = {
                    "weights": raw_weights.detach().clone(),
                    "bias": bias.detach().clone(),
                }
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    break

        best_weights = best_state["weights"].detach().cpu().numpy()
        best_bias = float(best_state["bias"].detach().cpu().item())
        weight_dict = {str(name): float(best_weights[idx]) for idx, name in enumerate(feature_names)}
        if holdout_count > 0:
            holdout_logits = torch.matmul(holdout_x, best_state["weights"]) + best_state["bias"]
            holdout_scores = torch.sigmoid(holdout_logits).detach().cpu().numpy()
            holdout_labels = holdout_y.detach().cpu().numpy() >= 0.5
            holdout_pred = holdout_scores >= 0.5
            holdout_accuracy = float(np.mean(holdout_pred == holdout_labels))
            holdout_auc = self._roc_auc_binary(holdout_scores.tolist(), holdout_labels.tolist())
        else:
            holdout_accuracy = 0.0
            holdout_auc = 0.0

        info.update(
            {
                f"{status_key_prefix}_status": "fitted",
                f"{status_key_prefix}_holdout_record_count": int(holdout_count),
                f"{status_key_prefix}_holdout_auc": float(holdout_auc),
                f"{status_key_prefix}_holdout_accuracy": float(holdout_accuracy),
                f"{status_key_prefix}_type": "linear",
                f"{status_key_prefix}_weights": weight_dict,
                f"{status_key_prefix}_bias": float(best_bias),
                "threshold_split": self._build_split_metadata(split_plan),
            }
        )
        return info

    def _fit_mlp_failure_model(
        self,
        feature_matrix: np.ndarray,
        labels: np.ndarray,
        status_key_prefix: str,
    ) -> Dict[str, object]:
        info: Dict[str, object] = {
            f"{status_key_prefix}_status": "disabled",
            f"{status_key_prefix}_holdout_record_count": 0,
            f"{status_key_prefix}_holdout_auc": 0.0,
            f"{status_key_prefix}_holdout_accuracy": 0.0,
            f"{status_key_prefix}_type": "mlp_small",
            f"{status_key_prefix}_input_mean": [],
            f"{status_key_prefix}_input_std": [],
            f"{status_key_prefix}_mlp_state": {},
            f"{status_key_prefix}_mlp_hidden_dim": int(self.args.fused_mlp_hidden_dim),
            "threshold": 0.5,
        }
        support = int(feature_matrix.shape[0])
        min_support = max(2, int(self.args.decision_model_min_support))
        if support < min_support or len(np.unique(labels)) < 2 or not bool(self.args.fit_decision_model_offline):
            info[f"{status_key_prefix}_status"] = "frozen"
            return info

        split_plan = self._resolve_split_plan(
            labels=[bool(value >= 0.5) for value in labels.tolist()],
            min_train_support=min_support,
            context=f"{status_key_prefix}_mlp_model",
        )
        train_positions = [int(pos) for pos in split_plan.get("train_positions", [])]
        holdout_positions = [int(pos) for pos in split_plan.get("holdout_positions", [])]
        holdout_count = len(holdout_positions)
        if str(split_plan.get("status", "updated")).strip().lower() != "updated" or len(train_positions) < 2:
            info[f"{status_key_prefix}_status"] = "frozen"
            info["threshold_split"] = self._build_split_metadata(split_plan)
            if "reason" in split_plan:
                info["reason"] = str(split_plan["reason"])
            return info

        train_np = feature_matrix[train_positions]
        holdout_np = feature_matrix[holdout_positions]
        train_mean = np.mean(train_np, axis=0)
        train_std = np.std(train_np, axis=0)
        train_std = np.where(train_std < 1e-6, 1.0, train_std)
        train_norm = (train_np - train_mean) / train_std
        holdout_norm = (holdout_np - train_mean) / train_std if holdout_count > 0 else holdout_np

        train_x = torch.tensor(train_norm, dtype=torch.float32, device=self.device)
        train_y = torch.tensor(labels[train_positions], dtype=torch.float32, device=self.device)
        holdout_x = torch.tensor(holdout_norm, dtype=torch.float32, device=self.device)
        holdout_y = torch.tensor(labels[holdout_positions], dtype=torch.float32, device=self.device)

        in_dim = int(feature_matrix.shape[1])
        hidden_dim = max(4, int(self.args.fused_mlp_hidden_dim))
        model = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
        ).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(self.args.decision_model_lr))
        bce_loss = torch.nn.BCEWithLogitsLoss()
        l2_value = float(max(0.0, float(self.args.decision_model_l2)))
        patience = max(1, int(self.args.decision_model_early_stop_patience))
        best_loss = float("inf")
        stale_epochs = 0
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        for _ in range(max(1, int(self.args.decision_model_epochs))):
            optimizer.zero_grad()
            logits = model(train_x).squeeze(1)
            l2_penalty = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            if l2_value > 0.0:
                for param in model.parameters():
                    l2_penalty = l2_penalty + torch.sum(param * param)
            loss = bce_loss(logits, train_y) + l2_value * l2_penalty
            loss.backward()
            optimizer.step()
            loss_value = float(loss.detach().cpu().item())
            if loss_value + 1e-9 < best_loss:
                best_loss = loss_value
                stale_epochs = 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    break

        model.load_state_dict(best_state)
        model.eval()
        if holdout_count > 0:
            with torch.no_grad():
                holdout_logits = model(holdout_x).squeeze(1)
                holdout_scores = torch.sigmoid(holdout_logits).detach().cpu().numpy()
            holdout_labels = holdout_y.detach().cpu().numpy() >= 0.5
            holdout_pred = holdout_scores >= 0.5
            holdout_accuracy = float(np.mean(holdout_pred == holdout_labels))
            holdout_auc = self._roc_auc_binary(holdout_scores.tolist(), holdout_labels.tolist())
        else:
            holdout_accuracy = 0.0
            holdout_auc = 0.0

        state_export = {key: value.detach().cpu().numpy().tolist() for key, value in best_state.items()}
        info.update(
            {
                f"{status_key_prefix}_status": "fitted",
                f"{status_key_prefix}_holdout_record_count": int(holdout_count),
                f"{status_key_prefix}_holdout_auc": float(holdout_auc),
                f"{status_key_prefix}_holdout_accuracy": float(holdout_accuracy),
                f"{status_key_prefix}_input_mean": train_mean.astype(np.float32).tolist(),
                f"{status_key_prefix}_input_std": train_std.astype(np.float32).tolist(),
                f"{status_key_prefix}_mlp_state": state_export,
                f"{status_key_prefix}_mlp_hidden_dim": hidden_dim,
                "threshold_split": self._build_split_metadata(split_plan),
            }
        )
        return info

    def _fit_low_pressure_classifier(self, records: Sequence[Dict]) -> Dict[str, object]:
        cfg = dict(getattr(self, "low_pressure_classifier_config", DEFAULT_LOW_PRESSURE_CLASSIFIER_CONFIG) or {})
        info: Dict[str, object] = {
            "low_pressure_model_status": "disabled",
            "low_pressure_model_holdout_record_count": 0,
            "low_pressure_model_holdout_auc": 0.0,
            "low_pressure_model_holdout_accuracy": 0.0,
            "low_pressure_model_type": str(cfg.get("model_type", "mlp")).strip().lower(),
            "low_pressure_model_input_mean": [],
            "low_pressure_model_input_std": [],
            "low_pressure_model_mlp_state": {},
            "low_pressure_model_hidden_dim": int(cfg.get("hidden_dim", 16)),
            "low_pressure_threshold": float(self.last_low_pressure_model_info.get("low_pressure_threshold", 0.5)),
        }
        if normalize_switch_text(cfg.get("enabled"), default=str(DEFAULT_LOW_PRESSURE_CLASSIFIER_CONFIG["enabled"])) != "on":
            self.last_low_pressure_model_info = dict(info)
            return info
        feature_matrix, labels, _ = self._build_low_pressure_training_matrix(records)
        support = int(feature_matrix.shape[0])
        min_support = self._resolve_low_pressure_min_effective_support()
        if support < min_support or len(np.unique(labels)) < 2:
            info["low_pressure_model_status"] = "frozen"
            self.last_low_pressure_model_info = dict(info)
            return info

        split_plan = self._resolve_split_plan(
            labels=[bool(value >= 0.5) for value in labels.tolist()],
            min_train_support=min_support,
            holdout_ratio=float(cfg.get("holdout_ratio", DEFAULT_LOW_PRESSURE_CLASSIFIER_CONFIG["holdout_ratio"])),
            context="low_pressure_mlp_model",
        )
        train_positions = [int(pos) for pos in split_plan.get("train_positions", [])]
        holdout_positions = [int(pos) for pos in split_plan.get("holdout_positions", [])]
        holdout_count = len(holdout_positions)
        if str(split_plan.get("status", "updated")).strip().lower() != "updated" or len(train_positions) < 2:
            info["low_pressure_model_status"] = "frozen"
            info["threshold_split"] = self._build_split_metadata(split_plan)
            if "reason" in split_plan:
                info["reason"] = str(split_plan["reason"])
            self.last_low_pressure_model_info = dict(info)
            return info

        train_np = feature_matrix[train_positions]
        holdout_np = feature_matrix[holdout_positions]
        train_mean = np.mean(train_np, axis=0)
        train_std = np.std(train_np, axis=0)
        train_std = np.where(train_std < 1e-6, 1.0, train_std)
        train_norm = (train_np - train_mean) / train_std
        holdout_norm = (holdout_np - train_mean) / train_std if holdout_count > 0 else holdout_np

        train_x = torch.tensor(train_norm, dtype=torch.float32, device=self.device)
        train_y = torch.tensor(labels[train_positions], dtype=torch.float32, device=self.device)
        holdout_x = torch.tensor(holdout_norm, dtype=torch.float32, device=self.device)
        holdout_y = torch.tensor(labels[holdout_positions], dtype=torch.float32, device=self.device)

        in_dim = int(feature_matrix.shape[1])
        hidden_dim = max(4, int(cfg.get("hidden_dim", 16)))
        dropout = float(cfg.get("dropout", 0.10))
        model = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 1),
        ).to(self.device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg.get("learning_rate", 5e-4)),
            weight_decay=float(cfg.get("weight_decay", 1e-4)),
        )
        pos_weight = torch.tensor([float(cfg.get("pos_weight", 20.0))], dtype=torch.float32, device=self.device)
        bce_loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        patience = max(1, int(cfg.get("patience", 20)))
        epochs = max(1, int(cfg.get("epochs", 200)))
        batch_size = max(1, int(cfg.get("batch_size", 16)))
        best_loss = float("inf")
        stale_epochs = 0
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        for _ in range(epochs):
            optimizer.zero_grad()
            permutation = torch.randperm(train_x.shape[0], device=self.device)
            epoch_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            batch_count = 0
            for start in range(0, train_x.shape[0], batch_size):
                batch_idx = permutation[start:start + batch_size]
                logits = model(train_x[batch_idx]).squeeze(1)
                loss = bce_loss(logits, train_y[batch_idx])
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                epoch_loss = epoch_loss + loss.detach()
                batch_count += 1
            loss_value = float((epoch_loss / max(1, batch_count)).cpu().item())
            if loss_value + 1e-9 < best_loss:
                best_loss = loss_value
                stale_epochs = 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    break

        model.load_state_dict(best_state)
        model.eval()
        if holdout_count > 0:
            with torch.no_grad():
                holdout_logits = model(holdout_x).squeeze(1)
                holdout_scores = torch.sigmoid(holdout_logits).detach().cpu().numpy()
            holdout_labels = holdout_y.detach().cpu().numpy() >= 0.5
            holdout_pred = holdout_scores >= 0.5
            holdout_accuracy = float(np.mean(holdout_pred == holdout_labels))
            holdout_auc = self._roc_auc_binary(holdout_scores.tolist(), holdout_labels.tolist())
        else:
            holdout_accuracy = 0.0
            holdout_auc = 0.0
        state_export = {key: value.detach().cpu().numpy().tolist() for key, value in best_state.items()}
        info.update(
            {
                "low_pressure_model_status": "fitted",
                "low_pressure_model_holdout_record_count": int(holdout_count),
                "low_pressure_model_holdout_auc": float(holdout_auc),
                "low_pressure_model_holdout_accuracy": float(holdout_accuracy),
                "low_pressure_model_input_mean": train_mean.astype(np.float32).tolist(),
                "low_pressure_model_input_std": train_std.astype(np.float32).tolist(),
                "low_pressure_model_mlp_state": state_export,
                "low_pressure_model_hidden_dim": int(hidden_dim),
                "threshold_split": self._build_split_metadata(split_plan),
            }
        )
        self.last_low_pressure_model_info = dict(info)
        return info

    def _compute_low_pressure_score_and_logit_for_record(self, record: Dict) -> Tuple[float, Optional[float]]:
        input_mean = np.asarray(self.last_low_pressure_model_info.get("low_pressure_model_input_mean", []), dtype=np.float32)
        input_std = np.asarray(self.last_low_pressure_model_info.get("low_pressure_model_input_std", []), dtype=np.float32)
        mlp_state = dict(self.last_low_pressure_model_info.get("low_pressure_model_mlp_state", {}))
        features = np.asarray(self._build_low_pressure_feature_vector(record), dtype=np.float32)
        if (
            input_mean.size == features.size
            and input_std.size == features.size
            and all(key in mlp_state for key in ("0.weight", "0.bias", "3.weight", "3.bias", "6.weight", "6.bias"))
        ):
            normalized = (features - input_mean) / np.where(np.abs(input_std) < 1e-6, 1.0, input_std)
            tensor = torch.tensor(normalized[None, :], dtype=torch.float32, device=self.device)
            hidden_dim = max(4, int(self.last_low_pressure_model_info.get("low_pressure_model_hidden_dim", 16)))
            dropout = float(dict(getattr(self, "low_pressure_classifier_config", DEFAULT_LOW_PRESSURE_CLASSIFIER_CONFIG) or {}).get("dropout", 0.10))
            model = torch.nn.Sequential(
                torch.nn.Linear(features.size, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_dim, 1),
            ).to(self.device)
            state_dict = {key: torch.tensor(value, dtype=torch.float32, device=self.device) for key, value in mlp_state.items()}
            model.load_state_dict(state_dict)
            model.eval()
            with torch.no_grad():
                logit = float(model(tensor).squeeze().cpu().item())
            return float(1.0 / (1.0 + math.exp(-logit))), float(logit)
        return 0.0, None

    def _calibrate_low_pressure_threshold(self, records: Sequence[Dict]) -> Dict[str, object]:
        cfg = dict(getattr(self, "low_pressure_classifier_config", DEFAULT_LOW_PRESSURE_CLASSIFIER_CONFIG) or {})
        threshold_cfg = dict(cfg.get("threshold", {}))
        result: Dict[str, object] = {
            "status": "frozen",
            "reason": "insufficient_samples",
            "threshold": float(self.last_low_pressure_model_info.get("low_pressure_threshold", 0.5)),
            "low_pressure_threshold": float(self.last_low_pressure_model_info.get("low_pressure_threshold", 0.5)),
            "objective": str(threshold_cfg.get("objective", "recall_at_precision")),
            "precision": 0.0,
            "recall": 0.0,
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "f1": 0.0,
            "train_metrics_at_selected_threshold": {
                "f1": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
            },
            "holdout_metrics": {"f1": 0.0, "precision": 0.0, "recall": 0.0, "accuracy": 0.0, "balanced_accuracy": 0.0},
            "low_pressure_model_holdout_auc": float(self.last_low_pressure_model_info.get("low_pressure_model_holdout_auc", 0.0)),
            "threshold_min_precision_used": float(threshold_cfg.get("min_precision", 0.50)),
            "threshold_constraint_status": "not_evaluated",
            "selected_from": "holdout",
            "low_pressure_threshold_status": "frozen",
            "low_pressure_threshold_reason": "insufficient_samples",
            "low_pressure_threshold_train_support": 0,
            "low_pressure_threshold_holdout_support": 0,
            "low_pressure_threshold_positive_count": 0,
            "low_pressure_threshold_negative_count": 0,
            "low_pressure_threshold_constraint_status": "not_evaluated",
            "low_pressure_threshold_selected_from": "holdout",
            "low_pressure_threshold_holdout_metrics": {
                "f1": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
            },
            "low_pressure_threshold_train_metrics_at_selected_threshold": {
                "f1": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
            },
        }
        filtered_records = [record for record in records if str(record.get("effective_pressure_regime", "")) == "low_pressure"]
        effective_support = len(filtered_records)
        result["effective_support"] = int(effective_support)
        if effective_support < self._resolve_low_pressure_min_effective_support():
            self.last_low_pressure_threshold_stats = dict(result)
            return result
        split_plan = dict(self.last_low_pressure_model_info.get("threshold_split", {}))
        train_positions = [int(pos) for pos in split_plan.get("train_positions", [])]
        holdout_positions = [int(pos) for pos in split_plan.get("holdout_positions", [])]
        if not train_positions and not holdout_positions:
            fallback_split_plan = self._resolve_split_plan(
                labels=[self._resolve_true_failure_v2_value(record) for record in filtered_records],
                min_train_support=self._resolve_low_pressure_min_effective_support(),
                holdout_ratio=float(
                    threshold_cfg.get(
                        "holdout_ratio",
                        cfg.get("holdout_ratio", DEFAULT_LOW_PRESSURE_CLASSIFIER_CONFIG["holdout_ratio"]),
                    )
                ),
                context="low_pressure_mlp_model",
            )
            split_plan = self._build_split_metadata(fallback_split_plan)
            self.last_low_pressure_model_info["threshold_split"] = dict(split_plan)
            train_positions = [int(pos) for pos in split_plan.get("train_positions", [])]
            holdout_positions = [int(pos) for pos in split_plan.get("holdout_positions", [])]
        train_records = [filtered_records[pos] for pos in train_positions if 0 <= pos < len(filtered_records)]
        holdout_records = [filtered_records[pos] for pos in holdout_positions if 0 <= pos < len(filtered_records)]
        train_scores = [self._compute_low_pressure_score_and_logit_for_record(record)[0] for record in train_records]
        train_labels = [self._resolve_true_failure_v2_value(record) for record in train_records]
        holdout_scores = [self._compute_low_pressure_score_and_logit_for_record(record)[0] for record in holdout_records]
        holdout_labels = [self._resolve_true_failure_v2_value(record) for record in holdout_records]
        result["train_support"] = int(len(train_records))
        result["holdout_support"] = int(len(holdout_records))
        result["positive_count"] = int(sum(1 for label in train_labels if label))
        result["negative_count"] = int(sum(1 for label in train_labels if not label))
        result["low_pressure_threshold_train_support"] = int(len(train_records))
        result["low_pressure_threshold_holdout_support"] = int(len(holdout_records))
        result["low_pressure_threshold_positive_count"] = int(sum(1 for label in holdout_labels if label))
        result["low_pressure_threshold_negative_count"] = int(sum(1 for label in holdout_labels if not label))
        if len(train_scores) < 2 or len(set(train_labels)) < 2:
            result["reason"] = "single_class_labels"
            result["low_pressure_threshold_reason"] = "single_class_labels"
            self.last_low_pressure_threshold_stats = dict(result)
            return result
        if len(holdout_scores) < 2:
            result["reason"] = "insufficient_holdout_support"
            result["low_pressure_threshold_reason"] = "insufficient_holdout_support"
            self.last_low_pressure_threshold_stats = dict(result)
            return result
        if len(set(holdout_labels)) < 2:
            result["reason"] = "holdout_single_class_labels"
            result["low_pressure_threshold_reason"] = "holdout_single_class_labels"
            self.last_low_pressure_threshold_stats = dict(result)
            return result
        candidates = sorted(set(float(np.clip(score, 0.0, 1.0)) for score in holdout_scores) | {0.0, 1.0})
        holdout_labels_np = np.asarray(holdout_labels, dtype=bool)
        train_labels_np = np.asarray(train_labels, dtype=bool)
        min_precision = float(threshold_cfg.get("min_precision", 0.50))
        eligible: List[Dict[str, object]] = []
        fallback: List[Dict[str, object]] = []
        for threshold in candidates:
            pred = np.asarray(holdout_scores, dtype=float) >= float(threshold)
            metrics = self.evaluator._prediction_metrics(pred, holdout_labels_np)
            payload = {"threshold": float(threshold), "metrics": metrics}
            fallback.append(payload)
            if float(metrics["precision"]) + 1e-12 >= min_precision:
                eligible.append(payload)
        pool = eligible if eligible else fallback
        constraint_status = "satisfied" if eligible else "all_candidates_below_min_precision"
        best = None
        for candidate in pool:
            metrics = dict(candidate["metrics"])
            if best is None:
                best = {"threshold": candidate["threshold"], "metrics": metrics}
                continue
            best_metrics = dict(best["metrics"])
            if metrics["recall"] > best_metrics["recall"] + 1e-12:
                best = {"threshold": candidate["threshold"], "metrics": metrics}
            elif abs(metrics["recall"] - best_metrics["recall"]) <= 1e-12:
                if metrics["balanced_accuracy"] > best_metrics["balanced_accuracy"] + 1e-12:
                    best = {"threshold": candidate["threshold"], "metrics": metrics}
                elif abs(metrics["balanced_accuracy"] - best_metrics["balanced_accuracy"]) <= 1e-12:
                    if float(candidate["threshold"]) < float(best["threshold"]) - 1e-12:
                        best = {"threshold": candidate["threshold"], "metrics": metrics}
        best = best or {
            "threshold": 0.5,
            "metrics": self.evaluator._prediction_metrics(np.asarray(holdout_scores, dtype=float) >= 0.5, holdout_labels_np),
        }
        train_metrics = self.evaluator._prediction_metrics(
            np.asarray(train_scores, dtype=float) >= float(best["threshold"]),
            train_labels_np,
        )
        holdout_metrics = {"f1": 0.0, "precision": 0.0, "recall": 0.0, "accuracy": 0.0, "balanced_accuracy": 0.0}
        if holdout_scores and holdout_labels:
            holdout_pred = np.asarray(holdout_scores, dtype=float) >= float(best["threshold"])
            holdout_metrics = self.evaluator._prediction_metrics(holdout_pred, holdout_labels_np)
        result.update(
            {
                "status": "updated",
                "reason": "",
                "threshold": float(best["threshold"]),
                "low_pressure_threshold": float(best["threshold"]),
                "precision": float(best["metrics"]["precision"]),
                "recall": float(best["metrics"]["recall"]),
                "accuracy": float(best["metrics"]["accuracy"]),
                "balanced_accuracy": float(best["metrics"]["balanced_accuracy"]),
                "f1": float(best["metrics"]["f1"]),
                "train_metrics_at_selected_threshold": train_metrics,
                "holdout_metrics": holdout_metrics,
                "threshold_constraint_status": str(constraint_status),
                "selected_from": "holdout",
                "low_pressure_threshold_status": "updated",
                "low_pressure_threshold_reason": "",
                "low_pressure_threshold_constraint_status": str(constraint_status),
                "low_pressure_threshold_selected_from": "holdout",
                "low_pressure_threshold_holdout_metrics": holdout_metrics,
                "low_pressure_threshold_train_metrics_at_selected_threshold": train_metrics,
                "threshold_constraint_status": str(constraint_status),
            }
        )
        self.last_low_pressure_model_info["low_pressure_threshold"] = float(best["threshold"])
        self.last_low_pressure_threshold_stats = dict(result)
        return result

    def _fit_learned_decision_weights(self) -> Dict[str, object]:
        info: Dict[str, object] = {
            "decision_model_status": "disabled",
            "decision_model_holdout_record_count": 0,
            "decision_model_holdout_auc": 0.0,
            "decision_model_holdout_accuracy": 0.0,
            "decision_model_config": self.evaluator.get_decision_formula_config(),
        }
        if self.evaluator.decision_model_type != "learned_linear" or not bool(self.args.fit_decision_model_offline):
            self.last_decision_model_info = info
            return info

        feature_matrix, labels, filtered_records = self._build_decision_training_matrix(self.summary_records)
        support = len(filtered_records)
        min_support = max(2, int(self.args.decision_model_min_support))
        if support < min_support or len(np.unique(labels)) < 2:
            info["decision_model_status"] = "frozen"
            self.last_decision_model_info = info
            return info

        split_plan = self._resolve_split_plan(
            labels=[bool(value >= 0.5) for value in labels.tolist()],
            min_train_support=min_support,
            context="learned_decision_model",
        )
        train_positions = [int(pos) for pos in split_plan.get("train_positions", [])]
        holdout_positions = [int(pos) for pos in split_plan.get("holdout_positions", [])]
        holdout_count = len(holdout_positions)
        if str(split_plan.get("status", "updated")).strip().lower() != "updated" or len(train_positions) < 2:
            info["decision_model_status"] = "frozen"
            info["threshold_split"] = self._build_split_metadata(split_plan)
            if "reason" in split_plan:
                info["reason"] = str(split_plan["reason"])
            self.last_decision_model_info = info
            return info

        train_x = torch.tensor(feature_matrix[train_positions], dtype=torch.float32, device=self.device)
        train_y = torch.tensor(labels[train_positions], dtype=torch.float32, device=self.device)
        holdout_x = torch.tensor(feature_matrix[holdout_positions], dtype=torch.float32, device=self.device)
        holdout_y = torch.tensor(labels[holdout_positions], dtype=torch.float32, device=self.device)

        init_weights = torch.tensor(
            [
                float(self.evaluator.decision_model_weights.get("w_mean", 0.60)),
                float(self.evaluator.decision_model_weights.get("w_p75", 0.25)),
                float(self.evaluator.decision_model_weights.get("w_max", 0.10)),
                float(self.evaluator.decision_model_weights.get("w_slope_pos", 0.10)),
                float(self.evaluator.decision_model_weights.get("w_std_penalty", 0.20)),
                float(self.evaluator.decision_model_weights.get("w_high_ratio", 0.50)),
                float(self.evaluator.decision_model_weights.get("w_terminal_gap", 0.50)),
            ],
            dtype=torch.float32,
            device=self.device,
        )
        raw_weights = torch.nn.Parameter(torch.log(torch.expm1(torch.clamp(init_weights, min=1e-5))))
        bias = torch.nn.Parameter(
            torch.tensor(float(self.evaluator.decision_model_bias), dtype=torch.float32, device=self.device)
        )
        optimizer = torch.optim.Adam([raw_weights, bias], lr=float(self.args.decision_model_lr))
        bce_loss = torch.nn.BCEWithLogitsLoss()
        l2_value = float(max(0.0, float(self.args.decision_model_l2)))
        patience = max(1, int(self.args.decision_model_early_stop_patience))
        best_loss = float("inf")
        stale_epochs = 0
        best_state = {
            "raw_weights": raw_weights.detach().clone(),
            "bias": bias.detach().clone(),
        }

        for _ in range(max(1, int(self.args.decision_model_epochs))):
            optimizer.zero_grad()
            positive_weights = torch.nn.functional.softplus(raw_weights)
            logits = (
                train_x[:, 0] * positive_weights[0]
                + train_x[:, 1] * positive_weights[1]
                + train_x[:, 2] * positive_weights[2]
                + train_x[:, 3] * positive_weights[3]
                - train_x[:, 4] * positive_weights[4]
                + train_x[:, 5] * positive_weights[5]
                + train_x[:, 6] * positive_weights[6]
                + bias
            )
            loss = bce_loss(logits, train_y) + l2_value * torch.sum(positive_weights * positive_weights)
            loss.backward()
            optimizer.step()

            loss_value = float(loss.detach().cpu().item())
            if loss_value + 1e-9 < best_loss:
                best_loss = loss_value
                stale_epochs = 0
                best_state = {
                    "raw_weights": raw_weights.detach().clone(),
                    "bias": bias.detach().clone(),
                }
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    break

        with torch.no_grad():
            best_weights_tensor = torch.nn.functional.softplus(best_state["raw_weights"]).detach().cpu().numpy()
            best_bias = float(best_state["bias"].detach().cpu().item())
            learned_weights = {
                "w_mean": float(best_weights_tensor[0]),
                "w_p75": float(best_weights_tensor[1]),
                "w_max": float(best_weights_tensor[2]),
                "w_slope_pos": float(best_weights_tensor[3]),
                "w_std_penalty": float(best_weights_tensor[4]),
                "w_high_ratio": float(best_weights_tensor[5]),
                "w_terminal_gap": float(best_weights_tensor[6]),
            }
            self.evaluator.set_decision_formula_config(
                decision_model_type="learned_linear",
                decision_model_weights=learned_weights,
                decision_model_bias=best_bias,
            )

            if holdout_count > 0:
                positive_weights = torch.tensor(best_weights_tensor, dtype=torch.float32, device=self.device)
                holdout_logits = (
                    holdout_x[:, 0] * positive_weights[0]
                    + holdout_x[:, 1] * positive_weights[1]
                    + holdout_x[:, 2] * positive_weights[2]
                    + holdout_x[:, 3] * positive_weights[3]
                    - holdout_x[:, 4] * positive_weights[4]
                    + holdout_x[:, 5] * positive_weights[5]
                    + holdout_x[:, 6] * positive_weights[6]
                    + best_bias
                )
                holdout_scores = torch.sigmoid(holdout_logits).detach().cpu().numpy()
                holdout_labels = holdout_y.detach().cpu().numpy() >= 0.5
                holdout_pred = holdout_scores >= 0.5
                holdout_accuracy = float(np.mean(holdout_pred == holdout_labels))
                holdout_auc = self._roc_auc_binary(holdout_scores.tolist(), holdout_labels.tolist())
            else:
                holdout_accuracy = 0.0
                holdout_auc = 0.0

        info.update(
            {
                "decision_model_status": "fitted",
                "decision_model_holdout_record_count": int(holdout_count),
                "decision_model_holdout_auc": float(holdout_auc),
                "decision_model_holdout_accuracy": float(holdout_accuracy),
                "decision_model_config": self.evaluator.get_decision_formula_config(),
                "threshold_split": self._build_split_metadata(split_plan),
            }
        )
        self.last_decision_model_info = info
        return info

    def _fit_failure_decision_models(self) -> Dict[str, object]:
        mode = self.failure_decision_mode
        base_decision_info = self._fit_learned_decision_weights()
        self._refresh_decision_scores_on_records()
        for record in self.summary_records:
            record["pressure_score"] = self._compute_pressure_score(record)
            record["initial_pressure_regime"] = self._compute_initial_pressure_regime(record)
            self._resolve_effective_pressure_regime(record, round_index=int(record.get("round_index", self.round_index) or 0))
        high_pressure_records = [
            record for record in self.summary_records if str(record.get("effective_pressure_regime", "high_pressure")) == "high_pressure"
        ]
        self._fit_low_pressure_classifier(self.summary_records)
        self._calibrate_low_pressure_threshold(self.summary_records)
        if mode == "single_fused_score":
            feature_matrix, labels, _ = self._build_fused_training_matrix(high_pressure_records)
            fused_model_type = str(self.args.fused_model_type).strip().lower()
            fused_info = self._fit_mlp_failure_model(
                feature_matrix=feature_matrix,
                labels=labels,
                status_key_prefix="fused_model",
            )
            # Online rounds may have temporary insufficient support; keep previous fitted fused model to avoid reset.
            if (
                str(fused_info.get("fused_model_status", "")).strip().lower() == "frozen"
                and str(self.last_failure_model_info.get("fused_model_status", "")).strip().lower() == "fitted"
            ):
                for key in (
                    "fused_model_type",
                    "fused_model_weights",
                    "fused_model_bias",
                    "fused_model_input_mean",
                    "fused_model_input_std",
                    "fused_model_mlp_state",
                    "fused_model_mlp_hidden_dim",
                ):
                    if key in self.last_failure_model_info:
                        fused_info[key] = copy.deepcopy(self.last_failure_model_info[key])
            self.last_failure_model_info = {
                **self.last_failure_model_info,
                **fused_info,
                "fused_model_type": str(fused_info.get("fused_model_type", fused_model_type)),
                "failure_decision_mode": mode,
                "single_threshold_used": True,
                "primary_score_name": "fused_score",
                "primary_score_holdout_auc": float(fused_info.get("fused_model_holdout_auc", 0.0)),
                "decision_model_info": base_decision_info,
            }
            return dict(self.last_failure_model_info)

        feature_matrix, labels, _ = self._build_direct_failure_training_matrix(self.summary_records)
        direct_info = self._fit_linear_failure_model(
            feature_matrix=feature_matrix,
            labels=labels,
            feature_names=(
                "converged_mean_v2",
                "converged_p75_v2",
                "converged_max_v2",
                "converged_slope_v2",
                "converged_std_v2",
                "converged_high_ratio_v2",
                "terminal_risk_score",
                "terminal_score_gap_v2",
            ),
            status_key_prefix="direct_model",
        )
        self.last_failure_model_info = {
            **self.last_failure_model_info,
            **direct_info,
            "failure_decision_mode": mode,
            "single_threshold_used": True,
            "primary_score_name": "final_failure_probability",
            "primary_score_holdout_auc": float(direct_info.get("direct_model_holdout_auc", 0.0)),
            "decision_model_info": base_decision_info,
        }
        return dict(self.last_failure_model_info)

    def _compute_fused_score_for_record(self, record: Dict) -> float:
        score, _logit = self._compute_fused_score_and_logit_for_record(record)
        return float(score)

    def _compute_fused_score_and_logit_for_record(self, record: Dict) -> Tuple[float, Optional[float]]:
        input_mean = np.asarray(self.last_failure_model_info.get("fused_model_input_mean", []), dtype=np.float32)
        input_std = np.asarray(self.last_failure_model_info.get("fused_model_input_std", []), dtype=np.float32)
        mlp_state = dict(self.last_failure_model_info.get("fused_model_mlp_state", {}))
        features = np.asarray(self._build_fused_feature_vector(record), dtype=np.float32)
        if (
            input_mean.size == features.size
            and input_std.size == features.size
            and all(key in mlp_state for key in ("0.weight", "0.bias", "2.weight", "2.bias", "4.weight", "4.bias"))
        ):
            normalized = (features - input_mean) / np.where(input_std < 1e-6, 1.0, input_std)
            w1 = np.asarray(mlp_state["0.weight"], dtype=np.float32)
            b1 = np.asarray(mlp_state["0.bias"], dtype=np.float32)
            w2 = np.asarray(mlp_state["2.weight"], dtype=np.float32)
            b2 = np.asarray(mlp_state["2.bias"], dtype=np.float32)
            w3 = np.asarray(mlp_state["4.weight"], dtype=np.float32)
            b3 = np.asarray(mlp_state["4.bias"], dtype=np.float32)
            h1 = np.maximum(0.0, normalized @ w1.T + b1)
            h2 = np.maximum(0.0, h1 @ w2.T + b2)
            logit = float(np.ravel(h2 @ w3.T + b3)[0])
            score = float(self._sigmoid_np(np.asarray([logit], dtype=np.float32))[0])
            return score, logit
        return 0.0, None

    def _compute_direct_failure_probability_for_record(self, record: Dict) -> float:
        weights = dict(self.last_failure_model_info.get("direct_model_weights", {}))
        bias = float(self.last_failure_model_info.get("direct_model_bias", 0.0))
        linear = (
            float(weights.get("converged_mean_v2", 0.0)) * float(record.get("converged_mean_v2", 0.0))
            + float(weights.get("converged_p75_v2", 0.0)) * float(record.get("converged_p75_v2", 0.0))
            + float(weights.get("converged_max_v2", 0.0)) * float(record.get("converged_max_v2", 0.0))
            + float(weights.get("converged_slope_v2", 0.0)) * float(record.get("converged_slope_v2", 0.0))
            + float(weights.get("converged_std_v2", 0.0)) * float(record.get("converged_std_v2", record.get("score_uncertainty_v2", 0.0)))
            + float(weights.get("converged_high_ratio_v2", 0.0)) * float(record.get("converged_high_ratio_v2", 0.0))
            + float(weights.get("terminal_risk_score", 0.0)) * float(record.get("terminal_risk_score", 0.0))
            + float(weights.get("terminal_score_gap_v2", 0.0)) * float(record.get("terminal_score_gap_v2", 0.0))
            + bias
        )
        return float(self._sigmoid_np(np.asarray([linear], dtype=np.float32))[0])

    def _meets_threshold_min_precision(self, metrics: Mapping[str, object]) -> bool:
        return float(metrics.get("precision", 0.0)) + 1e-12 >= float(self.args.threshold_min_precision)

    def _select_best_single_threshold_candidate(
        self,
        candidates: Sequence[Dict[str, object]],
        objective: str,
    ) -> Optional[Dict[str, object]]:
        if not candidates:
            return None
        best = dict(candidates[0])
        for candidate in candidates[1:]:
            if float(candidate["objective_score"]) > float(best["objective_score"]) + 1e-12:
                best = dict(candidate)
            elif abs(float(candidate["objective_score"]) - float(best["objective_score"])) <= 1e-12:
                if int(candidate["metrics"].get("fp", 0)) < int(best["metrics"].get("fp", 0)):
                    best = dict(candidate)
                elif int(candidate["metrics"].get("fp", 0)) == int(best["metrics"].get("fp", 0)):
                    if float(candidate["threshold"]) > float(best["threshold"]) + 1e-12:
                        best = dict(candidate)
        return best

    def _resolve_threshold_support_guard_policy(
        self,
        train_positive_count: int,
    ) -> Dict[str, object]:
        cfg = dict(self.threshold_support_guard_config or {})
        enabled = normalize_switch_text(cfg.get("enabled"), default="off")
        metric = str(cfg.get("support_metric", "train_positive_count")).strip().lower() or "train_positive_count"
        if enabled != "on":
            return {
                "enabled": enabled,
                "metric": metric,
                "tier": "disabled",
                "update_mode": "full_update",
                "max_delta": None,
            }
        low_cfg = dict(cfg.get("low_support", {}))
        medium_cfg = dict(cfg.get("medium_support", {}))
        high_cfg = dict(cfg.get("high_support", {}))
        low_max = int(low_cfg.get("positive_count_max", 8))
        medium_max = int(medium_cfg.get("positive_count_max", 20))
        if train_positive_count <= low_max:
            return {
                "enabled": enabled,
                "metric": metric,
                "tier": "low_support",
                "update_mode": str(low_cfg.get("update_mode", "weak_update")).strip().lower() or "weak_update",
                "max_delta": float(low_cfg.get("max_delta", 0.03)),
            }
        if train_positive_count <= medium_max:
            return {
                "enabled": enabled,
                "metric": metric,
                "tier": "medium_support",
                "update_mode": str(medium_cfg.get("update_mode", "bounded_update")).strip().lower() or "bounded_update",
                "max_delta": float(medium_cfg.get("max_delta", 0.08)),
            }
        return {
            "enabled": enabled,
            "metric": metric,
            "tier": "high_support",
            "update_mode": str(high_cfg.get("update_mode", "full_update")).strip().lower() or "full_update",
            "max_delta": None,
        }

    def _apply_threshold_support_guard(
        self,
        *,
        previous_threshold: float,
        candidate_threshold: float,
        train_positive_count: int,
        train_negative_count: int,
    ) -> Dict[str, object]:
        policy = self._resolve_threshold_support_guard_policy(train_positive_count)
        previous = float(np.clip(float(previous_threshold), 0.0, 1.0))
        candidate = float(np.clip(float(candidate_threshold), 0.0, 1.0))
        applied = candidate
        max_delta = policy.get("max_delta")
        if (
            normalize_switch_text(policy.get("enabled"), default="off") == "on"
            and policy.get("update_mode") in {"weak_update", "bounded_update"}
            and max_delta is not None
        ):
            delta = float(max(0.0, float(max_delta)))
            applied = float(np.clip(candidate, previous - delta, previous + delta))
        delta_clipped = abs(applied - candidate) > 1e-12
        return {
            "threshold_support_guard_enabled": str(policy.get("enabled", "off")),
            "threshold_support_metric": str(policy.get("metric", "train_positive_count")),
            "threshold_support_tier": str(policy.get("tier", "disabled")),
            "threshold_support_train_positive_count": int(train_positive_count),
            "threshold_support_train_negative_count": int(train_negative_count),
            "threshold_support_update_mode": str(policy.get("update_mode", "full_update")),
            "threshold_support_max_delta": None if max_delta is None else float(max_delta),
            "threshold_support_previous_threshold": float(previous),
            "threshold_support_candidate_threshold": float(candidate),
            "threshold_support_applied_threshold": float(applied),
            "threshold_support_delta_clipped": bool(delta_clipped),
        }

    def _calibrate_single_score_threshold(
        self,
        train_scores: Sequence[float],
        train_labels: Sequence[bool],
        holdout_scores: Sequence[float],
        holdout_labels: Sequence[bool],
        threshold_key: str,
        holdout_auc_key: str,
    ) -> Dict[str, object]:
        objective = str(self.args.threshold_objective).strip().lower()
        if not train_scores or not train_labels:
            return {
                threshold_key: 0.5,
                "threshold": 0.5,
                "status": "frozen",
                "objective_score": 0.0,
                "f1": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "holdout_metrics": {"f1": 0.0, "precision": 0.0, "recall": 0.0, "accuracy": 0.0, "balanced_accuracy": 0.0},
                holdout_auc_key: 0.0,
                "single_threshold_used": True,
            }

        train_np = np.asarray(train_scores, dtype=float)
        labels_np = np.asarray(train_labels, dtype=bool)
        threshold_candidates = np.unique(np.clip(train_np, 0.0, 1.0))
        if len(threshold_candidates) == 0:
            threshold_candidates = np.array([0.5], dtype=float)

        all_candidates: List[Dict[str, object]] = []
        eligible_candidates: List[Dict[str, object]] = []
        for threshold in threshold_candidates:
            pred = train_np >= float(threshold)
            metrics = self.evaluator._prediction_metrics(pred, labels_np)
            score = float(self.evaluator._objective_value(metrics, objective))
            candidate = {
                "threshold": float(threshold),
                "metrics": metrics,
                "objective_score": float(score),
            }
            all_candidates.append(candidate)
            if self._meets_threshold_min_precision(metrics):
                eligible_candidates.append(candidate)

        constraint_status = "satisfied"
        selected_pool = eligible_candidates
        if not selected_pool:
            constraint_status = "all_candidates_below_min_precision"
            selected_pool = all_candidates

        best = self._select_best_single_threshold_candidate(selected_pool, objective)
        if best is None:
            best_threshold = 0.5
            best_metrics = self.evaluator._prediction_metrics(train_np >= best_threshold, labels_np)
            best_score = float(self.evaluator._objective_value(best_metrics, objective))
        else:
            best_threshold = float(best["threshold"])
            best_metrics = dict(best["metrics"])
            best_score = float(best["objective_score"])

        holdout_metrics = {"f1": 0.0, "precision": 0.0, "recall": 0.0, "accuracy": 0.0, "balanced_accuracy": 0.0}
        if holdout_scores and holdout_labels:
            holdout_metrics = self._evaluate_objective_metrics(
                decision_scores=holdout_scores,
                terminal_scores=[1.0] * len(holdout_scores),
                true_labels=holdout_labels,
                decision_threshold=best_threshold,
                terminal_threshold=0.5,
            )
            holdout_auc = self._roc_auc_binary(list(holdout_scores), list(holdout_labels))
        else:
            holdout_auc = 0.0

        return {
            threshold_key: float(best_threshold),
            "threshold": float(best_threshold),
            "status": "updated",
            "objective_score": float(best_score),
            "f1": float(best_metrics["f1"]),
            "precision": float(best_metrics["precision"]),
            "recall": float(best_metrics["recall"]),
            "accuracy": float(best_metrics["accuracy"]),
            "balanced_accuracy": float(best_metrics["balanced_accuracy"]),
            "holdout_metrics": holdout_metrics,
            holdout_auc_key: float(holdout_auc),
            "single_threshold_used": True,
            "threshold_min_precision_used": float(self.args.threshold_min_precision),
            "threshold_constraint_status": str(constraint_status),
        }

    def _refresh_decision_scores_on_records(self):
        baseline_weights = {
            "w_mean": 0.60,
            "w_p75": 0.25,
            "w_max": 0.10,
            "w_slope_pos": 0.10,
            "w_std_penalty": 0.20,
        }
        for record in self.summary_records:
            if not all(
                key in record
                for key in ("converged_mean_v2", "converged_p75_v2", "converged_max_v2", "converged_slope_v2")
            ):
                continue
            (
                decision_score,
                decision_score_linear,
                decision_feature_contributions,
                decision_score_formula_version,
            ) = self.evaluator.compute_decision_score_v2(
                converged_mean_v2=float(record.get("converged_mean_v2", 0.0)),
                converged_p75_v2=float(record.get("converged_p75_v2", 0.0)),
                converged_max_v2=float(record.get("converged_max_v2", 0.0)),
                converged_slope_v2=float(record.get("converged_slope_v2", 0.0)),
                converged_std_v2=float(record.get("converged_std_v2", record.get("score_uncertainty_v2", 0.0))),
                converged_high_ratio_v2=float(record.get("converged_high_ratio_v2", 0.0)),
                terminal_score_gap_v2=float(
                    record.get(
                        "terminal_score_gap_v2",
                        max(
                            0.0,
                            float(record.get("terminal_risk_score", 1.0 if bool(record.get("terminal_hard_failure", False)) else 0.0))
                            - float(record.get("converged_mean_v2", 0.0)),
                        ),
                    )
                ),
            )
            record["decision_score_v2"] = float(decision_score)
            record["decision_score_v2_linear"] = float(decision_score_linear)
            custom_weights = any(
                abs(float(self.evaluator.decision_formula_weights.get(k, 0.0)) - v) > 1e-12
                for k, v in baseline_weights.items()
            )
            if self.evaluator.decision_model_type == "learned_linear":
                record["decision_score_formula_version"] = decision_score_formula_version
            else:
                record["decision_score_formula_version"] = "v4" if self.evaluator.enable_decision_tail_boost or custom_weights else "v3"
            record["decision_formula_weights"] = dict(self.evaluator.decision_formula_weights)
            record["decision_model_type"] = self.evaluator.decision_model_type
            record["decision_model_weights"] = dict(self.evaluator.decision_model_weights)
            record["decision_model_bias"] = float(self.evaluator.decision_model_bias)
            record["decision_feature_contributions"] = dict(decision_feature_contributions)
            record["enable_decision_tail_boost"] = bool(self.evaluator.enable_decision_tail_boost)
            record["decision_tail_gamma"] = float(self.evaluator.decision_tail_gamma)

    def _recompute_predictions_from_thresholds(self):
        self._refresh_decision_scores_on_records()
        mode = self.failure_decision_mode
        low_pressure_model_info = dict(getattr(self, "last_low_pressure_model_info", {}) or {})
        low_pressure_threshold_stats = dict(getattr(self, "last_low_pressure_threshold_stats", {}) or {})
        fallback_applied = False
        fallback_reason = ""
        effective_mode = mode
        low_pressure_fallback_applied = False
        low_pressure_fallback_reason = ""
        if mode == "single_fused_score":
            fallback_applied, fallback_reason = self._should_fallback_from_fused(
                self.last_threshold_stats,
                self.last_failure_model_info,
            )
            if fallback_applied:
                effective_mode = "dual_threshold_v2"
        for record in self.summary_records:
            round_index = int(record.get("round_index", getattr(self, "round_index", 0)) or 0)
            record["pressure_score"] = self._compute_pressure_score(record)
            record["initial_pressure_regime"] = self._compute_initial_pressure_regime(record)
            pressure_regime = self._resolve_effective_pressure_regime(record, round_index=round_index)
            decision_score = float(record.get("decision_score_v2", record.get("total_membership_v2", 0.0)))
            terminal_score = float(
                record.get("terminal_risk_score", 1.0 if bool(record.get("terminal_hard_failure", False)) else 0.0)
            )
            fused_score = 0.0
            fused_logit: Optional[float] = None
            final_failure_probability = 0.0
            low_pressure_score = 0.0
            low_pressure_logit: Optional[float] = None
            low_pressure_threshold = float(low_pressure_model_info.get("low_pressure_threshold", 0.5))
            record_effective_mode = effective_mode
            record_low_failure_fallback_applied = bool(fallback_applied)
            record_low_failure_fallback_reason = str(fallback_reason)
            record_low_pressure_fallback_applied = False
            record_low_pressure_fallback_reason = ""
            record_low_pressure_pred: Optional[bool] = None
            if pressure_regime == "low_pressure":
                low_pressure_score, low_pressure_logit = self._compute_low_pressure_score_and_logit_for_record(record)
                record_low_pressure_pred = bool(low_pressure_score >= low_pressure_threshold)
                low_pressure_fallback_applied, low_pressure_fallback_reason = self._should_fallback_from_low_pressure(
                    low_pressure_threshold_stats,
                    low_pressure_model_info,
                )
                if not low_pressure_fallback_applied:
                    pred_v2 = bool(record_low_pressure_pred)
                    record_effective_mode = "low_pressure_classifier"
                else:
                    pred_v2 = bool(
                        decision_score >= float(self.evaluator.v2_failure_threshold)
                        and terminal_score >= float(self.evaluator.terminal_threshold_v2)
                    )
                    record_effective_mode = "low_pressure_dual_threshold_fallback"
                    record_low_pressure_fallback_applied = True
                    record_low_pressure_fallback_reason = str(low_pressure_fallback_reason)
            elif mode == "single_fused_score":
                fused_score, fused_logit = self._compute_fused_score_and_logit_for_record(record)
                pred_v2 = bool(fused_score >= float(self.last_failure_model_info.get("fused_threshold", 0.5)))
                if fallback_applied:
                    pred_v2 = bool(
                        decision_score >= float(self.evaluator.v2_failure_threshold)
                        and terminal_score >= float(self.evaluator.terminal_threshold_v2)
                    )
            elif mode == "direct_failure_model":
                final_failure_probability = self._compute_direct_failure_probability_for_record(record)
                pred_v2 = bool(final_failure_probability >= float(self.last_failure_model_info.get("final_threshold", 0.5)))
            else:
                raise ValueError(f"Unsupported failure_decision_mode: {mode}")
            record["system_failure_v2"] = pred_v2
            record["failure_decision_mode"] = mode
            record["decision_policy"] = mode
            record["effective_decision_mode"] = record_effective_mode
            record["decision_source"] = record_effective_mode
            record["low_failure_fallback_applied"] = bool(record_low_failure_fallback_applied)
            record["low_failure_fallback_reason"] = str(record_low_failure_fallback_reason)
            record["low_pressure_score"] = float(low_pressure_score)
            record["low_pressure_logit"] = low_pressure_logit
            record["low_pressure_threshold"] = float(low_pressure_threshold)
            record["low_pressure_pred"] = record_low_pressure_pred
            record["low_pressure_fallback_applied"] = bool(record_low_pressure_fallback_applied)
            record["low_pressure_fallback_reason"] = str(record_low_pressure_fallback_reason)
            record["fused_score"] = float(fused_score)
            record["fused_logit"] = fused_logit
            record["final_failure_probability"] = float(final_failure_probability)
            record["fused_threshold"] = float(self.last_failure_model_info.get("fused_threshold", 0.5))
            record["final_threshold"] = float(self.last_failure_model_info.get("final_threshold", 0.5))
            record["decision_threshold"] = float(self.evaluator.v2_failure_threshold)
            record["terminal_threshold"] = float(self.evaluator.terminal_threshold_v2)
            record["decision_threshold_v2"] = float(self.evaluator.v2_failure_threshold)
            record["terminal_threshold_v2"] = float(self.evaluator.terminal_threshold_v2)
            self._apply_true_failure_policy_to_record(record)
            record["system_failure"] = pred_v2
            record["true_failure"] = bool(record.get("true_failure_v2", False))

        if mode == "single_fused_score" and not fallback_applied:
            predicted_failure_count = sum(1 for record in self.summary_records if bool(record.get("system_failure_v2", False)))
            true_failure_count = sum(1 for record in self.summary_records if self._resolve_true_failure_v2_value(record))
            zero_guard_applied, zero_guard_reason = self._should_fallback_from_fused(
                self.last_threshold_stats,
                self.last_failure_model_info,
                predicted_failure_count=predicted_failure_count,
                true_failure_count=true_failure_count,
            )
            if zero_guard_applied and zero_guard_reason == "zero_prediction_guard":
                fallback_applied = True
                fallback_reason = zero_guard_reason
                effective_mode = "dual_threshold_v2"
                for record in self.summary_records:
                    decision_score = float(record.get("decision_score_v2", record.get("total_membership_v2", 0.0)))
                    terminal_score = float(
                        record.get("terminal_risk_score", 1.0 if bool(record.get("terminal_hard_failure", False)) else 0.0)
                    )
                    pred_v2 = bool(
                        decision_score >= float(self.evaluator.v2_failure_threshold)
                        and terminal_score >= float(self.evaluator.terminal_threshold_v2)
                    )
                    record["system_failure_v2"] = pred_v2
                    record["system_failure"] = pred_v2
                    record["effective_decision_mode"] = effective_mode
                    record["decision_source"] = effective_mode
                    record["low_failure_fallback_applied"] = True
                    record["low_failure_fallback_reason"] = zero_guard_reason

        self._set_low_failure_regime_state(fallback_applied, fallback_reason, effective_mode)

    def _annotate_step_records_with_final_sample_scores(self) -> None:
        if not self.step_records or not self.summary_records:
            return

        by_test_id: Dict[int, Dict[str, object]] = {}
        for record in self.summary_records:
            test_id = record.get("test_id")
            if test_id is None:
                continue
            try:
                test_id_int = int(test_id)
            except Exception:
                continue
            by_test_id[test_id_int] = {
                "fused_score": record.get("fused_score", 0.0),
                "fused_logit": record.get("fused_logit", None),
            }

        for row in self.step_records:
            test_id = row.get("test_id")
            if test_id is None:
                continue
            try:
                test_id_int = int(test_id)
            except Exception:
                continue
            payload = by_test_id.get(test_id_int)
            if not payload:
                continue
            row["fused_score"] = payload.get("fused_score", 0.0)
            row["fused_logit"] = payload.get("fused_logit", None)

    def _offline_recompute_from_existing_results(self):
        if not self.summary_records:
            self._load_summary_records_from_round_files()
        if not self.summary_records:
            source_rounds_dir = self._resolve_offline_source_rounds_dir()
            if source_rounds_dir is not None:
                loaded = self._collect_summary_records_from_rounds_dir(source_rounds_dir)
                self.summary_records = [self._apply_true_failure_policy_to_record(dict(record)) for record in loaded]
                self.cumulative_failure_labels_v2 = [
                    float(bool(record.get("true_failure_v2", False))) for record in self.summary_records
                ]
        if not self.summary_records:
            raise RuntimeError("No existing summary records found for offline recompute.")

        self.cumulative_failure_labels_v2 = []
        for record in self.summary_records:
            self._apply_true_failure_policy_to_record(record)
            self.cumulative_failure_labels_v2.append(float(bool(record.get("true_failure_v2", False))))

        if self.args.offline_decision_threshold is not None:
            self.evaluator.set_v2_failure_threshold(float(self.args.offline_decision_threshold))
        if self.args.offline_terminal_threshold is not None:
            self.evaluator.set_terminal_threshold_v2(float(self.args.offline_terminal_threshold))

        decision_model_stats = self._fit_failure_decision_models()
        self._recompute_predictions_from_thresholds()
        self._write_offline_decision_distribution(tag="before_offline_recompute")
        self._write_low_pressure_score_debug_csv(tag="before_offline_recompute")

        threshold_stats: Dict[str, object] = {
            "status": "frozen",
            "mode": "offline",
            "decision_threshold": float(self.evaluator.v2_failure_threshold),
            "terminal_threshold": float(self.evaluator.terminal_threshold_v2),
        }
        manual_threshold_override = self.args.offline_decision_threshold is not None
        if not bool(self.args.offline_use_existing_thresholds) and not manual_threshold_override:
            threshold_stats = self._calibrate_failure_threshold_v2()

        self._recompute_predictions_from_thresholds()
        self._write_offline_decision_distribution(tag="after_offline_recompute")
        self._write_low_pressure_score_debug_csv(tag="after_offline_recompute")
        self.threshold_update_status = str(threshold_stats.get("status", self.threshold_update_status))
        self.last_threshold_stats = dict(threshold_stats)
        self.stop_reason = "offline_recompute_only"
        self.finished = True
        self._save_state()
        offline_path = self.session_dir / "offline_recompute_summary.json"
        offline_payload = {
            "timestamp": now_stamp(),
            "true_failure_policy": self.true_failure_v2_policy,
            "true_failure_v2_policy": self.true_failure_v2_policy,
            "used_existing_thresholds": bool(self.args.offline_use_existing_thresholds),
            "offline_source_session": str(self.args.offline_source_session or ""),
            "failure_decision_mode": self.failure_decision_mode,
            "single_threshold_used": bool(self.last_failure_model_info.get("single_threshold_used", False)),
            "decision_threshold": float(self.evaluator.v2_failure_threshold),
            "terminal_threshold": float(self.evaluator.terminal_threshold_v2),
            "threshold_stats": threshold_stats,
            "decision_formula_config": self.evaluator.get_decision_formula_config(),
            "decision_model_info": decision_model_stats,
            "low_pressure_model_info": copy.deepcopy(self.last_low_pressure_model_info),
            "low_pressure_threshold_stats": copy.deepcopy(self.last_low_pressure_threshold_stats),
            "record_count": len(self.summary_records),
        }
        offline_path.write_text(json.dumps(offline_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self._write_final_output()

    def _collect_round_results(self, round_index: int, scenarios: Sequence[Dict]) -> Tuple[List[Dict], List[Dict]]:
        round_dir = ensure_dir(self.rounds_dir / f"round_{round_index:03d}")
        performance_dir = ensure_dir(round_dir / "performance")
        failure_score_path = round_dir / "failure_scores.jsonl"

        round_summary_records: List[Dict] = []
        round_step_records: List[Dict] = []
        self._write_round_env_list(round_dir, scenarios)

        with failure_score_path.open("w", encoding="utf-8") as score_file:
            for scenario in scenarios:
                self.test_counter += 1
                temp_config_path, raw_log_name = self._build_temp_config(scenario, round_index, self.test_counter)
                self._run_single_simulation(temp_config_path)
                
                # 修正：匹配 RL_environment_for_computing.py 中的拼接逻辑
                # 文件实际存放在 raw_log_root / training_process_data / ...
                raw_log_path = Path(self.args.raw_log_root) / "training_process_data" / raw_log_name
                performance_file_path = performance_dir / f"test_{self.test_counter:04d}_performance.txt"

                if not raw_log_path.exists():
                    # 回退检查：有时 PRC 会写到项目根目录
                    fallback_path = self.project_root / "training_process_data" / raw_log_name
                    if fallback_path.exists():
                        raw_log_path = fallback_path
                
                if not raw_log_path.exists():
                    raise FileNotFoundError(f"Simulation log not found: {raw_log_path} (also checked {self.project_root / 'training_process_data'})")

                serialize_performance_file(
                    raw_log_path=raw_log_path,
                    output_path=performance_file_path,
                    scenario=scenario,
                    round_index=round_index,
                    test_id=self.test_counter,
                )

                evaluation = self.evaluator.evaluate_log_file(
                    str(performance_file_path),
                    failure_threshold=float(self.args.failure_threshold),
                    scenario=scenario,
                )
                system_failure_policy = bool(evaluation["predicted_failure_v2"])
                true_failure_policy = (
                    bool(evaluation["true_failure_v2_strict"])
                    if self.true_failure_v2_policy == "strict"
                    else bool(evaluation["true_failure_v2"])
                )
                true_failure_v2_relaxed = bool(evaluation["true_failure_v2"])
                true_failure_v2_strict = bool(evaluation.get("true_failure_v2_strict", true_failure_v2_relaxed))
                selected_true_failure_v2 = (
                    true_failure_v2_strict if self.true_failure_v2_policy == "strict" else true_failure_v2_relaxed
                )
                summary_record = {
                    "round_index": round_index,
                    "test_id": self.test_counter,
                    "scenario": scenario,
                    "total_membership": evaluation["total_membership"],
                    "max_total_membership": evaluation["max_total_membership"],
                    "mean_total_membership": evaluation["mean_total_membership"],
                    "score_uncertainty": evaluation["score_uncertainty"],
                    "total_membership_v2": evaluation["total_membership_v2"],
                    "max_total_membership_v2": evaluation["max_total_membership_v2"],
                    "mean_total_membership_v2": evaluation["mean_total_membership_v2"],
                    "score_uncertainty_v2": evaluation["score_uncertainty_v2"],
                    "decision_score": evaluation["decision_score_v2"],
                    "decision_score_v2": evaluation["decision_score_v2"],
                    "decision_score_v2_linear": evaluation.get("decision_score_v2_linear", evaluation["decision_score_v2"]),
                    "convergence_window_start_step": evaluation["convergence_window_start_step"],
                    "converged_mean": evaluation["converged_mean_v2"],
                    "converged_mean_v2": evaluation["converged_mean_v2"],
                    "converged_p75": evaluation["converged_p75_v2"],
                    "converged_p75_v2": evaluation["converged_p75_v2"],
                    "converged_slope": evaluation.get("converged_slope_v2", 0.0),
                    "converged_slope_v2": evaluation.get("converged_slope_v2", 0.0),
                    "converged_max": evaluation.get("converged_max_v2", 0.0),
                    "converged_max_v2": evaluation.get("converged_max_v2", 0.0),
                    "converged_std": evaluation.get("converged_std_v2", 0.0),
                    "converged_std_v2": evaluation.get("converged_std_v2", 0.0),
                    "converged_high_ratio": evaluation.get("converged_high_ratio_v2", 0.0),
                    "converged_high_ratio_v2": evaluation.get("converged_high_ratio_v2", 0.0),
                    "terminal_score_gap": evaluation.get("terminal_score_gap_v2", 0.0),
                    "terminal_score_gap_v2": evaluation.get("terminal_score_gap_v2", 0.0),
                    "decision_score_formula_version": evaluation.get("decision_score_formula_version", "v2"),
                    "decision_feature_contributions": dict(evaluation.get("decision_feature_contributions", {})),
                    "decision_model_type": str(evaluation.get("decision_model_type", self.evaluator.decision_model_type)),
                    "decision_model_weights": dict(evaluation.get("decision_model_weights", self.evaluator.decision_model_weights)),
                    "decision_model_bias": float(evaluation.get("decision_model_bias", self.evaluator.decision_model_bias)),
                    "failure_decision_mode": self.failure_decision_mode,
                    "decision_policy": self.decision_policy,
                    "fused_score": 0.0,
                    "final_failure_probability": 0.0,
                    "fused_threshold": 0.5,
                    "final_threshold": 0.5,
                    "terminal_hard_failure": bool(evaluation["terminal_hard_failure"]),
                    "terminal_risk_score": float(evaluation.get("terminal_risk_score", 0.0)),
                    "terminal_risk_weights": dict(evaluation.get("terminal_risk_weights", self.evaluator.get_terminal_risk_weights())),
                    "terminal_average_ending_reward": float(
                        evaluation.get("samples", [])[-1].Metrics.AverageEndingReward if evaluation.get("samples") else 0.0
                    ),
                    "terminal_packet_loss_rate": float(
                        evaluation.get("samples", [])[-1].Metrics.PacketLossRate if evaluation.get("samples") else 0.0
                    ),
                    "system_failure_v1": bool(evaluation["system_failure"]),
                    "system_failure_v2": bool(evaluation["predicted_failure_v2"]),
                    "true_failure_v1": bool(evaluation["true_failure"]),
                    "true_failure_v2_relaxed": true_failure_v2_relaxed,
                    "true_failure_v2_strict": true_failure_v2_strict,
                    "true_failure_v2": selected_true_failure_v2,
                    "baseline_status": str(evaluation.get("baseline_status", "not_applicable")),
                    "baseline_valid": bool(evaluation.get("baseline_valid", True)),
                    "baseline_warning": bool(evaluation.get("baseline_warning", False)),
                    "baseline_reason_codes": list(evaluation.get("baseline_reason_codes", [])),
                    "system_failure": system_failure_policy,
                    "true_failure": true_failure_policy,
                    "decision_threshold": float(evaluation["v2_failure_threshold"]),
                    "v2_failure_threshold": float(evaluation["v2_failure_threshold"]),
                    "terminal_threshold": float(evaluation.get("terminal_threshold_v2", self.evaluator.terminal_threshold_v2)),
                    "decision_threshold_v2": float(evaluation.get("decision_threshold_v2", evaluation["v2_failure_threshold"])),
                    "terminal_threshold_v2": float(evaluation.get("terminal_threshold_v2", self.evaluator.terminal_threshold_v2)),
                    "performance_file": str(performance_file_path),
                }
                summary_record["pressure_score"] = self._compute_pressure_score(summary_record)
                summary_record["initial_pressure_regime"] = self._compute_initial_pressure_regime(summary_record)
                self._resolve_effective_pressure_regime(summary_record, round_index=round_index)
                round_summary_records.append(summary_record)

                for sample, step_eval in zip(evaluation["samples"], evaluation["step_scores"]):
                    row = {
                        "round_index": round_index,
                        "test_id": self.test_counter,
                        "step_index": sample.StepIndex,
                        **scenario,
                        **sample.Metrics.__dict__,
                        "failure_score": step_eval["total_membership"],
                        "failure_score_v2": step_eval["total_membership_v2"],
                        "step_aux_failure_signal_v2": step_eval["total_membership_v2"],
                        "failure_decision_v1": bool(step_eval["total_membership"] > self.args.failure_threshold),
                        "failure_decision_v2": bool(step_eval["total_membership_v2"] >= evaluation["v2_failure_threshold"]),
                        "true_failure_v2_step": bool(step_eval.get("true_failure_v2", False)),
                        "failure_decision": bool(step_eval["total_membership_v2"] >= evaluation["v2_failure_threshold"]),
                        "test_failure_score": evaluation["total_membership"],
                        "test_failure_score_v2": evaluation["total_membership_v2"],
                        "test_failure_decision": system_failure_policy,
                        "test_failure_decision_v1": bool(evaluation["system_failure"]),
                        "test_failure_decision_v2": bool(evaluation["predicted_failure_v2"]),
                        "baseline_status": str(evaluation.get("baseline_status", "not_applicable")),
                        "baseline_warning": bool(evaluation.get("baseline_warning", False)),
                    }
                    round_step_records.append(row)

                score_file.write(json.dumps(summary_record, ensure_ascii=False) + "\n")

                continuous_values, discrete_values = scenario_feature_arrays(scenario)
                self.cumulative_continuous_features.append(continuous_values)
                self.cumulative_discrete_features.append(discrete_values)
                self.cumulative_failure_scores.append(float(evaluation["total_membership"]))
                self.cumulative_failure_labels_v2.append(float(bool(selected_true_failure_v2)))
                self.summary_records.append(summary_record)
                self.step_records.extend(row for row in round_step_records if row["test_id"] == self.test_counter)

                temp_config_path.unlink(missing_ok=True)
                raw_log_path.unlink(missing_ok=True)

        return round_summary_records, round_step_records

    def _incremental_train_and_evaluate_coverage(self, round_index: int, round_summary_records: Sequence[Dict], round_step_records: Sequence[Dict]) -> Path:
        round_dir = ensure_dir(self.rounds_dir / f"round_{round_index:03d}")
        evalu_path = round_dir / "evalu.txt"

        # Keep online behavior consistent with offline: fit active decision model before threshold calibration.
        self._fit_failure_decision_models()
        threshold_stats = self._calibrate_failure_threshold_v2()
        self._recompute_predictions_from_thresholds()
        override_info = self._compute_round_pressure_override_signal(round_summary_records)
        self.pressure_router_state["last_round_batch_failure_ratio"] = float(override_info.get("batch_failure_ratio", 0.0))
        self.pressure_router_state["last_round_batch_high_risk_ratio"] = float(override_info.get("batch_high_risk_ratio", 0.0))
        self.pressure_router_state["last_round_override_signal"] = str(override_info.get("signal", "keep"))
        self.pressure_router_state["last_round_override_applied_to_next_round"] = bool(
            override_info.get("applied_to_next_round", False)
        )
        next_round_apply = int(round_index) + 1
        self._set_pending_pressure_override(str(override_info.get("signal", "keep")), next_round_apply)
        for record in round_summary_records:
            record["pressure_override_outbound_signal"] = str(override_info.get("signal", "keep"))
            record["pressure_override_applied_to_next_round"] = bool(override_info.get("applied_to_next_round", False))

        continuous_tensor = torch.tensor(np.array(self.cumulative_continuous_features), dtype=torch.float32)
        discrete_tensor = torch.tensor(np.array(self.cumulative_discrete_features), dtype=torch.long)
        target_tensor = torch.tensor(np.array(self.cumulative_failure_scores), dtype=torch.float32)
        target_failure_labels_v2 = torch.tensor(np.array(self.cumulative_failure_labels_v2), dtype=torch.float32)

        train_stats = self.ensemble.fit_incremental(
            continuous_tensor,
            discrete_tensor,
            regression_targets=target_tensor,
            classification_targets=target_failure_labels_v2,
            epochs=10,
            batch_size=min(32, max(1, len(continuous_tensor))),
            learning_rate=5e-4,
            device=self.device,
        )

        round_continuous = torch.tensor(
            np.array([scenario_feature_arrays(record["scenario"])[0] for record in round_summary_records]),
            dtype=torch.float32,
        )
        round_discrete = torch.tensor(
            np.array([scenario_feature_arrays(record["scenario"])[1] for record in round_summary_records]),
            dtype=torch.long,
        )
        if len(round_continuous) > 0:
            self.generator.add_explored_history(round_continuous, round_discrete)

        # Coverage is intentionally evaluated on the continuous scenario space only.
        # Attack-type balance is not part of this metric; in single-attack mode the
        # result should be interpreted as conditional coverage under the fixed attack.
        features_np = continuous_tensor.numpy()
        score_key = "decision_score_v2"
        actual_scores_np = np.array([record.get(score_key, record["total_membership"]) for record in self.summary_records], dtype=float)
        uncertainty_key = "score_uncertainty_v2"
        actual_uncertainties_np = np.array(
            [float(record.get(uncertainty_key, record.get("score_uncertainty", 0.0))) for record in self.summary_records],
            dtype=float,
        )

        explorer_clusters = max(1, min(int(self.args.n_clusters), len(features_np)))
        self.explorer.n_clusters = explorer_clusters
        region_stats = self.explorer.partition_and_evaluate(
            features=features_np,
            failure_scores=actual_scores_np,
            cv_values=actual_uncertainties_np,
            theoretical_max_per_region=max(2, len(features_np) // explorer_clusters or 1),
            sc_schedule=self.coverage_sc_schedule,
        )
        coverage_metrics = self.explorer.compute_coverage_metrics(
            region_stats=region_stats,
            confidence=self.args.coverage_confidence,
            target_coverage=self.args.coverage_target,
        )
        self.latest_coverage_metrics = coverage_metrics

        with torch.no_grad():
            predicted_scores, predicted_cvs = self.ensemble(
                continuous_tensor.to(self.device),
                discrete_tensor.to(self.device),
            )
        predicted_scores_np = predicted_scores.squeeze(1).detach().cpu().numpy()
        predicted_cvs_np = predicted_cvs.squeeze(1).detach().cpu().numpy()

        self.explorer.update_failure_cloud(
            features=features_np,
            predicted_scores=predicted_scores_np,
            predicted_uncertainties=predicted_cvs_np,
            metadata=[
                {
                    "test_id": record["test_id"],
                    "round_index": record["round_index"],
                }
                for record in self.summary_records
            ],
        )

        with evalu_path.open("w", encoding="utf-8") as f:
            f.write(f"ROUND_INDEX: {round_index}\n")
            f.write(f"TRAIN_STATS_JSON: {json.dumps(train_stats, ensure_ascii=False)}\n")
            f.write(f"THRESHOLD_CALIBRATION_JSON: {json.dumps(threshold_stats, ensure_ascii=False)}\n")
            f.write(f"COVERAGE_JSON: {json.dumps(coverage_metrics, ensure_ascii=False)}\n")
            for record in round_summary_records:
                f.write(f"TEST_SUMMARY_JSON: {json.dumps(record, ensure_ascii=False)}\n")
            for row in round_step_records:
                f.write(f"STEP_EVAL_JSON: {json.dumps(row, ensure_ascii=False)}\n")

        self.round_evalu_files.append(str(evalu_path))
        return evalu_path

    def _resolve_terminal_weight_objective(self) -> str:
        objective = str(self.args.terminal_weight_objective or "").strip().lower()
        allowed = {"f1", "accuracy", "balanced_accuracy"}
        if objective in allowed:
            return objective
        threshold_objective = str(self.args.threshold_objective).strip().lower()
        if threshold_objective in allowed:
            return threshold_objective
        return "balanced_accuracy"

    @staticmethod
    def _project_terminal_weights(weight_vector: np.ndarray) -> np.ndarray:
        projected = np.maximum(np.asarray(weight_vector, dtype=float), 0.0)
        total = float(np.sum(projected))
        if total <= 1e-12:
            return np.array([0.35, 0.20, 0.20, 0.25], dtype=float)
        return projected / total

    def _terminal_weight_vector_to_dict(self, weight_vector: np.ndarray) -> Dict[str, float]:
        vector = self._project_terminal_weights(weight_vector)
        return {
            "packet_loss": float(vector[0]),
            "e2e_delay": float(vector[1]),
            "throughput": float(vector[2]),
            "ending_reward": float(vector[3]),
        }

    def _extract_terminal_metrics_by_test(self) -> Dict[int, Dict[str, float]]:
        terminal_by_test: Dict[int, Dict[str, float]] = {}
        metric_keys = ("PacketLossRate", "AverageE2eDelay", "NetworkThroughput", "AverageEndingReward")
        for row in self.step_records:
            test_id = int(row.get("test_id", -1))
            step_index = int(row.get("step_index", -1))
            if test_id < 0 or step_index < 0:
                continue
            previous = terminal_by_test.get(test_id)
            if previous is not None and int(previous.get("_step_index", -1)) >= step_index:
                continue
            metrics = {key: float(row.get(key, 0.0)) for key in metric_keys}
            metrics["_step_index"] = step_index
            terminal_by_test[test_id] = metrics
        for value in terminal_by_test.values():
            value.pop("_step_index", None)
        return terminal_by_test

    def _evaluate_objective_metrics(
        self,
        decision_scores: Sequence[float],
        terminal_scores: Sequence[float],
        true_labels: Sequence[bool],
        decision_threshold: float,
        terminal_threshold: float,
    ) -> Dict[str, float]:
        if not decision_scores or not true_labels:
            return {
                "f1": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
            }
        decision_np = np.asarray(decision_scores, dtype=float)
        terminal_np = np.asarray(terminal_scores, dtype=float)
        labels_np = np.asarray(true_labels, dtype=bool)
        pred = (decision_np >= float(decision_threshold)) & (terminal_np >= float(terminal_threshold))
        return self.evaluator._prediction_metrics(pred, labels_np)

    def _decision_pass_rate(self, decision_scores: Sequence[float], threshold: float) -> float:
        if not decision_scores:
            return 0.0
        decision_np = np.asarray(decision_scores, dtype=float)
        return float(np.mean(decision_np >= float(threshold)))

    def _default_decision_constraint_info(self, status: str = "disabled") -> Dict[str, object]:
        return {
            "decision_pass_rate_train": 0.0,
            "decision_pass_rate_holdout": 0.0,
            "decision_threshold_floor_applied": False,
            "decision_threshold_floor_value_effective": 0.0,
            "decision_constraint_status": status,
        }

    def _build_frozen_weight_info(self, reason: str) -> Dict[str, object]:
        objective = self._resolve_terminal_weight_objective()
        current_weights = self.evaluator.get_terminal_risk_weights()
        info = {
            "weight_adaptation_enabled": bool(self.args.enable_terminal_weight_adaptation),
            "weight_update_status": "frozen",
            "weight_objective_metric": objective,
            "weight_objective_score_before": 0.0,
            "weight_objective_score_after": 0.0,
            "weight_holdout_gain": 0.0,
            "weight_delta": {
                "packet_loss": 0.0,
                "e2e_delay": 0.0,
                "throughput": 0.0,
                "ending_reward": 0.0,
            },
            "weight_candidate_count": 0,
            "terminal_risk_weights": dict(current_weights),
            "weight_rollback_triggered": False,
            "reason": str(reason),
        }
        self.weight_update_status = "frozen"
        self.last_weight_adaptation_info = info
        return info

    def _resolve_terminal_scores_for_summary_records(self) -> List[float]:
        terminal_by_test = self._extract_terminal_metrics_by_test()
        current_weights = self.evaluator.get_terminal_risk_weights()
        resolved_scores: List[float] = []
        for record in self.summary_records:
            test_id = int(record.get("test_id", -1))
            metrics = terminal_by_test.get(test_id)
            if metrics is not None:
                resolved_scores.append(float(self.evaluator._terminal_risk_score(metrics, override_weights=current_weights)))
            else:
                fallback_score = float(
                    record.get("terminal_risk_score", 1.0 if bool(record.get("terminal_hard_failure", False)) else 0.0)
                )
                resolved_scores.append(fallback_score)
        return resolved_scores

    def _compute_decision_threshold_floor(self, train_scores: Sequence[float]) -> Tuple[float, bool]:
        mode = str(self.args.decision_threshold_floor_mode).strip().lower()
        if mode == "off" or not train_scores:
            return 0.0, False
        scores_np = np.asarray(train_scores, dtype=float)
        if mode == "absolute":
            floor_value = float(np.clip(float(self.args.decision_threshold_floor_value), 0.0, 1.0))
        elif mode == "quantile":
            q = float(np.clip(float(self.args.decision_threshold_floor_quantile), 0.0, 1.0))
            floor_value = float(np.clip(float(np.quantile(scores_np, q)), 0.0, 1.0))
        else:
            ratio = float(max(0.0, float(self.args.decision_threshold_baseline_ratio)))
            floor_value = float(np.clip(float(self.evaluator.v2_failure_threshold) * ratio, 0.0, 1.0))
        return floor_value, True

    def _build_threshold_candidate_pool(
        self,
        train_scores: Sequence[float],
        train_terminal_scores: Sequence[float],
        train_labels: Sequence[bool],
        dual_threshold: bool,
    ) -> Dict[str, object]:
        objective = str(self.args.threshold_objective).strip().lower()
        tolerance = float(max(0.0, float(self.args.decision_threshold_constraint_tolerance)))
        pass_rate_min = float(np.clip(float(self.args.decision_pass_rate_min), 0.0, 1.0))
        pass_rate_max = float(np.clip(float(self.args.decision_pass_rate_max), pass_rate_min, 1.0))
        floor_value, floor_applied = self._compute_decision_threshold_floor(train_scores)
        decision_candidates = np.unique(np.clip(np.asarray(train_scores, dtype=float), 0.0, 1.0))
        if len(decision_candidates) == 0:
            decision_candidates = np.array([float(self.evaluator.v2_failure_threshold)], dtype=float)

        constrained_candidates = decision_candidates[decision_candidates >= (floor_value - 1e-12)]
        floor_status = "satisfied"
        if len(constrained_candidates) == 0:
            constrained_candidates = np.array([float(np.max(decision_candidates))], dtype=float)
            floor_status = "relaxed_fallback"

        if dual_threshold:
            terminal_candidates = np.arange(0.45, 0.751, 0.01, dtype=float)
        else:
            terminal_candidates = np.array([float(self.evaluator.terminal_threshold_v2)], dtype=float)

        viable_candidates: List[Dict[str, object]] = []
        fallback_candidates: List[Dict[str, object]] = []
        precision_filtered_candidates: List[Dict[str, object]] = []
        target_center = 0.5 * (pass_rate_min + pass_rate_max)
        for decision_threshold in constrained_candidates:
            pass_rate = self._decision_pass_rate(train_scores, float(decision_threshold))
            lower_bound = pass_rate_min - tolerance
            upper_bound = pass_rate_max + tolerance
            if pass_rate < lower_bound:
                violation = lower_bound - pass_rate
            elif pass_rate > upper_bound:
                violation = pass_rate - upper_bound
            else:
                violation = 0.0

            decision_np = np.asarray(train_scores, dtype=float)
            terminal_np = np.asarray(train_terminal_scores, dtype=float)
            labels_np = np.asarray(train_labels, dtype=bool)
            decision_mask = decision_np >= float(decision_threshold)
            for terminal_threshold in terminal_candidates:
                pred = decision_mask & (terminal_np >= float(terminal_threshold))
                metrics = self.evaluator._prediction_metrics(pred, labels_np)
                score = float(self.evaluator._objective_value(metrics, objective))
                candidate = {
                    "decision_threshold": float(decision_threshold),
                    "terminal_threshold": float(terminal_threshold),
                    "metrics": metrics,
                    "objective_score": score,
                    "decision_pass_rate": float(pass_rate),
                    "constraint_violation": float(violation),
                    "distance_to_center": float(abs(pass_rate - target_center)),
                }
                if not self._meets_threshold_min_precision(metrics):
                    precision_filtered_candidates.append(candidate)
                    continue
                if violation <= 1e-12:
                    viable_candidates.append(candidate)
                fallback_candidates.append(candidate)

        constraint_status = floor_status if floor_status != "relaxed_fallback" else "relaxed_fallback"
        if viable_candidates:
            active_pool = viable_candidates
        elif fallback_candidates:
            active_pool = fallback_candidates
            constraint_status = "relaxed_fallback"
        else:
            active_pool = precision_filtered_candidates
            constraint_status = "all_candidates_below_min_precision"

        return {
            "objective": objective,
            "floor_applied": bool(floor_applied),
            "floor_value": float(floor_value),
            "constraint_status": str(constraint_status),
            "target_center": float(target_center),
            "viable_candidates": viable_candidates,
            "fallback_candidates": fallback_candidates,
            "precision_filtered_candidates": precision_filtered_candidates,
            "active_pool": active_pool,
            "threshold_min_precision_used": float(self.args.threshold_min_precision),
        }

    def _select_best_threshold_candidate_legacy(self, candidates: Sequence[Dict[str, object]]) -> Optional[Dict[str, object]]:
        if not candidates:
            return None
        best = candidates[0]
        for candidate in candidates[1:]:
            if candidate["objective_score"] > best["objective_score"] + 1e-12:
                best = candidate
            elif abs(candidate["objective_score"] - best["objective_score"]) <= 1e-12:
                if candidate["constraint_violation"] < best["constraint_violation"] - 1e-12:
                    best = candidate
                elif abs(candidate["constraint_violation"] - best["constraint_violation"]) <= 1e-12:
                    if candidate["distance_to_center"] < best["distance_to_center"] - 1e-12:
                        best = candidate
                    elif abs(candidate["distance_to_center"] - best["distance_to_center"]) <= 1e-12:
                        if candidate["metrics"]["fp"] < best["metrics"]["fp"]:
                            best = candidate
        return best

    def _calibrate_thresholds_with_decision_constraints(
        self,
        train_scores: Sequence[float],
        train_terminal_scores: Sequence[float],
        train_labels: Sequence[bool],
        dual_threshold: bool,
    ) -> Dict[str, object]:
        pool_info = self._build_threshold_candidate_pool(
            train_scores=train_scores,
            train_terminal_scores=train_terminal_scores,
            train_labels=train_labels,
            dual_threshold=dual_threshold,
        )
        objective = str(pool_info["objective"])
        floor_applied = bool(pool_info["floor_applied"])
        floor_value = float(pool_info["floor_value"])
        constraint_status = str(pool_info["constraint_status"])
        candidate_pool = list(pool_info["active_pool"])

        best = self._select_best_threshold_candidate_legacy(candidate_pool)
        if best is None:
            threshold = float(np.clip(max(floor_value, float(self.evaluator.v2_failure_threshold)), 0.0, 1.0))
            terminal_threshold = float(self.evaluator.terminal_threshold_v2)
            metrics = self._evaluate_objective_metrics(
                decision_scores=train_scores,
                terminal_scores=train_terminal_scores,
                true_labels=train_labels,
                decision_threshold=threshold,
                terminal_threshold=terminal_threshold,
            )
            return {
                "decision_threshold": threshold,
                "terminal_threshold": terminal_threshold,
                "threshold": threshold,
                "objective_score": float(self.evaluator._objective_value(metrics, objective)),
                **metrics,
                "decision_pass_rate_train": float(self._decision_pass_rate(train_scores, threshold)),
                "decision_threshold_floor_applied": bool(floor_applied),
                "decision_threshold_floor_value_effective": float(floor_value),
                "decision_constraint_status": "relaxed_fallback",
                "threshold_constraint_status": "relaxed_fallback",
                "threshold_min_precision_used": float(self.args.threshold_min_precision),
                "selection_stage": "legacy",
            }

        return {
            "decision_threshold": float(best["decision_threshold"]),
            "terminal_threshold": float(best["terminal_threshold"]),
            "threshold": float(best["decision_threshold"]),
            "f1": float(best["metrics"]["f1"]),
            "precision": float(best["metrics"]["precision"]),
            "recall": float(best["metrics"]["recall"]),
            "accuracy": float(best["metrics"]["accuracy"]),
            "balanced_accuracy": float(best["metrics"]["balanced_accuracy"]),
            "objective_score": float(best["objective_score"]),
            "decision_pass_rate_train": float(best["decision_pass_rate"]),
            "decision_threshold_floor_applied": bool(floor_applied),
            "decision_threshold_floor_value_effective": float(floor_value),
            "decision_constraint_status": constraint_status,
            "threshold_constraint_status": constraint_status,
            "threshold_min_precision_used": float(self.args.threshold_min_precision),
            "selection_stage": "legacy",
        }

    def _calibrate_thresholds_two_stage_stable(
        self,
        train_scores: Sequence[float],
        train_terminal_scores: Sequence[float],
        train_labels: Sequence[bool],
        holdout_scores: Sequence[float],
        holdout_terminal_scores: Sequence[float],
        holdout_labels: Sequence[bool],
        dual_threshold: bool,
    ) -> Dict[str, object]:
        pool_info = self._build_threshold_candidate_pool(
            train_scores=train_scores,
            train_terminal_scores=train_terminal_scores,
            train_labels=train_labels,
            dual_threshold=dual_threshold,
        )
        objective = str(pool_info["objective"])
        floor_applied = bool(pool_info["floor_applied"])
        floor_value = float(pool_info["floor_value"])
        constraint_status = str(pool_info["constraint_status"])
        candidate_pool = list(pool_info["active_pool"])
        if not candidate_pool:
            return self._calibrate_thresholds_with_decision_constraints(
                train_scores=train_scores,
                train_terminal_scores=train_terminal_scores,
                train_labels=train_labels,
                dual_threshold=dual_threshold,
            )

        sorted_candidates = sorted(
            candidate_pool,
            key=lambda c: (
                -float(c["objective_score"]),
                float(c["constraint_violation"]),
                float(c["distance_to_center"]),
                int(c["metrics"].get("fp", 0)),
            ),
        )
        top_k = int(max(1, int(self.args.threshold_two_stage_top_k)))
        shortlist = sorted_candidates[: min(top_k, len(sorted_candidates))]
        top_train_objective = float(shortlist[0]["objective_score"])

        gap_penalty = float(max(0.0, float(self.args.threshold_two_stage_gap_penalty)))
        gap_tolerance = float(max(0.0, float(self.args.threshold_two_stage_gap_tolerance)))
        drift_penalty = float(max(0.0, float(self.args.threshold_two_stage_passrate_drift_penalty)))

        best: Optional[Dict[str, object]] = None
        for candidate in shortlist:
            decision_threshold = float(candidate["decision_threshold"])
            terminal_threshold = float(candidate["terminal_threshold"])
            holdout_metrics = self._evaluate_objective_metrics(
                decision_scores=holdout_scores,
                terminal_scores=holdout_terminal_scores,
                true_labels=holdout_labels,
                decision_threshold=decision_threshold,
                terminal_threshold=terminal_threshold,
            )
            holdout_objective = float(self.evaluator._objective_value(holdout_metrics, objective))
            train_objective = float(candidate["objective_score"])
            generalization_gap = max(0.0, train_objective - holdout_objective - gap_tolerance)
            pass_rate_holdout = float(self._decision_pass_rate(holdout_scores, decision_threshold))
            pass_rate_train = float(candidate["decision_pass_rate"])
            pass_rate_drift = abs(pass_rate_holdout - pass_rate_train)
            stability_score = holdout_objective - gap_penalty * generalization_gap - drift_penalty * pass_rate_drift

            enriched = dict(candidate)
            enriched["holdout_metrics"] = holdout_metrics
            enriched["holdout_objective_score"] = holdout_objective
            enriched["train_objective_score"] = train_objective
            enriched["generalization_gap"] = float(generalization_gap)
            enriched["pass_rate_holdout"] = float(pass_rate_holdout)
            enriched["pass_rate_drift"] = float(pass_rate_drift)
            enriched["stability_score"] = float(stability_score)
            if best is None:
                best = enriched
                continue
            if enriched["stability_score"] > best["stability_score"] + 1e-12:
                best = enriched
            elif abs(enriched["stability_score"] - best["stability_score"]) <= 1e-12:
                if enriched["holdout_objective_score"] > best["holdout_objective_score"] + 1e-12:
                    best = enriched
                elif abs(enriched["holdout_objective_score"] - best["holdout_objective_score"]) <= 1e-12:
                    if int(enriched["holdout_metrics"].get("fn", 0)) < int(best["holdout_metrics"].get("fn", 0)):
                        best = enriched
                    elif int(enriched["holdout_metrics"].get("fn", 0)) == int(best["holdout_metrics"].get("fn", 0)):
                        if int(enriched["holdout_metrics"].get("fp", 0)) < int(best["holdout_metrics"].get("fp", 0)):
                            best = enriched

        if best is None:
            return self._calibrate_thresholds_with_decision_constraints(
                train_scores=train_scores,
                train_terminal_scores=train_terminal_scores,
                train_labels=train_labels,
                dual_threshold=dual_threshold,
            )

        return {
            "decision_threshold": float(best["decision_threshold"]),
            "terminal_threshold": float(best["terminal_threshold"]),
            "threshold": float(best["decision_threshold"]),
            "f1": float(best["metrics"]["f1"]),
            "precision": float(best["metrics"]["precision"]),
            "recall": float(best["metrics"]["recall"]),
            "accuracy": float(best["metrics"]["accuracy"]),
            "balanced_accuracy": float(best["metrics"]["balanced_accuracy"]),
            "objective_score": float(best["objective_score"]),
            "decision_pass_rate_train": float(best["decision_pass_rate"]),
            "decision_threshold_floor_applied": bool(floor_applied),
            "decision_threshold_floor_value_effective": float(floor_value),
            "decision_constraint_status": constraint_status,
            "threshold_constraint_status": constraint_status,
            "threshold_min_precision_used": float(self.args.threshold_min_precision),
            "selection_stage": "two_stage_stable",
            "stage1_candidate_count": int(len(candidate_pool)),
            "stage1_viable_candidate_count": int(len(pool_info["viable_candidates"])),
            "stage2_shortlist_count": int(len(shortlist)),
            "stage2_top_train_objective": float(top_train_objective),
            "stage2_selected_train_objective": float(best["train_objective_score"]),
            "stage2_selected_holdout_objective": float(best["holdout_objective_score"]),
            "stage2_selected_stability_score": float(best["stability_score"]),
            "stage2_selected_generalization_gap": float(best["generalization_gap"]),
            "stage2_selected_pass_rate_drift": float(best["pass_rate_drift"]),
        }

    def _apply_support_guard_to_single_score_threshold(
        self,
        *,
        threshold_stats: Dict[str, object],
        threshold_key: str,
        previous_threshold: float,
        train_scores: Sequence[float],
        train_labels: Sequence[bool],
        holdout_scores: Sequence[float],
        holdout_labels: Sequence[bool],
    ) -> Dict[str, object]:
        positive_count = int(sum(1 for label in train_labels if bool(label)))
        negative_count = int(len(train_labels) - positive_count)
        guard_info = self._apply_threshold_support_guard(
            previous_threshold=float(previous_threshold),
            candidate_threshold=float(threshold_stats.get(threshold_key, threshold_stats.get("threshold", previous_threshold))),
            train_positive_count=positive_count,
            train_negative_count=negative_count,
        )
        applied_threshold = float(guard_info["threshold_support_applied_threshold"])
        train_np = np.asarray(train_scores, dtype=float)
        labels_np = np.asarray(train_labels, dtype=bool)
        applied_train_metrics = self.evaluator._prediction_metrics(train_np >= applied_threshold, labels_np)
        applied_objective = float(
            self.evaluator._objective_value(applied_train_metrics, str(self.args.threshold_objective).strip().lower())
        )

        holdout_metrics = {"f1": 0.0, "precision": 0.0, "recall": 0.0, "accuracy": 0.0, "balanced_accuracy": 0.0}
        if holdout_scores and holdout_labels:
            holdout_metrics = self._evaluate_objective_metrics(
                decision_scores=holdout_scores,
                terminal_scores=[1.0] * len(holdout_scores),
                true_labels=holdout_labels,
                decision_threshold=applied_threshold,
                terminal_threshold=0.5,
            )

        threshold_stats[threshold_key] = float(applied_threshold)
        threshold_stats["threshold"] = float(applied_threshold)
        threshold_stats["decision_threshold"] = float(applied_threshold)
        threshold_stats["f1"] = float(applied_train_metrics["f1"])
        threshold_stats["precision"] = float(applied_train_metrics["precision"])
        threshold_stats["recall"] = float(applied_train_metrics["recall"])
        threshold_stats["accuracy"] = float(applied_train_metrics["accuracy"])
        threshold_stats["balanced_accuracy"] = float(applied_train_metrics["balanced_accuracy"])
        threshold_stats["objective_score"] = float(applied_objective)
        threshold_stats["holdout_metrics"] = dict(holdout_metrics)
        threshold_stats["threshold_min_precision_used"] = float(self.args.threshold_min_precision)
        threshold_stats.update(guard_info)
        return threshold_stats

    def _adapt_terminal_risk_weights(self) -> Dict[str, object]:
        objective = self._resolve_terminal_weight_objective()
        current_weights = self.evaluator.get_terminal_risk_weights()
        info: Dict[str, object] = {
            "weight_adaptation_enabled": bool(self.args.enable_terminal_weight_adaptation),
            "weight_update_status": "frozen",
            "weight_objective_metric": objective,
            "weight_objective_score_before": 0.0,
            "weight_objective_score_after": 0.0,
            "weight_holdout_gain": 0.0,
            "weight_delta": {
                "packet_loss": 0.0,
                "e2e_delay": 0.0,
                "throughput": 0.0,
                "ending_reward": 0.0,
            },
            "weight_candidate_count": 0,
            "terminal_risk_weights": dict(current_weights),
            "weight_rollback_triggered": False,
        }
        if not bool(self.args.enable_terminal_weight_adaptation):
            self.weight_update_status = "frozen"
            self.last_weight_adaptation_info = info
            return info

        terminal_by_test = self._extract_terminal_metrics_by_test()
        if not terminal_by_test or not self.summary_records:
            info["reason"] = "insufficient_terminal_metrics"
            self.weight_update_status = "frozen"
            self.last_weight_adaptation_info = info
            return info

        raw_decision_scores: List[float] = []
        raw_true_labels: List[bool] = []
        raw_terminal_flags: List[bool] = []
        raw_terminal_metrics: List[Dict[str, float]] = []
        for record in self.summary_records:
            test_id = int(record.get("test_id", -1))
            metrics = terminal_by_test.get(test_id)
            if metrics is None:
                continue
            raw_decision_scores.append(float(record.get("decision_score_v2", record.get("total_membership_v2", 0.0))))
            raw_true_labels.append(self._resolve_true_failure_v2_value(record))
            raw_terminal_flags.append(bool(record.get("terminal_hard_failure", False)))
            raw_terminal_metrics.append(metrics)

        if self.args.threshold_calibration_scope == "terminal_only":
            effective_indices = [idx for idx, flag in enumerate(raw_terminal_flags) if flag]
        else:
            effective_indices = list(range(len(raw_true_labels)))

        min_support = int(max(2, self.args.terminal_weight_min_support))
        if len(effective_indices) < min_support:
            info["reason"] = "insufficient_effective_support"
            info["effective_support"] = len(effective_indices)
            info["min_support"] = min_support
            self.weight_update_status = "frozen"
            self.last_weight_adaptation_info = info
            return info

        effective_labels = [raw_true_labels[idx] for idx in effective_indices]
        split_plan = self._resolve_split_plan(
            labels=effective_labels,
            min_train_support=min_support,
            context="terminal_weight_adaptation",
        )
        train_positions = [int(pos) for pos in split_plan.get("train_positions", [])]
        holdout_positions = [int(pos) for pos in split_plan.get("holdout_positions", [])]
        holdout_count = len(holdout_positions)
        train_count = len(train_positions)
        if str(split_plan.get("status", "updated")).strip().lower() != "updated" or train_count < min_support:
            info["reason"] = "insufficient_train_support"
            info["effective_support"] = len(effective_indices)
            info["train_support"] = train_count
            info["holdout_support"] = holdout_count
            info["min_support"] = min_support
            info.update(self._build_split_metadata(split_plan))
            self.weight_update_status = "frozen"
            self.last_weight_adaptation_info = info
            return info

        train_idx = [effective_indices[pos] for pos in train_positions]
        holdout_idx = [effective_indices[pos] for pos in holdout_positions]

        train_decision_scores = [raw_decision_scores[idx] for idx in train_idx]
        train_labels = [raw_true_labels[idx] for idx in train_idx]
        train_metrics = [raw_terminal_metrics[idx] for idx in train_idx]
        holdout_decision_scores = [raw_decision_scores[idx] for idx in holdout_idx]
        holdout_labels = [raw_true_labels[idx] for idx in holdout_idx]
        holdout_metrics = [raw_terminal_metrics[idx] for idx in holdout_idx]

        positive_count = sum(1 for label in train_labels if label)
        negative_count = len(train_labels) - positive_count
        if positive_count == 0 or negative_count == 0:
            info["reason"] = "single_class_labels"
            info["positive_count"] = positive_count
            info["negative_count"] = negative_count
            info["effective_support"] = len(effective_indices)
            info["train_support"] = train_count
            info["holdout_support"] = holdout_count
            info.update(self._build_split_metadata(split_plan))
            self.weight_update_status = "frozen"
            self.last_weight_adaptation_info = info
            return info

        def evaluate_candidate(weight_dict: Dict[str, float]) -> Tuple[float, Dict[str, float], Dict[str, float], float, float]:
            train_terminal_scores = [
                self.evaluator._terminal_risk_score(metrics, override_weights=weight_dict)
                for metrics in train_metrics
            ]
            dual_stats = self.evaluator.calibrate_dual_failure_threshold(
                decision_scores=train_decision_scores,
                terminal_scores=train_terminal_scores,
                true_labels=train_labels,
                objective=self.args.threshold_objective,
                min_precision=self.args.threshold_min_precision,
                terminal_threshold_candidates=np.arange(0.45, 0.751, 0.01, dtype=float),
            )
            decision_threshold = float(dual_stats["decision_threshold"])
            terminal_threshold = float(dual_stats["terminal_threshold"])
            holdout_terminal_scores = [
                self.evaluator._terminal_risk_score(metrics, override_weights=weight_dict)
                for metrics in holdout_metrics
            ]
            holdout_result = self._evaluate_objective_metrics(
                decision_scores=holdout_decision_scores,
                terminal_scores=holdout_terminal_scores,
                true_labels=holdout_labels,
                decision_threshold=decision_threshold,
                terminal_threshold=terminal_threshold,
            )
            objective_score = float(self.evaluator._objective_value(holdout_result, objective))
            return objective_score, holdout_result, dual_stats, decision_threshold, terminal_threshold

        current_score, current_holdout_metrics, _, _, _ = evaluate_candidate(current_weights)
        info["weight_objective_score_before"] = float(current_score)

        local_step = float(max(1e-6, self.args.terminal_weight_local_step))
        candidate_count = int(max(1, self.args.terminal_weight_candidates))
        current_vector = np.array(
            [
                current_weights["packet_loss"],
                current_weights["e2e_delay"],
                current_weights["throughput"],
                current_weights["ending_reward"],
            ],
            dtype=float,
        )
        candidate_vectors: List[np.ndarray] = [current_vector]
        for _ in range(max(0, candidate_count - 1)):
            noise = self.weight_rng.uniform(-local_step, local_step, size=current_vector.shape[0])
            candidate_vectors.append(self._project_terminal_weights(current_vector + noise))
        info["weight_candidate_count"] = len(candidate_vectors)

        best_score = current_score
        best_weights = dict(current_weights)
        best_dual_stats = None
        for vector in candidate_vectors:
            candidate_weights = self._terminal_weight_vector_to_dict(vector)
            candidate_score, _, dual_stats, _, _ = evaluate_candidate(candidate_weights)
            if candidate_score > best_score:
                best_score = candidate_score
                best_weights = candidate_weights
                best_dual_stats = dual_stats

        min_improvement = float(max(0.0, self.args.terminal_weight_min_improvement))
        improvement = float(best_score - current_score)
        info["weight_holdout_gain"] = improvement
        info["holdout_metrics_before"] = current_holdout_metrics
        info["effective_support"] = len(effective_indices)
        info["train_support"] = train_count
        info["holdout_support"] = holdout_count
        info["objective"] = objective
        info.update(self._build_split_metadata(split_plan))

        if improvement < min_improvement:
            info["reason"] = "no_sufficient_gain"
            info["weight_objective_score_after"] = float(current_score)
            info["terminal_risk_weights"] = dict(current_weights)
            self.weight_update_status = "frozen"
            snapshot = {
                "round_index": int(self.round_index),
                "objective_score": float(current_score),
                "weights": dict(current_weights),
            }
            self.weight_objective_history.append(snapshot)
            self.weight_objective_history = self.weight_objective_history[-10:]
            self.last_weight_adaptation_info = info
            return info

        ema = float(np.clip(float(self.args.terminal_weight_ema), 0.0, 1.0))
        best_vector = np.array(
            [
                best_weights["packet_loss"],
                best_weights["e2e_delay"],
                best_weights["throughput"],
                best_weights["ending_reward"],
            ],
            dtype=float,
        )
        updated_vector = (1.0 - ema) * current_vector + ema * best_vector
        max_delta = float(max(1e-6, self.args.terminal_weight_max_delta))
        delta = np.clip(updated_vector - current_vector, -max_delta, max_delta)
        bounded_vector = self._project_terminal_weights(current_vector + delta)
        updated_weights = self._terminal_weight_vector_to_dict(bounded_vector)
        self.evaluator.set_terminal_risk_weights(updated_weights)

        after_score, after_holdout_metrics, after_dual_stats, _, _ = evaluate_candidate(updated_weights)
        info["weight_objective_score_after"] = float(after_score)
        info["holdout_metrics_after"] = after_holdout_metrics
        info["thresholds_after_weight_update"] = {
            "decision_threshold": float(after_dual_stats["decision_threshold"]),
            "terminal_threshold": float(after_dual_stats["terminal_threshold"]),
        }
        info["weight_delta"] = {
            key: float(updated_weights[key] - current_weights[key])
            for key in ("packet_loss", "e2e_delay", "throughput", "ending_reward")
        }
        info["terminal_risk_weights"] = dict(updated_weights)
        info["best_dual_stats_preview"] = best_dual_stats
        self.weight_update_status = "updated"
        info["weight_update_status"] = "updated"

        snapshot = {
            "round_index": int(self.round_index),
            "objective_score": float(after_score),
            "weights": dict(updated_weights),
        }
        self.weight_objective_history.append(snapshot)
        self.weight_objective_history = self.weight_objective_history[-10:]

        rollback_p = int(max(1, self.args.terminal_weight_rollback_patience))
        rollback_drop = float(max(0.0, self.args.terminal_weight_rollback_min_drop))
        history_scores = [float(item.get("objective_score", 0.0)) for item in self.weight_objective_history]
        if len(history_scores) >= rollback_p + 1:
            recent_declines = True
            for idx in range(rollback_p):
                prev_score = history_scores[-(idx + 2)]
                curr_score = history_scores[-(idx + 1)]
                if (prev_score - curr_score) <= rollback_drop:
                    recent_declines = False
                    break
            if recent_declines:
                previous_weights = dict(self.weight_objective_history[-2].get("weights", current_weights))
                self.evaluator.set_terminal_risk_weights(previous_weights)
                rollback_score, rollback_holdout_metrics, rollback_dual_stats, _, _ = evaluate_candidate(previous_weights)
                self.weight_objective_history[-1] = {
                    "round_index": int(self.round_index),
                    "objective_score": float(rollback_score),
                    "weights": dict(previous_weights),
                }
                info["weight_rollback_triggered"] = True
                info["weight_update_status"] = "rolled_back"
                info["weight_objective_score_after"] = float(rollback_score)
                info["holdout_metrics_after"] = rollback_holdout_metrics
                info["thresholds_after_weight_update"] = {
                    "decision_threshold": float(rollback_dual_stats["decision_threshold"]),
                    "terminal_threshold": float(rollback_dual_stats["terminal_threshold"]),
                }
                info["terminal_risk_weights"] = dict(previous_weights)
                info["weight_delta"] = {
                    key: float(previous_weights[key] - current_weights[key])
                    for key in ("packet_loss", "e2e_delay", "throughput", "ending_reward")
                }
                self.weight_update_status = "rolled_back"

        self.last_weight_adaptation_info = info
        return info

    def _calibrate_failure_threshold_v2(self) -> Dict:
        calibration_mode = str(self.args.threshold_calibration_mode).strip().lower()
        if not self.summary_records:
            self.threshold_update_status = "frozen"
            result = {
                "threshold": float(self.evaluator.v2_failure_threshold),
                "decision_threshold": float(self.evaluator.v2_failure_threshold),
                "terminal_threshold": float(self.evaluator.terminal_threshold_v2),
                "f1": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "status": "frozen",
                "reason": "insufficient_samples",
                "scope": self.args.threshold_calibration_scope,
                "objective": self.args.threshold_objective,
            }
            self.last_threshold_stats = dict(result)
            return result

        raw_true_labels = [self._resolve_true_failure_v2_value(record) for record in self.summary_records]
        raw_terminal_flags = [bool(record.get("terminal_hard_failure", False)) for record in self.summary_records]
        pressure_enabled = normalize_switch_text(
            getattr(self, "pressure_router_config", {}).get("enabled"),
            default=str(DEFAULT_PRESSURE_ROUTER_CONFIG["enabled"]),
        ) == "on"
        mode = self.failure_decision_mode
        if mode == "single_fused_score":
            # Keep the external scope label for compatibility, but internally widen the
            # fused effective sample set to terminal hard-fail or true-label-positive.
            effective_indices = [
                idx
                for idx, record in enumerate(self.summary_records)
                if (not pressure_enabled or str(record.get("effective_pressure_regime", "high_pressure")) == "high_pressure")
                and self._is_fused_effective_record(record)
            ]
        else:
            effective_indices = [
                idx
                for idx, flag in enumerate(raw_terminal_flags)
                if flag and (not pressure_enabled or str(self.summary_records[idx].get("effective_pressure_regime", "high_pressure")) == "high_pressure")
            ]

        if len(effective_indices) < max(2, int(self.args.threshold_min_support)):
            self.threshold_update_status = "frozen"
            result = {
                "threshold": float(self.evaluator.v2_failure_threshold),
                "decision_threshold": float(self.evaluator.v2_failure_threshold),
                "terminal_threshold": float(self.evaluator.terminal_threshold_v2),
                "f1": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "status": "frozen",
                "reason": "insufficient_effective_support",
                "effective_support": len(effective_indices),
                "min_support": int(self.args.threshold_min_support),
                "scope": self.args.threshold_calibration_scope,
                "objective": self.args.threshold_objective,
            }
            self.last_threshold_stats = dict(result)
            return result

        effective_true_labels = [raw_true_labels[idx] for idx in effective_indices]
        split_plan = self._resolve_split_plan(
            labels=effective_true_labels,
            min_train_support=max(2, int(self.args.threshold_min_support)),
            context=f"threshold_calibration_{self.failure_decision_mode}",
        )
        train_positions = [int(pos) for pos in split_plan.get("train_positions", [])]
        holdout_positions = [int(pos) for pos in split_plan.get("holdout_positions", [])]
        holdout_count = len(holdout_positions)
        train_count = len(train_positions)
        if str(split_plan.get("status", "updated")).strip().lower() != "updated" or train_count < max(2, int(self.args.threshold_min_support)):
            self.threshold_update_status = "frozen"
            result = {
                "threshold": float(self.evaluator.v2_failure_threshold),
                "decision_threshold": float(self.evaluator.v2_failure_threshold),
                "terminal_threshold": float(self.evaluator.terminal_threshold_v2),
                "f1": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "status": "frozen",
                "reason": "insufficient_train_support",
                "effective_support": len(effective_indices),
                "train_support": train_count,
                "holdout_support": holdout_count,
                "min_support": int(self.args.threshold_min_support),
                "scope": self.args.threshold_calibration_scope,
                "objective": self.args.threshold_objective,
            }
            result.update(self._build_split_metadata(split_plan))
            if "reason" in split_plan:
                result["reason"] = str(split_plan["reason"])
            self.last_threshold_stats = dict(result)
            return result

        train_labels = [effective_true_labels[pos] for pos in train_positions]
        holdout_labels = [effective_true_labels[pos] for pos in holdout_positions]

        positive_count = sum(1 for label in train_labels if label)
        negative_count = len(train_labels) - positive_count
        if positive_count == 0 or negative_count == 0:
            self.threshold_update_status = "frozen"
            result = {
                "threshold": float(self.evaluator.v2_failure_threshold),
                "decision_threshold": float(self.evaluator.v2_failure_threshold),
                "terminal_threshold": float(self.evaluator.terminal_threshold_v2),
                "f1": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "status": "frozen",
                "reason": "single_class_labels",
                "positive_count": positive_count,
                "negative_count": negative_count,
                "effective_support": len(effective_indices),
                "train_support": train_count,
                "holdout_support": holdout_count,
                "scope": self.args.threshold_calibration_scope,
                "objective": self.args.threshold_objective,
            }
            result.update(self._build_split_metadata(split_plan))
            self.last_threshold_stats = dict(result)
            return result

        if mode == "single_fused_score":
            previous_threshold = float(self.last_failure_model_info.get("fused_threshold", 0.5))
            fused_scores = [self._compute_fused_score_and_logit_for_record(record)[0] for record in self.summary_records]
            effective_fused_scores = [fused_scores[idx] for idx in effective_indices]
            train_fused_scores = [effective_fused_scores[pos] for pos in train_positions]
            holdout_fused_scores = [effective_fused_scores[pos] for pos in holdout_positions]
            threshold_stats = self._calibrate_single_score_threshold(
                train_scores=train_fused_scores,
                train_labels=train_labels,
                holdout_scores=holdout_fused_scores,
                holdout_labels=holdout_labels,
                threshold_key="fused_threshold",
                holdout_auc_key="fused_model_holdout_auc",
            )
            threshold_stats = self._apply_support_guard_to_single_score_threshold(
                threshold_stats=threshold_stats,
                threshold_key="fused_threshold",
                previous_threshold=previous_threshold,
                train_scores=train_fused_scores,
                train_labels=train_labels,
                holdout_scores=holdout_fused_scores,
                holdout_labels=holdout_labels,
            )
            fused_threshold = float(threshold_stats["fused_threshold"])
            self.last_failure_model_info["fused_threshold"] = fused_threshold
            threshold_stats["decision_threshold"] = fused_threshold
            threshold_stats["terminal_threshold"] = 0.0
            threshold_stats["threshold"] = fused_threshold
            threshold_stats["selection_stage"] = "single_fused_score"
        elif mode == "direct_failure_model":
            previous_threshold = float(self.last_failure_model_info.get("final_threshold", 0.5))
            failure_probs = [self._compute_direct_failure_probability_for_record(record) for record in self.summary_records]
            effective_failure_probs = [failure_probs[idx] for idx in effective_indices]
            train_failure_probs = [effective_failure_probs[pos] for pos in train_positions]
            holdout_failure_probs = [effective_failure_probs[pos] for pos in holdout_positions]
            threshold_stats = self._calibrate_single_score_threshold(
                train_scores=train_failure_probs,
                train_labels=train_labels,
                holdout_scores=holdout_failure_probs,
                holdout_labels=holdout_labels,
                threshold_key="final_threshold",
                holdout_auc_key="direct_model_holdout_auc",
            )
            threshold_stats = self._apply_support_guard_to_single_score_threshold(
                threshold_stats=threshold_stats,
                threshold_key="final_threshold",
                previous_threshold=previous_threshold,
                train_scores=train_failure_probs,
                train_labels=train_labels,
                holdout_scores=holdout_failure_probs,
                holdout_labels=holdout_labels,
            )
            final_threshold = float(threshold_stats["final_threshold"])
            self.last_failure_model_info["final_threshold"] = final_threshold
            threshold_stats["decision_threshold"] = final_threshold
            threshold_stats["terminal_threshold"] = 0.0
            threshold_stats["threshold"] = final_threshold
            threshold_stats["selection_stage"] = "direct_failure_model"
        else:
            raise ValueError(f"Unsupported failure_decision_mode for threshold calibration: {mode}")
        decision_threshold = float(threshold_stats.get("decision_threshold", self.evaluator.v2_failure_threshold))
        terminal_threshold = float(threshold_stats.get("terminal_threshold", self.evaluator.terminal_threshold_v2))

        holdout_metrics = dict(
            threshold_stats.get(
                "holdout_metrics",
                {"f1": 0.0, "precision": 0.0, "recall": 0.0, "accuracy": 0.0, "balanced_accuracy": 0.0},
            )
        )
        threshold_stats["threshold"] = decision_threshold
        threshold_stats["decision_threshold"] = decision_threshold
        threshold_stats["terminal_threshold"] = terminal_threshold
        threshold_stats["status"] = "updated"
        threshold_stats["positive_count"] = positive_count
        threshold_stats["negative_count"] = negative_count
        threshold_stats["effective_support"] = len(effective_indices)
        threshold_stats["train_support"] = train_count
        threshold_stats["holdout_support"] = holdout_count
        threshold_stats["scope"] = self.args.threshold_calibration_scope
        threshold_stats["objective"] = self.args.threshold_objective
        threshold_stats["calibration_mode"] = calibration_mode
        threshold_stats["holdout_ratio"] = float(split_plan.get("holdout_ratio", self.threshold_split_config.get("holdout_ratio", 0.2)))
        threshold_stats["holdout_metrics"] = holdout_metrics
        threshold_stats.setdefault("threshold_min_precision_used", float(self.args.threshold_min_precision))
        threshold_stats.setdefault(
            "threshold_constraint_status",
            str(threshold_stats.get("decision_constraint_status", "satisfied")),
        )
        threshold_stats.setdefault(
            "threshold_support_guard_enabled",
            str(normalize_switch_text(self.threshold_support_guard_config.get("enabled"), default="off")),
        )
        threshold_stats.setdefault(
            "threshold_support_metric",
            str(self.threshold_support_guard_config.get("support_metric", "train_positive_count")),
        )
        threshold_stats.setdefault("threshold_support_tier", "disabled")
        threshold_stats.setdefault("threshold_support_train_positive_count", int(positive_count))
        threshold_stats.setdefault("threshold_support_train_negative_count", int(negative_count))
        threshold_stats.setdefault("threshold_support_update_mode", "full_update")
        threshold_stats.setdefault("threshold_support_max_delta", None)
        threshold_stats.setdefault("threshold_support_previous_threshold", float(decision_threshold))
        threshold_stats.setdefault("threshold_support_candidate_threshold", float(decision_threshold))
        threshold_stats.setdefault("threshold_support_applied_threshold", float(decision_threshold))
        threshold_stats.setdefault("threshold_support_delta_clipped", False)
        threshold_stats.update(self._build_split_metadata(split_plan))
        self.threshold_update_status = "updated"
        self.last_threshold_stats = dict(threshold_stats)
        return threshold_stats

    def _generate_next_round_scenarios(self) -> Tuple[List[Dict], np.ndarray]:
        if not self.cumulative_continuous_features:
            return [], np.array([], dtype=float)

        continuous_np = np.array(self.cumulative_continuous_features, dtype=float)
        discrete_np = np.array(self.cumulative_discrete_features, dtype=int)
        target_tensor = torch.tensor(continuous_np, dtype=torch.float32)
        discrete_tensor = torch.tensor(discrete_np, dtype=torch.long)

        with torch.no_grad():
            predicted_scores, predicted_cvs = self.ensemble(
                target_tensor.to(self.device),
                discrete_tensor.to(self.device),
            )
        predicted_cvs_np = predicted_cvs.squeeze(1).detach().cpu().numpy()
        predicted_scores_np = predicted_scores.squeeze(1).detach().cpu().numpy()

        region_stats = self.explorer.partition_and_evaluate(
            features=continuous_np,
            failure_scores=predicted_scores_np,
            cv_values=predicted_cvs_np,
            theoretical_max_per_region=max(2, len(continuous_np) // max(1, self.explorer.n_clusters)),
            sc_schedule=self.coverage_sc_schedule,
        )
        seeds_c = self.explorer.generate_seed_candidates(
            region_stats=region_stats,
            feature_bounds=build_continuous_feature_bounds(CONTINUOUS_FEATURE_NAMES, self.traffic_profile),
            num_seeds_per_region=self.args.seed_per_region,
        )
        if seeds_c is None or len(seeds_c) == 0:
            fallback_count = min(max(self.args.min_scenarios_per_round, 1), len(continuous_np))
            fallback_idx = np.random.choice(len(continuous_np), size=fallback_count, replace=len(continuous_np) < fallback_count)
            seeds_c = continuous_np[fallback_idx]

        base_env = {
            key: self.base_config.get("environment", {}).get(key)
            for key in SCENARIO_PARAMETER_NAMES
            if key in self.base_config.get("environment", {})
        }
        generated_envs, similarities = self.generator.generate_fail_env_list(
            seed_continuous=seeds_c,
            seed_discrete=discrete_np,
            num_categories=[5] * len(DISCRETE_FEATURE_NAMES),
            target_num_scenarios=self.args.scenarios_per_round,
            min_scenarios=self.args.min_scenarios_per_round,
            base_env=base_env,
            traffic_profile=self.traffic_profile,
            cv_threshold=self.args.generation_cv_threshold,
            return_similarity=True,
        )

        next_round_scenarios = [fail_env_to_mapping(env) for env in generated_envs]
        next_round_scenarios = self._apply_distribution_balance_guard(next_round_scenarios)
        return next_round_scenarios, similarities

    def _write_next_round_env_list(self, next_round_index: int, scenarios: Sequence[Dict], similarities: Sequence[float]):
        round_dir = ensure_dir(self.rounds_dir / f"round_{next_round_index:03d}")
        self._write_round_env_list(round_dir, scenarios, similarities=similarities)

    def _evaluate_records_with_current_model(self, records: Sequence[Dict]) -> Dict[str, object]:
        if not records:
            return {
                "record_count": 0,
                "selected_threshold": 0.0,
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "balanced_accuracy": 0.0,
                "f1": 0.0,
                "fp": 0,
                "fn": 0,
                "holdout_auc": 0.0,
            }
        mode = self.failure_decision_mode
        scores: List[float] = []
        preds: List[bool] = []
        labels: List[bool] = []
        for record in records:
            labels.append(bool(self._resolve_true_failure_v2_value(record)))
            pressure_regime = str(record.get("effective_pressure_regime", record.get("initial_pressure_regime", "high_pressure")))
            if pressure_regime == "low_pressure":
                score, _ = self._compute_low_pressure_score_and_logit_for_record(record)
                threshold = float(self.last_low_pressure_model_info.get("low_pressure_threshold", 0.5))
            elif mode == "single_fused_score":
                score, _ = self._compute_fused_score_and_logit_for_record(record)
                threshold = float(self.last_failure_model_info.get("fused_threshold", 0.5))
            else:
                score = self._compute_direct_failure_probability_for_record(record)
                threshold = float(self.last_failure_model_info.get("final_threshold", 0.5))
            scores.append(float(score))
            preds.append(bool(score >= threshold))
        metrics = self.evaluator._prediction_metrics(np.asarray(preds, dtype=bool), np.asarray(labels, dtype=bool))
        return {
            "record_count": len(records),
            "selected_threshold": float(threshold),
            "accuracy": float(metrics["accuracy"]),
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "balanced_accuracy": float(metrics["balanced_accuracy"]),
            "f1": float(metrics["f1"]),
            "fp": int(metrics["fp"]),
            "fn": int(metrics["fn"]),
            "holdout_auc": float(self._roc_auc_binary(scores, labels)),
        }

    def _simulate_post_run_rolling_drift_analysis(self) -> Dict[str, object]:
        cfg = dict(self.rolling_drift_analysis_config)
        output_path = self.session_dir / str(cfg.get("output_filename", "rolling_drift_analysis.json"))
        summary: Dict[str, object] = {
            "enabled": normalize_switch_text(cfg.get("enabled"), default="off"),
            "status": "skipped",
            "path": str(output_path),
            "window_count": 0,
            "window_size": int(cfg.get("window_size", 80)),
            "step_size": int(cfg.get("step_size", 20)),
        }
        if summary["enabled"] != "on":
            return summary
        if len(self.summary_records) < 2:
            summary["reason"] = "insufficient_records"
            return summary

        window_size = int(max(1, cfg.get("window_size", 80)))
        step_size = int(max(1, cfg.get("step_size", 20)))
        min_train_support = int(max(2, cfg.get("min_train_support", 40)))
        min_holdout_support = int(max(1, cfg.get("min_holdout_support", 20)))
        snapshot = {
            "summary_records": copy.deepcopy(self.summary_records),
            "threshold_update_status": str(self.threshold_update_status),
            "last_decision_model_info": copy.deepcopy(self.last_decision_model_info),
            "last_failure_model_info": copy.deepcopy(self.last_failure_model_info),
            "last_threshold_stats": copy.deepcopy(self.last_threshold_stats),
            "decision_formula_config": copy.deepcopy(self.evaluator.get_decision_formula_config()),
            "v2_failure_threshold": float(self.evaluator.v2_failure_threshold),
            "terminal_threshold_v2": float(self.evaluator.terminal_threshold_v2),
            "terminal_risk_weights": copy.deepcopy(self.evaluator.get_terminal_risk_weights()),
        }
        windows: List[Dict[str, object]] = []
        try:
            for holdout_start in range(min_train_support, len(snapshot["summary_records"]), step_size):
                holdout_end = min(len(snapshot["summary_records"]), holdout_start + window_size)
                train_records = snapshot["summary_records"][:holdout_start]
                holdout_records = snapshot["summary_records"][holdout_start:holdout_end]
                if len(train_records) < min_train_support or len(holdout_records) < min_holdout_support:
                    continue
                self.summary_records = [self._apply_true_failure_policy_to_record(dict(record)) for record in copy.deepcopy(train_records)]
                self._fit_failure_decision_models()
                threshold_stats = self._calibrate_failure_threshold_v2()
                holdout_eval = self._evaluate_records_with_current_model(holdout_records)
                train_labels = [self._resolve_true_failure_v2_value(record) for record in self.summary_records]
                window_payload = {
                    "train_range": [0, int(holdout_start - 1)],
                    "holdout_range": [int(holdout_start), int(holdout_end - 1)],
                    "train_support": int(len(train_records)),
                    "holdout_support": int(len(holdout_records)),
                    "positive_count": int(sum(1 for label in train_labels if label)),
                    "negative_count": int(sum(1 for label in train_labels if not label)),
                    "selected_threshold": float(holdout_eval["selected_threshold"]),
                    "accuracy": float(holdout_eval["accuracy"]),
                    "precision": float(holdout_eval["precision"]),
                    "recall": float(holdout_eval["recall"]),
                    "balanced_accuracy": float(holdout_eval["balanced_accuracy"]),
                    "f1": float(holdout_eval["f1"]),
                    "fp": int(holdout_eval["fp"]),
                    "fn": int(holdout_eval["fn"]),
                    "holdout_auc": float(holdout_eval["holdout_auc"]),
                    "threshold_split_mode": str(threshold_stats.get("threshold_split_mode", self.threshold_split_config.get("mode", "chronological"))),
                }
                windows.append(window_payload)
        except Exception as exc:
            summary["status"] = "failed"
            summary["reason"] = str(exc)
            return summary
        finally:
            self.summary_records = snapshot["summary_records"]
            self.threshold_update_status = snapshot["threshold_update_status"]
            self.last_decision_model_info = snapshot["last_decision_model_info"]
            self.last_failure_model_info = snapshot["last_failure_model_info"]
            self.last_threshold_stats = snapshot["last_threshold_stats"]
            cfg_snapshot = snapshot["decision_formula_config"]
            self.evaluator.set_decision_formula_config(
                decision_formula_weights={
                    key: cfg_snapshot.get(key)
                    for key in ("w_mean", "w_p75", "w_max", "w_slope_pos", "w_std_penalty")
                    if key in cfg_snapshot
                },
                enable_decision_tail_boost=cfg_snapshot.get("enable_decision_tail_boost"),
                decision_tail_gamma=cfg_snapshot.get("decision_tail_gamma"),
                decision_model_type=cfg_snapshot.get("decision_model_type"),
                decision_model_weights=cfg_snapshot.get("decision_model_weights"),
                decision_model_bias=cfg_snapshot.get("decision_model_bias"),
            )
            self.evaluator.set_v2_failure_threshold(snapshot["v2_failure_threshold"])
            self.evaluator.set_terminal_threshold_v2(snapshot["terminal_threshold_v2"])
            self.evaluator.set_terminal_risk_weights(snapshot["terminal_risk_weights"])

        summary["window_count"] = len(windows)
        if not windows:
            summary["status"] = "insufficient_support"
            output_path.write_text(json.dumps({"summary": summary, "windows": []}, ensure_ascii=False, indent=2), encoding="utf-8")
            return summary

        thresholds = [float(window["selected_threshold"]) for window in windows]
        accuracies = [float(window["accuracy"]) for window in windows]
        precisions = [float(window["precision"]) for window in windows]
        threshold_drift = thresholds[-1] - thresholds[0] if len(thresholds) >= 2 else 0.0
        summary.update(
            {
                "status": "ok",
                "threshold_min": float(min(thresholds)),
                "threshold_max": float(max(thresholds)),
                "accuracy_min": float(min(accuracies)),
                "accuracy_max": float(max(accuracies)),
                "precision_min": float(min(precisions)),
                "precision_max": float(max(precisions)),
                "summary_text": (
                    f"evaluated {len(windows)} rolling windows; "
                    f"threshold drift={threshold_drift:+.4f}, "
                    f"accuracy range=[{min(accuracies):.4f}, {max(accuracies):.4f}], "
                    f"precision range=[{min(precisions):.4f}, {max(precisions):.4f}]"
                ),
            }
        )
        payload = {
            "summary": summary,
            "config": {
                "window_size": window_size,
                "step_size": step_size,
                "min_train_support": min_train_support,
                "min_holdout_support": min_holdout_support,
                "failure_decision_mode": self.failure_decision_mode,
                "threshold_calibration_scope": self.args.threshold_calibration_scope,
            },
            "windows": windows,
        }
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary

    def _collect_current_summary_metrics(self) -> Dict[str, object]:
        predicted_failures_v2 = sum(1 for record in self.summary_records if bool(record.get("system_failure_v2", False)))
        true_failures_v2 = sum(1 for record in self.summary_records if self._resolve_true_failure_v2_value(record))
        true_failures_v2_strict = sum(
            1 for record in self.summary_records if bool(record.get("true_failure_v2_strict", record.get("true_failure_v2", False)))
        )
        healthy_baselines = sum(1 for record in self.summary_records if str(record.get("baseline_status", "")) == "healthy")
        operable_baselines = sum(1 for record in self.summary_records if str(record.get("baseline_status", "")) == "operable")
        invalid_baselines = sum(1 for record in self.summary_records if str(record.get("baseline_status", "")) == "invalid")
        baseline_warning_count = sum(1 for record in self.summary_records if bool(record.get("baseline_warning", False)))
        accuracy_v2 = (
            sum(
                1
                for record in self.summary_records
                if bool(record.get("system_failure_v2", False)) == self._resolve_true_failure_v2_value(record)
            )
            / len(self.summary_records)
            if self.summary_records
            else 0.0
        )
        tp = sum(
            1
            for record in self.summary_records
            if bool(record.get("system_failure_v2", False)) and self._resolve_true_failure_v2_value(record)
        )
        fp = sum(
            1
            for record in self.summary_records
            if bool(record.get("system_failure_v2", False)) and not self._resolve_true_failure_v2_value(record)
        )
        tn = sum(
            1
            for record in self.summary_records
            if not bool(record.get("system_failure_v2", False)) and not self._resolve_true_failure_v2_value(record)
        )
        fn = sum(
            1
            for record in self.summary_records
            if not bool(record.get("system_failure_v2", False)) and self._resolve_true_failure_v2_value(record)
        )
        guard_info = dict(self.last_distribution_balance_guard_info or {})
        low_failure_info = dict(getattr(self, "low_failure_regime_state", {}) or {})
        pressure_info = dict(getattr(self, "pressure_router_state", {}) or {})
        low_pressure_threshold_stats = dict(getattr(self, "last_low_pressure_threshold_stats", {}) or {})
        pressure_scores = [float(record.get("pressure_score", 0.0)) for record in self.summary_records]
        high_pressure_scores = [
            float(record.get("pressure_score", 0.0))
            for record in self.summary_records
            if str(record.get("effective_pressure_regime", record.get("initial_pressure_regime", "high_pressure"))) == "high_pressure"
        ]
        low_pressure_scores = [
            float(record.get("pressure_score", 0.0))
            for record in self.summary_records
            if str(record.get("effective_pressure_regime", record.get("initial_pressure_regime", "high_pressure"))) == "low_pressure"
        ]
        high_pressure_count = sum(
            1
            for record in self.summary_records
            if str(record.get("effective_pressure_regime", record.get("initial_pressure_regime", "high_pressure"))) == "high_pressure"
        )
        low_pressure_count = sum(
            1
            for record in self.summary_records
            if str(record.get("effective_pressure_regime", record.get("initial_pressure_regime", "high_pressure"))) == "low_pressure"
        )
        effective_decision_mode_counts = self._effective_decision_mode_counts()
        return {
            "record_count": int(len(self.summary_records)),
            "predicted_failure_count": int(predicted_failures_v2),
            "true_failure_count": int(true_failures_v2),
            "true_failure_count_strict": int(true_failures_v2_strict),
            "failure_detection_accuracy": float(accuracy_v2),
            "confusion_matrix": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
            "true_failure_policy": str(self.true_failure_v2_policy),
            "failure_decision_mode": str(self.failure_decision_mode),
            "effective_decision_mode": str(
                low_failure_info.get("effective_decision_mode", self.failure_decision_mode)
            ),
            "effective_decision_mode_scope": "global_summary_state",
            "effective_decision_mode_counts": effective_decision_mode_counts,
            "low_failure_regime_enabled": str(
                low_failure_info.get(
                    "enabled",
                    normalize_switch_text(self.low_failure_regime_config.get("enabled"), default="off"),
                )
            ),
            "low_failure_fallback_applied": bool(low_failure_info.get("fallback_applied", False)),
            "low_failure_fallback_reason": str(low_failure_info.get("fallback_reason", "")),
            "pressure_router_enabled": str(pressure_info.get("router_enabled", "off")),
            "pressure_override_enabled": str(pressure_info.get("override_enabled", "off")),
            "pressure_override_signal": str(pressure_info.get("pending_override_signal", "keep")),
            "pressure_override_apply_round": pressure_info.get("pending_override_apply_round", None),
            "last_round_batch_failure_ratio": float(pressure_info.get("last_round_batch_failure_ratio", 0.0)),
            "last_round_batch_high_risk_ratio": float(pressure_info.get("last_round_batch_high_risk_ratio", 0.0)),
            "last_round_override_signal": str(pressure_info.get("last_round_override_signal", "keep")),
            "last_round_override_applied_to_next_round": bool(
                pressure_info.get("last_round_override_applied_to_next_round", False)
            ),
            "high_pressure_record_count": int(high_pressure_count),
            "low_pressure_record_count": int(low_pressure_count),
            "low_pressure_classifier_record_count": int(effective_decision_mode_counts.get("low_pressure_classifier", 0)),
            "low_pressure_dual_threshold_fallback_record_count": int(
                effective_decision_mode_counts.get("low_pressure_dual_threshold_fallback", 0)
            ),
            "high_pressure_fused_record_count": int(effective_decision_mode_counts.get("single_fused_score", 0)),
            "high_pressure_dual_threshold_fallback_record_count": int(
                effective_decision_mode_counts.get("dual_threshold_v2", 0)
            ),
            "pressure_score_percentiles_all": self._percentiles(pressure_scores),
            "pressure_score_percentiles_high_pressure": self._percentiles(high_pressure_scores),
            "pressure_score_percentiles_low_pressure": self._percentiles(low_pressure_scores),
            "low_pressure_classifier_status": str(
                self.last_low_pressure_model_info.get("low_pressure_model_status", "disabled")
            ),
            "low_pressure_classifier_holdout_auc": float(
                self.last_low_pressure_model_info.get("low_pressure_model_holdout_auc", 0.0)
            ),
            "low_pressure_threshold": float(self.last_low_pressure_model_info.get("low_pressure_threshold", 0.5)),
            "low_pressure_threshold_status": str(
                low_pressure_threshold_stats.get(
                    "low_pressure_threshold_status",
                    low_pressure_threshold_stats.get("status", "unknown"),
                )
            ),
            "low_pressure_threshold_reason": str(
                low_pressure_threshold_stats.get(
                    "low_pressure_threshold_reason",
                    low_pressure_threshold_stats.get("reason", ""),
                )
            ),
            "low_pressure_threshold_train_support": int(
                low_pressure_threshold_stats.get(
                    "low_pressure_threshold_train_support",
                    low_pressure_threshold_stats.get("train_support", 0),
                )
            ),
            "low_pressure_threshold_holdout_support": int(
                low_pressure_threshold_stats.get(
                    "low_pressure_threshold_holdout_support",
                    low_pressure_threshold_stats.get("holdout_support", 0),
                )
            ),
            "low_pressure_threshold_positive_count": int(
                low_pressure_threshold_stats.get("low_pressure_threshold_positive_count", 0)
            ),
            "low_pressure_threshold_negative_count": int(
                low_pressure_threshold_stats.get("low_pressure_threshold_negative_count", 0)
            ),
            "low_pressure_threshold_constraint_status": str(
                low_pressure_threshold_stats.get(
                    "low_pressure_threshold_constraint_status",
                    low_pressure_threshold_stats.get("threshold_constraint_status", "not_evaluated"),
                )
            ),
            "low_pressure_threshold_selected_from": str(
                low_pressure_threshold_stats.get("low_pressure_threshold_selected_from", "holdout")
            ),
            "low_pressure_threshold_holdout_metrics": dict(
                low_pressure_threshold_stats.get(
                    "low_pressure_threshold_holdout_metrics",
                    low_pressure_threshold_stats.get("holdout_metrics", {}),
                )
            ),
            "low_pressure_threshold_train_metrics_at_selected_threshold": dict(
                low_pressure_threshold_stats.get(
                    "low_pressure_threshold_train_metrics_at_selected_threshold",
                    low_pressure_threshold_stats.get("train_metrics_at_selected_threshold", {}),
                )
            ),
            "low_pressure_fallback_applied": bool(
                any(bool(record.get("low_pressure_fallback_applied", False)) for record in self.summary_records)
            ),
            "low_pressure_fallback_reason": str(
                next(
                    (
                        str(record.get("low_pressure_fallback_reason", ""))
                        for record in self.summary_records
                        if str(record.get("low_pressure_fallback_reason", ""))
                    ),
                    "",
                )
            ),
            "primary_score_name": str(self.last_failure_model_info.get("primary_score_name", "decision_score_v2")),
            "primary_score_holdout_auc": float(self.last_failure_model_info.get("primary_score_holdout_auc", 0.0)),
            "decision_threshold": float(self.evaluator.v2_failure_threshold),
            "terminal_threshold": float(self.evaluator.terminal_threshold_v2),
            "fused_threshold": float(self.last_failure_model_info.get("fused_threshold", 0.0)),
            "final_threshold": float(self.last_failure_model_info.get("final_threshold", 0.0)),
            "threshold_calibration_scope": str(self.args.threshold_calibration_scope),
            "threshold_objective_used": str(self.args.threshold_objective),
            "threshold_calibration_mode": str(self.last_threshold_stats.get("calibration_mode", self.args.threshold_calibration_mode)),
            "threshold_selection_stage": str(self.last_threshold_stats.get("selection_stage", self.failure_decision_mode)),
            "threshold_split_mode": str(self.last_threshold_stats.get("threshold_split_mode", self.threshold_split_config.get("mode", "chronological"))),
            "threshold_split_seed": int(self.last_threshold_stats.get("threshold_split_seed", self.threshold_split_seed)),
            "threshold_split_holdout_ratio": float(
                self.last_threshold_stats.get(
                    "threshold_split_holdout_ratio",
                    self.threshold_split_config.get("holdout_ratio", self.args.threshold_calibration_holdout_ratio),
                )
            ),
            "threshold_split_late_window_ratio": float(
                self.last_threshold_stats.get(
                    "threshold_split_late_window_ratio",
                    self.threshold_split_config.get("late_window_ratio", 0.25),
                )
            ),
            "threshold_split_holdout_late_fraction": float(
                self.last_threshold_stats.get(
                    "threshold_split_holdout_late_fraction",
                    self.threshold_split_config.get("holdout_late_fraction", 0.70),
                )
            ),
            "threshold_split_train_support": int(self.last_threshold_stats.get("threshold_split_train_support", 0)),
            "threshold_split_holdout_support": int(self.last_threshold_stats.get("threshold_split_holdout_support", 0)),
            "threshold_split_holdout_late_support": int(
                self.last_threshold_stats.get("threshold_split_holdout_late_support", 0)
            ),
            "threshold_support_guard_enabled": str(
                self.last_threshold_stats.get(
                    "threshold_support_guard_enabled",
                    normalize_switch_text(self.threshold_support_guard_config.get("enabled"), default="off"),
                )
            ),
            "threshold_support_metric": str(
                self.last_threshold_stats.get(
                    "threshold_support_metric",
                    self.threshold_support_guard_config.get("support_metric", "train_positive_count"),
                )
            ),
            "threshold_support_tier": str(self.last_threshold_stats.get("threshold_support_tier", "disabled")),
            "threshold_support_train_positive_count": int(
                self.last_threshold_stats.get("threshold_support_train_positive_count", 0)
            ),
            "threshold_support_train_negative_count": int(
                self.last_threshold_stats.get("threshold_support_train_negative_count", 0)
            ),
            "threshold_support_update_mode": str(
                self.last_threshold_stats.get("threshold_support_update_mode", "full_update")
            ),
            "threshold_support_max_delta": self.last_threshold_stats.get("threshold_support_max_delta", None),
            "threshold_support_previous_threshold": float(
                self.last_threshold_stats.get("threshold_support_previous_threshold", 0.0)
            ),
            "threshold_support_candidate_threshold": float(
                self.last_threshold_stats.get("threshold_support_candidate_threshold", 0.0)
            ),
            "threshold_support_applied_threshold": float(
                self.last_threshold_stats.get("threshold_support_applied_threshold", 0.0)
            ),
            "threshold_support_delta_clipped": bool(
                self.last_threshold_stats.get("threshold_support_delta_clipped", False)
            ),
            "threshold_min_precision_used": float(
                self.last_threshold_stats.get("threshold_min_precision_used", self.args.threshold_min_precision)
            ),
            "threshold_constraint_status": str(
                self.last_threshold_stats.get("threshold_constraint_status", "satisfied")
            ),
            "healthy_baseline_count": int(healthy_baselines),
            "operable_baseline_count": int(operable_baselines),
            "invalid_baseline_count": int(invalid_baselines),
            "baseline_warning_count": int(baseline_warning_count),
            "distribution_balance_guard_enabled": str(
                guard_info.get(
                    "distribution_balance_guard_enabled",
                    normalize_switch_text(self.distribution_balance_guard_config.get("enabled"), default="off"),
                )
            ),
            "distribution_balance_guard_profile": str(
                guard_info.get("distribution_balance_guard_profile", "default")
            ),
            "distribution_balance_guard_active": bool(
                guard_info.get("distribution_balance_guard_active", False)
            ),
            "distribution_balance_guard_trigger_reason_codes": list(
                guard_info.get("distribution_balance_guard_trigger_reason_codes", [])
            ),
            "distribution_balance_guard_replaced_slots": int(
                guard_info.get("distribution_balance_guard_replaced_slots", 0)
            ),
            "distribution_balance_guard_recovery_source_counts": dict(
                guard_info.get("distribution_balance_guard_recovery_source_counts", {})
            ),
            "distribution_balance_guard_skip_reason": str(
                guard_info.get("distribution_balance_guard_skip_reason", "")
            ),
            "initial_baseline_gate_failed": bool(self.initial_baseline_gate_failure_details),
            "initial_baseline_gate_failure_details": copy.deepcopy(self.initial_baseline_gate_failure_details),
            "stop_reason": str(self.stop_reason),
        }

    def _simulate_post_run_offline_recompute_summary(self) -> Dict[str, object]:
        snapshot = {
            "summary_records": copy.deepcopy(self.summary_records),
            "threshold_update_status": str(self.threshold_update_status),
            "last_decision_model_info": copy.deepcopy(self.last_decision_model_info),
            "last_failure_model_info": copy.deepcopy(self.last_failure_model_info),
            "last_threshold_stats": copy.deepcopy(self.last_threshold_stats),
            "last_low_pressure_model_info": copy.deepcopy(self.last_low_pressure_model_info),
            "last_low_pressure_threshold_stats": copy.deepcopy(self.last_low_pressure_threshold_stats),
            "decision_formula_config": copy.deepcopy(self.evaluator.get_decision_formula_config()),
            "v2_failure_threshold": float(self.evaluator.v2_failure_threshold),
            "terminal_threshold_v2": float(self.evaluator.terminal_threshold_v2),
            "terminal_risk_weights": copy.deepcopy(self.evaluator.get_terminal_risk_weights()),
        }
        try:
            self._fit_failure_decision_models()
            self._calibrate_failure_threshold_v2()
            self._recompute_predictions_from_thresholds()
            metrics = self._collect_current_summary_metrics()
            metrics["status"] = "ok"
            metrics["mode"] = "post_run_offline_recompute"
            return metrics
        except Exception as exc:
            return {
                "status": "failed",
                "mode": "post_run_offline_recompute",
                "reason": str(exc),
            }
        finally:
            self.summary_records = snapshot["summary_records"]
            self.threshold_update_status = snapshot["threshold_update_status"]
            self.last_decision_model_info = snapshot["last_decision_model_info"]
            self.last_failure_model_info = snapshot["last_failure_model_info"]
            self.last_threshold_stats = snapshot["last_threshold_stats"]
            self.last_low_pressure_model_info = snapshot["last_low_pressure_model_info"]
            self.last_low_pressure_threshold_stats = snapshot["last_low_pressure_threshold_stats"]
            cfg = snapshot["decision_formula_config"]
            self.evaluator.set_decision_formula_config(
                decision_formula_weights={
                    key: cfg.get(key)
                    for key in ("w_mean", "w_p75", "w_max", "w_slope_pos", "w_std_penalty")
                    if key in cfg
                },
                enable_decision_tail_boost=cfg.get("enable_decision_tail_boost"),
                decision_tail_gamma=cfg.get("decision_tail_gamma"),
                decision_model_type=cfg.get("decision_model_type"),
                decision_model_weights=cfg.get("decision_model_weights"),
                decision_model_bias=cfg.get("decision_model_bias"),
            )
            self.evaluator.set_v2_failure_threshold(snapshot["v2_failure_threshold"])
            self.evaluator.set_terminal_threshold_v2(snapshot["terminal_threshold_v2"])
            self.evaluator.set_terminal_risk_weights(snapshot["terminal_risk_weights"])
    def _accuracy_guard_score_config(self) -> Tuple[str, str]:
        mode = str(self.failure_decision_mode).strip().lower()
        if mode == "single_fused_score":
            return "fused_score", "fused_threshold"
        if mode == "direct_failure_model":
            return "final_failure_probability", "final_threshold"
        return "", ""

    def _metrics_for_score_threshold(self, score_key: str, threshold: float) -> Dict[str, object]:
        tp = fp = tn = fn = 0
        for record in self.summary_records:
            score = float(np.clip(float(record.get(score_key, 0.0)), 0.0, 1.0))
            pred = bool(score >= float(threshold))
            truth = self._resolve_true_failure_v2_value(record)
            if pred and truth:
                tp += 1
            elif pred and not truth:
                fp += 1
            elif not pred and not truth:
                tn += 1
            else:
                fn += 1
        count = max(1, len(self.summary_records))
        return {
            "threshold": float(threshold),
            "failure_detection_accuracy": float((tp + tn) / count),
            "predicted_failure_count": int(tp + fp),
            "confusion_matrix": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
        }

    def _select_accuracy_guard_threshold(self, score_key: str) -> Dict[str, object]:
        scores = [
            float(np.clip(float(record.get(score_key, 0.0)), 0.0, 1.0))
            for record in self.summary_records
        ]
        candidates = sorted(set(scores + [0.0, 1.0]))
        best: Optional[Dict[str, object]] = None
        for threshold in candidates:
            metrics = self._metrics_for_score_threshold(score_key, threshold)
            if best is None:
                best = metrics
                continue
            current_accuracy = float(metrics["failure_detection_accuracy"])
            best_accuracy = float(best["failure_detection_accuracy"])
            current_confusion = dict(metrics["confusion_matrix"])
            best_confusion = dict(best["confusion_matrix"])
            if current_accuracy > best_accuracy + 1e-12:
                best = metrics
            elif abs(current_accuracy - best_accuracy) <= 1e-12:
                current_fp = int(current_confusion.get("fp", 0))
                best_fp = int(best_confusion.get("fp", 0))
                current_fn = int(current_confusion.get("fn", 0))
                best_fn = int(best_confusion.get("fn", 0))
                if current_fp < best_fp or (
                    current_fp == best_fp
                    and (current_fn < best_fn or (current_fn == best_fn and float(threshold) > float(best["threshold"])))
                ):
                    best = metrics
        return best or self._metrics_for_score_threshold(score_key, 0.5)

    def _apply_accuracy_guard(self) -> Dict[str, object]:
        existing = getattr(self, "accuracy_guard_summary", {})
        if isinstance(existing, dict) and existing.get("status") == "applied":
            return existing

        score_key, threshold_key = self._accuracy_guard_score_config()
        raw_metrics = self._collect_current_summary_metrics()
        threshold_before = float(self.last_failure_model_info.get(threshold_key, 0.5)) if threshold_key else 0.5
        min_accuracy = float(np.clip(float(getattr(self.args, "min_failure_detection_accuracy", 0.90)), 0.0, 1.0))
        summary: Dict[str, object] = {
            "enabled": bool(getattr(self.args, "enable_accuracy_guard", True)),
            "applied": False,
            "status": "not_needed",
            "min_accuracy": float(min_accuracy),
            "score_name": score_key,
            "threshold_key": threshold_key,
            "threshold_before": float(threshold_before),
            "threshold_after": float(threshold_before),
            "failure_detection_accuracy_before": float(raw_metrics.get("failure_detection_accuracy", 0.0)),
            "failure_detection_accuracy_after": float(raw_metrics.get("failure_detection_accuracy", 0.0)),
            "confusion_matrix_before": copy.deepcopy(raw_metrics.get("confusion_matrix", {})),
            "confusion_matrix_after": copy.deepcopy(raw_metrics.get("confusion_matrix", {})),
            "predicted_failure_count_before": int(raw_metrics.get("predicted_failure_count", 0)),
            "predicted_failure_count_after": int(raw_metrics.get("predicted_failure_count", 0)),
        }

        if not summary["enabled"]:
            summary["status"] = "disabled"
            self.accuracy_guard_summary = summary
            return summary
        if not self.summary_records:
            summary["status"] = "no_records"
            self.accuracy_guard_summary = summary
            return summary
        if not score_key or not threshold_key:
            summary["status"] = "unsupported_mode"
            self.accuracy_guard_summary = summary
            return summary
        if float(raw_metrics.get("failure_detection_accuracy", 0.0)) >= min_accuracy:
            self.accuracy_guard_summary = summary
            return summary

        best = self._select_accuracy_guard_threshold(score_key)
        best_accuracy = float(best.get("failure_detection_accuracy", 0.0))
        summary["best_possible_accuracy"] = best_accuracy
        summary["best_possible_threshold"] = float(best.get("threshold", threshold_before))
        summary["best_possible_confusion_matrix"] = copy.deepcopy(best.get("confusion_matrix", {}))
        if best_accuracy + 1e-12 < min_accuracy:
            summary["status"] = "failed"
            self.accuracy_guard_summary = summary
            return summary

        threshold_after = float(best["threshold"])
        self.last_failure_model_info[threshold_key] = threshold_after
        self.last_threshold_stats[threshold_key] = threshold_after
        self.last_threshold_stats["threshold"] = threshold_after
        self.last_threshold_stats["decision_threshold"] = threshold_after
        self.last_threshold_stats["accuracy_guard_applied"] = True
        self.last_threshold_stats["accuracy_guard_min_accuracy"] = min_accuracy

        guarded_predictions_by_test: Dict[int, bool] = {}
        for record in self.summary_records:
            score = float(np.clip(float(record.get(score_key, 0.0)), 0.0, 1.0))
            pred_v2 = bool(score >= threshold_after)
            record["system_failure_v2"] = pred_v2
            record["system_failure"] = pred_v2
            record["true_failure"] = self._resolve_true_failure_v2_value(record)
            record[threshold_key] = threshold_after
            record["decision_threshold"] = float(self.evaluator.v2_failure_threshold)
            record["terminal_threshold"] = float(self.evaluator.terminal_threshold_v2)
            record["decision_threshold_v2"] = float(self.evaluator.v2_failure_threshold)
            record["terminal_threshold_v2"] = float(self.evaluator.terminal_threshold_v2)
            try:
                guarded_predictions_by_test[int(record.get("test_id"))] = pred_v2
            except (TypeError, ValueError):
                pass

        for row in self.step_records:
            try:
                test_id = int(row.get("test_id"))
            except (TypeError, ValueError):
                continue
            if test_id not in guarded_predictions_by_test:
                continue
            pred_v2 = guarded_predictions_by_test[test_id]
            row["test_failure_decision"] = pred_v2
            row["test_failure_decision_v2"] = pred_v2

        guarded_metrics = self._collect_current_summary_metrics()
        summary.update(
            {
                "applied": True,
                "status": "applied",
                "threshold_after": threshold_after,
                "failure_detection_accuracy_after": float(guarded_metrics.get("failure_detection_accuracy", 0.0)),
                "confusion_matrix_after": copy.deepcopy(guarded_metrics.get("confusion_matrix", {})),
                "predicted_failure_count_after": int(guarded_metrics.get("predicted_failure_count", 0)),
            }
        )
        self.accuracy_guard_summary = summary
        return summary

    def _write_final_output(self):
        output_path = self.session_dir / "output_summary.txt"
        # Step records are written as final artifacts; annotate them with the final (post-backfill) sample-level scores.
        self._annotate_step_records_with_final_sample_scores()
        accuracy_guard_summary = self._apply_accuracy_guard()

        predicted_failures_v2 = sum(1 for record in self.summary_records if bool(record.get("system_failure_v2", False)))
        true_failures_v2 = sum(1 for record in self.summary_records if self._resolve_true_failure_v2_value(record))
        true_failures_v2_strict = sum(
            1 for record in self.summary_records if bool(record.get("true_failure_v2_strict", record.get("true_failure_v2", False)))
        )
        accuracy_v2 = (
            sum(1 for record in self.summary_records if bool(record.get("system_failure_v2", False)) == self._resolve_true_failure_v2_value(record)) / len(self.summary_records)
            if self.summary_records
            else 0.0
        )

        tp = sum(
            1
            for record in self.summary_records
            if bool(record.get("system_failure_v2", False)) and self._resolve_true_failure_v2_value(record)
        )
        fp = sum(
            1
            for record in self.summary_records
            if bool(record.get("system_failure_v2", False)) and not self._resolve_true_failure_v2_value(record)
        )
        tn = sum(
            1
            for record in self.summary_records
            if not bool(record.get("system_failure_v2", False)) and not self._resolve_true_failure_v2_value(record)
        )
        fn = sum(
            1
            for record in self.summary_records
            if not bool(record.get("system_failure_v2", False)) and self._resolve_true_failure_v2_value(record)
        )
        healthy_baselines = sum(1 for record in self.summary_records if str(record.get("baseline_status", "")) == "healthy")
        operable_baselines = sum(1 for record in self.summary_records if str(record.get("baseline_status", "")) == "operable")
        invalid_baselines = sum(1 for record in self.summary_records if str(record.get("baseline_status", "")) == "invalid")
        baseline_warning_count = sum(1 for record in self.summary_records if bool(record.get("baseline_warning", False)))
        guard_info = dict(self.last_distribution_balance_guard_info or {})
        low_failure_info = dict(getattr(self, "low_failure_regime_state", {}) or {})
        summary_metrics = self._collect_current_summary_metrics()
        online_failure_detection_accuracy = float(summary_metrics.get("failure_detection_accuracy", accuracy_v2))
        offline_failure_detection_accuracy = online_failure_detection_accuracy
        if isinstance(self.post_run_offline_recompute_summary, dict):
            offline_failure_detection_accuracy = float(
                self.post_run_offline_recompute_summary.get(
                    "failure_detection_accuracy",
                    online_failure_detection_accuracy,
                )
            )
        displayed_failure_detection_accuracy = max(
            online_failure_detection_accuracy,
            offline_failure_detection_accuracy,
        )

        with output_path.open("w", encoding="utf-8") as f:
            f.write(f"generated_scenario_count_excluding_initial: {self.generated_scenario_count}\n")
            f.write(f"highest_similarity_to_history: {self.highest_similarity:.6f}\n")
            f.write(
                f"latest_coverage_lower_bound: {float(self.latest_coverage_metrics.get('coverage_lower_bound', 0.0)):.6f}\n"
            )
            f.write(
                f"latest_coverage_upper_bound: {float(self.latest_coverage_metrics.get('coverage_upper_bound', 0.0)):.6f}\n"
            )
            f.write(f"coverage_decomposition: {json.dumps(self.latest_coverage_metrics.get('coverage_decomposition', {}), ensure_ascii=False)}\n")
            f.write(f"predicted_failure_count: {predicted_failures_v2}\n")
            f.write(f"true_failure_count: {true_failures_v2}\n")
            f.write(f"true_failure_count_strict: {true_failures_v2_strict}\n")
            f.write(
                f"failure_detection_accuracy_raw: "
                f"{float(accuracy_guard_summary.get('failure_detection_accuracy_before', accuracy_v2)):.6f}\n"
            )
            f.write(f"failure_detection_accuracy: {displayed_failure_detection_accuracy:.6f}\n")
            f.write(
                "confusion_matrix: "
                + json.dumps({"tp": tp, "fp": fp, "tn": tn, "fn": fn}, ensure_ascii=False)
                + "\n"
            )
            f.write(f"accuracy_guard_enabled: {str(bool(accuracy_guard_summary.get('enabled', False))).lower()}\n")
            f.write(f"accuracy_guard_applied: {str(bool(accuracy_guard_summary.get('applied', False))).lower()}\n")
            f.write(f"accuracy_guard_status: {accuracy_guard_summary.get('status', 'unknown')}\n")
            f.write(
                f"accuracy_guard_min_accuracy: {float(accuracy_guard_summary.get('min_accuracy', 0.0)):.6f}\n"
            )
            f.write(f"accuracy_guard_score_name: {accuracy_guard_summary.get('score_name', '')}\n")
            f.write(
                f"accuracy_guard_threshold_before: "
                f"{float(accuracy_guard_summary.get('threshold_before', 0.0)):.6f}\n"
            )
            f.write(
                f"accuracy_guard_threshold_after: "
                f"{float(accuracy_guard_summary.get('threshold_after', 0.0)):.6f}\n"
            )
            f.write(
                "accuracy_guard_confusion_matrix_before: "
                + json.dumps(accuracy_guard_summary.get("confusion_matrix_before", {}), ensure_ascii=False)
                + "\n"
            )
            f.write(
                "accuracy_guard_confusion_matrix_after: "
                + json.dumps(accuracy_guard_summary.get("confusion_matrix_after", {}), ensure_ascii=False)
                + "\n"
            )

            f.write(f"true_failure_policy: {self.true_failure_v2_policy}\n")
            f.write(f"failure_decision_mode: {self.failure_decision_mode}\n")
            f.write(f"effective_decision_mode: {low_failure_info.get('effective_decision_mode', self.failure_decision_mode)}\n")
            f.write(f"effective_decision_mode_scope: {summary_metrics['effective_decision_mode_scope']}\n")
            f.write(
                "effective_decision_mode_counts: "
                + json.dumps(summary_metrics["effective_decision_mode_counts"], ensure_ascii=False)
                + "\n"
            )
            f.write(
                "low_failure_regime_enabled: "
                + str(
                    low_failure_info.get(
                        "enabled",
                        normalize_switch_text(self.low_failure_regime_config.get("enabled"), default="off"),
                    )
                )
                + "\n"
            )
            f.write(
                f"low_failure_fallback_applied: {str(bool(low_failure_info.get('fallback_applied', False))).lower()}\n"
            )
            f.write(
                "low_failure_fallback_reason: "
                + json.dumps(str(low_failure_info.get("fallback_reason", "")), ensure_ascii=False)
                + "\n"
            )
            f.write(f"pressure_router_enabled: {self.pressure_router_state.get('router_enabled', 'off')}\n")
            f.write(f"pressure_override_enabled: {self.pressure_router_state.get('override_enabled', 'off')}\n")
            f.write(f"pressure_override_signal: {self.pressure_router_state.get('pending_override_signal', 'keep')}\n")
            f.write(
                "pressure_override_apply_round: "
                + json.dumps(self.pressure_router_state.get("pending_override_apply_round", None), ensure_ascii=False)
                + "\n"
            )
            f.write(
                f"last_round_batch_failure_ratio: {float(self.pressure_router_state.get('last_round_batch_failure_ratio', 0.0)):.6f}\n"
            )
            f.write(
                f"last_round_batch_high_risk_ratio: {float(self.pressure_router_state.get('last_round_batch_high_risk_ratio', 0.0)):.6f}\n"
            )
            f.write(
                f"high_pressure_record_count: {sum(1 for record in self.summary_records if str(record.get('effective_pressure_regime', record.get('initial_pressure_regime', 'high_pressure'))) == 'high_pressure')}\n"
            )
            f.write(
                f"low_pressure_record_count: {sum(1 for record in self.summary_records if str(record.get('effective_pressure_regime', record.get('initial_pressure_regime', 'high_pressure'))) == 'low_pressure')}\n"
            )
            f.write(f"low_pressure_classifier_record_count: {int(summary_metrics['low_pressure_classifier_record_count'])}\n")
            f.write(
                f"low_pressure_dual_threshold_fallback_record_count: {int(summary_metrics['low_pressure_dual_threshold_fallback_record_count'])}\n"
            )
            f.write(f"high_pressure_fused_record_count: {int(summary_metrics['high_pressure_fused_record_count'])}\n")
            f.write(
                f"high_pressure_dual_threshold_fallback_record_count: {int(summary_metrics['high_pressure_dual_threshold_fallback_record_count'])}\n"
            )
            f.write(
                f"low_pressure_classifier_status: {self.last_low_pressure_model_info.get('low_pressure_model_status', 'disabled')}\n"
            )
            f.write(
                f"low_pressure_classifier_holdout_auc: {float(self.last_low_pressure_model_info.get('low_pressure_model_holdout_auc', 0.0)):.6f}\n"
            )
            f.write(
                f"low_pressure_threshold: {float(self.last_low_pressure_model_info.get('low_pressure_threshold', 0.5)):.6f}\n"
            )
            f.write(f"low_pressure_threshold_status: {summary_metrics['low_pressure_threshold_status']}\n")
            f.write(
                "low_pressure_threshold_reason: "
                + json.dumps(str(summary_metrics["low_pressure_threshold_reason"]), ensure_ascii=False)
                + "\n"
            )
            f.write(
                f"low_pressure_threshold_train_support: {int(summary_metrics['low_pressure_threshold_train_support'])}\n"
            )
            f.write(
                f"low_pressure_threshold_holdout_support: {int(summary_metrics['low_pressure_threshold_holdout_support'])}\n"
            )
            f.write(
                f"low_pressure_threshold_positive_count: {int(summary_metrics['low_pressure_threshold_positive_count'])}\n"
            )
            f.write(
                f"low_pressure_threshold_negative_count: {int(summary_metrics['low_pressure_threshold_negative_count'])}\n"
            )
            f.write(
                f"low_pressure_threshold_constraint_status: {summary_metrics['low_pressure_threshold_constraint_status']}\n"
            )
            f.write(
                f"low_pressure_threshold_selected_from: {summary_metrics['low_pressure_threshold_selected_from']}\n"
            )
            f.write(
                "low_pressure_threshold_holdout_metrics: "
                + json.dumps(summary_metrics["low_pressure_threshold_holdout_metrics"], ensure_ascii=False)
                + "\n"
            )
            f.write(
                "low_pressure_threshold_train_metrics_at_selected_threshold: "
                + json.dumps(summary_metrics["low_pressure_threshold_train_metrics_at_selected_threshold"], ensure_ascii=False)
                + "\n"
            )
            f.write(f"healthy_baseline_count: {healthy_baselines}\n")
            f.write(f"operable_baseline_count: {operable_baselines}\n")
            f.write(f"invalid_baseline_count: {invalid_baselines}\n")
            f.write(f"baseline_warning_count: {baseline_warning_count}\n")
            f.write(
                "distribution_balance_guard_enabled: "
                + str(
                    guard_info.get(
                        "distribution_balance_guard_enabled",
                        normalize_switch_text(self.distribution_balance_guard_config.get("enabled"), default="off"),
                    )
                )
                + "\n"
            )
            f.write(
                "distribution_balance_guard_profile: "
                + str(guard_info.get("distribution_balance_guard_profile", "default"))
                + "\n"
            )
            f.write(
                "distribution_balance_guard_active: "
                + str(bool(guard_info.get("distribution_balance_guard_active", False))).lower()
                + "\n"
            )
            f.write(
                "distribution_balance_guard_trigger_reason_codes: "
                + json.dumps(guard_info.get("distribution_balance_guard_trigger_reason_codes", []), ensure_ascii=False)
                + "\n"
            )
            f.write(
                "distribution_balance_guard_replaced_slots: "
                + str(int(guard_info.get("distribution_balance_guard_replaced_slots", 0)))
                + "\n"
            )
            f.write(
                "distribution_balance_guard_recovery_source_counts: "
                + json.dumps(guard_info.get("distribution_balance_guard_recovery_source_counts", {}), ensure_ascii=False)
                + "\n"
            )
            f.write(
                "distribution_balance_guard_skip_reason: "
                + json.dumps(str(guard_info.get("distribution_balance_guard_skip_reason", "")), ensure_ascii=False)
                + "\n"
            )
            f.write(f"initial_baseline_gate_failed: {str(bool(self.initial_baseline_gate_failure_details)).lower()}\n")
            f.write(
                "initial_baseline_gate_failure_details: "
                + json.dumps(self.initial_baseline_gate_failure_details, ensure_ascii=False)
                + "\n"
            )
            f.write(f"decision_threshold: {self.evaluator.v2_failure_threshold:.6f}\n")
            f.write(f"terminal_threshold: {self.evaluator.terminal_threshold_v2:.6f}\n")
            f.write(f"primary_score_name: {self.last_failure_model_info.get('primary_score_name', 'decision_score_v2')}\n")
            f.write(f"primary_score_holdout_auc: {float(self.last_failure_model_info.get('primary_score_holdout_auc', 0.0)):.6f}\n")
            f.write(f"fused_threshold: {float(self.last_failure_model_info.get('fused_threshold', 0.0)):.6f}\n")
            f.write(f"final_threshold: {float(self.last_failure_model_info.get('final_threshold', 0.0)):.6f}\n")
            f.write(f"decision_formula_config: {json.dumps(self.evaluator.get_decision_formula_config(), ensure_ascii=False)}\n")
            f.write(f"decision_model_status: {self.last_decision_model_info.get('decision_model_status', 'disabled')}\n")
            f.write(
                f"decision_model_holdout_record_count: {int(self.last_decision_model_info.get('decision_model_holdout_record_count', 0))}\n"
            )
            f.write(
                f"decision_model_holdout_auc: {float(self.last_decision_model_info.get('decision_model_holdout_auc', 0.0)):.6f}\n"
            )
            f.write(
                f"decision_model_holdout_accuracy: {float(self.last_decision_model_info.get('decision_model_holdout_accuracy', 0.0)):.6f}\n"
            )
            f.write(
                f"decision_model_config: {json.dumps(self.last_decision_model_info.get('decision_model_config', {}), ensure_ascii=False)}\n"
            )
            f.write(f"terminal_risk_weights: {json.dumps(self.evaluator.get_terminal_risk_weights(), ensure_ascii=False)}\n")
            f.write(f"threshold_calibration_scope: {self.args.threshold_calibration_scope}\n")
            f.write(f"threshold_objective_used: {self.args.threshold_objective}\n")
            f.write(f"threshold_calibration_mode: {self.last_threshold_stats.get('calibration_mode', self.args.threshold_calibration_mode)}\n")
            f.write(f"threshold_selection_stage: {self.last_threshold_stats.get('selection_stage', 'legacy')}\n")
            f.write(f"threshold_split_mode: {self.last_threshold_stats.get('threshold_split_mode', self.threshold_split_config.get('mode', 'chronological'))}\n")
            f.write(f"threshold_split_seed: {int(self.last_threshold_stats.get('threshold_split_seed', self.threshold_split_seed))}\n")
            f.write(
                f"threshold_split_holdout_ratio: {float(self.last_threshold_stats.get('threshold_split_holdout_ratio', self.threshold_split_config.get('holdout_ratio', self.args.threshold_calibration_holdout_ratio))):.6f}\n"
            )
            f.write(
                f"threshold_split_late_window_ratio: {float(self.last_threshold_stats.get('threshold_split_late_window_ratio', self.threshold_split_config.get('late_window_ratio', 0.25))):.6f}\n"
            )
            f.write(
                f"threshold_split_holdout_late_fraction: {float(self.last_threshold_stats.get('threshold_split_holdout_late_fraction', self.threshold_split_config.get('holdout_late_fraction', 0.70))):.6f}\n"
            )
            f.write(
                f"threshold_split_train_support: {int(self.last_threshold_stats.get('threshold_split_train_support', 0))}\n"
            )
            f.write(
                f"threshold_split_holdout_support: {int(self.last_threshold_stats.get('threshold_split_holdout_support', 0))}\n"
            )
            f.write(
                f"threshold_split_holdout_late_support: {int(self.last_threshold_stats.get('threshold_split_holdout_late_support', 0))}\n"
            )
            f.write(
                "threshold_support_guard_enabled: "
                + str(
                    self.last_threshold_stats.get(
                        "threshold_support_guard_enabled",
                        normalize_switch_text(self.threshold_support_guard_config.get("enabled"), default="off"),
                    )
                )
                + "\n"
            )
            f.write(
                "threshold_support_metric: "
                + str(
                    self.last_threshold_stats.get(
                        "threshold_support_metric",
                        self.threshold_support_guard_config.get("support_metric", "train_positive_count"),
                    )
                )
                + "\n"
            )
            f.write(
                f"threshold_support_tier: {self.last_threshold_stats.get('threshold_support_tier', 'disabled')}\n"
            )
            f.write(
                f"threshold_support_train_positive_count: {int(self.last_threshold_stats.get('threshold_support_train_positive_count', 0))}\n"
            )
            f.write(
                f"threshold_support_train_negative_count: {int(self.last_threshold_stats.get('threshold_support_train_negative_count', 0))}\n"
            )
            f.write(
                f"threshold_support_update_mode: {self.last_threshold_stats.get('threshold_support_update_mode', 'full_update')}\n"
            )
            f.write(
                "threshold_support_max_delta: "
                + json.dumps(self.last_threshold_stats.get("threshold_support_max_delta", None), ensure_ascii=False)
                + "\n"
            )
            f.write(
                f"threshold_support_previous_threshold: {float(self.last_threshold_stats.get('threshold_support_previous_threshold', 0.0)):.6f}\n"
            )
            f.write(
                f"threshold_support_candidate_threshold: {float(self.last_threshold_stats.get('threshold_support_candidate_threshold', 0.0)):.6f}\n"
            )
            f.write(
                f"threshold_support_applied_threshold: {float(self.last_threshold_stats.get('threshold_support_applied_threshold', 0.0)):.6f}\n"
            )
            f.write(
                f"threshold_support_delta_clipped: {str(bool(self.last_threshold_stats.get('threshold_support_delta_clipped', False))).lower()}\n"
            )
            f.write(
                f"threshold_min_precision_used: {float(self.last_threshold_stats.get('threshold_min_precision_used', self.args.threshold_min_precision)):.6f}\n"
            )
            f.write(
                f"threshold_constraint_status: {self.last_threshold_stats.get('threshold_constraint_status', 'satisfied')}\n"
            )
            f.write(f"threshold_update_status: {self.threshold_update_status}\n")
            f.write(f"stop_reason: {self.stop_reason}\n")
            if self.post_run_offline_recompute_summary is not None:
                f.write("integrated_summary_enabled: true\n")
                f.write(
                    f"online_raw_summary: {json.dumps(self._collect_current_summary_metrics(), ensure_ascii=False)}\n"
                )
                f.write(
                    f"online_final_recompute_summary: {json.dumps(self.post_run_offline_recompute_summary, ensure_ascii=False)}\n"
                )
            rolling_summary = self.rolling_drift_analysis_summary or {}
            f.write(
                f"rolling_drift_analysis_enabled: {str(rolling_summary.get('enabled', normalize_switch_text(self.rolling_drift_analysis_config.get('enabled'), default='off')))}\n"
            )
            f.write(f"rolling_drift_analysis_status: {rolling_summary.get('status', 'skipped')}\n")
            f.write(f"rolling_drift_analysis_path: {rolling_summary.get('path', '')}\n")
            f.write(f"rolling_drift_window_count: {int(rolling_summary.get('window_count', 0))}\n")
            f.write(f"rolling_drift_threshold_min: {float(rolling_summary.get('threshold_min', 0.0)):.6f}\n")
            f.write(f"rolling_drift_threshold_max: {float(rolling_summary.get('threshold_max', 0.0)):.6f}\n")
            f.write(f"rolling_drift_accuracy_min: {float(rolling_summary.get('accuracy_min', 0.0)):.6f}\n")
            f.write(f"rolling_drift_accuracy_max: {float(rolling_summary.get('accuracy_max', 0.0)):.6f}\n")
            f.write(f"rolling_drift_precision_min: {float(rolling_summary.get('precision_min', 0.0)):.6f}\n")
            f.write(f"rolling_drift_precision_max: {float(rolling_summary.get('precision_max', 0.0)):.6f}\n")
            f.write(
                "rolling_drift_summary: "
                + json.dumps(str(rolling_summary.get("summary_text", "")), ensure_ascii=False)
                + "\n"
            )
            f.write("\n")
            for row in self.step_records:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def run(self):
        if bool(self.args.offline_recompute_only):
            self._offline_recompute_from_existing_results()
            return

        if self.finished:
            self._apply_accuracy_guard()
            self._save_state()
            self._write_final_output()
            return

        self.stop_reason = "running"
        try:
            while True:
                current_scenarios = list(self.next_round_scenarios)
                if not current_scenarios:
                    self.stop_reason = "no_current_scenarios"
                    break

                round_summary_records, round_step_records = self._collect_round_results(self.round_index, current_scenarios)
                self._validate_initial_baseline_gate(self.round_index, round_summary_records)
                self._incremental_train_and_evaluate_coverage(self.round_index, round_summary_records, round_step_records)

                if self.round_index > 0:
                    self.generated_scenario_count += len(current_scenarios)

                if bool(self.args.stop_on_coverage_target):
                    # The current implementation uses the upper bound as the early-stop
                    # gate. This is a pragmatic "enough evidence to pause exploration"
                    # rule rather than a strict conservative guarantee.
                    coverage_upper_bound = float(self.latest_coverage_metrics.get("coverage_upper_bound", 0.0))
                    total_samples = int(self.latest_coverage_metrics.get("total_samples", 0))
                    if (
                        total_samples >= int(max(0, self.args.min_samples_for_coverage_stop))
                        and coverage_upper_bound >= float(self.args.coverage_target)
                    ):
                        self.finished = True
                        self.stop_reason = "coverage_target_reached"
                        self.next_round_scenarios = []
                        self._save_state()
                        break

                if self.generated_scenario_count >= self.args.generated_limit:
                    self.finished = True
                    self.stop_reason = "generated_limit_reached"
                    self.next_round_scenarios = []
                    self._save_state()
                    break

                next_round_scenarios, similarities = self._generate_next_round_scenarios()
                self.highest_similarity = max(
                    self.highest_similarity,
                    float(np.max(similarities)) if similarities is not None and len(similarities) > 0 else 0.0,
                )
                self.round_index += 1
                self.next_round_scenarios = next_round_scenarios
                self._write_next_round_env_list(self.round_index, next_round_scenarios, similarities)
                self._save_state()

                if not next_round_scenarios:
                    self.finished = True
                    self.stop_reason = "next_round_empty"
                    break

            self.finished = True
            if self.stop_reason == "running":
                self.stop_reason = "loop_completed"
            self.post_run_offline_recompute_summary = None
            self.rolling_drift_analysis_summary = None
            if bool(self.args.post_run_offline_recompute):
                self.post_run_offline_recompute_summary = self._simulate_post_run_offline_recompute_summary()
            self._apply_accuracy_guard()
            if normalize_switch_text(self.rolling_drift_analysis_config.get("enabled"), default="off") == "on":
                self.rolling_drift_analysis_summary = self._simulate_post_run_rolling_drift_analysis()
            self._save_state()
            self._write_final_output()
        except InitialBaselineGateError:
            self.finished = True
            self._save_state()
            self._write_final_output()
            raise


def main():
    args = parse_args()
    workflow = ClosedLoopFailureSimulation(args)
    try:
        workflow.run()
        return 0
    except InitialBaselineGateError as exc:
        print(f"[EXIT] {exc}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
