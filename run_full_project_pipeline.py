#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run the complete simulation -> failure detection -> attack/attribution pipeline.

This script is intentionally thin: it calls the existing closed-loop simulator
and the updated 2.2 post-analysis pipeline, then validates the data contract
between them before each handoff.
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


BOOTSTRAP_PROJECT_ROOT = Path(__file__).resolve().parent
BOOTSTRAP_SRC_ROOT = BOOTSTRAP_PROJECT_ROOT / "src"
for path in (BOOTSTRAP_PROJECT_ROOT, BOOTSTRAP_SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from project_paths import (  # noqa: E402
    ANALYSIS_PIPELINE_CONFIG_PATH,
    ANALYSIS_PIPELINE_SCRIPT,
    ATTACK_ARTIFACT_PATH,
    ATTRIBUTION_ARTIFACT_PATH,
    DATA_ARCHIVE_ROOT,
    DEFAULT_TRAIN_CONFIG_PATH,
    ENV_CONFIG_PATH,
    FULL_PROJECT_RUNS_ROOT,
    ITERATIVE_FAILURE_SIMULATION_SCRIPT,
    PART3_PIPELINE_SCRIPT,
    PROJECT_ROOT,
    SCENARIO_EXPLORATION_CONFIG_PATH,
)
from iterative_testing.gpu_runtime import select_torch_device  # noqa: E402
from failure_and_attribution_analysis.parameter_interfaces import (  # noqa: E402
    CONTINUOUS_FEATURE_NAMES,
    DISCRETE_FEATURE_NAMES,
    METRIC_NAMES,
    SCENARIO_PARAMETER_NAMES,
)
from iterative_testing.run_batch_experiments import parse_experiment_md  # noqa: E402


DEFAULT_CONFIG = DEFAULT_TRAIN_CONFIG_PATH
DEFAULT_ENV_MD = ENV_CONFIG_PATH
DEFAULT_ENV_CONFIG: dict[str, Any] = {
    "ConstellationConfig": 4,
    "DegradedEdgeRatio": 0.08,
    "EdgeDisconnectRatio": 0.1,
    "EdgeBandwidthMeanDecreaseRatio": 0.2,
    "EdgeBandwidthDecreaseStd": 0.1,
    "TrafficProfile": "low",
    "PacketSizeMean": 400000000,
    "PacketSizeStd": 115.47e6,
}
REQUIRED_ENV_CONFIG_KEYS = (
    "ConstellationConfig",
    "DegradedEdgeRatio",
    "EdgeDisconnectRatio",
    "EdgeBandwidthMeanDecreaseRatio",
    "EdgeBandwidthDecreaseStd",
    "TrafficProfile",
    "PacketSizeMean",
    "PacketSizeStd",
)
DEFAULT_ATTACK_LEVELS: dict[str, int] = {
    "StateObservationAttack_level": 0,
    "ActionAttack_level": 0,
    "RewardAttack_level": 0,
    "StateTransferAttack_level": 0,
    "ExperiencePoolAttack_level": 0,
    "ModelTampAttack_level": 0,
}
ENV_CONFIG_WRITE_ORDER = (
    *REQUIRED_ENV_CONFIG_KEYS,
    *DEFAULT_ATTACK_LEVELS.keys(),
)
DEFAULT_RUN_ROOT = FULL_PROJECT_RUNS_ROOT
DEFAULT_ANALYSIS_SCRIPT = ANALYSIS_PIPELINE_SCRIPT
DEFAULT_ANALYSIS_CONFIG = ANALYSIS_PIPELINE_CONFIG_PATH
DEFAULT_EXPLORATION_CONFIG = SCENARIO_EXPLORATION_CONFIG_PATH
DEFAULT_PART3_SCRIPT = PART3_PIPELINE_SCRIPT
DEFAULT_ATTACK_ARTIFACT = ATTACK_ARTIFACT_PATH
DEFAULT_ATTRIBUTION_ARTIFACT = ATTRIBUTION_ARTIFACT_PATH
DEFAULT_ARCHIVE_ROOT = DATA_ARCHIVE_ROOT
VALID_ATTRIBUTION_TARGETS = ("total_membership_v2", "decision_score_v2", "fused_score")
ATTACK_TYPE_TO_ID = {
    "NoAttack": 0,
    "StateObservationAttack": 1,
    "ActionAttack": 2,
    "StateTransferAttack": 3,
    "RewardAttack": 4,
    "ExperiencePoolAttack": 5,
    "ModelTampAttack": 6,
}
STEP_EVALUATE_TABLE = "step_evaluate"
TEST_SCENARIO_CONFIG_TABLE = "test_scenario_config"
FAILURE_ANALYSIS_TABLE = "failure_analysis"
SELF_HEALING_TABLE = "self_healing"

SCENARIO_SIMILARITY_FILENAME = "scenario_similarity.csv"
SCENARIO_SIMILARITY_HISTORY_CSV = PROJECT_ROOT / "step_evaluate.csv"
SCENARIO_SIMILARITY_CONTINUOUS_DISTANCE_THRESHOLD = 0.165

DEFAULT_CONTINUOUS_PARAMETER_RANGES = {
    "DegradedEdgeRatio": 1.0,
    "EdgeDisconnectRatio": 1.0,
    "EdgeBandwidthMeanDecreaseRatio": 1.0,
    "EdgeBandwidthDecreaseStd": 0.2,
    "PoissonRate": 6.0,
    "MeanIntervalTime": 8.0,
    "PacketGenerationInterval": 1.0,
    "PacketSizeMean": 1900000000.0,
    "PacketSizeStd": 500000000.0,
}

TEST_SCENARIO_CONFIG_COLUMNS = (
    "start_timestamp",
    "test_id",
    "scenario_similarity",
    "latest_coverage",
    "failure_detection_accuracy",
    *SCENARIO_PARAMETER_NAMES,
)

STEP_EVALUATE_COLUMNS = (
    "start_timestamp",
    "round_index",
    "test_id",
    "step_index",
    *METRIC_NAMES,
    "failure_score_v2",
    "fused_score",
    "timestamp",
)
FAILURE_ANALYSIS_COLUMNS = (
    "start_timestamp",
    "original_round_index",
    "original_test_id",
    "merged_round_index",
    "merged_test_id",
    "step_evaluation",
    "true_attack_type",
    "predicted_attack_type",
    "target_field",
    "target_value",
    "predicted_fail_score",
    "absolute_error",
    *SCENARIO_PARAMETER_NAMES,
    "timestamp",
)
SELF_HEALING_COLUMNS = (
    "start_timestamp",
    "test_id",
    "timestamp",
    "ConstellationConfig",
    "node_id",
    "success",
    "healing_level",
    "healing_time",
    "message",
    "fail_score",
    "attack_type",
    "attack_label",
    "total_time",
    "final_level",
)
DATABASE_TABLE_COLUMNS = {
    TEST_SCENARIO_CONFIG_TABLE: TEST_SCENARIO_CONFIG_COLUMNS,
    STEP_EVALUATE_TABLE: STEP_EVALUATE_COLUMNS,
    FAILURE_ANALYSIS_TABLE: FAILURE_ANALYSIS_COLUMNS,
    SELF_HEALING_TABLE: SELF_HEALING_COLUMNS,
}


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}")


def bool_text(value: bool) -> str:
    return "true" if bool(value) else "false"


def resolve_path(path_text: str | Path, base_dir: Path = PROJECT_ROOT) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.load(handle, Loader=yaml.FullLoader)
    if not isinstance(payload, dict):
        raise ValueError(f"YAML config must be a mapping: {path}")
    return payload


def load_exploration_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Scenario exploration config not found: {path}")
    payload = load_yaml(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Scenario exploration config must be a mapping: {path}")
    if "single_attack_type" in payload:
        raise ValueError(
            f"{path} contains deprecated field 'single_attack_type'; "
            "please rename it to 'single_attack_types'."
        )
    return payload


def resolve_single_attack_types(
    explicit_list_value: str,
    exploration_config_path: Path,
) -> list[str]:
    def _parse_attack_list(raw_value: Any) -> list[str]:
        if raw_value is None:
            return []
        if isinstance(raw_value, str):
            return [item.strip() for item in raw_value.split(",") if item.strip()]
        if isinstance(raw_value, (list, tuple, set)):
            return [str(item).strip() for item in raw_value if str(item).strip()]
        return [str(raw_value).strip()] if str(raw_value).strip() else []

    attack_types = _parse_attack_list(explicit_list_value)
    if attack_types:
        return attack_types
    config_payload = load_exploration_config(exploration_config_path)
    return _parse_attack_list(config_payload.get("single_attack_types", []))


def resolve_exploration_cli_override(text: str | None) -> str:
    return str(text or "").strip()


def parse_env_config_text(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        try:
            payload = ast.literal_eval(text)
        except (SyntaxError, ValueError) as exc:
            raise ValueError("--env-config-json must be a JSON object or Python dict literal.") from exc
    if not isinstance(payload, dict):
        raise ValueError("--env-config-json must evaluate to a mapping.")
    return payload


def load_env_config_file(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        payload = load_yaml(path)
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Environment config file must contain a mapping: {path}")
    return payload


def normalize_env_config_mapping(raw_config: Mapping[str, Any]) -> dict[str, Any]:
    raw_mapping: Mapping[str, Any] = raw_config
    nested_env = raw_config.get("environment")
    if isinstance(nested_env, Mapping):
        raw_mapping = nested_env

    env_config = {str(key): value for key, value in raw_mapping.items()}
    missing = [key for key in REQUIRED_ENV_CONFIG_KEYS if key not in env_config]
    if missing:
        raise KeyError(f"Environment config dictionary missing required keys: {missing}")

    normalized = dict(env_config)
    for key, value in DEFAULT_ATTACK_LEVELS.items():
        normalized.setdefault(key, value)
    return normalized


def resolve_env_config_dict(
    args: argparse.Namespace,
    env_config: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    dict_sources = int(env_config is not None)
    dict_sources += int(bool(args.env_config_json.strip()))
    dict_sources += int(bool(args.env_config_file.strip()))
    if dict_sources > 1:
        raise ValueError("Use only one environment dictionary source: env_config, --env-config-json, or --env-config-file.")
    if dict_sources and args.env_md:
        raise ValueError("--env-md is a legacy input and cannot be combined with dictionary environment input.")

    if env_config is not None:
        return normalize_env_config_mapping(env_config)
    if args.env_config_json.strip():
        return normalize_env_config_mapping(parse_env_config_text(args.env_config_json))
    if args.env_config_file.strip():
        return normalize_env_config_mapping(load_env_config_file(resolve_path(args.env_config_file)))
    if args.env_md:
        return None
    return normalize_env_config_mapping(DEFAULT_ENV_CONFIG)


def env_md_scalar_to_text(value: Any) -> str:
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def env_md_value_to_text(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return ",".join(env_md_scalar_to_text(item) for item in value)
    return env_md_scalar_to_text(value)


def write_env_config_md(path: Path, env_config: Mapping[str, Any]) -> None:
    ordered_keys = [key for key in ENV_CONFIG_WRITE_ORDER if key in env_config]
    ordered_keys.extend(key for key in env_config if key not in set(ordered_keys))
    lines = [f"{key}:{env_md_value_to_text(env_config[key])}" for key in ordered_keys]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def strip_jsonc_comments(text: str) -> str:
    cleaned_lines: list[str] = []
    for line in text.splitlines():
        if line.lstrip().startswith("//"):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def load_jsonc(path: Path) -> dict[str, Any]:
    payload = json.loads(strip_jsonc_comments(path.read_text(encoding="utf-8")))
    if not isinstance(payload, dict):
        raise ValueError(f"JSONC config must be a mapping: {path}")
    return payload


def dump_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.dump(payload, handle, allow_unicode=True, sort_keys=False)


def path_for_config(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def read_jsonl_records(path: Path, prefix: str | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if prefix is not None:
                if not line.startswith(prefix):
                    continue
                line = line.split(":", 1)[1].strip()
            elif not line.startswith("{"):
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
            if isinstance(payload, dict):
                records.append(payload)
    return records


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def project_database_row(columns: Sequence[str], values: Mapping[str, Any]) -> dict[str, Any]:
    return {column: values.get(column) for column in columns}


def format_step_evaluation_id(start_timestamp: str, round_index: Any, test_id: Any) -> str:
    return f"{start_timestamp}|round_{int(round_index):03d}|test_{int(test_id):04d}"


def database_stage_enabled(database_writer: Any | None) -> bool:
    return database_writer is not None


def write_database_rows(
    database_writer: Any | None,
    table_name: str,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if table_name not in DATABASE_TABLE_COLUMNS:
        raise ValueError(f"Unsupported database table: {table_name}")
    materialized_rows = [
        project_database_row(DATABASE_TABLE_COLUMNS[table_name], row)
        for row in rows
    ]
    summary: dict[str, Any] = {
        "table": table_name,
        "enabled": database_stage_enabled(database_writer),
        "row_count": len(materialized_rows),
    }
    if database_writer is None:
        return summary

    stage_method_name = f"write_{table_name}"
    if hasattr(database_writer, stage_method_name):
        result = getattr(database_writer, stage_method_name)(materialized_rows)
    elif hasattr(database_writer, "write_rows"):
        result = database_writer.write_rows(table_name, materialized_rows)
    elif callable(database_writer):
        result = database_writer(table_name, materialized_rows)
    else:
        raise TypeError(
            "database_writer must implement write_rows(table_name, rows), "
            "write_<table_name>(rows), or be callable(table_name, rows)."
        )
    if result is not None:
        try:
            json.dumps(result, ensure_ascii=False)
            summary["writer_result"] = result
        except TypeError:
            summary["writer_result"] = repr(result)
    return summary


def optional_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def required_float(value: Any, field_name: str) -> float:
    number = optional_float(value)
    if number is None:
        raise ValueError(f"Missing required continuous parameter: {field_name}")
    return number


def load_continuous_parameter_ranges(history_csv_path: Path) -> dict[str, float]:
    import csv
    ranges = dict(DEFAULT_CONTINUOUS_PARAMETER_RANGES)
    minimums: dict[str, float] = {}
    maximums: dict[str, float] = {}
    if history_csv_path.exists():
        with history_csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                for name in CONTINUOUS_FEATURE_NAMES:
                    val = optional_float(row.get(name))
                    if val is not None:
                        minimums[name] = min(minimums.get(name, val), val)
                        maximums[name] = max(maximums.get(name, val), val)

    for name in CONTINUOUS_FEATURE_NAMES:
        if name in minimums and name in maximums:
            ranges[name] = max(ranges[name], maximums[name] - minimums[name])
        # Ensure never 0
        if ranges[name] <= 0:
            ranges[name] = 1.0
    return ranges


def load_test_scenario_parameters(output_summary_path: Path) -> list[dict[str, float]]:
    scenarios_by_test: dict[tuple[int, int], dict[str, float]] = {}
    for record in read_jsonl_records(output_summary_path):
        key = (int(record["round_index"]), int(record["test_id"]))
        if key not in scenarios_by_test:
            scenario = {}
            for param in SCENARIO_PARAMETER_NAMES:
                scenario[param] = required_float(record[param], param)
            scenarios_by_test[key] = scenario
    return [scenarios_by_test[key] for key in sorted(scenarios_by_test)]


def continuous_parameter_distance(
    left: Mapping[str, float],
    right: Mapping[str, float],
    continuous_ranges: Mapping[str, float],
) -> float:
    total = 0.0
    for name in CONTINUOUS_FEATURE_NAMES:
        range_val = continuous_ranges.get(name, 1.0)
        diff = abs(left[name] - right[name])
        total += (diff / range_val) if range_val > 0 else 0.0
    return total / len(CONTINUOUS_FEATURE_NAMES)


def scenario_parameters_are_similar(
    left: Mapping[str, float],
    right: Mapping[str, float],
    continuous_ranges: Mapping[str, float],
) -> bool:
    for name in DISCRETE_FEATURE_NAMES:
        if left[name] != right[name]:
            return False
    return (
        continuous_parameter_distance(left, right, continuous_ranges)
        <= SCENARIO_SIMILARITY_CONTINUOUS_DISTANCE_THRESHOLD
    )


def build_scenario_similarity_report(
    output_summary_path: Path,
    history_csv_path: Path = SCENARIO_SIMILARITY_HISTORY_CSV,
) -> dict[str, Any]:
    scenarios = load_test_scenario_parameters(output_summary_path)
    continuous_ranges = load_continuous_parameter_ranges(history_csv_path)
    total_pair_count = len(scenarios) * (len(scenarios) - 1) // 2
    similar_pair_count = 0
    for left_index in range(len(scenarios)):
        for right_index in range(left_index + 1, len(scenarios)):
            if scenario_parameters_are_similar(scenarios[left_index], scenarios[right_index], continuous_ranges):
                similar_pair_count += 1
    similarity = similar_pair_count / total_pair_count if total_pair_count > 0 else 0.0
    return {
        "similarity": similarity,
        "test_count": len(scenarios),
        "similar_pair_count": similar_pair_count,
        "total_pair_count": total_pair_count,
        "history_csv": str(history_csv_path),
        "continuous_distance_threshold": SCENARIO_SIMILARITY_CONTINUOUS_DISTANCE_THRESHOLD,
        "continuous_parameter_ranges": continuous_ranges,
        "discrete_parameters": list(DISCRETE_FEATURE_NAMES),
        "continuous_parameters": list(CONTINUOUS_FEATURE_NAMES),
    }


def write_scenario_similarity_csv(path: Path, similarity: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("similarity\n")
        handle.write(f"{similarity}\n")

def read_output_summary_run_metrics(output_summary_path: Path) -> dict[str, float | None]:
    metrics = {
        "latest_coverage": None,
        "failure_detection_accuracy": None,
    }
    with output_summary_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("{") or ":" not in line:
                continue
            key, value = line.split(":", 1)
            if key == "latest_coverage_upper_bound":
                metrics["latest_coverage"] = optional_float(value)
            elif key == "failure_detection_accuracy":
                metrics["failure_detection_accuracy"] = optional_float(value)
    return metrics

def build_step_evaluate_rows(
    output_summary_path: Path,
    start_timestamp: str,
    timestamp: str | None = None,
    scenario_similarity: float | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    row_timestamp = timestamp or utc_timestamp()
    run_metrics = read_output_summary_run_metrics(output_summary_path)

    
    config_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    
    seen_configs = set()

    for record in read_jsonl_records(output_summary_path):
        values: dict[str, Any] = {
            "start_timestamp": start_timestamp,
            "round_index": int(record["round_index"]),
            "test_id": int(record["test_id"]),
            "step_index": int(record["step_index"]),
            "failure_score_v2": record.get("failure_score_v2"),
            "fused_score": record.get("fused_score"),
            "timestamp": row_timestamp,
            "scenario_similarity": scenario_similarity,
            "latest_coverage": run_metrics["latest_coverage"],
            "failure_detection_accuracy": run_metrics["failure_detection_accuracy"],

        }
        for name in SCENARIO_PARAMETER_NAMES:
            values[name] = record.get(name)
        for name in METRIC_NAMES:
            values[name] = record.get(name)
            
        test_id = int(record["test_id"])
        if test_id not in seen_configs:
            config_rows.append(project_database_row(TEST_SCENARIO_CONFIG_COLUMNS, values))
            seen_configs.add(test_id)
            
        step_rows.append(project_database_row(STEP_EVALUATE_COLUMNS, values))
    return config_rows, step_rows


def build_failure_analysis_rows(
    integrated_results_path: Path,
    start_timestamp: str,
    timestamp: str | None = None,
) -> list[dict[str, Any]]:
    row_timestamp = timestamp or utc_timestamp()
    rows: list[dict[str, Any]] = []
    for record in iter_jsonl(integrated_results_path):
        attack_record = record.get("attack_classifier")
        if not isinstance(attack_record, Mapping):
            attack_record = {}
        attribution_record = record.get("attribution_analysis")
        if not isinstance(attribution_record, Mapping):
            attribution_record = {}
        contribution_by_feature = attribution_record.get("contribution_by_feature")
        if not isinstance(contribution_by_feature, Mapping):
            contribution_by_feature = {}

        original_round_index = record.get("original_round_index")
        original_test_id = record.get("original_test_id")
        values: dict[str, Any] = {
            "start_timestamp": start_timestamp,
            "original_round_index": None if original_round_index is None else int(original_round_index),
            "original_test_id": None if original_test_id is None else int(original_test_id),
            "merged_round_index": (
                None
                if record.get("merged_round_index") is None
                else int(record["merged_round_index"])
            ),
            "merged_test_id": (
                None
                if record.get("merged_test_id") is None
                else int(record["merged_test_id"])
            ),
            "true_attack_type": attack_record.get("true_attack_type"),
            "predicted_attack_type": attack_record.get("predicted_attack_type"),
            "target_field": attribution_record.get("target_field"),
            "target_value": attribution_record.get("target_value"),
            "predicted_fail_score": attribution_record.get("predicted_fail_score"),
            "absolute_error": attribution_record.get("absolute_error"),
            "timestamp": row_timestamp,
        }
        if original_round_index is not None and original_test_id is not None:
            values["step_evaluation"] = format_step_evaluation_id(
                start_timestamp,
                original_round_index,
                original_test_id,
            )
        else:
            values["step_evaluation"] = None
        for name in SCENARIO_PARAMETER_NAMES:
            values[name] = contribution_by_feature.get(name)
        rows.append(project_database_row(FAILURE_ANALYSIS_COLUMNS, values))
    return rows


def resolve_report_path(path_text: Any) -> Path | None:
    text = str(path_text or "").strip()
    if not text:
        return None
    return resolve_path(text)


def load_json_report(path_text: Any) -> dict[str, Any] | None:
    path = resolve_report_path(path_text)
    if path is None or not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def collect_part3_scenario_reports(part3_report: Mapping[str, Any]) -> list[dict[str, Any]]:
    outputs = part3_report.get("outputs", {})
    if not isinstance(outputs, Mapping):
        outputs = {}

    report_paths = outputs.get("scenario_report_jsons")
    reports: list[dict[str, Any]] = []
    if isinstance(report_paths, Mapping):
        for path_text in report_paths.values():
            loaded = load_json_report(path_text)
            if loaded is not None:
                reports.append(loaded)
    if reports:
        return reports

    if outputs.get("healing_results_json"):
        return [dict(part3_report)]
    return []


def map_attack_label_to_id(label: Any) -> int | None:
    if label is None:
        return None
    return ATTACK_TYPE_TO_ID.get(str(label))


def build_self_healing_rows(
    part3_report: Mapping[str, Any],
    start_timestamp: str,
    timestamp: str | None = None,
) -> list[dict[str, Any]]:
    row_timestamp = timestamp or utc_timestamp()
    rows: list[dict[str, Any]] = []

    for scenario_report in collect_part3_scenario_reports(part3_report):
        outputs = scenario_report.get("outputs", {})
        if not isinstance(outputs, Mapping):
            outputs = {}
        healing_payload = load_json_report(outputs.get("healing_results_json"))
        if healing_payload is None:
            healing = scenario_report.get("healing", {})
            if isinstance(healing, Mapping):
                healing_payload = load_json_report(healing.get("results_json"))
        if healing_payload is None:
            continue

        selected = scenario_report.get("selected_scenario", {})
        if not isinstance(selected, Mapping):
            selected = {}
        resolved_targets = scenario_report.get("resolved_targets", {})
        if not isinstance(resolved_targets, Mapping):
            resolved_targets = {}
        observations = resolved_targets.get("observations_by_satellite", {})
        if not isinstance(observations, Mapping):
            observations = {}

        result_by_satellite = healing_payload.get("result_by_satellite", {})
        if not isinstance(result_by_satellite, Mapping):
            continue

        for node_id, result in result_by_satellite.items():
            if not isinstance(result, Mapping):
                continue
            observation = observations.get(str(node_id), {})
            if not isinstance(observation, Mapping):
                observation = {}
            logger = result.get("logger")
            if not isinstance(logger, Mapping):
                logger = {}

            attack_label = (
                observation.get("attack_type_name")
                or observation.get("attack_label")
                or selected.get("predicted_attack_type")
            )
            attack_type = observation.get("attack_type_id")
            if attack_type is None:
                attack_type = map_attack_label_to_id(attack_label)
            healing_time = result.get("healing_time")
            values = {
                "start_timestamp": start_timestamp,
                "test_id": selected.get("test_id"),
                "timestamp": row_timestamp,
                "ConstellationConfig": selected.get("constellation_id"),
                "node_id": result.get("node_id", node_id),
                "success": int(bool(result.get("success"))),
                "healing_level": result.get("healing_level"),
                "healing_time": healing_time,
                "message": result.get("message"),
                "fail_score": observation.get("fail_score", selected.get("fail_score")),
                "attack_type": attack_type,
                "attack_label": attack_label,
                "total_time": logger.get("total_time", healing_time),
                "final_level": logger.get("final_level", result.get("healing_level")),
            }
            rows.append(project_database_row(SELF_HEALING_COLUMNS, values))
    return rows


def require_file(path: Path, description: str) -> None:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")


def require_dir(path: Path, description: str) -> None:
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"{description} not found: {path}")


def require_keys(mapping: dict[str, Any], keys: Iterable[str], context: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise KeyError(f"{context} missing required keys: {missing}")


def resolve_fixed_constellation_config(config: dict[str, Any], env_md_path: Path) -> int:
    param_space = parse_experiment_md(str(env_md_path))
    md_values = param_space.get("ConstellationConfig")
    if md_values is not None:
        normalized = sorted({int(value) for value in md_values})
        if len(normalized) != 1:
            raise ValueError(
                "env_config.md must define exactly one ConstellationConfig for the closed-loop workflow. "
                f"Got: {normalized}"
            )
        return int(normalized[0])

    env_cfg = config.get("environment", {})
    if "ConstellationConfig" not in env_cfg:
        raise ValueError("ConstellationConfig must be defined in env_config.md or the YAML environment section.")
    return int(env_cfg["ConstellationConfig"])


def prepare_runtime_config(
    source_config_path: Path,
    output_path: Path,
    constellation_id: int,
    constellation2_agent_scope: str,
    region_orbit_block_size: int,
    region_sat_block_size: int,
) -> dict[str, Any]:
    runtime_config = copy.deepcopy(load_yaml(source_config_path))
    env_cfg = runtime_config.setdefault("environment", {})
    agent_cfg = runtime_config.setdefault("agent", {})
    env_cfg["ConstellationConfig"] = int(constellation_id)
    changes: dict[str, Any] = {
        "source_config": str(source_config_path),
        "runtime_config": str(output_path),
        "constellation_id": int(constellation_id),
        "environment_constellation_synced_from_env_md": True,
        "constellation2_agent_scope": constellation2_agent_scope,
        "region_agent_enabled_by_script": False,
    }

    if constellation_id == 2 and constellation2_agent_scope == "region":
        raw_ids = agent_cfg.get("region_agent_constellation_ids", [2])
        if isinstance(raw_ids, (str, int, float)):
            raw_ids = [raw_ids]
        region_ids = sorted({int(value) for value in raw_ids} | {2})
        agent_cfg["enable_region_agent_for_large_constellation"] = True
        agent_cfg["region_agent_constellation_ids"] = region_ids
        agent_cfg["region_orbit_block_size"] = int(region_orbit_block_size)
        agent_cfg["region_sat_block_size"] = int(region_sat_block_size)
        agent_cfg["intra_region_routing"] = "dijkstra"
        changes["region_agent_enabled_by_script"] = True
        changes["region_orbit_block_size"] = int(region_orbit_block_size)
        changes["region_sat_block_size"] = int(region_sat_block_size)

    raw_region_ids = agent_cfg.get("region_agent_constellation_ids", [2])
    if isinstance(raw_region_ids, (str, int, float)):
        raw_region_ids = [raw_region_ids]
    effective_region_ids = {int(value) for value in raw_region_ids}
    changes["region_agent_enabled_effective"] = (
        constellation_id == 2
        and parse_bool(agent_cfg.get("enable_region_agent_for_large_constellation", False))
        and 2 in effective_region_ids
    )

    dump_yaml(output_path, runtime_config)
    return changes


def prepare_runtime_analysis_config(
    source_config_path: Path,
    output_path: Path,
    analysis_data_root: Path,
    analysis_output_root: Path,
    archive_processed: bool,
    archive_root: Path,
    retraining_enabled: bool,
    attack_artifact: Path,
    attribution_artifact: Path,
    target_field: str,
    skip_attack: bool,
    skip_attribution: bool,
    retraining_registry_path: Path | None = None,
    retraining_candidate_root: Path | None = None,
    retraining_production_root: Path | None = None,
) -> dict[str, Any]:
    runtime_config = copy.deepcopy(load_jsonc(source_config_path))

    global_config = dict(runtime_config.get("global", {}))
    global_config["data_root"] = path_for_config(analysis_data_root)
    global_config["output_root"] = path_for_config(analysis_output_root)
    global_config["archive_processed"] = bool(archive_processed)
    global_config["archive_root"] = path_for_config(archive_root)
    runtime_config["global"] = global_config

    attack_config = dict(runtime_config.get("attack_classifier", {}))
    attack_config["enabled"] = not bool(skip_attack)
    attack_config["model_type"] = "random_forest"
    attack_config["feature_mode"] = "weak_scene"
    attack_config["artifact_path"] = path_for_config(attack_artifact)
    attack_config["input_csv"] = ""
    runtime_config["attack_classifier"] = attack_config

    attribution_config = dict(runtime_config.get("attribution_analysis", {}))
    attribution_config["enabled"] = not bool(skip_attribution)
    attribution_config["artifact_path"] = path_for_config(attribution_artifact)
    attribution_config["target_field"] = target_field
    attribution_config["input_csv"] = ""
    attribution_config["rounds_root"] = ""
    runtime_config["attribution_analysis"] = attribution_config

    retraining_config = dict(runtime_config.get("retraining", {}))
    retraining_config["enabled"] = bool(retraining_enabled)
    if retraining_registry_path is not None:
        retraining_config["registry_path"] = path_for_config(retraining_registry_path)
    if retraining_candidate_root is not None:
        retraining_config["candidate_root"] = path_for_config(retraining_candidate_root)
    if retraining_production_root is not None:
        retraining_config["production_root"] = path_for_config(retraining_production_root)
    runtime_config["retraining"] = retraining_config

    write_json(output_path, runtime_config)
    return {
        "source_analysis_config": str(source_config_path),
        "runtime_analysis_config": str(output_path),
        "analysis_data_root": str(analysis_data_root),
        "analysis_output_root": str(analysis_output_root),
        "archive_processed": bool(archive_processed),
        "archive_root": str(archive_root),
        "retraining_enabled": bool(retraining_enabled),
        "attack_model_type": "random_forest",
        "attack_feature_mode": "weak_scene",
        "attack_artifact": str(attack_artifact),
        "attribution_artifact": str(attribution_artifact),
        "target_field": target_field,
    }


def infer_analysis_data_root(simulation_output_root: Path, constellation_id: int) -> Path:
    resolved = simulation_output_root.resolve()
    if resolved.parent.name == str(constellation_id):
        return resolved.parent.parent
    return resolved


def validate_archive_ready_layout(
    analysis_data_root: Path,
    simulation_output_root: Path,
    constellation_id: int,
) -> None:
    try:
        relative_simulation_root = simulation_output_root.resolve().relative_to(analysis_data_root.resolve())
    except ValueError as exc:
        raise ValueError(
            "For archive/retraining, simulation output root must be under the analysis data root. "
            f"Got simulation_output_root={simulation_output_root}, analysis_data_root={analysis_data_root}."
        ) from exc

    parts = relative_simulation_root.parts
    if len(parts) != 2 or parts[0] != str(constellation_id):
        raise ValueError(
            "For archive/retraining, simulation output root must have the layout "
            "<analysis_data_root>/<ConstellationConfig>/<experiment_id>. "
            f"Got relative layout: {relative_simulation_root.as_posix()!r}."
        )


def command_to_text(cmd: Sequence[str]) -> str:
    return " ".join(str(part) for part in cmd)


def run_command(label: str, cmd: Sequence[str], cwd: Path, log_path: Path, dry_run: bool) -> None:
    print(f"\n[{label}] {command_to_text(cmd)}", flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        log_path.write_text(command_to_text(cmd) + "\n", encoding="utf-8")
        return

    with log_path.open("w", encoding="utf-8", errors="ignore") as log_handle:
        log_handle.write(command_to_text(cmd) + "\n\n")
        log_handle.flush()
        process = subprocess.Popen(
            list(cmd),
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_handle.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, list(cmd))


def build_simulation_cmd(
    args: argparse.Namespace,
    runtime_config_path: Path,
    env_md_path: Path,
    simulation_output_root: Path,
    single_attack_types: list[str],
) -> list[str]:
    cmd = [
        sys.executable,
        str(ITERATIVE_FAILURE_SIMULATION_SCRIPT),
        "--config",
        str(runtime_config_path),
        "--env-md",
        str(env_md_path),
        "--exploration-config",
        str(args.exploration_config),
        "--output-root",
        str(simulation_output_root),
        "--raw-log-root",
        str(simulation_output_root),
        "--generated-limit",
        str(args.generated_limit),
        "--scenarios-per-round",
        str(args.scenarios_per_round),
        "--seed-per-region",
        str(args.seed_per_region),
        "--coverage-target",
        str(args.coverage_target),
        "--min-samples-for-coverage-stop",
        str(args.min_samples_for_coverage_stop),
        "--stop-on-coverage-target",
        bool_text(args.stop_on_coverage_target),
        "--true-failure-policy",
        "strict",
        "--failure-decision-mode",
        args.failure_decision_mode,
        "--fused-model-type",
        "mlp_small",
        "--fit-decision-model-offline",
        "true",
        "--threshold-calibration-scope",
        "terminal_only",
        "--threshold-calibration-mode",
        "two_stage_stable",
        "--allow-multi-attacks-per-scenario",
        "false",
    ]
    if single_attack_types:
        cmd.extend(["--single-attack-types", ",".join(single_attack_types)])
    if resolve_exploration_cli_override(getattr(args, "threshold_objective", "")):
        cmd.extend(["--threshold-objective", str(args.threshold_objective)])
    if getattr(args, "threshold_min_precision", None) is not None:
        cmd.extend(["--threshold-min-precision", str(args.threshold_min_precision)])
    if resolve_exploration_cli_override(getattr(args, "threshold_split_mode", "")):
        cmd.extend(["--threshold-split-mode", str(args.threshold_split_mode)])
    if getattr(args, "threshold_split_holdout_ratio", None) is not None:
        cmd.extend(["--threshold-split-holdout-ratio", str(args.threshold_split_holdout_ratio)])
    if getattr(args, "threshold_split_late_window_ratio", None) is not None:
        cmd.extend(["--threshold-split-late-window-ratio", str(args.threshold_split_late_window_ratio)])
    if getattr(args, "threshold_split_holdout_late_fraction", None) is not None:
        cmd.extend(["--threshold-split-holdout-late-fraction", str(args.threshold_split_holdout_late_fraction)])
    if resolve_exploration_cli_override(getattr(args, "rolling_drift_analysis", "")):
        cmd.extend(["--rolling-drift-analysis", str(args.rolling_drift_analysis)])
    if getattr(args, "rolling_window_size", None) is not None:
        cmd.extend(["--rolling-window-size", str(args.rolling_window_size)])
    if getattr(args, "rolling_step_size", None) is not None:
        cmd.extend(["--rolling-step-size", str(args.rolling_step_size)])
    if getattr(args, "rolling_min_train_support", None) is not None:
        cmd.extend(["--rolling-min-train-support", str(args.rolling_min_train_support)])
    if getattr(args, "rolling_min_holdout_support", None) is not None:
        cmd.extend(["--rolling-min-holdout-support", str(args.rolling_min_holdout_support)])
    cmd.extend([
        "--online-backfill-after-each-round",
        "true",
        "--post-run-offline-recompute",
        "true",
        "--enable-accuracy-guard",
        bool_text(args.enable_accuracy_guard),
        "--min-failure-detection-accuracy",
        str(args.min_failure_detection_accuracy)
    ])
    if not args.resume:
        cmd.append("--reset-state")
    return cmd


def build_analysis_cmd(
    args: argparse.Namespace,
    analysis_config_path: Path,
    analysis_data_root: Path,
    analysis_output_root: Path,
    archive_processed: bool,
    archive_root: Path,
    attack_artifact: Path,
    attribution_artifact: Path,
) -> list[str]:
    cmd = [
        sys.executable,
        str(DEFAULT_ANALYSIS_SCRIPT),
        "--config",
        str(analysis_config_path),
        "--data-root",
        str(analysis_data_root),
        "--output-root",
        str(analysis_output_root),
        "--target-field",
        args.target_field,
    ]
    if archive_processed:
        cmd.append("--archive-processed")
        cmd.extend(["--archive-root", str(archive_root)])
    if args.skip_attack:
        cmd.append("--skip-attack")
    else:
        cmd.extend(["--attack-artifact", str(attack_artifact)])
    if args.skip_attribution:
        cmd.append("--skip-attribution")
    else:
        cmd.extend(["--attribution-artifact", str(attribution_artifact)])
    return cmd


def build_part3_cmd(
    args: argparse.Namespace,
    full_run_root: Path,
    run_root: Path,
    analysis_output_root: Path,
    part3_output_root: Path,
) -> list[str]:
    return [
        sys.executable,
        str(DEFAULT_PART3_SCRIPT),
        "--project-root",
        str(PROJECT_ROOT),
        "--full-run-root",
        str(full_run_root),
        "--run-root",
        str(run_root),
        "--analysis-output-root",
        str(analysis_output_root),
        "--output-root",
        str(part3_output_root),
        "--target-field",
        args.target_field,
        "--scenario-selection",
        args.part3_scenario_selection,
        "--region-orbit-block-size",
        str(args.region_orbit_block_size),
        "--region-sat-block-size",
        str(args.region_sat_block_size),
        "--constellation2-default-satellites-per-agent",
        str(args.part3_constellation2_default_satellites_per_agent),
        "--constellation2-max-satellites-per-agent",
        str(args.part3_constellation2_max_satellites_per_agent),
    ]


def validate_simulation_outputs(
    simulation_output_root: Path,
    target_field: str,
    constellation_id: int,
    region_agent_enabled: bool,
) -> dict[str, Any]:
    session_dir = simulation_output_root / "current_session"
    rounds_dir = session_dir / "rounds"
    output_summary_path = session_dir / "output_summary.txt"
    require_dir(session_dir, "closed-loop current_session directory")
    require_dir(rounds_dir, "closed-loop rounds directory")
    require_file(output_summary_path, "closed-loop output_summary.txt")

    step_records = read_jsonl_records(output_summary_path)
    if not step_records:
        raise ValueError(f"No JSON step records found in {output_summary_path}")

    step_required = ["round_index", "test_id", "step_index", *SCENARIO_PARAMETER_NAMES, *METRIC_NAMES]
    step_keys: set[tuple[int, int]] = set()
    for index, record in enumerate(step_records, start=1):
        require_keys(record, step_required, f"{output_summary_path} JSON record #{index}")
        step_keys.add((int(record["round_index"]), int(record["test_id"])))

    evalu_paths = sorted(rounds_dir.glob("round_*/evalu.txt"))
    if not evalu_paths:
        raise ValueError(f"No round_*/evalu.txt files found under {rounds_dir}")

    summary_records: list[dict[str, Any]] = []
    step_eval_count = 0
    for evalu_path in evalu_paths:
        for record in read_jsonl_records(evalu_path, prefix="TEST_SUMMARY_JSON:"):
            scenario = record.get("scenario")
            if not isinstance(scenario, dict):
                raise ValueError(f"TEST_SUMMARY_JSON missing scenario dict in {evalu_path}")
            require_keys(scenario, SCENARIO_PARAMETER_NAMES, f"{evalu_path} scenario")
            require_keys(record, ["round_index", "test_id", target_field], f"{evalu_path} TEST_SUMMARY_JSON")
            summary_records.append(record)
        step_eval_count += len(read_jsonl_records(evalu_path, prefix="STEP_EVAL_JSON:"))

    if not summary_records:
        raise ValueError(f"No TEST_SUMMARY_JSON rows found under {rounds_dir}")

    summary_keys = {
        (int(record["round_index"]), int(record["test_id"]))
        for record in summary_records
    }
    missing_from_summary = sorted(step_keys - summary_keys)
    missing_from_steps = sorted(summary_keys - step_keys)
    if missing_from_summary or missing_from_steps:
        raise ValueError(
            "Simulation output interface mismatch between output_summary.txt and round evalu files. "
            f"Missing from TEST_SUMMARY_JSON: {missing_from_summary[:8]}; "
            f"missing from output_summary JSON records: {missing_from_steps[:8]}"
        )

    env_list_count = sum(1 for _ in rounds_dir.glob("round_*/env_list.jsonl"))
    failure_score_count = sum(1 for _ in rounds_dir.glob("round_*/failure_scores.jsonl"))
    detected_constellations = sorted(
        {
            int(record["scenario"]["ConstellationConfig"])
            for record in summary_records
            if isinstance(record.get("scenario"), dict)
        }
    )
    if 2 in detected_constellations:
        constellation2_agent_scope = "region_agent" if region_agent_enabled else "project_config_original"
    else:
        constellation2_agent_scope = "not_applicable"
    report = {
        "stage": "simulation_to_analysis",
        "status": "passed",
        "analysis_subject": "agent",
        "satellite_detail_policy": (
            "ignored: downstream analysis consumes scenario/test-level metrics and does not read "
            "AttackSummary satellite lines or ActionLog satellite traces"
        ),
        "configured_constellation_id": int(constellation_id),
        "detected_constellation_configs": detected_constellations,
        "constellation2_agent_scope": constellation2_agent_scope,
        "configured_constellation_detected": int(constellation_id) in detected_constellations,
        "session_dir": str(session_dir),
        "rounds_dir": str(rounds_dir),
        "output_summary_path": str(output_summary_path),
        "round_evalu_count": len(evalu_paths),
        "env_list_file_count": env_list_count,
        "failure_scores_file_count": failure_score_count,
        "step_record_count_for_attack_classifier": len(step_records),
        "test_sample_count": len(summary_keys),
        "test_summary_count_for_attribution": len(summary_records),
        "step_eval_count": step_eval_count,
        "required_scenario_fields": list(SCENARIO_PARAMETER_NAMES),
        "required_metric_fields": list(METRIC_NAMES),
        "attribution_target_field": target_field,
    }
    return report


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL row at {path}:{line_no}") from exc
            if isinstance(payload, dict):
                yield payload


def find_disallowed_satellite_keys(payload: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if str(key).strip().lower() in {"satellite", "satellite_name"}:
                found.append(path)
            found.extend(find_disallowed_satellite_keys(value, path))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            found.extend(find_disallowed_satellite_keys(value, f"{prefix}[{index}]"))
    return found


def validate_analysis_outputs(
    analysis_output_root: Path,
    simulation_report: dict[str, Any],
    attack_enabled: bool,
    attribution_enabled: bool,
) -> dict[str, Any]:
    summary_path = analysis_output_root / "pipeline_summary.json"
    require_file(summary_path, "2.2 pipeline summary")
    pipeline_summary = json.loads(summary_path.read_text(encoding="utf-8"))

    integrated = pipeline_summary.get("integrated_results")
    if not isinstance(integrated, dict) or not integrated.get("output_jsonl"):
        raise ValueError("pipeline_summary.json missing integrated_results.output_jsonl")
    integrated_path = resolve_path(str(integrated["output_jsonl"]))
    require_file(integrated_path, "integrated attack/attribution JSONL")

    test_count = int(simulation_report["test_sample_count"])
    attack_summary = pipeline_summary.get("attack_classifier", {})
    attribution_summary = pipeline_summary.get("attribution_analysis", {})

    if attack_enabled:
        attack_count = int(attack_summary.get("sample_count", -1))
        if attack_count != test_count:
            raise ValueError(f"Attack classifier sample_count={attack_count}, expected {test_count}.")
    if attribution_enabled:
        attribution_count = int(attribution_summary.get("sample_count", -1))
        if attribution_count != test_count:
            raise ValueError(f"Attribution sample_count={attribution_count}, expected {test_count}.")

    rows = list(iter_jsonl(integrated_path))
    disallowed_keys: list[str] = []
    for index, row in enumerate(rows[:20], start=1):
        disallowed_keys.extend(f"row[{index}].{path}" for path in find_disallowed_satellite_keys(row))
    if disallowed_keys:
        raise ValueError(
            "Integrated analysis output contains satellite-level fields, but this pipeline is agent-oriented: "
            + ", ".join(disallowed_keys[:8])
        )

    if attack_enabled and attribution_enabled:
        matched_count = int(integrated.get("matched_count", -1))
        if matched_count != test_count:
            raise ValueError(f"Integrated matched_count={matched_count}, expected {test_count}.")

    return {
        "stage": "analysis_outputs",
        "status": "passed",
        "pipeline_summary_path": str(summary_path),
        "integrated_results_path": str(integrated_path),
        "integrated_row_count": len(rows),
        "attack_enabled": bool(attack_enabled),
        "attack_sample_count": attack_summary.get("sample_count"),
        "attribution_enabled": bool(attribution_enabled),
        "attribution_sample_count": attribution_summary.get("sample_count"),
        "integrated_results": integrated,
        "archive_summary": pipeline_summary.get("archive_summary"),
        "retraining_summary": pipeline_summary.get("retraining_summary"),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full project workflow: closed-loop simulation, failure judgement, attack type analysis, and fail-score attribution."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Initial YAML config passed to the simulator.")
    parser.add_argument(
        "--env-config-json",
        default="",
        help="Scenario parameter dictionary as a JSON object or Python dict literal.",
    )
    parser.add_argument(
        "--env-config-file",
        default="",
        help="JSON/YAML file containing the scenario parameter dictionary.",
    )
    parser.add_argument(
        "--env-md",
        default="",
        help=(
            "Legacy scenario parameter md file. "
            f"If omitted and no dictionary environment input is provided, defaults to {DEFAULT_ENV_MD}."
        ),
    )
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT), help="Root directory for full pipeline runs.")
    parser.add_argument("--run-id", default="", help="Optional run id. Defaults to a timestamp.")
    parser.add_argument("--experiment-id", default="", help="Optional experiment id under the analysis data root.")
    parser.add_argument("--simulation-output-root", default="", help="Override simulation output root.")
    parser.add_argument(
        "--analysis-data-root",
        default="",
        help="Override the data root scanned by the 2.2 pipeline. Defaults to <run_root>/simulation_data.",
    )
    parser.add_argument("--analysis-output-root", default="", help="Override 2.2 analysis output root.")
    parser.add_argument(
        "--exploration-config",
        default=str(DEFAULT_EXPLORATION_CONFIG),
        help="YAML config for scenario exploration options such as single_attack_types.",
    )
    parser.add_argument(
        "--analysis-config",
        default=str(DEFAULT_ANALYSIS_CONFIG),
        help="Base 2.2 JSONC config used to build the runtime analysis config.",
    )
    parser.add_argument("--skip-simulation", action="store_true", help="Use an existing simulation output root.")
    parser.add_argument("--resume", action="store_true", help="Resume current_session instead of resetting it.")
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU execution. By default the full pipeline requires CUDA to avoid slow accidental CPU runs.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands and write manifest only.")

    parser.add_argument("--generated-limit", type=int, default=500)
    parser.add_argument("--scenarios-per-round", type=int, default=16)
    parser.add_argument("--seed-per-region", type=int, default=48)
    parser.add_argument(
        "--coverage-target",
        type=float,
        default=0.90,
        help=(
            "Target conditional coverage for the current exploration constraints. "
            "Coverage is computed over the continuous scenario space of sampled scenarios."
        ),
    )
    parser.add_argument(
        "--min-samples-for-coverage-stop",
        type=int,
        default=100,
        help="Minimum total samples required before allowing coverage-target early stop.",
    )
    parser.add_argument(
        "--stop-on-coverage-target",
        type=parse_bool,
        default=True,
        help="true: stop early when the conditional coverage upper bound reaches target with enough samples.",
    )
    parser.add_argument(
        "--single-attack-types",
        type=str,
        default="",
        help="Single-attack whitelist; leave blank to avoid restricting attack types.",
    )
    parser.add_argument(
        "--threshold-objective",
        choices=("f1", "recall_at_precision", "accuracy", "balanced_accuracy"),
        default="",
        help="Temporary CLI override for online threshold objective; the exploration config remains the default source.",
    )
    parser.add_argument(
        "--threshold-min-precision",
        type=float,
        default=None,
        help="Temporary CLI override for online threshold minimum precision floor; the exploration config remains the default source.",
    )
    parser.add_argument(
        "--threshold-split-mode",
        choices=("chronological", "stratified_random", "stratified_late_holdout"),
        default="",
        help="Temporary CLI override for online threshold split mode; the exploration config remains the default source.",
    )
    parser.add_argument("--threshold-split-holdout-ratio", type=float, default=None)
    parser.add_argument("--threshold-split-late-window-ratio", type=float, default=None)
    parser.add_argument("--threshold-split-holdout-late-fraction", type=float, default=None)
    parser.add_argument(
        "--rolling-drift-analysis",
        choices=("on", "off"),
        default=None,
        help="Temporary CLI override for post-run rolling drift analysis switch.",
    )
    parser.add_argument("--rolling-window-size", type=int, default=None)
    parser.add_argument("--rolling-step-size", type=int, default=None)
    parser.add_argument("--rolling-min-train-support", type=int, default=None)
    parser.add_argument("--rolling-min-holdout-support", type=int, default=None)
    parser.add_argument(
        "--failure-decision-mode",
        choices=("single_fused_score", "direct_failure_model"),
        default="single_fused_score",
    )
    parser.add_argument("--enable-accuracy-guard", type=parse_bool, default=True)
    parser.add_argument("--min-failure-detection-accuracy", type=float, default=0.90)

    parser.add_argument("--skip-attack", action="store_true", help="Skip attack type inference.")
    parser.add_argument("--skip-attribution", action="store_true", help="Skip fail-score attribution inference.")
    parser.add_argument("--attack-artifact", default=str(DEFAULT_ATTACK_ARTIFACT))
    parser.add_argument("--attribution-artifact", default=str(DEFAULT_ATTRIBUTION_ARTIFACT))
    parser.add_argument("--target-field", choices=VALID_ATTRIBUTION_TARGETS, default="fused_score")
    parser.add_argument(
        "--no-archive-processed",
        action="store_true",
        help="Keep processed simulation experiments in place instead of moving them to archive_root.",
    )
    parser.add_argument("--archive-root", default=str(DEFAULT_ARCHIVE_ROOT), help="Archive root for processed runs.")
    parser.add_argument(
        "--skip-retraining",
        action="store_true",
        help="Run inference/attribution only; do not launch candidate retraining after archiving.",
    )
    parser.add_argument("--retraining-registry-path", default="", help="Override retraining registry JSON path.")
    parser.add_argument("--candidate-root", default="", help="Override candidate artifact root.")
    parser.add_argument("--production-root", default="", help="Override production artifact root.")
    parser.add_argument("--skip-part3", action="store_true", help="Skip the embedded online self-healing stage.")
    parser.add_argument("--part3-output-root", default="", help="Override online self-healing output root.")
    parser.add_argument(
        "--part3-scenario-selection",
        choices=("per_healing_level", "highest_score_with_attack", "highest_score"),
        default="per_healing_level",
        help=(
            "Scenario selection policy for the online self-healing stage. "
            "per_healing_level selects one scenario for each Level 1-4 healing class; "
            "legacy highest_score* values are deprecated aliases in Part3."
        ),
    )
    parser.add_argument("--part3-constellation2-default-satellites-per-agent", type=int, default=5)
    parser.add_argument("--part3-constellation2-max-satellites-per-agent", type=int, default=10)

    parser.add_argument(
        "--constellation2-agent-scope",
        choices=("region", "original"),
        default="region",
        help="For ConstellationConfig=2, region enables region-level agents in the runtime config.",
    )
    parser.add_argument("--region-orbit-block-size", type=int, default=5)
    parser.add_argument("--region-sat-block-size", type=int, default=5)
    return parser.parse_args(argv)


def main(
    env_config: Mapping[str, Any] | None = None,
    argv: Sequence[str] | None = None,
    database_writer: Any | None = None,
) -> int:
    args = parse_args(argv)
    if not args.allow_cpu:
        os.environ["ISATCR_REQUIRE_CUDA"] = "1"
    select_torch_device("full project pipeline")
    if args.skip_attack and args.skip_attribution:
        raise ValueError("At least one downstream analysis module must be enabled.")

    config_path = resolve_path(args.config)
    analysis_config_path = resolve_path(args.analysis_config)
    exploration_config_path = resolve_path(args.exploration_config)
    env_config_dict = resolve_env_config_dict(args, env_config)
    single_attack_types = resolve_single_attack_types(
        args.single_attack_types,
        exploration_config_path,
    )
    legacy_env_md_path = resolve_path(args.env_md) if args.env_md else None
    if legacy_env_md_path is None and env_config_dict is None:
        legacy_env_md_path = DEFAULT_ENV_MD
    require_file(config_path, "initial YAML config")
    if legacy_env_md_path is not None:
        require_file(legacy_env_md_path, "initial env md config")
    require_file(exploration_config_path, "scenario exploration config")
    require_file(DEFAULT_ANALYSIS_SCRIPT, "2.2 analysis script")
    require_file(analysis_config_path, "2.2 base pipeline config")
    if not args.skip_part3:
        require_file(DEFAULT_PART3_SCRIPT, "online self-healing pipeline entry script")
    if args.part3_constellation2_max_satellites_per_agent < 1 or args.part3_constellation2_max_satellites_per_agent > 10:
        raise ValueError("--part3-constellation2-max-satellites-per-agent must be in [1, 10].")
    if args.part3_constellation2_default_satellites_per_agent < 1:
        raise ValueError("--part3-constellation2-default-satellites-per-agent must be >= 1.")

    attack_artifact = resolve_path(args.attack_artifact)
    attribution_artifact = resolve_path(args.attribution_artifact)
    archive_root = resolve_path(args.archive_root)
    retraining_registry_path = (
        resolve_path(args.retraining_registry_path) if args.retraining_registry_path else None
    )
    retraining_candidate_root = resolve_path(args.candidate_root) if args.candidate_root else None
    retraining_production_root = resolve_path(args.production_root) if args.production_root else None
    archive_processed = not bool(args.no_archive_processed)
    retraining_enabled = not bool(args.skip_retraining)
    if retraining_enabled and not archive_processed:
        raise ValueError("--skip-retraining is required when --no-archive-processed is used.")
    if not args.skip_attack:
        require_file(attack_artifact, "attack classifier artifact")
    if not args.skip_attribution:
        require_file(attribution_artifact, "fail-score attribution artifact")
        
    if database_writer and hasattr(database_writer, "task_id"):
        run_id = database_writer.task_id
    else:
        run_id = args.run_id.strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
        
    run_root = resolve_path(args.run_root) / run_id
    experiment_id = args.experiment_id.strip() or run_id
    analysis_data_root = (
        resolve_path(args.analysis_data_root)
        if args.analysis_data_root
        else run_root / "simulation_data"
    )
    simulation_output_root = (
        resolve_path(args.simulation_output_root)
        if args.simulation_output_root
        else analysis_data_root / "pending_constellation" / experiment_id
    )
    analysis_output_root = (
        resolve_path(args.analysis_output_root)
        if args.analysis_output_root
        else run_root / "analysis"
    )
    part3_output_root = (
        resolve_path(args.part3_output_root)
        if args.part3_output_root
        else run_root / "part3_rebuild"
    )
    logs_dir = run_root / "logs"
    runtime_config_path = run_root / "runtime_config.yaml"
    env_md_path = legacy_env_md_path or run_root / "runtime_env_config.md"
    runtime_analysis_config_path = run_root / "runtime_pipeline_config.jsonc"
    manifest_path = run_root / "run_manifest.json"
    interface_report_path = run_root / "interface_contract_report.json"
    if env_config_dict is not None:
        write_env_config_md(env_md_path, env_config_dict)

    source_config = load_yaml(config_path)
    constellation_id = resolve_fixed_constellation_config(source_config, env_md_path)
    if not args.simulation_output_root:
        simulation_output_root = analysis_data_root / str(constellation_id) / experiment_id
    elif not args.analysis_data_root:
        analysis_data_root = infer_analysis_data_root(simulation_output_root, constellation_id)
    if archive_processed:
        validate_archive_ready_layout(
            analysis_data_root=analysis_data_root,
            simulation_output_root=simulation_output_root,
            constellation_id=constellation_id,
        )
    config_changes = prepare_runtime_config(
        source_config_path=config_path,
        output_path=runtime_config_path,
        constellation_id=constellation_id,
        constellation2_agent_scope=args.constellation2_agent_scope,
        region_orbit_block_size=args.region_orbit_block_size,
        region_sat_block_size=args.region_sat_block_size,
    )
    analysis_config_changes = prepare_runtime_analysis_config(
        source_config_path=analysis_config_path,
        output_path=runtime_analysis_config_path,
        analysis_data_root=analysis_data_root,
        analysis_output_root=analysis_output_root,
        archive_processed=archive_processed,
        archive_root=archive_root,
        retraining_enabled=retraining_enabled,
        attack_artifact=attack_artifact,
        attribution_artifact=attribution_artifact,
        target_field=args.target_field,
        skip_attack=args.skip_attack,
        skip_attribution=args.skip_attribution,
        retraining_registry_path=retraining_registry_path,
        retraining_candidate_root=retraining_candidate_root,
        retraining_production_root=retraining_production_root,
    )

    simulation_cmd = build_simulation_cmd(
        args,
        runtime_config_path,
        env_md_path,
        simulation_output_root,
        single_attack_types=single_attack_types,
    )
    analysis_cmd = build_analysis_cmd(
        args,
        runtime_analysis_config_path,
        analysis_data_root,
        analysis_output_root,
        archive_processed,
        archive_root,
        attack_artifact,
        attribution_artifact,
    )
    part3_cmd = (
        []
        if args.skip_part3
        else build_part3_cmd(
            args=args,
            full_run_root=resolve_path(args.run_root),
            run_root=run_root,
            analysis_output_root=analysis_output_root,
            part3_output_root=part3_output_root,
        )
    )
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "project_root": str(PROJECT_ROOT),
        "run_root": str(run_root),
        "experiment_id": experiment_id,
        "initial_config": str(config_path),
        "runtime_config": str(runtime_config_path),
        "analysis_config": str(analysis_config_path),
        "runtime_analysis_config": str(runtime_analysis_config_path),
        "env_md": str(env_md_path),
        "analysis_data_root": str(analysis_data_root),
        "simulation_output_root": str(simulation_output_root),
        "analysis_output_root": str(analysis_output_root),
        "part3_output_root": str(part3_output_root),
        "archive_processed": bool(archive_processed),
        "archive_root": str(archive_root),
        "retraining_enabled": bool(retraining_enabled),
        "interface_report": str(interface_report_path),
        "config_changes": config_changes,
        "analysis_config_changes": analysis_config_changes,
        "commands": {
            "simulation": simulation_cmd,
            "analysis": analysis_cmd,
            "part3": part3_cmd,
        },
        "dry_run": bool(args.dry_run),
        "skip_simulation": bool(args.skip_simulation),
        "skip_attack": bool(args.skip_attack),
        "skip_attribution": bool(args.skip_attribution),
        "skip_part3": bool(args.skip_part3),
        "database_writes": [],
    }
    write_json(manifest_path, manifest)

    if not args.skip_simulation:
        run_command(
            "SIMULATION",
            simulation_cmd,
            cwd=PROJECT_ROOT,
            log_path=logs_dir / "simulation.log",
            dry_run=args.dry_run,
        )
    elif not args.simulation_output_root:
        raise ValueError("--skip-simulation requires --simulation-output-root.")

    if args.dry_run:
        run_command(
            "ANALYSIS",
            analysis_cmd,
            cwd=PROJECT_ROOT,
            log_path=logs_dir / "analysis.log",
            dry_run=True,
        )
        if not args.skip_part3:
            run_command(
                "PART3_REBUILD",
                part3_cmd,
                cwd=PROJECT_ROOT,
                log_path=logs_dir / "part3_rebuild.log",
                dry_run=True,
            )
        print(f"\n[DRY-RUN] Manifest written to {manifest_path}")
        return 0

    simulation_report = validate_simulation_outputs(
        simulation_output_root=simulation_output_root,
        target_field=args.target_field,
        constellation_id=constellation_id,
        region_agent_enabled=bool(config_changes["region_agent_enabled_effective"]),
    )
    write_json(interface_report_path, {"simulation": simulation_report})
    database_write_summaries: list[dict[str, Any]] = manifest["database_writes"]
    """TODO 返回前端结果，并且写入数据库的StepEvaluate表中 和TestScenarioConfig表格中
    """
    if database_stage_enabled(database_writer):
        # 1. 计算当前这批场景配置的“相似度”
        similarity_report = build_scenario_similarity_report(
            output_summary_path=Path(simulation_report["output_summary_path"])
        )
        sim_val = similarity_report.get("similarity", 0.0)

        # 2. 将 similarity 值传入表拆分函数，以便随配置一起落盘
        config_rows, step_rows = build_step_evaluate_rows(
            output_summary_path=Path(simulation_report["output_summary_path"]),
            start_timestamp=run_id,
            scenario_similarity=sim_val,
        )
        database_write_summaries.append(
            write_database_rows(database_writer, TEST_SCENARIO_CONFIG_TABLE, config_rows)
        )
        database_write_summaries.append(
            write_database_rows(database_writer, STEP_EVALUATE_TABLE, step_rows)
        )
        # ========= 新增：推进状态进度条 =========
        if hasattr(database_writer, 'update_stage'):
            database_writer.update_stage('Analysis') 
        write_json(manifest_path, manifest)

    run_command(
        "ANALYSIS",
        analysis_cmd,
        cwd=PROJECT_ROOT,
        log_path=logs_dir / "analysis.log",
        dry_run=False,
    )

    analysis_report = validate_analysis_outputs(
        analysis_output_root=analysis_output_root,
        simulation_report=simulation_report,
        attack_enabled=not args.skip_attack,
        attribution_enabled=not args.skip_attribution,
    )
    interface_payload: dict[str, Any] = {"simulation": simulation_report, "analysis": analysis_report}
    """
    TODO 返回前端结果，并且写入数据库的FailureAnalysis表中
     integrated_results_path字段在analysis_report中，路径为analysis_output_root/integrated_results.jsonl
     integrated_results.jsonl文件中每一行是一个测试样本的分析结果，包含scenario参数、attack分析结果、attribution分析结果等
    """
    if database_stage_enabled(database_writer):
        failure_rows = build_failure_analysis_rows(
            integrated_results_path=Path(analysis_report["integrated_results_path"]),
            start_timestamp=run_id,
        )
        database_write_summaries.append(
            write_database_rows(database_writer, FAILURE_ANALYSIS_TABLE, failure_rows)
        )

        # ========= 新增：推进状态进度条 =========
        if hasattr(database_writer, 'update_stage'):
            database_writer.update_stage('SelfHealing')
        write_json(manifest_path, manifest)

    part3_report: dict[str, Any] | None = None
    if not args.skip_part3:
        run_command(
            "PART3_REBUILD",
            part3_cmd,
            cwd=PROJECT_ROOT,
            log_path=logs_dir / "part3_rebuild.log",
            dry_run=False,
        )
        part3_report_path = part3_output_root / "part3_report.json"
        require_file(part3_report_path, "online self-healing report")
        part3_report = json.loads(part3_report_path.read_text(encoding="utf-8"))
        interface_payload["part3_rebuild"] = {
            "stage": "part3_rebuild_self_healing",
            "status": part3_report.get("status"),
            "report_path": str(part3_report_path),
            "outputs": part3_report.get("outputs", {}),
            "selection_mode": part3_report.get("selection_mode"),
            "selection_summary": part3_report.get("selection_summary"),
            "selected_scenarios": part3_report.get("selected_scenarios"),
            "selected_scenario": part3_report.get("selected_scenario"),
            "scenario_reports": part3_report.get("scenario_reports"),
            "constellation_branch": part3_report.get("constellation_branch"),
            "warnings": part3_report.get("warnings", []),
        }
        """
         TODO 返回前端结果，并且写入数据库的SelfHealing表中
         selected_scenario字段在part3_report中，表示在线自愈阶段选择的测试样本的scenario参数
         scenario_reports字段在part3_report中，是一个列表，每个元素是一个被选中进行在线自愈的测试样本的分析结果，包含scenario参数、healing_level、healing_actions、post_healing_analysis等
         constellation_branch字段在part3_report中，表示在线自愈阶段选择的测试样本所属
         """
        if database_stage_enabled(database_writer):
            self_healing_rows = build_self_healing_rows(
                part3_report=part3_report,
                start_timestamp=run_id,
            )
            database_write_summaries.append(
                write_database_rows(database_writer, SELF_HEALING_TABLE, self_healing_rows)
            )
            write_json(manifest_path, manifest)

    write_json(interface_report_path, interface_payload)

    manifest["status"] = "completed"
    manifest["completed_at"] = datetime.now().isoformat(timespec="seconds")
    manifest["pipeline_summary"] = analysis_report["pipeline_summary_path"]
    manifest["integrated_results"] = analysis_report["integrated_results_path"]
    if part3_report is not None:
        manifest["part3_report"] = str(part3_output_root / "part3_report.json")
        manifest["part3_status"] = part3_report.get("status")
        manifest["part3_outputs"] = part3_report.get("outputs", {})
    write_json(manifest_path, manifest)

    print("\n[DONE] Full project pipeline completed.")
    print(f"Run root: {run_root}")
    print(f"Interface report: {interface_report_path}")
    print(f"Pipeline summary: {analysis_report['pipeline_summary_path']}")
    print(f"Integrated results: {analysis_report['integrated_results_path']}")
    if part3_report is not None:
        print(f"Online self-healing report: {part3_output_root / 'part3_report.json'}")
        outputs = part3_report.get("outputs", {})
        if isinstance(outputs, dict) and outputs.get("visualization_html"):
            print(f"Online self-healing visualization: {outputs['visualization_html']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
