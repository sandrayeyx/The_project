#!/usr/bin/env python
"""Offline checker for fused-entry support using existing data-analysis outputs.

This script compares:
1. legacy terminal-only fused entry
2. scheme B fused entry: terminal_hard_failure OR resolved true label positive

It reads existing sample-level records from:
- rounds/round_*/failure_scores.jsonl

Optional metadata such as true-failure policy and split configuration is read
from output_summary.txt when available.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


DEFAULT_SPLIT_MODE = "stratified_late_holdout"
DEFAULT_HOLDOUT_RATIO = 0.20
DEFAULT_LATE_WINDOW_RATIO = 0.25
DEFAULT_HOLDOUT_LATE_FRACTION = 0.70
DEFAULT_THRESHOLD_MIN_SUPPORT = 30
DEFAULT_TRUE_FAILURE_POLICY = "strict"
DEFAULT_SPLIT_SEED = 42


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def _as_float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _as_int(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def parse_summary_header(summary_path: Path) -> Dict[str, object]:
    result: Dict[str, object] = {}
    if not summary_path.exists():
        return result

    with summary_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                break
            if line.startswith("{"):
                break
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                continue
            if value.startswith("{") or value.startswith("["):
                try:
                    result[key] = json.loads(value)
                    continue
                except json.JSONDecodeError:
                    pass
            if value.lower() in {"true", "false"}:
                result[key] = _as_bool(value)
                continue
            try:
                if any(ch in value for ch in (".", "e", "E")):
                    result[key] = float(value)
                else:
                    result[key] = int(value)
                continue
            except ValueError:
                result[key] = value
    return result


def discover_session_dirs(target: Path) -> List[Path]:
    if (target / "rounds").exists():
        return [target]
    children = [child for child in sorted(target.iterdir()) if child.is_dir() and (child / "rounds").exists()]
    return children


def load_round_records(session_dir: Path) -> List[Dict[str, object]]:
    round_files = sorted((session_dir / "rounds").glob("round_*/failure_scores.jsonl"))
    records: List[Dict[str, object]] = []
    for file_path in round_files:
        with file_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                records.append(payload)
    return records


def resolve_true_failure(record: Mapping[str, object], policy: str) -> bool:
    normalized = str(policy).strip().lower()
    if normalized == "strict":
        if "true_failure_v2_strict" in record:
            return _as_bool(record.get("true_failure_v2_strict"))
        return _as_bool(record.get("true_failure_v2"))
    if normalized == "relaxed":
        if "true_failure_v2_relaxed" in record:
            return _as_bool(record.get("true_failure_v2_relaxed"))
        return _as_bool(record.get("true_failure_v2"))
    return _as_bool(record.get("true_failure_v2"))


def is_terminal_only_effective(record: Mapping[str, object]) -> bool:
    return _as_bool(record.get("terminal_hard_failure", False))


def is_scheme_b_effective(record: Mapping[str, object], policy: str) -> bool:
    return is_terminal_only_effective(record) or resolve_true_failure(record, policy)


def make_split_rng(seed: int, context: str, support: int) -> random.Random:
    salt = sum(ord(ch) for ch in str(context))
    return random.Random(int(seed) + int(support) * 17 + salt)


def sample_positions_from_pool(
    pool_positions: Sequence[int],
    labels: Sequence[bool],
    target_count: int,
    rng: random.Random,
) -> List[int]:
    target = max(0, min(int(target_count), len(pool_positions)))
    if target <= 0 or not pool_positions:
        return []

    by_label = {
        False: [int(pos) for pos in pool_positions if not bool(labels[pos])],
        True: [int(pos) for pos in pool_positions if bool(labels[pos])],
    }
    for bucket in by_label.values():
        rng.shuffle(bucket)

    positive_ratio = float(len(by_label[True]) / len(pool_positions)) if pool_positions else 0.0
    true_target = min(int(round(target * positive_ratio)), len(by_label[True]))
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


def resolve_split_plan(
    labels: Sequence[bool],
    *,
    min_train_support: int,
    context: str,
    split_mode: str,
    holdout_ratio: float,
    late_window_ratio: float,
    holdout_late_fraction: float,
    seed: int,
) -> Dict[str, object]:
    support = len(labels)
    normalized_holdout_ratio = max(0.0, min(float(holdout_ratio), 0.49))
    holdout_count = int(math.floor(support * normalized_holdout_ratio))
    train_count = support - holdout_count
    result: Dict[str, object] = {
        "status": "updated",
        "mode": str(split_mode).strip().lower(),
        "seed": int(seed),
        "holdout_ratio": float(normalized_holdout_ratio),
        "late_window_ratio": float(late_window_ratio),
        "holdout_late_fraction": float(holdout_late_fraction),
        "support": int(support),
        "train_support": int(train_count),
        "holdout_support": int(holdout_count),
        "holdout_late_support": 0,
        "train_positions": [],
        "holdout_positions": [],
    }
    if train_count < max(2, int(min_train_support)):
        result["status"] = "frozen"
        result["reason"] = "insufficient_train_support"
        return result

    positions = list(range(support))
    mode = str(result["mode"])
    if holdout_count <= 0:
        result["train_positions"] = positions
        return result

    if mode == "chronological":
        result["train_positions"] = positions[:train_count]
        result["holdout_positions"] = positions[train_count:]
        late_window_size = max(1, int(math.ceil(support * float(result["late_window_ratio"]))))
        late_pool = set(positions[-late_window_size:])
        result["holdout_late_support"] = sum(1 for pos in result["holdout_positions"] if pos in late_pool)
        return result

    rng = make_split_rng(seed, context, support)
    if mode == "stratified_random":
        holdout_positions = sample_positions_from_pool(positions, labels, holdout_count, rng)
        train_positions = sorted(set(positions) - set(holdout_positions))
        result["train_positions"] = train_positions
        result["holdout_positions"] = holdout_positions
        return result

    late_window_ratio = max(0.05, min(float(late_window_ratio), 0.95))
    late_window_size = max(1, int(math.ceil(support * late_window_ratio)))
    late_pool = positions[-late_window_size:]
    early_pool = positions[:-late_window_size]
    holdout_late_target = int(round(holdout_count * max(0.0, min(float(holdout_late_fraction), 1.0))))
    holdout_late_target = min(holdout_count, holdout_late_target)
    holdout_positions = sample_positions_from_pool(late_pool, labels, holdout_late_target, rng)

    late_selected = set(holdout_positions)
    remaining_holdout = holdout_count - len(holdout_positions)
    if remaining_holdout > 0:
        early_selected = sample_positions_from_pool(
            [pos for pos in early_pool if pos not in late_selected],
            labels,
            remaining_holdout,
            rng,
        )
        holdout_positions.extend(early_selected)

    remaining_holdout = holdout_count - len(holdout_positions)
    if remaining_holdout > 0:
        supplemental_pool = [pos for pos in positions if pos not in set(holdout_positions)]
        supplemental = sample_positions_from_pool(supplemental_pool, labels, remaining_holdout, rng)
        holdout_positions.extend(supplemental)

    holdout_positions = sorted(set(int(pos) for pos in holdout_positions))
    if len(holdout_positions) > holdout_count:
        holdout_positions = holdout_positions[:holdout_count]
    train_positions = sorted(set(positions) - set(holdout_positions))
    result["train_positions"] = train_positions
    result["holdout_positions"] = holdout_positions
    result["train_support"] = len(train_positions)
    result["holdout_support"] = len(holdout_positions)
    late_pool_set = set(late_pool)
    result["holdout_late_support"] = sum(1 for pos in holdout_positions if pos in late_pool_set)
    if len(train_positions) < max(2, int(min_train_support)):
        result["status"] = "frozen"
        result["reason"] = "insufficient_train_support"
    return result


def evaluate_entry(
    records: Sequence[Mapping[str, object]],
    *,
    policy: str,
    entry_name: str,
    split_mode: str,
    holdout_ratio: float,
    late_window_ratio: float,
    holdout_late_fraction: float,
    threshold_min_support: int,
    split_seed: int,
) -> Dict[str, object]:
    if entry_name == "terminal_only":
        effective_records = [record for record in records if is_terminal_only_effective(record)]
    elif entry_name == "scheme_b":
        effective_records = [record for record in records if is_scheme_b_effective(record, policy)]
    else:
        raise ValueError(f"Unsupported entry_name: {entry_name}")

    labels = [resolve_true_failure(record, policy) for record in effective_records]
    terminal_count = sum(1 for record in effective_records if is_terminal_only_effective(record))
    positive_count = sum(1 for label in labels if label)
    negative_count = len(labels) - positive_count
    result: Dict[str, object] = {
        "entry_name": entry_name,
        "effective_support": len(effective_records),
        "terminal_hard_failure_support": terminal_count,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "status": "updated",
        "reason": "",
    }

    if len(effective_records) < max(2, int(threshold_min_support)):
        result["status"] = "frozen"
        result["reason"] = "insufficient_effective_support"
        result["train_support"] = 0
        result["holdout_support"] = 0
        result["holdout_late_support"] = 0
        return result

    split_plan = resolve_split_plan(
        labels=labels,
        min_train_support=max(2, int(threshold_min_support)),
        context="threshold_calibration_single_fused_score",
        split_mode=split_mode,
        holdout_ratio=holdout_ratio,
        late_window_ratio=late_window_ratio,
        holdout_late_fraction=holdout_late_fraction,
        seed=split_seed,
    )
    result["train_support"] = int(split_plan.get("train_support", 0))
    result["holdout_support"] = int(split_plan.get("holdout_support", 0))
    result["holdout_late_support"] = int(split_plan.get("holdout_late_support", 0))
    result["split_status"] = str(split_plan.get("status", "updated"))

    if str(split_plan.get("status", "updated")).strip().lower() != "updated":
        result["status"] = "frozen"
        result["reason"] = str(split_plan.get("reason", "insufficient_train_support"))
        return result

    train_positions = [int(pos) for pos in split_plan.get("train_positions", [])]
    train_labels = [labels[pos] for pos in train_positions]
    train_positive_count = sum(1 for label in train_labels if label)
    train_negative_count = len(train_labels) - train_positive_count
    result["train_positive_count"] = train_positive_count
    result["train_negative_count"] = train_negative_count
    if train_positive_count == 0 or train_negative_count == 0:
        result["status"] = "frozen"
        result["reason"] = "single_class_labels"
        return result

    return result


def collect_rescued_records(records: Sequence[Mapping[str, object]], policy: str) -> List[Dict[str, object]]:
    rescued: List[Dict[str, object]] = []
    for record in records:
        if is_terminal_only_effective(record):
            continue
        if not resolve_true_failure(record, policy):
            continue
        rescued.append(
            {
                "round_index": _as_int(record.get("round_index"), -1),
                "test_id": _as_int(record.get("test_id"), -1),
                "decision_score_v2": _as_float(record.get("decision_score_v2"), 0.0),
                "total_membership_v2": _as_float(record.get("total_membership_v2"), 0.0),
                "terminal_risk_score": _as_float(record.get("terminal_risk_score"), 0.0),
                "true_failure_v2": _as_bool(record.get("true_failure_v2")),
                "true_failure_v2_strict": _as_bool(record.get("true_failure_v2_strict")),
                "true_failure_v2_relaxed": _as_bool(record.get("true_failure_v2_relaxed")),
                "terminal_hard_failure": _as_bool(record.get("terminal_hard_failure")),
                "scenario": dict(record.get("scenario", {})) if isinstance(record.get("scenario"), Mapping) else {},
            }
        )
    return rescued


def analyze_session(session_dir: Path, rescued_limit: int) -> Dict[str, object]:
    header = parse_summary_header(session_dir / "output_summary.txt")
    policy = str(header.get("true_failure_policy", DEFAULT_TRUE_FAILURE_POLICY)).strip().lower() or DEFAULT_TRUE_FAILURE_POLICY
    split_mode = str(header.get("threshold_split_mode", DEFAULT_SPLIT_MODE)).strip().lower() or DEFAULT_SPLIT_MODE
    holdout_ratio = _as_float(header.get("threshold_split_holdout_ratio"), DEFAULT_HOLDOUT_RATIO)
    late_window_ratio = _as_float(header.get("threshold_split_late_window_ratio"), DEFAULT_LATE_WINDOW_RATIO)
    holdout_late_fraction = _as_float(header.get("threshold_split_holdout_late_fraction"), DEFAULT_HOLDOUT_LATE_FRACTION)
    threshold_min_support = _as_int(header.get("threshold_min_support"), DEFAULT_THRESHOLD_MIN_SUPPORT)
    split_seed = _as_int(header.get("threshold_split_seed"), DEFAULT_SPLIT_SEED)

    records = load_round_records(session_dir)
    global_label_stats = {
        "terminal_hard_failure_count": sum(1 for record in records if is_terminal_only_effective(record)),
        "true_failure_v2_count": sum(1 for record in records if _as_bool(record.get("true_failure_v2"))),
        "true_failure_v2_strict_count": sum(1 for record in records if _as_bool(record.get("true_failure_v2_strict"))),
        "true_failure_v2_relaxed_count": sum(1 for record in records if _as_bool(record.get("true_failure_v2_relaxed"))),
    }
    rescued_records = collect_rescued_records(records, policy)
    terminal_only = evaluate_entry(
        records,
        policy=policy,
        entry_name="terminal_only",
        split_mode=split_mode,
        holdout_ratio=holdout_ratio,
        late_window_ratio=late_window_ratio,
        holdout_late_fraction=holdout_late_fraction,
        threshold_min_support=threshold_min_support,
        split_seed=split_seed,
    )
    scheme_b = evaluate_entry(
        records,
        policy=policy,
        entry_name="scheme_b",
        split_mode=split_mode,
        holdout_ratio=holdout_ratio,
        late_window_ratio=late_window_ratio,
        holdout_late_fraction=holdout_late_fraction,
        threshold_min_support=threshold_min_support,
        split_seed=split_seed,
    )

    return {
        "session_dir": str(session_dir),
        "record_count": len(records),
        "true_failure_policy": policy,
        "split_mode": split_mode,
        "holdout_ratio": holdout_ratio,
        "late_window_ratio": late_window_ratio,
        "holdout_late_fraction": holdout_late_fraction,
        "threshold_min_support": threshold_min_support,
        "threshold_split_seed": split_seed,
        "header_predicted_failure_count": header.get("predicted_failure_count"),
        "header_true_failure_count": header.get("true_failure_count"),
        "header_threshold_update_status": header.get("threshold_update_status"),
        "global_label_stats": global_label_stats,
        "terminal_only": terminal_only,
        "scheme_b": scheme_b,
        "rescued_positive_count": len(rescued_records),
        "rescued_positive_examples": rescued_records[:rescued_limit],
    }


def print_report(report: Mapping[str, object]) -> None:
    print("=" * 80)
    print(f"Session: {report['session_dir']}")
    print(
        "Records={record_count} policy={true_failure_policy} split={split_mode} "
        "holdout_ratio={holdout_ratio:.2f} late_window_ratio={late_window_ratio:.2f} "
        "holdout_late_fraction={holdout_late_fraction:.2f} min_support={threshold_min_support}".format(**report)
    )
    print(
        "Header summary: predicted_failure_count={header_predicted_failure_count} "
        "true_failure_count={header_true_failure_count} threshold_update_status={header_threshold_update_status}".format(
            **report
        )
    )
    global_stats = report["global_label_stats"]
    print(
        "Global labels: terminal_hard={terminal_hard_failure_count} true_v2={true_failure_v2_count} "
        "strict={true_failure_v2_strict_count} relaxed={true_failure_v2_relaxed_count}".format(**global_stats)
    )
    for entry_key in ("terminal_only", "scheme_b"):
        entry = report[entry_key]
        print(
            f"[{entry['entry_name']}] status={entry.get('status')} reason={entry.get('reason', '') or '-'} "
            f"effective={entry.get('effective_support', 0)} train={entry.get('train_support', 0)} "
            f"holdout={entry.get('holdout_support', 0)} pos={entry.get('positive_count', 0)} "
            f"neg={entry.get('negative_count', 0)} train_pos={entry.get('train_positive_count', 0)} "
            f"train_neg={entry.get('train_negative_count', 0)}"
        )
    print(f"Rescued positives by scheme B: {report['rescued_positive_count']}")
    for example in report.get("rescued_positive_examples", []):
        print(
            "  - round={round_index} test_id={test_id} decision_score_v2={decision_score_v2:.4f} "
            "membership_v2={total_membership_v2:.4f} terminal_risk={terminal_risk_score:.4f} "
            "strict={true_failure_v2_strict} relaxed={true_failure_v2_relaxed}".format(**example)
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline fused-entry support checker")
    parser.add_argument(
        "target",
        nargs="?",
        default="data-analysis",
        help="Session dir with rounds/ or a parent dir that contains multiple sessions.",
    )
    parser.add_argument(
        "--rescued-limit",
        type=int,
        default=8,
        help="How many rescued positive samples to print.",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default="",
        help="Optional path to save the full report as JSON.",
    )
    parser.add_argument(
        "--policy-override",
        choices=("strict", "relaxed", "raw"),
        default="",
        help="Override the true-failure policy used by the offline comparison.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    target = Path(args.target).resolve()
    if not target.exists():
        raise FileNotFoundError(f"Target does not exist: {target}")

    session_dirs = discover_session_dirs(target)
    if not session_dirs:
        raise FileNotFoundError(f"No session directories with rounds/ found under: {target}")

    reports = []
    for session_dir in session_dirs:
        report = analyze_session(session_dir, rescued_limit=max(0, int(args.rescued_limit)))
        if args.policy_override:
            policy = str(args.policy_override)
            records = load_round_records(session_dir)
            report["true_failure_policy"] = policy
            report["terminal_only"] = evaluate_entry(
                records,
                policy=policy,
                entry_name="terminal_only",
                split_mode=str(report["split_mode"]),
                holdout_ratio=float(report["holdout_ratio"]),
                late_window_ratio=float(report["late_window_ratio"]),
                holdout_late_fraction=float(report["holdout_late_fraction"]),
                threshold_min_support=int(report["threshold_min_support"]),
                split_seed=int(report["threshold_split_seed"]),
            )
            report["scheme_b"] = evaluate_entry(
                records,
                policy=policy,
                entry_name="scheme_b",
                split_mode=str(report["split_mode"]),
                holdout_ratio=float(report["holdout_ratio"]),
                late_window_ratio=float(report["late_window_ratio"]),
                holdout_late_fraction=float(report["holdout_late_fraction"]),
                threshold_min_support=int(report["threshold_min_support"]),
                split_seed=int(report["threshold_split_seed"]),
            )
            rescued_records = collect_rescued_records(records, policy)
            report["rescued_positive_count"] = len(rescued_records)
            report["rescued_positive_examples"] = rescued_records[: max(0, int(args.rescued_limit))]
        reports.append(report)
    for report in reports:
        print_report(report)

    if args.json_out:
        output_path = Path(args.json_out).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"JSON report written to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
