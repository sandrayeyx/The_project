import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = next(parent for parent in CURRENT_FILE.parents if (parent / "src").is_dir())
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from project_paths import (  # noqa: E402
    DEFAULT_TRAIN_CONFIG_PATH,
    ENV_CONFIG_PATH,
    ITERATIVE_FAILURE_SIMULATION_SCRIPT,
    SMOKE_RUNS_ROOT,
)


DEFAULT_CONFIG = DEFAULT_TRAIN_CONFIG_PATH
DEFAULT_ENV_MD = ENV_CONFIG_PATH
DEFAULT_OUTPUT_ROOT = SMOKE_RUNS_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a near-real initial baseline flow check using the current env_config.md. "
            "The command executes round_000, triggers baseline gating, and stops before new scenario exploration."
        )
    )
    parser.add_argument("--python", default=sys.executable, help="Python interpreter used to launch the simulation.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Training YAML config.")
    parser.add_argument("--env-md", default=str(DEFAULT_ENV_MD), help="Environment markdown config.")
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root directory where this smoke run will create its timestamped output folder.",
    )
    parser.add_argument(
        "--run-name",
        default="",
        help="Optional fixed run directory name. Defaults to a timestamped folder under output-root.",
    )
    parser.add_argument(
        "--keep-output",
        action="store_true",
        help="Keep output files even when the run fails. By default failed runs are preserved as well; this flag is for clarity.",
    )
    parser.add_argument(
        "--cleanup-success",
        action="store_true",
        help="Delete the run directory after a successful smoke run.",
    )
    return parser.parse_args()


def build_command(args: argparse.Namespace, run_root: Path) -> list[str]:
    return [
        args.python,
        str(ITERATIVE_FAILURE_SIMULATION_SCRIPT),
        "--config",
        str(Path(args.config).resolve()),
        "--env-md",
        str(Path(args.env_md).resolve()),
        "--output-root",
        str(run_root),
        "--raw-log-root",
        str(run_root),
        "--generated-limit",
        "0",
        "--scenarios-per-round",
        "1",
        "--seed-per-region",
        "1",
        "--min-scenarios-per-round",
        "1",
        "--stop-on-coverage-target",
        "false",
        "--online-backfill-after-each-round",
        "false",
        "--post-run-offline-recompute",
        "false",
        "--allow-multi-attacks-per-scenario",
        "false",
        "--reset-state",
    ]


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    run_name = args.run_name.strip() or f"initial_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_root = output_root / run_name
    output_root.mkdir(parents=True, exist_ok=True)

    cmd = build_command(args, run_root)
    print(f"[SMOKE] project_root={PROJECT_ROOT}")
    print(f"[SMOKE] env_md={Path(args.env_md).resolve()}")
    print(f"[SMOKE] run_root={run_root}")
    print("[SMOKE] command=")
    print(" ".join(f'"{part}"' if " " in part else part for part in cmd))

    completed = subprocess.run(cmd, cwd=str(PROJECT_ROOT))

    if completed.returncode == 0:
        print("[SMOKE] result=PASS")
        print("[SMOKE] meaning=round_000 completed and the initial baseline gate did not abort the run")
        if args.cleanup_success:
            shutil.rmtree(run_root, ignore_errors=True)
            print(f"[SMOKE] cleanup=removed {run_root}")
        return 0

    print("[SMOKE] result=FAIL")
    if completed.returncode == 2:
        print("[SMOKE] meaning=the initial baseline gate rejected round_000 and the process exited cleanly without a traceback")
    else:
        print("[SMOKE] meaning=the near-real flow aborted due to a runtime error; inspect the command output above")
    print(f"[SMOKE] artifacts={run_root}")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
