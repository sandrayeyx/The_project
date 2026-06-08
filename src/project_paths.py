from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
DATA_ROOT = PROJECT_ROOT / "data"
CONFIG_ROOT = PROJECT_ROOT / "config"
LOG_ROOT = PROJECT_ROOT / "log"

GROUND_DATA_ROOT = DATA_ROOT / "Ground_Data"
SATELLITE_DATA_ROOT = DATA_ROOT / "Satellite_Data"
NE_DATA_ROOT = DATA_ROOT / "ne_data"
MODEL_WEIGHTS_ROOT = DATA_ROOT / "model_weights"
ONLINE_SELF_HEALING_MODEL_WEIGHTS_ROOT = MODEL_WEIGHTS_ROOT
TRAIN_CONFIG_ARCHIVE_ROOT = DATA_ROOT / "train_config_archive"
ANALYSIS_ARTIFACTS_ROOT = DATA_ROOT / "analysis_artifacts"
ATTACK_CLASSIFIER_ARTIFACTS_ROOT = ANALYSIS_ARTIFACTS_ROOT / "attack_classifier"
ATTRIBUTION_ANALYSIS_ARTIFACTS_ROOT = ANALYSIS_ARTIFACTS_ROOT / "attribution_analysis"

TMP_DATA_ROOT = DATA_ROOT / "tmp_data"
FULL_PROJECT_RUNS_ROOT = TMP_DATA_ROOT / "full_project_runs"
SMOKE_RUNS_ROOT = TMP_DATA_ROOT / "smoke_runs"
CLOSED_LOOP_OUTPUTS_ROOT = TMP_DATA_ROOT / "closed_loop_outputs"
DATA_ARCHIVE_ROOT = TMP_DATA_ROOT / "data_archive"
MODEL_ARTIFACTS_ROOT = TMP_DATA_ROOT / "model_artifacts"
TRAINING_PROCESS_DATA_ROOT = TMP_DATA_ROOT / "training_process_data"
INVALID_BATCH_ROOT = TMP_DATA_ROOT / "invalid_batch"
ARCHIVE_NON_MAINFLOW_ROOT = TMP_DATA_ROOT / "archive_non_mainflow"
DUPLICATE_FILE_ARCHIVE_ROOT = ARCHIVE_NON_MAINFLOW_ROOT / "duplicate_files"
EXPLORER_REPORT_ROOT = ARCHIVE_NON_MAINFLOW_ROOT / "explorer_reports"

ENVIRONMENT_CONFIG_ROOT = CONFIG_ROOT / "environment"
ENV_CONFIG_PATH = ENVIRONMENT_CONFIG_ROOT / "env_config.md"

ITERATIVE_TESTING_ROOT = SRC_ROOT / "iterative_testing"
FAILURE_AND_ATTRIBUTION_ROOT = SRC_ROOT / "failure_and_attribution_analysis"
ONLINE_SELF_HEALING_ROOT = SRC_ROOT / "online_self_healing"
TESTS_ROOT = SRC_ROOT / "tests"

DEFAULT_TRAIN_CONFIG_PATH = TRAIN_CONFIG_ARCHIVE_ROOT / "train_NewDDQN_dueling_shuffle.yaml"
SCENARIO_EXPLORATION_CONFIG_PATH = FAILURE_AND_ATTRIBUTION_ROOT / "config" / "scenario_exploration.yaml"
ITERATIVE_FAILURE_SIMULATION_SCRIPT = ITERATIVE_TESTING_ROOT / "iterative_failure_simulation.py"
ANALYSIS_PIPELINE_SCRIPT = FAILURE_AND_ATTRIBUTION_ROOT / "2.2module" / "run_2_2module_pipeline.py"
ANALYSIS_PIPELINE_CONFIG_PATH = FAILURE_AND_ATTRIBUTION_ROOT / "2.2module" / "pipeline_config.jsonc"
PART3_PIPELINE_SCRIPT = ONLINE_SELF_HEALING_ROOT / "pipeline_integration" / "run_from_full_pipeline.py"
PART3_AGENT_CONFIG_PATH = ONLINE_SELF_HEALING_ROOT / "train_NewDDQN_dueling_shuffle.yaml"
ATTACK_ARTIFACT_PATH = (
    ATTACK_CLASSIFIER_ARTIFACTS_ROOT
    / "attack_type_classifier_weak_scene_rf_sklearn17.pkl"
)
ATTRIBUTION_ARTIFACT_PATH = (
    ATTRIBUTION_ANALYSIS_ARTIFACTS_ROOT
    / "fail_score_contribution_model_fused_score_from_data_root.pkl"
)


def ensure_project_paths_on_syspath() -> None:
    for path in (PROJECT_ROOT, SRC_ROOT):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
