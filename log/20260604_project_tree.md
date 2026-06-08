# 项目树文档

生成日期：2026-06-04

本文档记录当前重构版项目的主要目录结构与模块用途。为保持可读性，已省略 `.git`、`.venv`、`__pycache__`、模型权重、卫星/地面原始数据、批量运行输出等大体量或临时目录的详细文件列表。

## 顶层结构

```text
algo-目录重构版/
|-- .gitattributes
|-- .gitignore
|-- agent.md
|-- c2_explore.md
|-- constellation_tle_order.py
|-- ENVIRONMENT_SETUP.md
|-- requirements.txt
|-- run_full_project_pipeline.py
|-- run_pipeline.ps1
|-- config/
|-- data/
|-- log/
`-- src/
```

## 目录树

```text
algo-目录重构版/
|-- config/
|   `-- environment/
|       `-- env_config.md
|-- data/
|   |-- Ground_Data/                         # 地面站/地面网络相关数据，详情省略
|   |-- model_weights/                       # 训练模型权重，详情省略
|   |-- ne_data/                             # 网络/地理等辅助数据
|   |-- Satellite_Data/                      # 卫星相关数据，详情省略
|   |-- tmp_data/
|   |   |-- archive_non_mainflow/
|   |   |-- closed_loop_outputs/             # 闭环运行输出，详情省略
|   |   |-- data_archive/                    # 归档数据，详情省略
|   |   |-- full_project_runs/               # 全流程运行输出，详情省略
|   |   |-- invalid_batch/
|   |   |-- model_artifacts/
|   |   |-- smoke_runs/                      # 冒烟测试输出，详情省略
|   |   `-- training_process_data/           # 训练过程数据，详情省略
|   `-- train_config_archive/
|       |-- train_NewDDQN_dueling.yaml
|       |-- train_NewDDQN_dueling_shuffle.yaml
|       |-- train_NewPPO_shuffle.yaml
|       |-- train_PureDDQN.yaml
|       |-- train_PureDDQN_dueling.yaml
|       |-- train_PureDDQN_dueling_shuffle.yaml
|       |-- train_PureDDQN_shuffle.yaml
|       |-- train_PurePPO.yaml
|       |-- train_PurePPO_shuffle.yaml
|       `-- train_WeakDQN.yaml
|-- log/
|   |-- 085229_online_threshold_issue.md
|   |-- 152543_distribution_imbalance_issue.md
|   |-- 20260604_new_old_version_diff_summary.md
|   |-- 20260604_project_tree.md
|   |-- 20260604_refactor_execution_log.md
|   |-- 20260604_refactor_migration_guide.md
|   |-- 20260604_refactor_problem_analysis.md
|   |-- 20260604_refactor_structure_plan.md
|   |-- coverage_vs_failure_boundary_semantics_issue.md
|   |-- single_attack_type_config_unification.md
|   |-- 修改.md
|   `-- 重构
`-- src/
    |-- project_paths.py
    |-- failure_and_attribution_analysis/
    |   |-- COVERAGE_SEMANTICS.md
    |   |-- agent_failure_evaluator.py
    |   |-- deep_ensemble_network.py
    |   |-- failure_boundary_explorer.py
    |   |-- parameter_interfaces.py
    |   |-- scenario_parameter_generator.py
    |   |-- config/
    |   |   `-- scenario_exploration.yaml
    |   `-- 2.2module/
    |       |-- build_fail_score_training_csv.py
    |       |-- pipeline_config.jsonc
    |       |-- pipeline_config.template.jsonc
    |       |-- run_2_2module_pipeline.py
    |       |-- run_candidate_retraining.py
    |       |-- attack_classifier/
    |       |   |-- attack_type_classifier.py
    |       |   |-- attack_type_classifier_weak_scene_rf_sklearn17.pkl
    |       |   |-- discrete_attack_labels.py
    |       |   |-- extract_output_csv.py
    |       |   `-- merge_output_summaries.py
    |       `-- attribution_analysis/
    |           |-- fail_score_contribution_model.py
    |           `-- fail_score_contribution_model_fused_score_from_data_root.pkl
    |-- iterative_testing/
    |   |-- __init__.py
    |   |-- Base_Agents.py
    |   |-- Draw_Graph_Quiker.py
    |   |-- gpu_runtime.py
    |   |-- iterative_failure_simulation.py
    |   |-- Make_Satellite_Graph.py
    |   |-- PRC.py
    |   |-- Read_Ground_Imformation.py
    |   |-- RL_environment_for_computing.py
    |   |-- run_batch_experiments.py
    |   |-- SatelliteNetworkSimulation.py
    |   |-- SatelliteNetworkSimulator_Beta.py
    |   |-- SatelliteNetworkSimulator_Computing.py
    |   `-- mdp_attacks/
    |       |-- __init__.py
    |       |-- attack_monitor.py
    |       |-- ExperiencePool_attack.py
    |       |-- mdp_action_attack.py
    |       |-- mdp_Reward_attack.py
    |       |-- mdp_StateObservation_attack.py
    |       |-- mdp_StateTransfer_attack.py
    |       `-- ModelTamp_attack.py
    |-- online_self_healing/
    |   |-- __init__.py
    |   |-- demo_consensus.py
    |   |-- demo_run.py
    |   |-- framework.py
    |   |-- node.py
    |   |-- orbit.py
    |   |-- topology.py
    |   |-- train_NewDDQN_dueling_shuffle.yaml
    |   |-- consensus_engine/
    |   |   |-- __init__.py
    |   |   |-- catalog.py
    |   |   |-- engine.py
    |   |   |-- explorer.py
    |   |   |-- models.py
    |   |   `-- output_reader.py
    |   |-- pipeline_integration/
    |   |   |-- __init__.py
    |   |   |-- runner.py
    |   |   `-- run_from_full_pipeline.py
    |   `-- self_healing_module/
    |       |-- batch_redundant_healing.py
    |       |-- federated_healing.py
    |       |-- healing_logger.py
    |       |-- healing_orchestrator.py
    |       |-- immune_healing.py
    |       |-- redundant_healing.py
    |       `-- self_healing_docs.md
    |-- tests/
    |   |-- run_initial_baseline_flow_smoke.py
    |   |-- test_checkpoint_state_cache.py
    |   `-- test_initial_baseline_gate.py
    `-- tools/
```

## 关键模块说明

- `run_full_project_pipeline.py`：项目全流程主入口脚本，负责串联仿真、失效分析、归因、自愈等流程。
- `run_pipeline.ps1`：Windows PowerShell 运行入口，用于便捷启动项目流水线。
- `src/project_paths.py`：统一项目路径解析模块，重构后各模块应优先通过它定位配置、数据和输出目录。
- `src/iterative_testing/`：迭代测试、卫星网络仿真、强化学习环境、批量实验与 MDP 攻击相关代码。
- `src/failure_and_attribution_analysis/`：失效评估、边界探索、参数接口、场景参数生成与归因分析模块。
- `src/failure_and_attribution_analysis/2.2module/`：2.2 模块流水线、候选重训练、攻击分类与 fail score 贡献模型相关代码。
- `src/online_self_healing/`：在线自愈框架、共识引擎、流水线集成与多类自愈策略实现。
- `src/tests/`：当前重构版保留的冒烟测试、缓存测试和 baseline gate 测试。
- `config/environment/`：环境配置说明文档。
- `data/train_config_archive/`：训练配置归档，保留多种 DQN/PPO 配置。
- `data/tmp_data/`：运行期中间数据、输出归档、冒烟测试输出和模型产物目录。
- `log/`：问题分析、迁移说明、执行记录、差异总结与本项目树文档。

## 省略规则

- 省略 `.git/`、`.venv/`、`__pycache__/` 等版本控制、虚拟环境和解释器缓存目录。
- 省略 `data/Ground_Data/`、`data/Satellite_Data/`、`data/model_weights/` 等大体量数据或权重目录的内部文件。
- 省略 `data/tmp_data/full_project_runs/`、`smoke_runs/`、`closed_loop_outputs/`、`training_process_data/` 等运行输出目录的内部文件。
- 保留关键 `.py`、`.md`、`.yaml`、`.jsonc`、`.pkl` 文件名，以便定位项目主要入口和模型产物。
