# 0608 fused 训练/校准入口条件调整

## 问题描述

本记录聚焦 `single_fused_score` 识别链路失效的问题，重点分析：

1. `terminal_hard_failure` 为什么被引入。
2. 它当前是否被过度用作 fused 训练/校准入口过滤条件。
3. 如果优先不动 `terminal_hard_failure` 规则本体，应该如何调整 fused 训练/校准入口。

## 背景 / 输入信息

- 当前运行模式：`single_fused_score`
- 当前阈值校准范围：`terminal_only`
- 当前结果摘要：
  - `predicted_failure_count: 0`
  - `true_failure_count: 14`
  - `threshold_split_train_support: 0`
  - `threshold_update_status: frozen`
  - `healthy_baseline_count: 0`
- 关键代码关系：
  - `terminal_hard_failure` 由终态 reward / packet loss / delay / throughput 规则生成
  - `true_failure_v2_strict = convergence_true_failure and terminal_hard_failure`
  - fused 训练与阈值校准前，会先按 `terminal_only` 过滤样本

## 问题列表

1. `terminal_hard_failure` 当前同时承担三种职责：
   - strict 标签闸门
   - no-attack baseline 有效性闸门
   - fused 训练/校准入口过滤器

2. fused 训练/校准入口过窄。
   当前实现只允许 `terminal_hard_failure=True` 的样本进入 fused 训练与阈值校准，导致一部分 `true_failure_v2=True` 但末态未触发硬失效规则的样本被排除。

3. 训练支持数被筛空。
   当前日志显示 `threshold_split_train_support: 0`，这说明 fused 入口过滤和当前样本分布不匹配，已经影响到训练与阈值更新。

4. 这次问题更像“入口条件设计不适配”，而不是“`terminal_hard_failure` 规则本身错误”。
   如果直接修改 `terminal_hard_failure` 规则，会同步影响：
   - strict 标签定义
   - baseline gate
   - constellation 2 fragile / invalid 的边界

## 采用方案

优先不修改 `_terminal_hard_failure()` 规则本体，先修改 fused 训练/校准入口条件。

目标：

- 保留 `terminal_hard_failure` 作为终态硬失效与 baseline gate 机制。
- 放宽 fused 训练/校准样本的进入条件。
- 恢复 fused 分数的有效训练与阈值更新。

## 关键决策及原因

### 决策 1

暂不改 `terminal_hard_failure` 规则。

原因：

- 规则存在明确代码语义和单元测试保护。
- 它原始职责更偏向“终态硬失败判定”和“baseline gate”，不是单纯为 fused 训练服务。
- 当前更强的证据指向 fused 入口过滤过窄。

### 决策 2

优先改 fused 训练/校准入口。

原因：

- 这是当前识别失败最直接的链路问题。
- 改动范围更集中，回归验证更清晰。
- 能先验证“support 被筛空”是否就是主因。

## 修改思路

### 方案 A：全量样本进入 fused 训练/校准

做法：

- 去掉 fused 训练矩阵构建和阈值校准阶段对 `terminal_hard_failure=True` 的强过滤。
- 保留 `terminal_hard_failure_flag` 作为 fused 特征输入，不删除这一维特征。

优点：

- 最直接恢复训练支持数。
- fused 模型能看到完整正负样本分布。

风险：

- 终态较弱异常样本也会进入训练，边界噪声会变大。

### 方案 B：放宽为“`terminal_hard_failure=True` 或 `true_failure_v2=True`”

做法：

- 终态硬失败样本仍然保留。
- 所有已被标记为 `true_failure_v2=True` 的样本也允许进入 fused 训练/校准。
- 负样本仍从全体非失效样本或有效候选里抽取。

优点：

- 比方案 A 更保守。
- 能补回这轮最关键的一批“真实失效但未触发硬失效规则”的样本。

风险：

- 如果负样本仍然过少或分布偏斜，support 改善可能有限。

### 方案 C：训练入口放宽，阈值校准分层

做法：

- fused 模型训练用更宽样本集。
- 阈值校准同时记录：
  - 全量样本阈值表现
  - `terminal_hard_failure` 子集阈值表现
- 最终按 objective 选择阈值。

优点：

- 兼顾训练支持数和终态强约束。

风险：

- 实现复杂度更高，不适合作为第一步。

### 当前推荐

优先实现方案 B。

如果方案 B 仍无法恢复训练支持数或 AUC 明显偏低，再升级到方案 A。

## 执行结果

- 已确认 `terminal_hard_failure` 的设计意图主要来自代码与测试。
- 已确认这次更适合修改 fused 训练/校准入口条件，而不是先改硬失效规则。
- 已确定优先方向：修改 fused 训练/校准入口条件。

## 待办事项

1. 统计本轮 `true_failure_v2=True` 且 `terminal_hard_failure=False` 的样本数量。
2. 实施方案 B，放宽 fused 训练/校准入口。
3. 修改后重点验证：
   - `threshold_split_train_support`
   - `primary_score_holdout_auc`
   - `fused_score` 分布
   - `predicted_failure_count`
4. 若 support 仍然不足，再评估是否切到方案 A。

## 更新时间

2026-06-08

## 追加执行记录

### 已完成修改

- 在 `ClosedLoopFailureSimulation` 中新增 `_is_fused_effective_record(record)`。
- `single_fused_score` 的 fused 训练矩阵入口已切换为：
  - `terminal_hard_failure=True`
  - 或 `_resolve_true_failure_v2_value(record)=True`
- `single_fused_score` 的阈值校准有效样本索引已切换为同一套入口规则。
- `direct_failure_model` 相关入口未改。
- 新增单元测试覆盖：
  - true failure but non-terminal-hard-fail
  - terminal hard fail but negative label
  - pure negative exclusion
  - strict policy label resolution
  - direct mode still terminal-only

### 当前验证结果

- 代码变更已完成。
- 目标测试已尝试运行，但当前机器的系统 Python 缺少依赖：
  - `pytest` 不存在
  - `unittest` 运行到测试导入阶段时缺少 `numpy`

### 后续验证建议

在具备项目依赖的 Python 环境中继续执行：

- `python -m unittest .\src\tests\test_initial_baseline_gate.py`
- 或项目既有虚拟环境中的等效测试命令

同时建议复跑当前 `data-analysis` 生成流程，重点核对：

- `threshold_split_train_support`
- `threshold_update_status`
- `primary_score_holdout_auc`
- `fused_score` 是否仍整列为 0
- `predicted_failure_count` 是否大于 0
