# 0608 offline fused entry check

## 问题描述

构型 2 全流程复跑耗时较长，需要一个离线脚本直接复用 `data-analysis` 目录下的现有输出，快速判断这次方案 B 修改是否有效。

重点不是离线复现整套模型训练，而是先回答下面几个问题：
- 旧的 `terminal_only` fused 入口是否会把有效样本筛空；
- 方案 B 是否能把 `true_failure_v2` 正样本补回 fused 训练/校准入口；
- 放宽入口后，`effective_support / train_support / 正负样本数` 是否恢复到可训练状态。

## 背景 / 输入信息

- 当前主流程修改已经落地：`single_fused_score` 的 fused 训练/校准入口改为
  - `terminal_hard_failure=True`
  - 或 `_resolve_true_failure_v2_value(record)=True`
- `data-analysis/2-healthy/` 下存在可复用结果：
  - `rounds/round_*/failure_scores.jsonl`
  - `output_summary.txt`
  - `closed_loop_state.pt`
- 用户希望尽量不要修改 `src/` 代码文件，而是在 `/test` 下单独建立测试脚本。

## 采用方案

新增顶层脚本：`test/offline_fused_entry_check.py`

脚本行为：
1. 优先读取 `rounds/round_*/failure_scores.jsonl` 中的 sample 级记录；
2. 从 `output_summary.txt` 头部读取：
   - `true_failure_policy`
   - `threshold_split_mode`
   - `threshold_split_holdout_ratio`
   - `threshold_split_late_window_ratio`
   - `threshold_split_holdout_late_fraction`
   - `threshold_split_seed`
3. 离线复刻阈值校准前半段的入口筛选和 split plan；
4. 输出两套口径的对比结果：
   - `terminal_only`
   - `scheme_b = terminal_hard_failure or resolved true label positive`

## 关键决策及原因

1. 不在离线脚本中训练 fused MLP
   - 这次的主要目标是验证“入口样本是否还会被筛空”。
   - 不训练模型可以让脚本更快、更稳，也更适合频繁试验。

2. 直接读取 round 级 `failure_scores.jsonl`
   - 该文件本身就是 sample 级记录，字段完整。
   - 比从 `output_summary.txt` 的 step 记录反推 sample 状态更可靠。

3. 不只统计总数，还复刻 split plan
   - 只看 `effective_support` 还不够。
   - 还需要看 `train_support`、`holdout_support`、`train_positive_count`、`train_negative_count`，这样才能更接近主流程里 freeze 的真实原因。

## 执行结果

- 已新增脚本：`test/offline_fused_entry_check.py`
- 该脚本不修改 `src/` 业务逻辑，只读取现有 `data-analysis` 结果做离线比较分析。
- 脚本输出内容包括：
  - `terminal_only` 与 `scheme_b` 的有效样本数
  - 训练/验证 split 支持数
  - 正负样本数
  - freeze 原因
  - 被方案 B “救回”的正样本示例

## 待办事项

1. 在项目虚拟环境中运行：
   - `.\.venv\Scripts\python.exe .\test\offline_fused_entry_check.py .\data-analysis\2-healthy`
2. 重点关注输出：
   - `terminal_only` 是否仍然 `insufficient_effective_support` 或 `single_class_labels`
   - `scheme_b` 是否恢复为可更新状态
   - `rescued_positive_count` 是否明显大于 0
3. 如果 `scheme_b` 离线检查仍然 support 不足，再评估是否升级到方案 A。

## 更新时间

2026-06-08
