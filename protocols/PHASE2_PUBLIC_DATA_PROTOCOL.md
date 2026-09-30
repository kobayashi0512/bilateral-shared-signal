# Phase 2：公开数据方法开发协议

> **Historical planning record.** This protocol was written before the final analysis
> roles were frozen. It records the candidate plan at that time, not the final
> dataset roles or current manuscript claims. Follow
> [`PAPILA_EXTERNAL_FREEZE_PROTOCOL_V1.md`](PAPILA_EXTERNAL_FREEZE_PROTOCOL_V1.md),
> [`REVISION_V6_PROTOCOL.md`](REVISION_V6_PROTOCOL.md), and the current result
> reports for the reported analysis. PAPILA is now a consumed, locked final
> external readout, while CFP–OCT is a previously inspected development stress set.

版本：2026-09-21  
目标：在等待受控数据授权的同时，完成可独立复现的患者级双眼方法原型。

## 数据范围

### 主开发集

- DeepDRiD 本地 regular training：300 位患者、左右眼多视图、DR/质量标签；
- PAPILA：官方公开包取得后加入，患者级双眼图像、青光眼临床标签、视盘/视杯结构标注。

DeepDRiD 的官方 CSV 没有独立 eye 列；本项目先运行
`normalize_deepdrid_csv.py`，只解析官方 `image_id` 中的 `l/r` token，并在
输出中写入 `eye_source=DeepDRiD_documented_image_id_schema`。通用审计器仍不
对任意文件名做推断。

### 辅助压力集

- 本地 CFP–OCT paired manifest：只用于缺失眼/跨模态输入压力测试；不把 DME 当成系统性风险标签。

### 暂不纳入主训练

ODIR、China-Fundus-CIMT、BRSET、mBRSET 在原始 metadata 和许可证完成审计前，只记录为候选，不参与模型调参。

## Split 规则

1. 以 `patient_id` 为最小不可拆分单位；
2. 同一患者的左右眼、同眼多视图、重复访视必须进入同一 split；
3. 采用固定随机种子 `20260921`；
4. 目标比例 train/validation/test = 70/15/15；
5. 先按患者级主标签分层，再随机分配；标签缺失者单独记录，不静默丢弃；
6. 不修改其他项目已有的 frozen test split；本项目输出放在 `bilateral_shared_signal/audit_outputs/phase2_splits/`。

DeepDRiD 每眼有多视图。manifest split 保留全部视图；仅在输入条件 dry-run
中选择每眼第一条记录，正式模型必须预先声明同眼多视图 pooling 规则。

## 四类最小实验

### E1：单眼与双眼融合

- single-eye encoder；
- mean/max/attention bilateral pooling；
- shared + eye-specific decomposition；
- 报告患者级而非图像级指标。

### E2：缺失眼

- 完整双眼；
- 随机缺失一眼；
- 质量门控后缺失一眼；
- 训练时完整、测试时缺失，以及训练/测试都允许缺失两种设定。

### E3：左右眼交换

- 原始配对；
- 随机跨患者交换另一只眼；
- 同患者双眼顺序交换；
- 期望模型对跨患者交换敏感、对同患者顺序交换稳定。

### E4：跨域与质量

- DeepDRiD source/domain 留一；
- PAPILA 结构辅助迁移；
- 低质量/人工遮挡/模糊扰动；
- 域身份 probe 作为泄漏和 shortcut 检查，不作为主要性能目标。

## 指标

- 任务性能：数据集自身标签的 AUROC/AUPRC/F1 或 MAE；
- 双眼共享性：同患者跨眼相似度 vs 跨患者相似度，报告 AUROC 与效应量；
- 缺失眼鲁棒性：完整双眼到单眼的性能下降 Δ；
- 交换敏感性：正确配对与跨患者交换的分离度；
- 质量鲁棒性：扰动前后性能差和校准差；
- 域泄漏：source/camera/domain probe 性能，并与主任务分开解释。

## 最低 baseline

1. 单眼独立模型；
2. 左右眼特征平均；
3. 左右眼拼接；
4. 共享编码器 + 眼别 token；
5. 共享表示约束 + 患者标签任务；
6. 缺失眼 mask 版本。

任何复杂模型必须至少击败 2 和 3，并在 E2/E3 不显著退化，才能进入论文主结果。

## 失败判据

- 仅在同数据集随机图像 split 上提升；
- 交换另一只眼后性能不变；
- 域 probe 几乎完美而主任务跨域崩溃；
- 缺失眼下降完全由图像数量而非表示策略解释；
- PAPILA/DeepDRiD 的标签定义差异未被分开报告。
