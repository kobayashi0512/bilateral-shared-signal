# R2.1 H-CBFD 冻结特征 screen v1

## 设计纪律

本实验是低容量机制筛选，不宣称新颖的 factorization 方法。严格保留 DeepDRiD 的
L1/L2/R1/R2；每个训练负对均在 **patient DR grade × 固定三档质量 tier** 内精确
derangement，不存在未匹配 fallback。所有模型仅以 DeepDRiD train 拟合；variant
选择只读取 validation 的 joint-matched pair-ID 与 non-collapse 检查。CFP–OCT
在 refit 后一次性冻结评估，绝不参与选择。

三个预注册 variant：`shared_only`（h 的对称 pair/classification baseline）、
`hierarchical_no_adversary`（h/d/e，d 预测 grade、e 对齐同眼视图、弱正交）、
`hierarchical_adversary`（前者加对 h 的 grade/quality gradient reversal）。每种 5 seeds。

## Validation：唯一的模型选择依据

| variant | joint_pair_auc_mean | joint_pair_auc_sd | pair_gap_mean | same_eye_e_mean | h_grade_probe_mean | h_quality_probe_mean | d_grade_probe_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hierarchical_adversary | 0.8554 | 0.0188 | 0.1787 | 0.9516 | 0.5135 | 0.6148 | 0.5191 |
| hierarchical_no_adversary | 0.8816 | 0.0174 | 0.2060 | 0.9499 | 0.4871 | 0.5622 | 0.5285 |
| shared_only | 0.8152 | 0.0411 | 0.2060 | 0.8656 | 0.5391 | 0.4948 | 0.5025 |

选择结果：`hierarchical_no_adversary`。理由：validation joint grade×quality pair-ID 均值
最高且 h/d/e 均未发生 variance collapse（最小 branch mean feature std=0.1722）。
这不是 test/CFP-OCT 驱动的选择。相同 eligible validation patients 上未训练 raw ResNet50
control 是 0.8425 ± 0.0118；
selected variant 为 0.8816，故通过预设的 internal non-inferiority gate。

## Refit train+val 后的一次性 test / 外部报告

| variant | test_full_bacc_mean | test_missing_left_bacc_mean | test_missing_right_bacc_mean | test_swap_bacc_mean | joint_pair_auc_mean | cfp_patientmax_auc_mean | cfp_eyemode_auc_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hierarchical_adversary | 0.6668 | 0.5874 | 0.5762 | 0.6668 | 0.8559 | 0.8413 | 0.8612 |
| hierarchical_no_adversary | 0.6857 | 0.6158 | 0.5673 | 0.6857 | 0.8579 | 0.8485 | 0.8701 |
| shared_only | 0.6863 | 0.6313 | 0.5625 | 0.6863 | 0.7689 | 0.7863 | 0.7983 |

`test_full/missing/swap` 是 patient grade endpoint 的辅助鲁棒性读数；本轮主终点仍是
joint-matched cross-eye correspondence。CFP–OCT 两列分别使用预定义 patient-max DME
和 legacy eye-mode schema 来限制外部 grade-matched negative，二者都仅作冻结外部读数。
相同 CFP-OCT eligible patients 的 raw controls 分别是 0.9015 和
0.9108；selected H-CBFD 分别为
0.8485/0.8701。其外部降幅为
0.0529/0.0408，超过第二轮预设的
0.02 tolerance，因此**不通过外部保持 gate**，不能作为跨域主方法。

## Nuisance / non-collapse 审计

`h_grade_probe` 与 `h_quality_probe` 是 train-fitted linear probe 在 validation/test
representation 上的 balanced accuracy；`d_grade_probe` 是 disease branch 的相应 probe。
这些指标用于判断“降低 h 中可读的 nuisance 是否只是删除全部信息”，不是单独的优化目标。
`same_eye_e` 是 e branch 的 L1-L2/R1-R2 cosine；它和各 branch std 共同防止把 e 写成
无信息残差。

## 边界与下一步

这只是受控 screen：若选择 variant 未在冻结外部 pair-ID 保持 raw baseline 附近，不能声称
domain-invariant patient factor。若 adversary 只降低 h probe、同时降低 correspondence 或 d/e
也塌缩，应按 information deletion 解释。完整逐 seed JSON/CSV 是唯一数值来源。
