# Bilateral Shared-Signal Benchmark：Phase 1 数据审计协议

建立日期：2026-09-21

## 目标

确认哪些公开眼底数据集可以支持患者级双眼 benchmark：

1. 左右眼能否通过可复现的 patient ID 配对；
2. 是否有可用的中心、相机、质量和采集信息；
3. 标签是否能支持系统性风险任务，或只能支持表征/域迁移任务；
4. 原始图像、标签和派生 split 是否可按许可证使用与发布。

本阶段只做元数据和许可审计，不训练模型、不读取隐藏测试标签、不修改现有项目的固定测试划分。

## 首批候选

| 数据集 | 计划角色 | 首要核验 |
|---|---|---|
| China-Fundus-CIMT | 主风险任务 | patient ID、双眼完整率、CIMT/年龄/性别字段、许可 |
| ODIR-5K | 双眼表征与多相机域 | 双眼记录、相机/中心字段、标签语义、下载条款 |
| BRSET | 跨中心/相机外部域 | patient ID、左右眼完整率、中心/相机、病史标签 |
| mBRSET | 移动相机压力测试 | patient ID、左右眼/重复检查、系统性病史、许可 |
| PAPILA | 解剖对齐辅助 | 双眼对应、视盘/视杯标注、样本量、许可 |
| DeepDRiD/FIVES | 质量/结构辅助 | 只用于辅助表征与质量，不冒充系统性外部终点 |
| UK Biobank | 远期强外部验证 | 申请状态、访问费用、双眼/OCT/纵向结局可用性 |

## 统一 metadata schema

原始字段不直接覆盖；每个数据集先保留原始字段，再映射到以下规范字段：

`dataset, patient_id, image_id, eye, visit_id, center, camera, acquisition_date, quality, systemic_labels, ocular_labels, source_path, license, pair_status`

允许值和规则：

- `eye`: `L`, `R`, `unknown`；无法确认时不得猜测；
- `pair_status`: `complete_pair`, `single_eye`, `multiple_same_eye`, `conflict`, `unknown`；
- `center/camera`: 未提供时写 `unknown`，不能用文件夹名推断临床中心；
- `systemic_labels`: 每个数据集保留原始定义，不把 CIMT、高血压、糖尿病或自述病史合并为一个标签；
- `quality`: 区分人工质量标签、自动质量分数和未标注；
- `visit_id`: 若无纵向访视字段，写 `unknown`，不能把文件名当作时间。

## 审计顺序

1. 读取官方说明和许可证；
2. 只获取可直接下载的 metadata/标签清单；
3. 统计患者数、图像数、左右眼数、完整双眼数和冲突数；
4. 统计中心、相机、质量与标签缺失；
5. 对需要申请、付费或受限访问的数据单独标记，不纳入第一版公开 benchmark 的硬依赖；
6. 生成数据集准入决定：`admit-core`、`admit-auxiliary`、`metadata-only` 或 `reject-for-now`。

## 准入门槛

### 核心数据集

- 有可复现 patient ID；
- 至少一项可用的患者级标签；
- 能构造患者互斥的 train/validation/test；
- 许可证允许研究使用；
- 不能依赖隐藏测试标签完成调参。

### 外部域数据集

- 与训练集存在明确且真实的域差异；
- patient ID 或至少可靠的患者分组字段；
- 标签定义可以单独报告；
- 不把同一来源的版本扩展误当作独立中心。

## Phase 1 输出

1. `dataset_registry.csv`：一行一个候选数据集；
2. 每个数据集的原始字段映射和许可证记录；
3. 双眼配对和标签完整性统计；
4. 一份准入/淘汰决定；
5. 下一阶段下载清单，包含固定版本、来源 URL、预估体积和校验方式。

## 明确禁止

- 图像级随机划分；
- 用文件名猜测患者或眼别；
- 把不同定义的系统性标签直接合并；
- 把没有 patient ID 的图像当作患者级外部验证；
- 因结果不理想而更换测试集或重新定义终点。
