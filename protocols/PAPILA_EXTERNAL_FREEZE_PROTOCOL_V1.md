# PAPILA final external bilateral protocol（v1）

状态：**已执行并锁定**（2026-09-22）  
目的：在不再依赖 mBRSET credentialed access 的前提下，为 Round 2 提供一个此前未用于方法选择的公开、完整双眼外部验证集。

## 数据与边界

- 规范数据来源与引用唯一为官方 [PAPILA v2 Figshare item](https://figshare.com/articles/dataset/PAPILA/14798004)，记录显示 GPL-3.0+。当前 CLI 无法从 Figshare 下载时，可使用**通过
  `audit_outputs/papila_kaggle_download_status.md` 全部 content/provenance gate 的**
  Kaggle GPL-3 分发副本；它不是一个新数据集，论文仍引用 PAPILA 官方论文和 Figshare。
  不使用 Google Drive、GitHub 或 Unknown/MIT/Apache/cropped/enhanced 的副本。
- 预期为 244 位患者、488 张 JPEG；每位一张 OD 与一张 OS。实际计数必须由 `normalize_papila_official.py` 的 preflight 决定。
- PAPILA 没有可比的图像质量标签，不能冒充 DeepDRiD 的 grade×quality control 外部复现；它检验的是**设备/病种变化下的同患者双眼 correspondence**。
- CFP–OCT 仍为已查看的 external development stress set，不取代 PAPILA 的盲态角色。

## 冻结步骤

1. 从官方页面下载、解压；保存 item version、license 截图/metadata 和 archive SHA-256。
2. 运行下列 ingestion；它必须得到一对一 OD/OS clinical-row/image 对应与完整患者计数，否则停止。

```bash
python3 normalize_papila_official.py /absolute/path/PAPILA_DBv1 \
  --out audit_outputs/phase2_public/papila/papila_external_frozen_manifest.csv
```

3. 审阅 `.preflight.json` 后冻结该 manifest 的 SHA-256。此后禁止根据 PAPILA 图像、diagnosis 或任何分数更改模型、特征变换、阈值、pooling 或负对构造。
4. 使用与 DeepDRiD 基线完全相同的 ImageNet ResNet-50 V2 图像变换和 frozen weights 提取特征；不 fine-tune。

```bash
../safe_oct_referral/.venv/bin/python extract_manifest_resnet50_features.py \
  --manifest audit_outputs/phase2_public/papila/papila_external_frozen_manifest.csv \
  --out audit_outputs/phase2_public/papila/papila_resnet50_features.npy
```

5. 运行唯一一次预设 external readout：

```bash
../safe_oct_referral/.venv/bin/python papila_frozen_external_probe.py \
  --manifest audit_outputs/phase2_public/papila/papila_external_frozen_manifest.csv \
  --features audit_outputs/phase2_public/papila/papila_resnet50_features.npy \
  --out audit_outputs/phase2_public/papila/papila_frozen_external_probe_v1.json
```

## 预设指标

|层级|指标|解释|
|---|---|---|
|主指标|随机跨患者负对上的 pair-identity AUROC（5 固定 seed）|同一患者 L/R 的 cosine 是否高于非同患者 L/R；不使用疾病标签。|
|敏感性分析|按 `(left diagnosis, right diagnosis)` 完全匹配的负对 AUROC|控制 glaucoma/healthy/suspect 组合；只有每个 signature 至少 2 位患者时纳入，报告入选人数。|
|审计|完整患者数、诊断缺失、OD/OS row/image 双向一致性、clinical XLSX SHA-256|确保不是 filename/排列偶然产生的左右眼对。|

不报告也不优化基于 PAPILA 选择出的 threshold、best seed、疾病分类 AUC 或因外部结果而变更的模型。若 preflight 不通过，则 PAPILA 仅作为数据审计记录，改回公开主线而不造出“外部验证”结论。

执行结果见 `audit_outputs/phase2_public/papila/PAPILA_FINAL_EXTERNAL_RESULT_V1.md`。PAPILA
现已被查看并消费为 final external readout，不能再用于下一轮方法的选择或反复试验。
