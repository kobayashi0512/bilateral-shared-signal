# Related work and defensible novelty boundary

This is a project-positioning note, not a systematic review. The cited studies should be rechecked against the final target venue and literature cutoff before submission.

## What is already established

- Bilateral retinal similarity and interretinal verification have prior work: Biswas et al., [Interretinal Symmetry in Color Fundus Photographs](https://doi.org/10.1109/EMBC44109.2020.9175444).
- Using same-patient left/right fundus images as contrastive positives is not new. Relevant retinal work includes Nguyen et al., [Self-FI](https://doi.org/10.3390/bioengineering10091089), and Xie et al., [iGWAS](https://doi.org/10.1371/journal.pgen.1011273).
- Common/shared versus view-specific representation branches and disentanglement objectives also have broad multi-view precedents, including Ke et al., [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Ke_Rethinking_Multi-view_Representation_Learning_via_Distilled_Disentangling_CVPR_2024_paper.html), and Sun et al., [Decoupled representation for multi-view learning](https://doi.org/10.1016/j.patcog.2024.110377).
- Matched or debiased contrastive negatives are an established general idea; see Chuang et al., [Debiased Contrastive Learning](https://proceedings.neurips.cc/paper/2020/hash/63c3ddcc7b23daa1e42dc41f9a44a873-Abstract.html), and Khosla et al., [Supervised Contrastive Learning](https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html).

## What this project can defend

The contribution is not the first bilateral positive pair, the first shared/specific decomposition, or proof of a disease-independent identity representation. The defensible focus is a conditional evaluation protocol for bilateral correspondence: preserve the explicit L1/L2/R1/R2 acquisition hierarchy, separate same-eye repeat-view from same-patient cross-eye comparisons, progressively match cross-patient negatives on observed disease and quality, audit nuisance readability and collapse, and report a frozen external readout with dataset roles stated symmetrically.

This framing matters because disease grade and image quality remain readable from visual features, and matching only the observed metadata cannot rule out unmeasured factors such as age, camera, site, field of view, artifacts, or disease duration. Use “operational bilateral correspondence under the specified controls,” not identity recognition, a biological patient factor, or complete confound removal.

## Method outcome as evidence, not a marketing claim

The low-capacity H-CBFD screen selected its non-adversarial hierarchical variant on DeepDRiD validation, but its CFP–OCT development-stress readout was below the raw frozen baseline. PAPILA tests the frozen raw baseline only. These outcomes narrow what can be claimed; they do not establish that factorization or adversarial nuisance removal is generally effective.

## Submission checklist

1. Search for bilateral fundus SSL and retina re-identification papers published after the current literature scan.
2. Compare protocol and controls directly against Self-FI/iGWAS and multi-view common/specific methods.
3. Keep an explicit claim-to-evidence table and report all predeclared denominators and negative results.
4. If a later method claims domain invariance, add multiple source domains and an external protocol fixed before outcome inspection.
