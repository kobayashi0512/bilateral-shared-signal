# Data access, provenance, and reuse boundaries

This repository does not redistribute the source datasets. Obtain data from the authoritative sources below, check their current terms, and keep any access credentials out of the repository.

| Dataset | Authoritative source | Project role | Release note |
|---|---|---|---|
| DeepDRiD | [Official repository](https://github.com/deepdrdoc/DeepDRiD), [Zenodo record](https://doi.org/10.5281/zenodo.6452623) | Internal development and multi-view conditional evaluation | The official repository identifies CC BY-SA 4.0. Reuse of source photos and derivatives must follow that license and attribution requirements. |
| PAPILA | [Official Figshare v2 item](https://figshare.com/articles/dataset/PAPILA/14798004), [dataset paper](https://doi.org/10.1038/s41597-022-01388-1) | Locked final external test of the frozen raw baseline | The Figshare record identifies GPL-3.0+. The experiment used an audited Kaggle distribution copy after the Figshare CLI route returned HTTP 403; it is the same PAPILA dataset, not an independent cohort. The paper cites the official PAPILA record and discloses the distribution source. |
| CFP–OCT | [Source repository](https://github.com/Traslational-Visual-Health-Laboratory/OCT-AND-EYE-FUNDUS-DATASET), [associated paper](https://doi.org/10.1016/j.compbiomed.2024.107979) | Previously inspected development stress readout | The reviewed repository did not provide an explicit image-reuse license. No CFP–OCT images are redistributed in this package. |
| mBRSET | [PhysioNet dataset record](https://physionet.org/content/mbrset/1.0.0/) | Candidate only; not part of the manuscript results | Access is credentialed. GitHub/Kaggle/Hugging Face mirrors were not treated as authorization and were not used as project data. |
| ODIR / China-Fundus-CIMT | Official dataset/provider pages and source publications | Candidate audit only | Neither contributes to the current manuscript's reported experiments. Consult the original provider terms before any reuse. |

## Figure image attribution

Figure 1 uses four selected DeepDRiD training images (two views per eye) and one selected PAPILA OD/OS pair. Figure 4 uses another selected PAPILA pair. These are a few illustrative examples, not a redistribution of the image archives; image reuse in the manuscript and source figure package is attributed to the respective dataset records/publications. DeepDRiD photographs are covered by the source CC BY-SA 4.0 terms; PAPILA photographs by the Figshare GPL-3.0+ terms. The examples are resized for layout and are otherwise unmodified. No CFP–OCT photos are shown.

Before any downstream redistribution, verify the current source license version, attribution requirements, and compatibility with a combined figure/manuscript. The dataset provider's terms take precedence over this summary.

## Files intentionally not released

- Original dataset archives or full image trees.
- Patient-level manifests, clinical tables, source IDs, or split assignments.
- Raw patient-level feature caches, trained checkpoints, or model weights.
- Kaggle API credentials, PhysioNet credentials, cookies, or download tokens.
- CFP–OCT image examples whose reuse license was not established.

The plotted point values in `results/figure_inputs_v1.json` contain no patient identifiers and are limited to figure reproduction. They are not a replacement for source data or clinical records.
