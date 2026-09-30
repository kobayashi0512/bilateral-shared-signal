#!/usr/bin/env python3
"""Source-grounded Figure 2–4 revisions and hierarchy evidence."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import build_main_figures_base as base


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "main_figures_generated"
INPUTS = ROOT / "results" / "figure_inputs_v1.json"
base.OUT = OUT


def patient_bootstrap_envelopes() -> dict:
    return json.loads(INPUTS.read_text(encoding="utf-8"))["patient_bootstrap"]


def hierarchy_data() -> dict:
    return json.loads(INPUTS.read_text(encoding="utf-8"))["hierarchy"]


def fig2_revised(data: dict, bootstrap: dict) -> plt.Figure:
    fig = base.fig2(data)
    fig.texts[1].set_text("Correspondence persists under stricter negatives")
    ax = fig.axes[0]
    ax.set_ylim(.805, .930)
    ax.set_yticks([.82, .86, .90])
    for x, key in enumerate(["grade_only", "grade_plus_quality_tier", "grade_plus_quality_nearest"]):
        low, high = bootstrap[key]["conservative_envelope"]
        xx = x + .115
        ax.vlines(xx, low, high, colors=base.SLATE, linewidth=1.05, zorder=3)
        ax.hlines([low, high], xx-.035, xx+.035, colors=base.SLATE, linewidth=1.05, zorder=3)
    fig.axes[2].set_xlabel("")  # Panel C heading already names the quantity; reserve space for uncertainty note.
    fig.texts[-1].set_text("Gray trajectories: five negative-pair seeds; gray offset whiskers: envelope of seedwise patient-bootstrap 95% CIs.")
    fig.texts[-1].set_position((.047, .049))
    fig.texts[-1].set_fontsize(6.3)
    fig.text(.047, .025, "Teal: mean ± across-seed sample SD (n = 5). AUROC chance = 0.5, outside the enlarged plotting range.",
             color=base.SLATE, fontsize=6.3)
    return fig


def fig4_hierarchy(hierarchy: dict) -> plt.Figure:
    fig = base.new_figure(135)
    base.heading(fig, "VIEW HIERARCHY", "Cross-eye similarity survives grade matching",
                 "Same-eye and cross-eye means are close; the uncertainty on their difference includes zero.")
    base.label(fig, .048, .790, "A", "Patient-level test distribution", size=9.2)
    fig.text(.050, .756, "DeepDRiD · 43 complete four-view test patients · frozen features", color=base.SLATE, fontsize=7.0)
    ax = fig.add_axes([.193, .555, .743, .177])
    base.style_numeric_ax(ax, grid="x", xlim=(.46, 1.01), ylim=(-.55, 2.55))
    ax.set_yticks([2, 1, 0], ["Same eye · different view", "Cross eye · same patient", "Cross patient · grade matched"])
    ax.set_xticks([.5, .6, .7, .8, .9, 1.0])
    ax.set_xlabel("Cosine similarity", fontsize=7.5)
    records = hierarchy["test_patient_level"]
    panels = [("same_eye", 2, base.OCHRE, "same_eye"),
              ("cross_eye", 1, base.TEAL, "cross_eye"),
              ("negative_mean_over_seeds", 0, base.SLATE, "negative")]
    jitter = np.linspace(-.20, .20, len(records))
    for field, y, color, mean_key in panels:
        values = [r[field] for r in records]
        ax.scatter(values, y+jitter, color=color, alpha=.35, s=14, linewidths=0, zorder=2)
        mean = hierarchy["test"]["similarity_means"][mean_key]
        ax.scatter([mean], [y], color=color, edgecolors=base.WHITE, linewidths=1.0,
                   s=95, marker="D", zorder=5)
        ax.text(.995, y, f"mean {mean:.3f}", ha="right", va="center", fontsize=7.2,
                color=color, weight="bold", bbox=dict(facecolor=base.WHITE, edgecolor="none", pad=1.0))

    base.line(fig, .047, .469, .953, .469, base.PALE, .8)
    base.label(fig, .048, .432, "B", "Paired gaps with patient-level uncertainty", size=9.0)
    fig.text(.050, .402, "Five fixed grade-matched negative seeds · 3,000 paired-patient bootstrap resamples per seed",
             color=base.SLATE, fontsize=7.0)
    bx = fig.add_axes([.410, .150, .526, .222])
    base.style_numeric_ax(bx, grid="x", xlim=(-.038, .132), ylim=(-.55, 3.55))
    bx.set_yticks([3, 2, 1, 0], ["Test: same-eye − cross-eye", "Test: cross-eye − negative",
                                    "Validation: same-eye − cross-eye", "Validation: cross-eye − negative"])
    bx.set_xticks([-.03, 0, .03, .06, .09, .12])
    bx.axvline(0, color="#7E8D96", linewidth=1.0, linestyle="--", zorder=1)
    bx.set_xlabel("Paired cosine-similarity difference", fontsize=7.4)
    rows = [("test", "same_minus_cross", 3, base.OCHRE),
            ("test", "cross_minus_negative", 2, base.TEAL),
            ("val", "same_minus_cross", 1, base.OCHRE),
            ("val", "cross_minus_negative", 0, base.TEAL)]
    for split, key, y, color in rows:
        item = hierarchy[split][key]
        for j, (point, ci) in enumerate(zip(item["values"], item["seedwise_patient_bootstrap_95ci"])):
            yy = y + (j-2)*.055
            bx.hlines(yy, ci[0], ci[1], color="#A7B4BB", linewidth=1.0, zorder=2)
            bx.plot([ci[0], ci[1]], [yy, yy], marker="|", color="#A7B4BB", markersize=3.7,
                    linewidth=0, zorder=2)
            bx.scatter(point, yy, color=color, s=11, alpha=.7, linewidths=0, zorder=3)
        bx.scatter(item["mean_over_seeds"], y, marker="D", color=color, edgecolors=base.WHITE,
                   linewidths=.8, s=58, zorder=5)
    fig.text(.047, .033, "Dots/thin whiskers: seed-specific gap and patient-bootstrap 95% CI. Diamond: mean gap over seeds; CIs are not pooled.",
             color=base.SLATE, fontsize=6.8)
    return fig


def captions() -> dict:
    return {
        "figure2_controls": (
            "Figure 2 | DeepDRiD pair correspondence under progressively stricter negative-patient controls. "
            "(A) Full-cohort diagnostic analysis of 300 complete bilateral patients. Gray trajectories show five fixed negative-pair assignments; teal points and whiskers show mean ± sample standard deviation (SD) across those seeds. "
            "Offset gray whiskers show the conservative envelope from the minimum lower to the maximum upper bound of five separately computed 95% paired-patient bootstrap confidence intervals (3,000 resamples per seed); this envelope is not itself a pooled confidence interval. The AUROC axis is enlarged and does not include chance level 0.5. "
            "(B) Four left–right view combinations on 43 eligible held-out test patients, with DR-grade-matched negatives; whiskers are across-seed SD. "
            "(C) Mean absolute source–negative difference in a dimensionless, normalized quality-composite score (overall quality, clarity, field definition and inverse artifact score); lower means tighter matching. "
            "(D) Training-fitted linear probes evaluated on 44 test patients. DR is diabetic retinopathy; AUROC is area under the receiver operating characteristic curve; OvR is one-versus-rest. Probe performance indicates readable metadata, not causal leakage. "
            "Panel A is not a held-out learned-model result."),
        "figure3_transfer": (
            "Figure 3 | Frozen external baseline and learned-candidate transfer boundary. "
            "(A) Unmodified PAPILA fundus photographs from illustrative patient RET014; OS denotes left eye and OD right eye. The example was chosen by identifier, not by score. "
            "(B) On 244 complete bilateral PAPILA patients, fixed ImageNet-pretrained ResNet-50 features yield pair-identity AUROC for random and exact bilateral glaucoma-diagnosis-signature-matched cross-patient negatives. Dots show five fixed negative-pair seeds; points and whiskers are their mean ± sample SD. No model, threshold, or negative rule was tuned on PAPILA. "
            "(C) On previously inspected CFP–OCT development data (137 eligible patients), the low-capacity hierarchical H-CBFD candidate scores below raw frozen features under both prespecified diabetic-macular-edema (DME) label schemas. Patient-max uses the maximum eye-level label per patient; eye-mode is the legacy patient-mode convention from eye-level labels. Points and whiskers show mean ± sample SD over five negative-pair seeds. Differences are H-CBFD minus raw AUROC. "
            "H-CBFD was not evaluated on PAPILA, so panel B does not validate its external generalization."),
        "figure4_hierarchy": (
            "Figure 4 | Observed acquisition hierarchy and paired uncertainty under grade-matched negatives. "
            "(A) On 43 eligible DeepDRiD test patients with explicit L1/L2/R1/R2 images, each small dot is a patient-level cosine similarity for same-eye different-view, cross-eye same-patient, or grade-matched cross-patient comparison. Cross-patient values are first averaged over the five fixed negative assignments within each patient; diamonds mark group means. "
            "(B) Same-eye minus cross-eye and cross-eye minus grade-matched cross-patient differences on test (43 patients) and validation (45 patients). Thin intervals are separate 95% paired-patient bootstrap confidence intervals from 3,000 resamples for each of five fixed negative seeds; colored diamonds are across-seed mean differences and do not carry a pooled confidence interval. "
            "The same-eye minus cross-eye intervals include zero, so their mean ordering is not established. All displayed cross-eye minus matched-negative intervals are above zero. These comparisons do not control image quality and therefore cannot establish a disease- and quality-independent identity factor."),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = base.source_values()
    bootstrap = patient_bootstrap_envelopes()
    hierarchy = hierarchy_data()
    draft_captions = captions()
    base.CAPTIONS.update({
        "figure2_hierarchy": draft_captions["figure4_hierarchy"].replace("Figure 4 |", "Figure 2 |", 1),
        "figure3_controls": draft_captions["figure2_controls"].replace("Figure 2 |", "Figure 3 |", 1),
        "figure4_transfer": draft_captions["figure3_transfer"].replace("Figure 3 |", "Figure 4 |", 1),
    })
    fig2 = fig2_revised(data, bootstrap)
    fig3 = base.fig3(data)
    fig4 = fig4_hierarchy(hierarchy)
    for slug, fig in [("figure2_hierarchy", fig4), ("figure3_controls", fig2), ("figure4_transfer", fig3)]:
        base.save_one(fig, slug)
    (OUT/"data_values.json").write_text(json.dumps({"prior_v2": data, "patient_bootstrap": bootstrap,
                                                     "hierarchy": hierarchy}, indent=2), encoding="utf-8")
    (OUT/"figure2_brief.json").write_text(json.dumps({
        "figure_goal": "Directly visualize acquisition hierarchy and uncertainty for the benchmark claim",
        "paper_claim": "Cross-eye same-patient similarity exceeds grade-matched cross-patient similarity; same-eye advantage over cross-eye is unresolved",
        "figure_type": "benchmark-plot", "mode": "plot",
        "panels": ["test patient-level cosine distributions", "test and validation paired gap forest"],
        "must_keep_labels": ["same eye", "cross eye", "grade-matched cross patient", "patient bootstrap"],
        "data": {"source": "results/figure_inputs_v1.json", "patient_identifiers_included": False},
        "style_constraints": ["180 mm width", "Arial", "white background", "color plus marker shape"],
        "output_formats": ["pdf", "svg", "png", "tif"],
        "verification_checklist": ["43 test and 45 validation patients", "five seedwise CIs not pooled",
                                   "same-eye-minus-cross-eye CI includes zero", "cross-eye-minus-negative CIs above zero"]
    }, indent=2), encoding="utf-8")
    (OUT/"CAPTIONS.md").write_text("# Figure captions\n\n"+"\n\n".join(base.CAPTIONS[k] for k in
        ["figure2_hierarchy", "figure3_controls", "figure4_transfer"])+"\n", encoding="utf-8")
    (OUT/"README.md").write_text(
        "# Current main-figure set\n\nFigure 1 is in `../figure1_framework/package_v4/`. "
        "Figure 2 directly plots the view hierarchy; Figure 3 adds patient-bootstrap CI envelopes to conditional controls; Figure 4 retains the locked PAPILA result with expanded abbreviations. "
        "All numeric marks are regenerated from the public aggregate JSON input. Patient identifiers and raw archives are excluded.\n\n"
        "| Figure | Preview | PDF | SVG | TIFF |\n| --- | --- | --- | --- | --- |\n"+
        "\n".join(f"| {slug} | [PNG]({slug}/{slug}.png) | [PDF]({slug}/{slug}.pdf) | "
                  f"[SVG]({slug}/{slug}.svg) | [TIFF]({slug}/{slug}.tif) |"
                  for slug in ["figure2_hierarchy", "figure3_controls", "figure4_transfer"])+
        "\n\n[Captions](CAPTIONS.md) · [Exact aggregate values](data_values.json) · [Figure 2 brief](figure2_brief.json).\n",
        encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
