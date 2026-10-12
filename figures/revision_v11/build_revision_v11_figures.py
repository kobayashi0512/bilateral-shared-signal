#!/usr/bin/env python3
"""Build the manuscript's revised data-grounded Figures 2–4."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
AUDIT = ROOT / "results" / "revision_v11" / "revision_v11_conditional_audit.json"
ASSETS = ROOT / "figures" / "figure4_transfer" / "assets"

NAVY = "#173246"
TEAL = "#147F78"
BLUE = "#24739D"
OCHRE = "#B78032"
CORAL = "#BC5B4B"
SLATE = "#667782"
PALE = "#D8E0E4"
GRID = "#E7ECEF"
WHITE = "#FFFFFF"
LIGHT_TEAL = "#EDF6F4"
LIGHT_BLUE = "#EEF4F8"
LIGHT_OCHRE = "#FAF5EC"

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 8,
    "text.color": NAVY,
    "axes.labelcolor": SLATE,
    "axes.edgecolor": PALE,
    "axes.linewidth": 0.65,
    "xtick.color": SLATE,
    "ytick.color": SLATE,
    "xtick.major.size": 2.5,
    "ytick.major.size": 0,
    "xtick.major.width": 0.65,
    "figure.facecolor": WHITE,
    "axes.facecolor": WHITE,
    "savefig.facecolor": WHITE,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "axes.unicode_minus": False,
})


def load_data() -> dict:
    return json.loads(AUDIT.read_text(encoding="utf-8"))


def make_fig(height_mm: float) -> plt.Figure:
    # Match the manuscript's 5.65 in image width on its 5.77 in text block.
    return plt.figure(figsize=(143.5 / 25.4, height_mm * (143.5 / 180) / 25.4), dpi=100)


def heading(fig: plt.Figure, kicker: str, title: str, subtitle: str) -> None:
    fig.text(.047, .963, kicker, color=TEAL, fontsize=7.2, weight="bold", va="center")
    fig.text(.047, .918, title, color=NAVY, fontsize=14.3, weight="bold", va="center")
    fig.text(.047, .876, subtitle, color=SLATE, fontsize=7.7, va="center")
    fig.add_artist(plt.Line2D([.047, .953], [.848, .848], transform=fig.transFigure,
                              color=PALE, linewidth=.75))


def panel(fig: plt.Figure, x: float, y: float, letter: str, title: str, note: str = "") -> None:
    fig.text(x, y, letter, color=TEAL, fontsize=10, weight="bold", va="center")
    fig.text(x + .027, y, title, color=NAVY, fontsize=8.8, weight="bold", va="center")
    if note:
        fig.text(x, y - .027, note, color=SLATE, fontsize=6.8, va="center")


def style_ax(ax: plt.Axes, grid: str = "y") -> None:
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(PALE)
    ax.tick_params(axis="both", labelsize=7, pad=3)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=.65, zorder=0)
    ax.set_axisbelow(True)


def envelope_from_rows(rows: list[dict], field: str) -> list[float]:
    intervals = [row[field] for row in rows]
    return [min(x[0] for x in intervals), max(x[1] for x in intervals)]


def fig2(d: dict) -> plt.Figure:
    """Strict four-view hierarchy under exact bilateral-grade and target-quality matching."""
    deep = d["deepdrid"]["four_view_hierarchy"]
    test = deep["test"]
    val = deep["validation"]
    fig = make_fig(122)
    heading(fig, "FOUR-VIEW HIERARCHY", "Cross-eye correspondence under stricter matching",
            "Exact ordered bilateral DR signature and right-eye quality tier; patient-node uncertainty, conditional on fixed assignments.")
    panel(fig, .048, .805, "A", "Test-patient score distributions",
          f"DeepDRiD · {test['n_common_eligible_patients']} eligible test patients · identity-free patient scores")
    ax = fig.add_axes([.205, .535, .700, .218])
    style_ax(ax, "y")
    ax.set_xlim(.48, 1.18)
    ax.set_ylim(-.7, 2.65)
    ax.set_yticks([2, 1, 0], ["Same eye · repeat view", "Cross eye · same patient", "Cross patient · matched"])
    ax.set_xticks([.5, .6, .7, .8, .9, 1.0])
    ax.set_xlabel("Cosine similarity", fontsize=7.5)
    distributions = test["identity_free_patient_score_distributions"]
    specs = [
        ("same_eye", 2, OCHRE, test["same_eye_patient_mean"], test["same_eye_patient_bootstrap_95ci"]),
        ("cross_eye", 1, TEAL, test["cross_eye_patient_mean"], test["cross_eye_patient_bootstrap_95ci"]),
        ("matched_negative_mean_over_seeds", 0, SLATE, test["matched_negative_mean_over_seeds"],
         test["matched_negative_node_bootstrap_interval_envelope"]),
    ]
    rng = np.random.default_rng(5814)
    for key, y, color, mean, ci in specs:
        values = np.asarray(distributions[key], dtype=float)
        jitter = rng.uniform(-.19, .19, size=len(values))
        ax.scatter(values, y + jitter, s=15, color=color, alpha=.40, linewidths=0, zorder=2)
        ax.errorbar(mean, y, xerr=[[mean-ci[0]], [ci[1]-mean]], fmt="D", ms=5.8,
                    color=color, ecolor=color, elinewidth=1.35, capsize=2.7,
                    markeredgecolor=WHITE, markeredgewidth=.7, zorder=4)
        ax.text(1.175, y, f"{mean:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]",
                ha="right", va="center", fontsize=6.5, color=color,
                bbox=dict(facecolor=WHITE, edgecolor="none", pad=.7))

    panel(fig, .048, .445, "B", "Paired differences with node-bootstrap interval envelopes",
          "Each envelope spans the five assignment-specific 95% patient-node bootstrap intervals; not a pooled interval.")
    bx = fig.add_axes([.355, .115, .550, .250])
    style_ax(bx, "x")
    bx.spines["bottom"].set_visible(False)
    bx.set_xlim(-.045, .155)
    bx.set_ylim(-.6, 3.6)
    bx.axvline(0, color=SLATE, linewidth=.9, linestyle=(0, (3, 2)), zorder=1)
    bx.set_xticks([-.04, 0, .04, .08, .12, .16])
    bx.set_yticks([3, 2, 1, 0], ["Test · same eye − cross eye", "Test · cross eye − negative",
                                    "Validation · same eye − cross eye", "Validation · cross eye − negative"])
    for split_name, obj, base_y in [("test", test, 2), ("validation", val, 0)]:
        rows = obj["rows"]
        contrasts = [
            ("same_eye_minus_cross_eye", "same_eye_minus_cross_eye_node_bootstrap_95ci", base_y + 1, OCHRE),
            ("cross_eye_minus_matched_negative", "cross_eye_minus_matched_negative_node_bootstrap_95ci", base_y, TEAL),
        ]
        for value_key, ci_key, y, color in contrasts:
            values = [r[value_key] for r in rows]
            mean = statistics.mean(values)
            low, high = envelope_from_rows(rows, ci_key)
            bx.errorbar(mean, y, xerr=[[mean-low], [high-mean]], fmt="D", ms=5.2,
                        color=color, ecolor=color, elinewidth=1.55, capsize=3,
                        markeredgecolor=WHITE, markeredgewidth=.7, zorder=3)
    fig.text(.047, .052,
             "The same-eye–cross-eye contrast remains unresolved; cross-eye scores exceed the matched-negative distribution in both splits.",
             color=SLATE, fontsize=6.6)
    return fig


def plot_condition_panel(ax: plt.Axes, entries: list[dict], labels: list[str], group_break: int,
                         title_note: str, split_id: str) -> None:
    style_ax(ax, "y")
    ax.set_ylim(.48, 1.025)
    ax.set_yticks([.5, .6, .7, .8, .9, 1.0])
    ax.set_ylabel("Pair-identity AUROC", fontsize=7.2)
    ax.axhline(.5, color=SLATE, linewidth=.9, linestyle=(0, (3, 2)), zorder=1)
    ax.axvline(group_break - .5, color=PALE, linewidth=.9, linestyle=(0, (2, 2)), zorder=1)
    ax.axvspan(-.5, group_break - .5, color=LIGHT_BLUE, alpha=.62, zorder=0)
    ax.axvspan(group_break - .5, len(entries) - .5, color=LIGHT_TEAL, alpha=.62, zorder=0)
    ax.set_xlim(-.55, len(entries) - .45)
    ax.set_xticks(range(len(entries)), labels, rotation=13, ha="right")
    ax.tick_params(axis="x", labelsize=6.4, pad=4)
    rng = np.random.default_rng(800 + (1 if split_id == "full" else 2))
    for i, entry in enumerate(entries):
        vals = [r["pair_identity_auroc"] for r in entry["rows"]]
        mean = entry["auroc_mean"]
        sd = entry["auroc_seed_sd"]
        low, high = entry["node_bootstrap_interval_envelope"]
        jitter = rng.uniform(-.105, .105, size=len(vals))
        color = [BLUE, BLUE, BLUE, TEAL, TEAL][i]
        ax.scatter(i + jitter, vals, s=20, color=color, alpha=.78,
                   edgecolors=WHITE, linewidths=.45, zorder=4)
        ax.errorbar(i, mean, yerr=sd, fmt="D", ms=5.3, color=color,
                    ecolor=color, elinewidth=1.5, capsize=2.6,
                    markeredgecolor=WHITE, markeredgewidth=.7, zorder=5)
        ax.errorbar(i + .19, mean, yerr=[[mean-low], [high-mean]], fmt="none",
                    ecolor=SLATE, elinewidth=1.05, capsize=2.0, alpha=.8, zorder=3)
        ax.text(i, .495, f"n={entry['n_patients']}", ha="center", va="bottom",
                fontsize=6.0, color=SLATE, zorder=6)


def fig3(d: dict) -> plt.Figure:
    deep = d["deepdrid"]
    full_old = deep["old_protocol_full_cohort"]
    test_old = deep["old_protocol_heldout_common_denominator"]["conditions"]
    strict = deep["strict_exact_signature_controls"]
    full_strict = strict["full_cohort"]
    test_strict = strict["test"]
    full_entries = full_old + [full_strict["signature_only"], full_strict["signature_plus_right_eye_quality"]]
    test_entries = test_old + [test_strict["signature_only"], test_strict["signature_plus_right_eye_quality"]]
    labels = ["Same patient\nDR grade", "DR grade +\npatient q tier", "DR grade +\nnearest q", "Exact ordered\nbilateral DR", "Bilateral DR +\nright-eye q tier"]
    fig = make_fig(139)
    heading(fig, "CONDITIONAL CONTROLS", "Correspondence persists under stricter eye-level controls",
            "Five fixed negative assignments per condition; points show seedwise AUROC, diamonds show mean ± sample SD.")
    panel(fig, .048, .805, "A", "Full-cohort diagnostic",
          "Original patient-level controls (n=300) versus post-hoc exact bilateral-signature controls (n=293).")
    ax1 = fig.add_axes([.095, .535, .835, .215])
    plot_condition_panel(ax1, full_entries, labels, 3, "Descriptive full-cohort analysis", "full")
    panel(fig, .048, .431, "B", "Held-out sensitivity analysis",
          "Post-hoc common-denominator analyses; the stricter protocol matches bilateral disease signature and right-eye quality tier.")
    ax2 = fig.add_axes([.095, .195, .835, .190])
    plot_condition_panel(ax2, test_entries, labels, 3, "Small held-out subsets: n=42 under original rules; n=34 under exact eye-level rules", "test")
    fig.text(.047, .055,
             "Thin offset whiskers: envelope of five separate 95% patient-node bootstrap intervals (conditional on fixed assignments; not pooled). "
             "The two matching schemes use different quality aggregation levels; AUROC differences are not causal effects.",
             color=SLATE, fontsize=6.3)
    return fig


def add_photo(ax: plt.Axes, path: Path, title: str, edge: str) -> None:
    with Image.open(path) as image:
        ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(edge)
        spine.set_linewidth(.9)
    ax.set_title(title, fontsize=7.0, color=NAVY, weight="bold", pad=3)


def fig4(d: dict) -> plt.Figure:
    fig = make_fig(136)
    heading(fig, "TRANSFER BOUNDARY", "External baseline signal; learned-head transfer remains unresolved",
            "PAPILA evaluates the frozen raw baseline only; H-CBFD is assessed on source-domain and already-inspected stress data.")
    panel(fig, .048, .805, "A", "PAPILA examples",
          "Representative OD/OS source images.")
    axl = fig.add_axes([.055, .565, .130, .180])
    axr = fig.add_axes([.204, .565, .130, .180])
    add_photo(axl, ASSETS / "figure4_left_eye.jpg", "Left eye (OS)", TEAL)
    add_photo(axr, ASSETS / "figure4_right_eye.jpg", "Right eye (OD)", BLUE)

    panel(fig, .392, .805, "B", "Frozen baseline on PAPILA",
          "n=244 · no PAPILA tuning.")
    pap = {row["condition"]: row for row in d["papila"]["conditions"]}
    ax = fig.add_axes([.430, .555, .470, .200])
    style_ax(ax, "y")
    ax.set_xlim(-.4, 1.4)
    ax.set_ylim(.79, .94)
    ax.set_yticks([.80, .84, .88, .92])
    ax.set_xticks([0, 1], ["Random\nnegative", "Exact bilateral\ndiagnosis signature"])
    ax.set_ylabel("Pair-identity AUROC", fontsize=7.0)
    pap_entries = [pap["unmatched_random"], pap["bilateral_diagnosis_signature_matched"]]
    for i, item in enumerate(pap_entries):
        vals = [r["pair_identity_auroc"] for r in item["rows"]]
        color = [SLATE, TEAL][i]
        jitter = np.linspace(-.075, .075, len(vals))
        ax.scatter(i + jitter, vals, s=23, color=color, edgecolors=WHITE, linewidths=.5, zorder=4)
        mean, sd = item["auroc_mean"], item["auroc_seed_sd"]
        lo, hi = item["node_bootstrap_interval_envelope"]
        ax.errorbar(i, mean, yerr=sd, fmt="D", ms=5.3, color=color, ecolor=color,
                    capsize=2.7, elinewidth=1.5, markeredgecolor=WHITE, markeredgewidth=.7, zorder=5)
        ax.errorbar(i + .18, mean, yerr=[[mean-lo], [hi-mean]], fmt="none", ecolor=SLATE,
                    capsize=2.0, elinewidth=1.0, alpha=.8, zorder=3)
        ax.text(i, .793, f"{mean:.3f} ± {sd:.3f}", ha="center", va="bottom", fontsize=6.3, color=color)

    panel(fig, .048, .470, "C", "H-CBFD candidate: internal comparison does not guarantee transfer",
          "Mean ± corrected across-seed sample SD; five fixed negative assignments per condition.")
    hcbfd = d["hcbfd"]["metrics"]
    axc = fig.add_axes([.095, .168, .835, .245])
    style_ax(axc, "y")
    axc.set_xlim(-.55, 3.55)
    axc.set_ylim(.65, .985)
    axc.set_yticks([.65, .70, .75, .80, .85, .90, .95])
    axc.set_ylabel("Pair-identity AUROC", fontsize=7.0)
    axc.set_xticks([0, 1, 2, 3], ["DeepDRiD\nvalidation", "DeepDRiD\nheld-out test", "CFP–OCT\npatient-max DME", "CFP–OCT\nlegacy eye-mode DME"])
    axc.axvline(1.5, color=PALE, linewidth=.9, linestyle=(0, (2, 2)))
    for i, item in enumerate(hcbfd):
        raw = np.asarray(item["raw_seed_aurocs"], dtype=float)
        learned = np.asarray(item["hcbfd_seed_aurocs"], dtype=float)
        raw_mean, raw_sd = item["raw_frozen"]["mean"], item["raw_frozen"]["sample_sd_ddof_1"]
        h_mean, h_sd = item["hcbfd_selected"]["mean"], item["hcbfd_selected"]["sample_sd_ddof_1"]
        for x, values, mean, sd, color in [(i - .105, raw, raw_mean, raw_sd, TEAL),
                                           (i + .105, learned, h_mean, h_sd, CORAL)]:
            jitter = np.linspace(-.030, .030, len(values))
            axc.scatter(np.full(len(values), x) + jitter, values, s=17, color=color,
                        edgecolors=WHITE, linewidths=.45, alpha=.82, zorder=4)
            axc.errorbar(x, mean, yerr=sd, fmt="D", ms=5.0, color=color, ecolor=color,
                         capsize=2.4, elinewidth=1.4, markeredgecolor=WHITE, markeredgewidth=.7, zorder=5)
    from matplotlib.lines import Line2D
    axc.legend(handles=[Line2D([0], [0], marker="D", color="none", markerfacecolor=TEAL,
                               markeredgecolor=WHITE, label="Raw frozen features", markersize=5),
                        Line2D([0], [0], marker="D", color="none", markerfacecolor=CORAL,
                               markeredgecolor=WHITE, label="H-CBFD", markersize=5)],
               loc="upper right", frameon=False, fontsize=6.4, ncol=2, handletextpad=.3, columnspacing=.8)
    fig.text(.047, .065,
             "CFP–OCT was inspected during development; PAPILA was not used to assess H-CBFD. Error bars in C show across-seed SD, not patient-sampling confidence intervals.",
             color=SLATE, fontsize=6.2)
    return fig


CAPTIONS = {
    "figure2_hierarchy": (
        "Figure 2. Four-view hierarchy under exact bilateral disease and target-eye quality matching. "
        "(A) Identity-free patient-level cosine scores for same-eye repeat views, same-patient cross-eye views, and assigned cross-patient negatives in the 33-patient held-out test subset. Diamonds and colored whiskers show group means and patient-node bootstrap 95% intervals; the negative interval is the envelope across five fixed assignments. "
        "(B) Same-eye minus cross-eye and cross-eye minus matched-negative contrasts in validation (n=35) and test (n=33). Points are means across five assignments; whiskers span the smallest lower and largest upper endpoints of assignment-specific patient-node bootstrap intervals. They are not pooled confidence intervals. Exact negative matching uses the ordered left/right diabetic-retinopathy grade signature and right-eye quality-composite tier; the anchor left-eye image is fixed within each positive/negative comparison. The same-eye versus cross-eye difference remains unresolved, whereas cross-eye scores exceed matched negatives in both splits. All intervals are conditional on fixed assignments and do not include rematching uncertainty."),
    "figure3_controls": (
        "Figure 3. Conditional correspondence under patient-level and stricter eye-specific negative controls. "
        "(A) Full-cohort descriptive diagnostic: three original patient-level controls in 300 patients and two post-hoc controls in 293 patients using the exact ordered bilateral diabetic-retinopathy signature, with or without exact right-eye quality-tier matching. "
        "(B) Held-out sensitivity analyses: the three original rules share a 42-patient denominator; stricter eye-specific rules share 34 patients. Colored dots are the five fixed assignment-specific AUROCs; diamonds and bold whiskers show mean ± sample SD across assignments; thin offset gray whiskers show the envelope of five separate 95% patient-node bootstrap intervals, not a pooled interval. The horizontal dashed line marks chance AUROC. The full-cohort analysis is descriptive, and the held-out controls were added post hoc. Quality aggregation differs between the original patient-level controls and the stricter target-eye rule; cross-condition changes are not causal estimates."),
    "figure4_transfer": (
        "Figure 4. External frozen-baseline evaluation and learned-candidate transfer boundary. "
        "(A) Representative PAPILA left-eye (oculus sinister, OS) and right-eye (oculus dexter, OD) source photographs; images were resized for layout only. "
        "(B) Frozen ImageNet-pretrained ResNet-50 pair-identity AUROC on 244 complete PAPILA patients for random and exact ordered bilateral glaucoma-diagnosis-signature-matched negatives. Dots show five fixed assignments; colored whiskers show mean ± across-seed sample SD; thin offset gray whiskers show patient-node bootstrap interval envelopes conditional on the fixed assignments. This is an external evaluation of the frozen baseline, not H-CBFD. "
        "(C) Raw frozen features versus H-CBFD on DeepDRiD validation and held-out readouts under patient-grade × patient-quality-tier matching, and on the previously inspected CFP–OCT development stress cohort under patient-max and legacy eye-mode diabetic-macular-edema label schemas. Points and whiskers show seedwise AUROC and mean ± sample SD across five assignments; these are not patient-sampling intervals. CFP–OCT is not blind external validation, and H-CBFD was not evaluated on PAPILA."),
}


def save_figure(fig: plt.Figure, slug: str) -> None:
    folder = OUT / slug
    folder.mkdir(parents=True, exist_ok=True)
    for ext, dpi in [("png", 450), ("tif", 600), ("svg", None), ("pdf", None)]:
        kwargs = {"bbox_inches": "tight", "facecolor": WHITE}
        if dpi:
            kwargs["dpi"] = dpi
        fig.savefig(folder / f"{slug}.{ext}", **kwargs)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = load_data()
    fig2(d)
    save_figure(plt.gcf(), "figure2_hierarchy")
    fig3(d)
    save_figure(plt.gcf(), "figure3_controls")
    fig4(d)
    save_figure(plt.gcf(), "figure4_transfer")
    (OUT / "CAPTIONS.md").write_text("# Figure captions\n\n" + "\n\n".join(CAPTIONS.values()) + "\n", encoding="utf-8")
    manifest = {
        "figure_set": "revision_v11",
        "source_data": "results/revision_v11/revision_v11_conditional_audit.json",
        "patient_identifiers_in_plot_data": False,
        "original_user_adjusted_figure_1_modified": False,
        "formats": ["PDF", "SVG", "PNG", "TIFF"],
        "figures": list(CAPTIONS),
        "figure_4_source_photographs": ["figure4_transfer/assets/figure4_left_eye.jpg", "figure4_transfer/assets/figure4_right_eye.jpg"],
        "notes": ["Numeric plots use frozen archived feature evaluations; no model retrained.",
                  "PAPILA panel is the external evaluation of the frozen raw baseline only.",
                  "No PAPILA patient identifier is displayed."]
    }
    (OUT / "figure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (OUT / "README.md").write_text(
        "# Revision v11 manuscript figures\n\n"
        "Figures 2–4 are regenerated from the revision v11 audit JSON. Figure 1 in the manuscript is preserved from the user-adjusted DOCX. "
        "Each numeric figure is available as editable SVG and vector PDF plus high-resolution PNG/TIFF previews. "
        "No patient identifiers are embedded in plot data or displayed.\n\n"
        "Source data: `../../results/revision_v11/revision_v11_conditional_audit.json`.\n\n"
        "Captions are in `CAPTIONS.md`; output and provenance notes are in `figure_manifest.json`.\n",
        encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
