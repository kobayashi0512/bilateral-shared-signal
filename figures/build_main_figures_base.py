#!/usr/bin/env python3
"""Asymmetric, data-faithful manuscript figures for bilateral correspondence."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results"
OUT = ROOT / "figures" / "main_figures_generated"
DEEPDRID_EXAMPLE = ROOT / "figures" / "figure1_framework" / "assets" / "deepdrid"
PAPILA_EXAMPLE = ROOT / "figures" / "figure4_transfer" / "assets"

NAVY = "#173246"
TEAL = "#147F78"
TEAL_LIGHT = "#E8F4F1"
BLUE = "#24739D"
BLUE_LIGHT = "#EAF3F8"
OCHRE = "#B78032"
OCHRE_LIGHT = "#F8F1E7"
CORAL = "#BC5B4B"
CORAL_LIGHT = "#FAECE9"
SLATE = "#667782"
PALE = "#D8E0E4"
GRID = "#E7ECEF"
WHITE = "#FFFFFF"

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 8,
    "text.color": NAVY,
    "axes.labelcolor": SLATE,
    "axes.edgecolor": PALE,
    "xtick.color": SLATE,
    "ytick.color": SLATE,
    "axes.linewidth": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 0,
    "xtick.major.width": 0.6,
    "figure.facecolor": WHITE,
    "axes.facecolor": WHITE,
    "savefig.facecolor": WHITE,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "axes.unicode_minus": False,
})


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_values() -> dict:
    return json.loads((DATA / "figure_inputs_v1.json").read_text(encoding="utf-8"))["prior_v2"]

    control_rows = rows(DATA / "deepdrid_r2_0b_d_grade_quality_controls_v1.pair_id.csv")
    balance_rows = rows(DATA / "deepdrid_r2_0b_d_grade_quality_controls_v1.balance.csv")
    view_rows = [r for r in rows(DATA / "deepdrid_r20_four_view_pair_probe_v1.csv") if r["split"] == "test"]
    probe_rows = [r for r in rows(DATA / "deepdrid_r2_0b_d_grade_quality_controls_v1.metadata_probe.csv")
                  if r["evaluation_split"] == "test"]
    papila_rows = rows(DATA / "papila" / "papila_frozen_external_probe_v1.csv")
    hcbfd_rows = [r for r in rows(DATA / "r21_hcbfd" / "deepdrid_r21_hcbfd_feature_screen_v1.csv")
                  if r["stage"] == "refit_test" and r["variant"] == "hierarchical_no_adversary"]
    hcbfd_json = json.loads((DATA / "r21_hcbfd" / "deepdrid_r21_hcbfd_feature_screen_v1.json").read_text(encoding="utf-8"))

    conditions = ["grade_only", "grade_plus_quality_tier", "grade_plus_quality_nearest"]
    control = {}
    balance = {}
    for condition in conditions:
        selected = [r for r in control_rows if r["condition"] == condition]
        quality = [r for r in balance_rows if r["condition"] == condition]
        assert len(selected) == len(quality) == 5
        v = [float(r["pair_identity_auroc"]) for r in selected]
        q = [float(r["mean_abs_quality_composite_difference"]) for r in quality]
        control[condition] = {"seed": [int(r["seed"]) for r in selected], "values": v,
                              "mean": mean_sd(v)[0], "sample_sd": mean_sd(v)[1]}
        balance[condition] = {"values": q, "mean": mean_sd(q)[0], "sample_sd": mean_sd(q)[1]}

    views = {}
    for label in ["L1-R1", "L1-R2", "L2-R1", "L2-R2"]:
        selected = [r for r in view_rows if r["view_combination"] == label]
        assert len(selected) == 5
        v = [float(r["pair_identity_auroc"]) for r in selected]
        views[label] = {"values": v, "mean": mean_sd(v)[0], "sample_sd": mean_sd(v)[1]}

    probes = {}
    for r in probe_rows:
        probes[r["task"]] = {"balanced_accuracy": float(r["balanced_accuracy"]),
                              "macro_ovr_auroc": float(r["macro_ovr_auroc"]),
                              "n_patients": int(r["n_patients"])}

    papila = {}
    for condition in ["unmatched_random", "bilateral_diagnosis_signature_matched"]:
        selected = [r for r in papila_rows if r["condition"] == condition]
        assert len(selected) == 5
        v = [float(r["pair_identity_auroc"]) for r in selected]
        papila[condition] = {"values": v, "mean": mean_sd(v)[0], "sample_sd": mean_sd(v)[1],
                             "n_patients": int(selected[0]["n_eligible_patients"])}

    cfp = {}
    for key, column in [("patient_max", "cfp_patientmax_auc"), ("eye_mode", "cfp_eyemode_auc")]:
        raw_key = "cfp_patientmax" if key == "patient_max" else "cfp_eyemode"
        raw_values = [float(x["pair_id_auroc"]) for x in hcbfd_json["raw_controls"][raw_key]["per_seed"]]
        trained_values = [float(x[column]) for x in hcbfd_rows]
        assert len(raw_values) == len(trained_values) == 5
        cfp[key] = {"raw": {"values": raw_values, "mean": mean_sd(raw_values)[0], "sample_sd": mean_sd(raw_values)[1]},
                    "hcbfd": {"values": trained_values, "mean": mean_sd(trained_values)[0], "sample_sd": mean_sd(trained_values)[1]},
                    "n_patients": int(hcbfd_json["raw_controls"][raw_key]["n_patients"])}

    assert abs(control["grade_plus_quality_tier"]["mean"] - .8660) < .0001
    assert abs(control["grade_plus_quality_nearest"]["sample_sd"] - .0044) < .0001
    assert abs(papila["unmatched_random"]["mean"] - .8669) < .0001
    assert abs(papila["bilateral_diagnosis_signature_matched"]["sample_sd"] - .0095) < .0001
    return {"control": control, "quality_matching": balance, "views": views,
            "metadata_probes": probes, "papila": papila, "cfp_stress": cfp}


def new_figure(height_mm: float):
    return plt.figure(figsize=(180 / 25.4, height_mm / 25.4), dpi=100)


def line(fig, x1, y1, x2, y2, color=PALE, lw=0.8, ls="-"):
    fig.add_artist(Line2D([x1, x2], [y1, y2], transform=fig.transFigure,
                          color=color, linewidth=lw, linestyle=ls, solid_capstyle="round"))


def card(fig, x, y, w, h, face, edge=PALE, radius=.015, lw=.7):
    patch = FancyBboxPatch((x, y), w, h, transform=fig.transFigure,
                           boxstyle=f"round,pad=0.004,rounding_size={radius}",
                           linewidth=lw, facecolor=face, edgecolor=edge)
    fig.add_artist(patch)
    return patch


def label(fig, x, y, letter, title, color=TEAL, size=9.2):
    fig.text(x, y, letter, color=color, fontsize=10.3, weight="bold", ha="left", va="center")
    fig.text(x + .027, y, title, color=NAVY, fontsize=size, weight="bold", ha="left", va="center")


def heading(fig, kicker: str, title: str, subtitle: str):
    fig.text(.047, .963, kicker, color=TEAL, fontsize=7.7, weight="bold", ha="left", va="center")
    fig.text(.047, .913, title, color=NAVY, fontsize=15.3, weight="bold", ha="left", va="center")
    fig.text(.047, .868, subtitle, color=SLATE, fontsize=8.0, ha="left", va="center")
    line(fig, .047, .839, .953, .839, PALE, .85)


def add_photo(fig, path: Path, bounds: tuple[float, float, float, float], label_text: str, edge: str):
    ax = fig.add_axes(bounds)
    with Image.open(path) as src:
        ax.imshow(src)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(.9)
        spine.set_edgecolor(edge)
    x, y, w, _ = bounds
    fig.text(x + w/2, y - .025, label_text, color=NAVY, fontsize=8.0,
             weight="bold", ha="center", va="center")
    return ax


def style_numeric_ax(ax, *, grid="x", xlim=None, ylim=None):
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(PALE)
    ax.tick_params(axis="both", labelsize=7.0, pad=2.5)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=.65, zorder=0)
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_axisbelow(True)


def fig1() -> plt.Figure:
    fig = new_figure(116)
    heading(fig, "STUDY DESIGN", "From four views to controlled bilateral evidence",
            "One patient contributes repeated views of both eyes; correspondence is tested against matched cross-patient pairs.")
    label(fig, .047, .793, "A", "Four-view acquisition", size=9.2)
    fig.text(.047, .754, "Illustrative DeepDRiD training patient 2", color=SLATE, fontsize=7.0, ha="left")
    label(fig, .654, .793, "B", "Pair construction", size=9.2)
    line(fig, .63, .29, .63, .805, PALE, .7)

    deep_paths = [DEEPDRID_EXAMPLE / f"{part.upper()}.jpg" for part in ["l1", "l2", "r1", "r2"]]
    x_positions = [.053, .189, .354, .490]
    for x, path, lab, edge in zip(x_positions, deep_paths, ["L1", "L2", "R1", "R2"],
                                  [OCHRE, OCHRE, BLUE, BLUE]):
        add_photo(fig, path, (x, .425, .119, .273), lab, edge)
    fig.text(.182, .723, "LEFT EYE", color=OCHRE, fontsize=7.2, weight="bold", ha="center")
    fig.text(.483, .723, "RIGHT EYE", color=BLUE, fontsize=7.2, weight="bold", ha="center")
    line(fig, .053, .387, .308, .387, OCHRE, 1.1)
    line(fig, .354, .387, .609, .387, BLUE, 1.1)
    fig.text(.182, .362, "two same-eye views", color=SLATE, fontsize=6.8, ha="center")
    fig.text(.483, .362, "two same-eye views", color=SLATE, fontsize=6.8, ha="center")

    ax = fig.add_axes([.655, .426, .300, .297])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    for y, right, title, color, linestyle in [(.71, r"$R_i$", "SAME PATIENT", TEAL, "-"),
                                               (.25, r"$R_j$", "DIFFERENT PATIENTS", SLATE, "--")]:
        for x, text, fc, ec in [(.04, r"$L_i$", OCHRE_LIGHT, OCHRE), (.75, right, BLUE_LIGHT, BLUE)]:
            patch = FancyBboxPatch((x, y-.10), .19, .19,
                                   boxstyle="round,pad=0.008,rounding_size=.03",
                                   facecolor=fc, edgecolor=ec, linewidth=1.0)
            ax.add_patch(patch)
            ax.text(x+.095, y, text, ha="center", va="center", color=NAVY, fontsize=9.0, weight="bold")
        ax.plot([.26, .72], [y, y], color=color, linewidth=2.0 if y>.5 else 1.25,
                linestyle=linestyle, solid_capstyle="round")
        ax.text(.50, y+.15, title, ha="center", va="bottom", color=color, fontsize=7.0, weight="bold")
    fig.text(.656, .384, "Cross-patient negative matching", color=NAVY, fontsize=7.5, weight="bold")
    for k,(yy,txt) in enumerate([(.346,"DR grade"),(.318,"Grade + quality tier"),(.290,"Grade + nearest quality")],1):
        fig.text(.658, yy, f"{k:02d}", color=OCHRE, fontsize=7.4, weight="bold", va="center")
        fig.text(.697, yy, txt, color=SLATE, fontsize=7.2, va="center")

    line(fig, .047, .267, .953, .267, PALE, .75)
    label(fig, .047, .239, "C", "Evidence roles", size=9.1)
    entries = [
        (.05, .335, "DeepDRiD", "300 paired patients · four views", "Protocol + internal test", TEAL),
        (.365, .650, "CFP–OCT", "137 eligible patients", "Development stress readout", OCHRE),
        (.680, .950, "PAPILA", "244 paired patients · one view/eye", "Locked external baseline", BLUE),
    ]
    for x1, x2, name, detail, role, color in entries:
        line(fig, x1, .207, x2, .207, color, 2.2)
        fig.text(x1, .176, name, color=NAVY, fontsize=9.4, weight="bold", ha="left")
        fig.text(x1, .131, detail, color=SLATE, fontsize=7.0, ha="left")
        fig.text(x1, .094, role, color=color, fontsize=7.3, weight="bold", ha="left")
    return fig


def fig2(data: dict) -> plt.Figure:
    fig = new_figure(128)
    heading(fig, "CONDITIONAL EVIDENCE", "The bilateral signal persists as negatives become harder",
            "Frozen ResNet-50 features; distinct panels report the full cohort, held-out view combinations and nuisance probes.")
    label(fig, .048, .786, "A", "Disease and quality matched negatives", size=9.1)
    fig.text(.050, .755, "Full DeepDRiD cohort · 300 patients · five fixed negative seeds", color=SLATE, fontsize=7.0)
    label(fig, .682, .786, "B", "Four cross-eye pairs", size=9.1)
    fig.text(.684, .755, "Held-out test · 43 eligible patients", color=SLATE, fontsize=7.0)
    line(fig, .652, .398, .652, .802, PALE, .7)

    control_keys = ["grade_only", "grade_plus_quality_tier", "grade_plus_quality_nearest"]
    conditions = ["DR grade", "Grade +\nquality tier", "Grade +\nnearest quality"]
    ax = fig.add_axes([.095, .455, .505, .270])
    style_numeric_ax(ax, grid="y", xlim=(-.2,2.2), ylim=(.825,.907))
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_color(PALE)
    ax.set_ylabel("Pair-identity AUROC", fontsize=7.6, labelpad=4)
    ax.set_xticks([0,1,2], conditions)
    ax.set_yticks([.83,.85,.87,.89])
    for seed_idx in range(5):
        values = [data["control"][key]["values"][seed_idx] for key in control_keys]
        ax.plot([0,1,2], values, color="#A5B4BA", linewidth=.85, alpha=.7, marker="o",
                markersize=3.3, zorder=2)
    means = [data["control"][key]["mean"] for key in control_keys]
    sds = [data["control"][key]["sample_sd"] for key in control_keys]
    ax.plot([0,1,2], means, color=TEAL, linewidth=2.15, zorder=4)
    ax.errorbar([0,1,2], means, yerr=sds, fmt="o", color=TEAL, ecolor=TEAL,
                elinewidth=1.35, capsize=3.0, markersize=6.2, markeredgecolor=WHITE,
                markeredgewidth=.9, zorder=5)
    for x,m in enumerate(means):
        ax.text(x, m+.010, f"{m:.3f}", ha="center", va="bottom", fontsize=7.5,
                weight="bold", color=TEAL)

    bx = fig.add_axes([.745, .457, .215, .266])
    style_numeric_ax(bx, grid="x", xlim=(.63,.90), ylim=(-.6,3.6))
    bx.set_yticks([3,2,1,0], ["L1–R1", "L1–R2", "L2–R1", "L2–R2"])
    bx.set_xticks([.65,.75,.85])
    bx.set_xlabel("AUROC", fontsize=7.3)
    for y,key in zip([3,2,1,0],["L1-R1","L1-R2","L2-R1","L2-R2"]):
        v=data["views"][key]
        for j,seed_value in enumerate(v["values"]):
            bx.scatter(seed_value,y+(-.13+j*.065),s=10,color="#AAB7BC",zorder=3,linewidths=0)
        bx.errorbar(v["mean"],y,xerr=v["sample_sd"],fmt="o",color=BLUE,ecolor=BLUE,
                    markersize=5.8,elinewidth=1.25,capsize=2.8,markeredgecolor=WHITE,
                    markeredgewidth=.8,zorder=4)
        bx.text(.895,y,f"{v['mean']:.3f}",ha="right",va="center",color=BLUE,fontsize=6.9,
                weight="bold",bbox=dict(facecolor=WHITE,edgecolor="none",pad=.5))

    line(fig,.047,.392,.953,.392,PALE,.75)
    label(fig,.048,.359,"C","Quality matching is tighter",size=9.0)
    label(fig,.566,.359,"D","Nuisance remains readable",size=9.0)
    fig.text(.048,.330,"Mean absolute quality-composite difference",color=SLATE,fontsize=7.0)
    fig.text(.566,.330,"Train-fitted probes on 44 test patients",color=SLATE,fontsize=7.0)
    line(fig,.530,.075,.530,.369,PALE,.7)

    cx=fig.add_axes([.157,.122,.324,.190])
    style_numeric_ax(cx,grid="x",xlim=(0,.225),ylim=(-.45,2.45))
    cx.set_yticks([2,1,0],["DR grade","+ quality tier","+ nearest quality"])
    cx.set_xticks([0,.10,.20])
    cx.set_xlabel("Mean |Δ quality|",fontsize=7.2)
    for y,key in zip([2,1,0],control_keys):
        v=data["quality_matching"][key]
        cx.hlines(y,0,v["mean"],color=OCHRE,linewidth=2.0,zorder=2)
        for j,seed_value in enumerate(v["values"]):
            cx.scatter(seed_value,y+(-.12+j*.06),s=9,color="#D5B487",zorder=3,linewidths=0)
        cx.scatter(v["mean"],y,s=54,color=OCHRE,edgecolors=WHITE,linewidths=.8,zorder=4)
        cx.text(min(v["mean"]+.012,.220),y,f"{v['mean']:.3f}",color=OCHRE,fontsize=7.0,
                ha="left" if v["mean"]+.012<.220 else "right",va="center",weight="bold")

    dx=fig.add_axes([.690,.122,.258,.190])
    style_numeric_ax(dx,grid="x",xlim=(.50,.96),ylim=(-.55,1.55))
    dx.set_yticks([1,0],["DR grade","Quality tier"])
    dx.set_xticks([.50,.70,.90])
    for y,key in [(1,"patient_DR_grade"),(0,"quality_composite_tier")]:
        v=data["metadata_probes"][key]
        b,a=v["balanced_accuracy"],v["macro_ovr_auroc"]
        dx.hlines(y,b,a,color="#C2CBCF",linewidth=1.6,zorder=2)
        dx.scatter(b,y,s=55,color=OCHRE,edgecolors=WHITE,linewidths=.7,zorder=4)
        dx.scatter(a,y,s=55,color=BLUE,edgecolors=WHITE,linewidths=.7,zorder=4)
        dx.text(b,y+.18,f"{b:.3f}",color=OCHRE,fontsize=6.8,ha="center",va="bottom")
        dx.text(a,y+.18,f"{a:.3f}",color=BLUE,fontsize=6.8,ha="center",va="bottom")
    fig.text(.572,.080,"●  Balanced accuracy",color=OCHRE,fontsize=6.8)
    fig.text(.761,.080,"●  Macro OvR AUROC",color=BLUE,fontsize=6.8)
    fig.text(.047,.034,"Small dots/gray lines: individual negative seeds. Large markers and whiskers: mean ± sample SD (n = 5).",
             color=SLATE,fontsize=6.8)
    return fig


def fig3(data: dict) -> plt.Figure:
    fig=new_figure(120)
    heading(fig,"EXTERNAL EVIDENCE","A frozen bilateral baseline replicates on PAPILA",
            "PAPILA is a locked glaucoma-cohort test; CFP–OCT is a separately labeled development stress readout.")
    label(fig,.048,.787,"A","New imaging cohort",size=9.2)
    fig.text(.050,.750,"PAPILA · 244 complete bilateral patients",color=SLATE,fontsize=7.0)
    label(fig,.502,.787,"B","Frozen external result",size=9.2)
    fig.text(.505,.750,"ImageNet ResNet-50 · one image per eye · no tuning",color=SLATE,fontsize=7.0)
    line(fig,.461,.410,.461,.804,PALE,.7)

    add_photo(fig,PAPILA_EXAMPLE/"figure4_left_eye.jpg",(.060,.480,.170,.220),"OS · left eye",BLUE)
    add_photo(fig,PAPILA_EXAMPLE/"figure4_right_eye.jpg",(.258,.480,.170,.220),"OD · right eye",BLUE)
    fig.text(.244,.423,"Illustrative patient RET014; selected without using scores",color=SLATE,
             fontsize=6.6,ha="center")

    p=ax=fig.add_axes([.660,.470,.279,.234])
    style_numeric_ax(p,grid="x",xlim=(.82,.90),ylim=(-.5,1.5))
    p.set_yticks([1,0],["Random cross-patient","Diagnosis-signature matched"])
    p.set_xticks([.82,.84,.86,.88,.90])
    p.set_xlabel("Pair-identity AUROC",fontsize=7.5)
    for y,key in [(1,"unmatched_random"),(0,"bilateral_diagnosis_signature_matched")]:
        v=data["papila"][key]
        for j,seed_value in enumerate(v["values"]):
            p.scatter(seed_value,y+(-.15+j*.075),s=15,color="#9FB4BF",linewidths=0,zorder=3)
        p.errorbar(v["mean"],y,xerr=v["sample_sd"],fmt="o",color=BLUE,ecolor=BLUE,
                   markersize=7.2,elinewidth=1.6,capsize=4.0,markeredgecolor=WHITE,
                   markeredgewidth=1.0,zorder=5)
        p.text(.899,y+.23,f"{v['mean']:.4f} ± {v['sample_sd']:.4f}",ha="right",va="center",
               color=BLUE,fontsize=7.2,weight="bold")

    line(fig,.047,.391,.953,.391,PALE,.8)
    label(fig,.048,.350,"C","CFP–OCT: learned-candidate stress test",color=CORAL,size=9.1)
    fig.text(.051,.316,"Exploratory · 137 eligible patients · two prespecified label schemas",color=SLATE,fontsize=7.0)
    fig.text(.684,.316,"●  Raw frozen",color=SLATE,fontsize=7.0)
    fig.text(.808,.316,"●  H-CBFD",color=CORAL,fontsize=7.0)
    c=fig.add_axes([.253,.145,.660,.145])
    style_numeric_ax(c,grid="x",xlim=(.80,.93),ylim=(-.48,1.48))
    c.set_yticks([1,0],["Patient-max DME","Eye-mode"])
    c.set_xticks([.80,.85,.90,.93])
    c.set_xlabel("Pair-identity AUROC",fontsize=7.4)
    for y,key in [(1,"patient_max"),(0,"eye_mode")]:
        rr=data["cfp_stress"][key]["raw"]
        hh=data["cfp_stress"][key]["hcbfd"]
        c.hlines(y,hh["mean"],rr["mean"],color="#AAB6BC",linewidth=1.5,zorder=2)
        c.errorbar(rr["mean"],y,xerr=rr["sample_sd"],fmt="o",color=SLATE,ecolor=SLATE,
                   markersize=6.3,elinewidth=1.1,capsize=2.8,markeredgecolor=WHITE,zorder=4)
        c.errorbar(hh["mean"],y,xerr=hh["sample_sd"],fmt="o",color=CORAL,ecolor=CORAL,
                   markersize=6.3,elinewidth=1.1,capsize=2.8,markeredgecolor=WHITE,zorder=4)
        c.text((rr["mean"]+hh["mean"])/2,y+.24,f"Δ {hh['mean']-rr['mean']:+.4f}",color=CORAL,
               fontsize=7.0,ha="center",weight="bold")
    fig.text(.047,.035,"PAPILA tests the frozen raw representation. H-CBFD was not evaluated on PAPILA.",
             color=SLATE,fontsize=6.9)
    return fig


CAPTIONS={
    "figure1_protocol": (
        "Figure 1 | Acquisition hierarchy and controlled bilateral comparison. "
        "(A) Four unmodified fundus images from one DeepDRiD training-split patient (image IDs 2_l1, 2_l2, 2_r1, 2_r2) illustrate two views of each eye; images are illustrative, not model outputs. "
        "(B) A same-patient cross-eye pair is compared with a cross-patient pair matched on patient DR grade, grade plus quality tier, or grade plus nearest quality. "
        "(C) DeepDRiD provides protocol development and internal testing, CFP–OCT is a development stress readout, and PAPILA is the locked external baseline evaluation. The endpoint is pair correspondence, not clinical patient identification or systemic disease inference."),
    "figure2_controls": (
        "Figure 2 | DeepDRiD correspondence under increasingly strict negative-pair controls. "
        "(A) Full-cohort diagnostic analysis (300 complete bilateral patients): each gray trajectory is one fixed negative-pair seed across grade-only, grade plus quality-tier, and grade plus nearest-quality matching. Teal points and whiskers are mean ± sample SD over five seeds. This panel is not a held-out learned-model result. "
        "(B) Pair-identity AUROC for four cross-eye combinations on the patient-disjoint test subset with complete L1/L2/R1/R2 (43 eligible patients), using grade-matched cross-patient negatives; points and whiskers summarize five seeds. "
        "(C) Mean absolute difference in the fixed quality-composite score between matched source and negative patients, averaged over five seeds. "
        "(D) Training-split-fitted linear probes evaluated on 44 test patients; the blue metric is macro one-versus-rest AUROC. These probes indicate readable grade and quality information, not causal leakage. All whiskers in A–C are across-seed sample SD, not patient bootstrap intervals."),
    "figure3_transfer": (
        "Figure 3 | Frozen external replication and learned-candidate transfer boundary. "
        "(A) Unmodified PAPILA OS/OD fundus images from patient RET014 illustrate the new acquisition domain; the example was chosen by identifier, independent of scores. "
        "(B) On 244 complete bilateral PAPILA patients, an ImageNet-pretrained frozen ResNet-50 yields pair-identity AUROC for random cross-patient negatives and exact bilateral glaucoma-diagnosis-signature-matched negatives. Small points show five fixed negative-pair seeds; large points and whiskers are mean ± sample SD. No model, threshold or negative construction was selected on PAPILA. "
        "(C) On the previously inspected CFP–OCT development stress set (137 eligible patients), the selected H-CBFD representation scores below raw frozen features under both prespecified label schemas. Differences are H-CBFD minus raw AUROC. H-CBFD was not evaluated on PAPILA; panel B does not establish its external generalization."),
}


def save_one(fig: plt.Figure, slug: str):
    base=OUT/slug
    base.mkdir(parents=True,exist_ok=True)
    fig.savefig(base/f"{slug}.pdf",format="pdf",dpi=300)
    fig.savefig(base/f"{slug}.svg",format="svg",dpi=300)
    fig.savefig(base/f"{slug}.png",format="png",dpi=300)
    fig.savefig(base/f"{slug}.tif",format="tiff",dpi=600,pil_kwargs={"compression":"tiff_lzw"})
    plt.close(fig)
    (base/"caption.md").write_text(CAPTIONS[slug]+"\n",encoding="utf-8")


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    data=source_values()
    image_paths=[DEEPDRID_EXAMPLE/f"{part.upper()}.jpg" for part in ["l1","l2","r1","r2"]]
    image_paths.extend([PAPILA_EXAMPLE/"figure4_left_eye.jpg",PAPILA_EXAMPLE/"figure4_right_eye.jpg"])
    provenance={str(path):sha256(path) for path in image_paths}
    (OUT/"data_values.json").write_text(json.dumps(data,indent=2),encoding="utf-8")
    (OUT/"image_provenance.json").write_text(json.dumps(provenance,indent=2),encoding="utf-8")
    for slug,figure in [("figure1_protocol",fig1()),("figure2_controls",fig2(data)),("figure3_transfer",fig3(data))]:
        save_one(figure,slug)
    (OUT/"CAPTIONS.md").write_text("# Main-figure captions\n\n"+"\n\n".join(CAPTIONS.values())+"\n",encoding="utf-8")
    (OUT/"README.md").write_text(
        "# Main figures v2\n\nThree redesigned manuscript figures with asymmetric layouts and source-grounded marks. "
        "The PDFs/SVGs retain vector text and plot geometry; fundus examples remain raster images. "
        "PNG is 300 dpi and LZW TIFF is 600 dpi.\n\n"
        "| Figure | Preview | PDF | SVG | TIFF |\n| --- | --- | --- | --- | --- |\n"
        +"\n".join(f"| {slug} | [PNG]({slug}/{slug}.png) | [PDF]({slug}/{slug}.pdf) | "
                    f"[SVG]({slug}/{slug}.svg) | [TIFF]({slug}/{slug}.tif) |"
                    for slug in ["figure1_protocol","figure2_controls","figure3_transfer"])
        +"\n\n[Captions](CAPTIONS.md) · [Source values](data_values.json) · [Image provenance](image_provenance.json) · "
         "[Design brief](../FIGURE_REDESIGN_BRIEF_V2.md).\n",encoding="utf-8")
    print(OUT)


if __name__=="__main__":
    main()
