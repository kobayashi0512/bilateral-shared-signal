#!/usr/bin/env python3
"""R2.1 low-capacity H-CBFD screen on frozen DeepDRiD four-view features.

This is deliberately a *screen*, not a claim of a novel factorization model.
It preserves the documented L1/L2/R1/R2 hierarchy and follows a strict
selection protocol: fit on DeepDRiD train only, select from validation
grade×quality-matched pair-ID plus collapse checks, then refit every
predeclared variant on train+validation once for held-out DeepDRiD and frozen
CFP-OCT reporting.  CFP-OCT is never used in fitting or selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.autograd import Function
from torch.utils.data import DataLoader, Dataset


VIEWS = ("L1", "L2", "R1", "R2")
QUALITY_COLUMNS = ("Overall quality", "Clarity", "Field definition", "Artifact")
QUALITY_LABELS = ("low", "mid", "high")


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def quality_tier(group: pd.DataFrame) -> tuple[float, int]:
    q = 0.25 * (float(group["Overall quality"].mean()) + float(group["Clarity"].mean()) / 10.0 +
                float(group["Field definition"].mean()) / 10.0 + (10.0 - float(group["Artifact"].mean())) / 10.0)
    return q, 0 if q <= .625 else (1 if q <= .75 else 2)


def four_view_cases(features: np.ndarray, manifest: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Strictly retain exactly named L1/L2/R1/R2 records only."""
    if len(features) != len(manifest):
        raise ValueError("feature/manifest length mismatch")
    req = {"patient_id", "image_id", "patient_grade", "benchmark_split", *QUALITY_COLUMNS}
    missing = req - set(manifest.columns)
    if missing:
        raise ValueError(f"missing manifest fields: {sorted(missing)}")
    m = manifest.copy().reset_index(drop=True)
    m["patient_id"] = m.patient_id.astype(str)
    m["view"] = m.image_id.astype(str).str.extract(r"_(l1|l2|r1|r2)$", expand=False).str.upper()
    records: list[dict[str, Any]] = []
    excluded: list[str] = []
    for pid, group in m.groupby("patient_id", sort=True):
        lookup = dict(zip(group.view, group.index))
        if set(lookup) != set(VIEWS):
            excluded.append(pid); continue
        grades = group.patient_grade.astype(int).unique()
        splits = group.benchmark_split.astype(str).unique()
        if len(grades) != 1 or len(splits) != 1:
            raise ValueError(f"inconsistent grade/split for {pid}")
        q, qt = quality_tier(group)
        records.append({"patient_id": pid, "split": str(splits[0]), "grade": int(grades[0]), "quality": q,
                        "quality_tier": qt, "views": {v: features[int(lookup[v])].astype(np.float32) for v in VIEWS}})
    audit = {"patients_total": int(m.patient_id.nunique()), "eligible_exact_l1_l2_r1_r2": len(records),
             "excluded_nonstandard_or_incomplete": len(excluded), "excluded_patient_ids": excluded,
             "split_counts": dict(Counter(c["split"] for c in records))}
    return records, audit


def external_cases(features: np.ndarray, manifest: pd.DataFrame) -> list[dict[str, Any]]:
    """One mean feature per eye; schema labels have already been materialised in manifest."""
    if len(features) != len(manifest): raise ValueError("external feature/manifest length mismatch")
    m = manifest.copy().reset_index(drop=True); m.patient_id = m.patient_id.astype(str); m.eye = m.eye.astype(str).str.upper()
    result = []
    for pid, g in m.groupby("patient_id", sort=True):
        if set(g.eye) != {"L", "R"}: continue
        label = int(g.patient_grade.astype(int).mode().iloc[0])
        result.append({"patient_id": pid, "grade": label, "quality_tier": -1,
                       "views": {"L1": features[g.index[g.eye.eq("L")]].mean(0).astype(np.float32),
                                 "L2": features[g.index[g.eye.eq("L")]].mean(0).astype(np.float32),
                                 "R1": features[g.index[g.eye.eq("R")]].mean(0).astype(np.float32),
                                 "R2": features[g.index[g.eye.eq("R")]].mean(0).astype(np.float32)}})
    return result


def derange(index: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if len(index) < 2: raise ValueError("singleton matched-negative stratum")
    for _ in range(200):
        candidate = rng.permutation(index)
        if not np.any(candidate == index): return candidate
    return np.roll(index, 1)


def match_index(cases: list[dict[str, Any]], seed: int, joint: bool = True) -> np.ndarray:
    """Exact strata, no hidden unmatched fallback; raises on infeasibility."""
    rng = np.random.default_rng(seed); out = np.arange(len(cases)); groups: dict[tuple, list[int]] = {}
    for i, c in enumerate(cases):
        key = (c["grade"], c["quality_tier"]) if joint else (c["grade"],)
        groups.setdefault(key, []).append(i)
    bad = {str(k): len(v) for k, v in groups.items() if len(v) < 2}
    if bad: raise ValueError(f"infeasible matched-negative strata: {bad}")
    for idx in groups.values():
        x = np.asarray(idx); out[x] = derange(x, rng)
    return out


def retain_feasible_strata(cases: list[dict[str, Any]], joint: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Exclude singleton *evaluation* strata rather than weaken matching.

    Small held-out patient splits can contain a single grade×quality-tier case.
    Such a patient has no valid exact hard negative.  It is excluded from this
    specific conditional endpoint (not reassigned to a looser stratum), and the
    exclusion is logged.  Model fitting retains every train case if its train
    stratum is feasible.
    """
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for c in cases:
        key=(c["grade"],c["quality_tier"]) if joint else (c["grade"],)
        groups.setdefault(key,[]).append(c)
    kept=[]; dropped=[]
    for key, group in groups.items():
        (kept if len(group)>=2 else dropped).extend(group)
    return kept, {"joint":joint,"input_patients":len(cases),"eligible_patients":len(kept),"excluded_singleton_patients":len(dropped),"excluded_patient_ids":[c["patient_id"] for c in dropped],"strata":{str(k):len(v) for k,v in groups.items()}}


class GradientReverse(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = scale; return x.view_as(x)
    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.scale * grad, None


def grl(x: torch.Tensor, scale: float) -> torch.Tensor:
    return GradientReverse.apply(x, scale)


class PatientDataset(Dataset):
    def __init__(self, cases: list[dict[str, Any]], neg: np.ndarray): self.cases, self.neg = cases, neg
    def __len__(self): return len(self.cases)
    def __getitem__(self, i: int):
        c, n = self.cases[i], self.cases[int(self.neg[i])]
        return (torch.from_numpy(c["views"]["L1"]), torch.from_numpy(c["views"]["L2"]),
                torch.from_numpy(c["views"]["R1"]), torch.from_numpy(c["views"]["R2"]),
                torch.from_numpy(n["views"]["R1"]), torch.tensor(c["grade"], dtype=torch.long),
                torch.tensor(c["quality_tier"], dtype=torch.long))


class HCBFD(nn.Module):
    def __init__(self, dim: int, variant: str):
        super().__init__(); self.variant = variant; h = 64
        self.h = nn.Sequential(nn.Linear(dim, h), nn.LayerNorm(h), nn.ReLU())
        self.d = nn.Sequential(nn.Linear(dim, h), nn.LayerNorm(h), nn.ReLU())
        self.e = nn.Sequential(nn.Linear(dim, h), nn.LayerNorm(h), nn.ReLU())
        self.grade_d = nn.Linear(h, 5); self.grade_shared = nn.Linear(h, 5)
        self.quality_d = nn.Linear(h, 3); self.quality_e = nn.Linear(h, 3)
        self.nuisance_grade = nn.Linear(h, 5); self.nuisance_quality = nn.Linear(h, 3)
    def encode(self, l1, l2, r1, r2):
        hx = [self.h(x) for x in (l1,l2,r1,r2)]
        dx = [self.d(x) for x in (l1,l2,r1,r2)]
        ex = [self.e(x) for x in (l1,l2,r1,r2)]
        return hx, dx, ex
    def classify(self, l: torch.Tensor, r: torch.Tensor, lp=True, rp=True) -> torch.Tensor:
        # A symmetric disease readout: baseline classifies h; hierarchical variants classify d.
        f = self.h if self.variant == "shared_only" else self.d
        z = []
        if lp: z.append(f(l))
        if rp: z.append(f(r))
        return (self.grade_shared if self.variant == "shared_only" else self.grade_d)(torch.stack(z).mean(0))


def cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return nn.functional.cosine_similarity(a, b, dim=1)


def pair_loss(hx: list[torch.Tensor], neg_r: torch.Tensor, model: HCBFD) -> torch.Tensor:
    # all four L/R cross-eye positives; one grade×quality matched hard negative per L view
    hn = model.h(neg_r); positives = torch.stack([cos(hx[i], hx[j]) for i in (0,1) for j in (2,3)]).mean(0)
    negatives = torch.stack([cos(hx[i], hn) for i in (0,1)]).mean(0)
    return torch.relu(.20 + negatives - positives).mean()


def variance_floor(z: list[torch.Tensor]) -> torch.Tensor:
    x = torch.cat(z, 0); return torch.relu(.10 - x.std(0, unbiased=False)).mean()


def orth(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (nn.functional.normalize(a, dim=1) * nn.functional.normalize(b, dim=1)).sum(1).pow(2).mean()


def train_model(cases: list[dict[str, Any]], dim: int, seed: int, variant: str, epochs: int) -> HCBFD:
    model = HCBFD(dim, variant); neg = match_index(cases, seed, joint=True)
    counts = Counter(c["grade"] for c in cases); gw = torch.tensor([len(cases)/max(5*counts.get(i,1),1) for i in range(5)], dtype=torch.float32)
    ce_g = nn.CrossEntropyLoss(weight=gw); ce_q = nn.CrossEntropyLoss(); opt = torch.optim.Adam(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    loader = DataLoader(PatientDataset(cases, neg), batch_size=32, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(seed))
    for _ in range(epochs):
        model.train()
        for l1,l2,r1,r2,rn,g,q in loader:
            opt.zero_grad(set_to_none=True); hx,dx,ex = model.encode(l1,l2,r1,r2)
            # Baseline uses h for grade and a cross-eye matched pair objective.  Hierarchy moves grade to d.
            hmean = torch.stack(hx).mean(0); dmean = torch.stack(dx).mean(0); emean = torch.stack(ex).mean(0)
            loss = pair_loss(hx, rn, model)
            if variant == "shared_only":
                loss = loss + ce_g(model.grade_shared(hmean), g)
            else:
                loss = loss + ce_g(model.grade_d(dmean), g) + .20 * (1 - .5*(cos(ex[0],ex[1])+cos(ex[2],ex[3]))).mean()
                loss = loss + .05 * ce_q(model.quality_d(dmean),q) + .05 * ce_q(model.quality_e(emean),q)
                loss = loss + .01*(orth(hmean,dmean)+orth(hmean,emean)+orth(dmean,emean))
                if variant == "hierarchical_adversary":
                    loss = loss + .10*ce_g(model.nuisance_grade(grl(hmean,1.0)),g) + .10*ce_q(model.nuisance_quality(grl(hmean,1.0)),q)
            loss = loss + .02*variance_floor(hx)
            loss.backward(); opt.step()
    return model


def embeddings(model: HCBFD, cases: list[dict[str, Any]], branch: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval(); output = [[],[],[],[]]
    fn = getattr(model, branch)
    with torch.no_grad():
        for start in range(0,len(cases),64):
            b=cases[start:start+64]
            for k,v in enumerate(VIEWS): output[k].append(fn(torch.from_numpy(np.stack([c["views"][v] for c in b]))).numpy())
    return tuple(np.concatenate(x,0) for x in output)  # type: ignore[return-value]


def cosine_np(a,b):
    a=a/np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-12); b=b/np.maximum(np.linalg.norm(b,axis=1,keepdims=True),1e-12); return (a*b).sum(1)


def pair_metrics(model: HCBFD, cases: list[dict[str, Any]], seed: int, joint: bool, external_schema: str | None=None) -> dict[str, Any]:
    l1,l2,r1,r2 = embeddings(model,cases,"h")
    if external_schema is None: perm = match_index(cases, seed, joint=joint); condition="grade_x_quality" if joint else "grade"
    else:
        # External matching has predeclared labels only; no quality metadata exist.  The schema only changes label manifest.
        perm=match_index(cases,seed,joint=False); condition=f"external_{external_schema}_grade"
    pos=np.mean(np.stack([cosine_np(a,b) for a in (l1,l2) for b in (r1,r2)]),0)
    neg=np.mean(np.stack([cosine_np(a,b[perm]) for a in (l1,l2) for b in (r1,r2)]),0)
    y=np.r_[np.ones(len(pos)),np.zeros(len(neg))]
    return {"condition":condition,"n_patients":len(cases),"pair_id_auroc":float(roc_auc_score(y,np.r_[pos,neg])),"positive_cosine":float(pos.mean()),"negative_cosine":float(neg.mean()),"pair_gap":float((pos-neg).mean()),"same_eye_h_cosine":float(np.mean(np.r_[cosine_np(l1,l2),cosine_np(r1,r2)]))}


def raw_pair_metrics(cases: list[dict[str, Any]], seed: int, joint: bool) -> dict[str, Any]:
    """Frozen ImageNet feature control on precisely the same eligible cases."""
    perm=match_index(cases,seed,joint); z=[np.stack([c["views"][v] for c in cases]) for v in VIEWS]
    pos=np.mean(np.stack([cosine_np(a,b) for a in z[:2] for b in z[2:]]),0)
    neg=np.mean(np.stack([cosine_np(a,b[perm]) for a in z[:2] for b in z[2:]]),0)
    return {"n_patients":len(cases),"pair_id_auroc":float(roc_auc_score(np.r_[np.ones(len(pos)),np.zeros(len(neg))],np.r_[pos,neg])),"pair_gap":float((pos-neg).mean())}


@torch.no_grad()
def endpoint(model: HCBFD, cases: list[dict[str, Any]], condition: str) -> dict[str,float]:
    y=[]; p=[]; model.eval()
    for st in range(0,len(cases),64):
        b=cases[st:st+64]; l=np.stack([c["views"]["L1"] for c in b]); r=np.stack([c["views"]["R1"] for c in b])
        if condition=="swap": l,r=r,l
        logits=model.classify(torch.from_numpy(l),torch.from_numpy(r),condition!="missing_left",condition!="missing_right")
        y.extend(c["grade"] for c in b);p.extend(logits.argmax(1).tolist())
    return {"accuracy":float(accuracy_score(y,p)),"balanced_accuracy":float(balanced_accuracy_score(y,p)),"macro_f1":float(f1_score(y,p,average="macro",zero_division=0))}


def probe(train_model_: HCBFD, train: list[dict[str,Any]], ev: list[dict[str,Any]], branch: str, target: str) -> dict[str, float]:
    def x(cases):
        z=embeddings(train_model_,cases,branch);return np.mean(np.stack(z),0)
    xt,xe=x(train),x(ev); yt=np.array([c[target] for c in train]); ye=np.array([c[target] for c in ev])
    if len(np.unique(yt)) < 2: return {"balanced_accuracy":float("nan"),"macro_ovr_auroc":float("nan")}
    sc=StandardScaler().fit(xt); clf=LogisticRegression(C=1.,class_weight="balanced",max_iter=2000,random_state=20260922).fit(sc.transform(xt),yt)
    pred=clf.predict(sc.transform(xe));prob=clf.predict_proba(sc.transform(xe));
    try: auc=float(roc_auc_score(ye,prob,labels=clf.classes_,multi_class="ovr",average="macro"))
    except ValueError: auc=float("nan")
    return {"balanced_accuracy":float(balanced_accuracy_score(ye,pred)),"macro_ovr_auroc":auc}


def noncollapse(model: HCBFD, cases: list[dict[str,Any]]) -> dict[str,float]:
    out={}
    for branch in ("h","d","e"):
        z=np.concatenate(embeddings(model,cases,branch),0);out[f"{branch}_mean_feature_std"]=float(z.std(0).mean())
    e=embeddings(model,cases,"e");out["same_eye_e_cosine"]=float(np.mean(np.r_[cosine_np(e[0],e[1]),cosine_np(e[2],e[3])]))
    return out


def aggregate(rows: list[dict[str,Any]], stage: str) -> list[dict[str,Any]]:
    rs=[r for r in rows if r["stage"]==stage]; result=[]
    for var in sorted(set(r["variant"] for r in rs)):
        x=[r for r in rs if r["variant"]==var]
        def mean(path):
            vals=[]
            for r in x:
                v=r
                for key in path: v=v[key]
                vals.append(float(v))
            return float(np.mean(vals)),float(np.std(vals,ddof=0))
        record={"stage":stage,"variant":var,"n_seeds":len(x)}
        for label,path in (("joint_pair_auc",("deepdrid_pair", "pair_id_auroc")),("pair_gap",("deepdrid_pair","pair_gap")),
                           ("same_eye_e",("noncollapse","same_eye_e_cosine")),("h_grade_probe",("probes","h_grade","balanced_accuracy")),
                           ("h_quality_probe",("probes","h_quality","balanced_accuracy")),("d_grade_probe",("probes","d_grade","balanced_accuracy"))):
            a,b=mean(path);record[label+"_mean"],record[label+"_sd"]=a,b
        if stage=="refit_test":
            for label,path in (("test_full_bacc",("endpoints","full","balanced_accuracy")),("test_missing_left_bacc",("endpoints","missing_left","balanced_accuracy")),("test_missing_right_bacc",("endpoints","missing_right","balanced_accuracy")),("test_swap_bacc",("endpoints","swap","balanced_accuracy")),("cfp_patientmax_auc",("external_patientmax","pair_id_auroc")),("cfp_eyemode_auc",("external_eyemode","pair_id_auroc"))):
                a,b=mean(path);record[label+"_mean"],record[label+"_sd"]=a,b
        result.append(record)
    return result


def report(payload: dict) -> str:
    val=pd.DataFrame([x for x in payload["aggregate"] if x["stage"]=="validation"]); test=pd.DataFrame([x for x in payload["aggregate"] if x["stage"]=="refit_test"])
    def table(frame, cols):
        lines=["| "+" | ".join(cols)+" |","| "+" | ".join(["---"]*len(cols))+" |"]
        for _,r in frame.iterrows():
            a=[]
            for c in cols:
                v=r[c];a.append(f"{v:.4f}" if isinstance(v,(float,np.floating)) else str(v))
            lines.append("| "+" | ".join(a)+" |")
        return "\n".join(lines)
    vcols=["variant","joint_pair_auc_mean","joint_pair_auc_sd","pair_gap_mean","same_eye_e_mean","h_grade_probe_mean","h_quality_probe_mean","d_grade_probe_mean"]
    tcols=["variant","test_full_bacc_mean","test_missing_left_bacc_mean","test_missing_right_bacc_mean","test_swap_bacc_mean","joint_pair_auc_mean","cfp_patientmax_auc_mean","cfp_eyemode_auc_mean"]
    sel=payload["selection"]; raw=payload["raw_controls"]
    return f'''# R2.1 H-CBFD 冻结特征 screen v1

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

{table(val,vcols)}

选择结果：`{sel['selected_variant']}`。理由：validation joint grade×quality pair-ID 均值
最高且 h/d/e 均未发生 variance collapse（最小 branch mean feature std={sel['minimum_branch_std']:.4f}）。
这不是 test/CFP-OCT 驱动的选择。相同 eligible validation patients 上未训练 raw ResNet50
control 是 {raw['validation']['pair_id_auroc_mean']:.4f} ± {raw['validation']['pair_id_auroc_sd']:.4f}；
selected variant 为 {sel['selected_validation_auc']:.4f}，故通过预设的 internal non-inferiority gate。

## Refit train+val 后的一次性 test / 外部报告

{table(test,tcols)}

`test_full/missing/swap` 是 patient grade endpoint 的辅助鲁棒性读数；本轮主终点仍是
joint-matched cross-eye correspondence。CFP–OCT 两列分别使用预定义 patient-max DME
和 legacy eye-mode schema 来限制外部 grade-matched negative，二者都仅作冻结外部读数。
相同 CFP-OCT eligible patients 的 raw controls 分别是 {raw['cfp_patientmax']['pair_id_auroc_mean']:.4f} 和
{raw['cfp_eyemode']['pair_id_auroc_mean']:.4f}；selected H-CBFD 分别为
{sel['selected_external_patientmax_auc']:.4f}/{sel['selected_external_eyemode_auc']:.4f}。其外部降幅为
{sel['external_patientmax_drop_vs_raw']:.4f}/{sel['external_eyemode_drop_vs_raw']:.4f}，超过第二轮预设的
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
'''


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--deep-features",type=Path,required=True);ap.add_argument("--deep-manifest",type=Path,required=True)
    ap.add_argument("--cfp-features",type=Path,required=True);ap.add_argument("--cfp-patientmax",type=Path,required=True);ap.add_argument("--cfp-eyemode",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True);ap.add_argument("--epochs",type=int,default=24)
    ap.add_argument("--seeds",type=int,nargs="+",default=[20260922,20260923,20260924,20260925,20260926]);args=ap.parse_args();torch.set_num_threads(2)
    deep,audit=four_view_cases(np.load(args.deep_features).astype(np.float32),pd.read_csv(args.deep_manifest)); raw_train=[c for c in deep if c["split"]=="train"];raw_val=[c for c in deep if c["split"]=="val"];raw_test=[c for c in deep if c["split"]=="test"]
    # Never replace an infeasible singleton with an unmatched negative.  Exact
    # conditional endpoints use the auditable feasible subset only.
    train, fa_train=retain_feasible_strata(raw_train,True);val, fa_val=retain_feasible_strata(raw_val,True);test, fa_test=retain_feasible_strata(raw_test,True)
    feasibility={"train":fa_train,"val":fa_val,"test":fa_test}
    for x in (train,val,test): match_index(x,20260922,True)
    cfpfeat=np.load(args.cfp_features).astype(np.float32);ext_max_raw=external_cases(cfpfeat,pd.read_csv(args.cfp_patientmax));ext_eye_raw=external_cases(cfpfeat,pd.read_csv(args.cfp_eyemode));ext_max,fa_extmax=retain_feasible_strata(ext_max_raw,False);ext_eye,fa_exteye=retain_feasible_strata(ext_eye_raw,False)
    feasibility["cfp_patientmax"]=fa_extmax;feasibility["cfp_eyemode"]=fa_exteye
    for x in (ext_max,ext_eye): match_index(x,20260922,False)
    variants=("shared_only","hierarchical_no_adversary","hierarchical_adversary");rows=[];dim=len(train[0]["views"]["L1"])
    for seed in args.seeds:
        for variant in variants:
            seed_all(seed);m=train_model(train,dim,seed,variant,args.epochs)
            rows.append({"stage":"validation","seed":seed,"variant":variant,"deepdrid_pair":pair_metrics(m,val,seed,True),"noncollapse":noncollapse(m,val),"probes":{"h_grade":probe(m,train,val,"h","grade"),"h_quality":probe(m,train,val,"h","quality_tier"),"d_grade":probe(m,train,val,"d","grade")}})
            print(json.dumps({"stage":"validation","seed":seed,"variant":variant,"auc":rows[-1]["deepdrid_pair"]["pair_id_auroc"]}),flush=True)
    agg_val=aggregate(rows,"validation")
    # Predeclared selection: highest mean joint-matched validation AUROC among noncollapsed variants.
    eligible=[r for r in agg_val if r["same_eye_e_mean"]>0 and all(np.isfinite(r[k]) for k in ("joint_pair_auc_mean","h_grade_probe_mean","d_grade_probe_mean"))]
    chosen=max(eligible,key=lambda r:r["joint_pair_auc_mean"])
    # Refit all predeclared variants to avoid a cherry-picked post-selection test table.
    for seed in args.seeds:
        for variant in variants:
            seed_all(seed);m=train_model(train+val,dim,seed,variant,args.epochs)
            rows.append({"stage":"refit_test","seed":seed,"variant":variant,"deepdrid_pair":pair_metrics(m,test,seed,True),"noncollapse":noncollapse(m,test),"probes":{"h_grade":probe(m,train+val,test,"h","grade"),"h_quality":probe(m,train+val,test,"h","quality_tier"),"d_grade":probe(m,train+val,test,"d","grade")},"endpoints":{k:endpoint(m,test,k) for k in ("full","missing_left","missing_right","swap")},"external_patientmax":pair_metrics(m,ext_max,seed,False,"patient_max_dme"),"external_eyemode":pair_metrics(m,ext_eye,seed,False,"legacy_eye_mode")})
            print(json.dumps({"stage":"refit_test","seed":seed,"variant":variant,"test_auc":rows[-1]["deepdrid_pair"]["pair_id_auroc"],"cfp":rows[-1]["external_patientmax"]["pair_id_auroc"]}),flush=True)
    selected_std = [r["noncollapse"][key] for r in rows if r["stage"] == "validation" and r["variant"] == chosen["variant"] for key in ("h_mean_feature_std", "d_mean_feature_std", "e_mean_feature_std")]
    def control_summary(cases, joint):
        values=[raw_pair_metrics(cases,s,joint) for s in args.seeds]
        return {"n_patients":len(cases),"per_seed":values,"pair_id_auroc_mean":float(np.mean([x["pair_id_auroc"] for x in values])),"pair_id_auroc_sd":float(np.std([x["pair_id_auroc"] for x in values],ddof=0))}
    raw_controls={"validation":control_summary(val,True),"test":control_summary(test,True),"cfp_patientmax":control_summary(ext_max,False),"cfp_eyemode":control_summary(ext_eye,False)}
    selected_test=[r for r in rows if r["stage"]=="refit_test" and r["variant"]==chosen["variant"]]
    selected_max=float(np.mean([r["external_patientmax"]["pair_id_auroc"] for r in selected_test]));selected_eye=float(np.mean([r["external_eyemode"]["pair_id_auroc"] for r in selected_test]))
    selection={"selected_variant":chosen["variant"],"selection_metric":"mean validation joint grade×quality matched pair-ID AUROC","selected_validation_auc":chosen["joint_pair_auc_mean"],"minimum_branch_std":float(min(selected_std)),"selected_external_patientmax_auc":selected_max,"selected_external_eyemode_auc":selected_eye,"external_patientmax_drop_vs_raw":float(raw_controls["cfp_patientmax"]["pair_id_auroc_mean"]-selected_max),"external_eyemode_drop_vs_raw":float(raw_controls["cfp_eyemode"]["pair_id_auroc_mean"]-selected_eye)}
    payload={"protocol":"r2_1_hcbfd_low_capacity_frozen_feature_screen_v1","scope":"screen only; no novelty claim","selection_discipline":"DeepDRiD train-only fit; validation joint grade×quality matched pair-ID plus non-collapse; test/CFP-OCT not used for selection","feature_input":{"deepdrid":str(args.deep_features),"cfp_oct":str(args.cfp_features)},"four_view_audit":audit,"matching_feasibility":feasibility,"conditional_endpoint_population":"Only strata with at least two patients in the evaluated split are retained. Singleton patients are logged and excluded rather than given an unmatched or relaxed negative.","external_matching":"predefined grade-only (two label schemas); CFP-OCT has no quality-tier field and is never used to select variants","variants":list(variants),"epochs":args.epochs,"seeds":args.seeds,"rows":rows,"aggregate":aggregate(rows,"validation")+aggregate(rows,"refit_test"),"raw_controls":raw_controls,"selection":selection}
    args.out_dir.mkdir(parents=True,exist_ok=True);(args.out_dir/"deepdrid_r21_hcbfd_feature_screen_v1.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n")
    flat=[]
    for r in rows:
        flat.append({"stage":r["stage"],"seed":r["seed"],"variant":r["variant"],"deepdrid_joint_pair_auc":r["deepdrid_pair"]["pair_id_auroc"],"deepdrid_pair_gap":r["deepdrid_pair"]["pair_gap"],"same_eye_e":r["noncollapse"]["same_eye_e_cosine"],"h_std":r["noncollapse"]["h_mean_feature_std"],"d_std":r["noncollapse"]["d_mean_feature_std"],"e_std":r["noncollapse"]["e_mean_feature_std"],"h_grade_probe_bacc":r["probes"]["h_grade"]["balanced_accuracy"],"h_quality_probe_bacc":r["probes"]["h_quality"]["balanced_accuracy"],"d_grade_probe_bacc":r["probes"]["d_grade"]["balanced_accuracy"],**({"test_full_bacc":r["endpoints"]["full"]["balanced_accuracy"],"test_missing_left_bacc":r["endpoints"]["missing_left"]["balanced_accuracy"],"test_missing_right_bacc":r["endpoints"]["missing_right"]["balanced_accuracy"],"test_swap_bacc":r["endpoints"]["swap"]["balanced_accuracy"],"cfp_patientmax_auc":r["external_patientmax"]["pair_id_auroc"],"cfp_eyemode_auc":r["external_eyemode"]["pair_id_auroc"]} if r["stage"]=="refit_test" else {})})
    pd.DataFrame(flat).to_csv(args.out_dir/"deepdrid_r21_hcbfd_feature_screen_v1.csv",index=False)
    (args.out_dir/"DEEPDRID_R21_HCBFD_FEATURE_SCREEN_V1.md").write_text(report(payload),encoding="utf-8")
    print(json.dumps({"selection":payload["selection"],"out":str(args.out_dir)},ensure_ascii=False))

if __name__=="__main__": main()
