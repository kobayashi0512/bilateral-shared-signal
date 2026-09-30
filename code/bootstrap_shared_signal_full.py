#!/usr/bin/env python3
"""Bootstrap CI for full-300 same-patient pair identity separation."""

from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score


def cosine(a,b):
    a=a/np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-12); b=b/np.maximum(np.linalg.norm(b,axis=1,keepdims=True),1e-12); return np.sum(a*b,axis=1)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--features',type=Path,required=True); ap.add_argument('--manifest',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--seed',type=int,default=20260922); ap.add_argument('--n-bootstrap',type=int,default=3000); args=ap.parse_args()
    df=pd.read_csv(args.manifest); x=np.load(args.features); patients=[]
    for pid,g in df.groupby('patient_id',sort=True):
        if set(g.eye)!={'L','R'}: continue
        l=x[g.index[g.eye=='L'].to_numpy()].mean(0); r=x[g.index[g.eye=='R'].to_numpy()].mean(0); grade=int(g.patient_grade.mode().iloc[0]); patients.append((str(pid),grade,l,r))
    left=np.stack([p[2] for p in patients]); right=np.stack([p[3] for p in patients]); grades=np.asarray([p[1] for p in patients]); pos=cosine(left,right); rng=np.random.default_rng(args.seed); aucs=[]; gaps=[]
    for _ in range(args.n_bootstrap):
        ix=rng.integers(0,len(patients),len(patients)); neg_idx=np.arange(len(patients));
        for g in sorted(set(grades)):
            idx=np.flatnonzero(grades==g); perm=rng.permutation(idx)
            if len(idx)>1 and np.any(perm==idx): perm=np.roll(idx,1)
            neg_idx[idx]=perm
        neg=cosine(left[ix],right[neg_idx[ix]]); pp=pos[ix]; aucs.append(roc_auc_score(np.r_[np.ones(len(pp)),np.zeros(len(neg))],np.r_[pp,neg])); gaps.append(float(pp.mean()-neg.mean()))
    out={'protocol':'full300_shared_signal_bootstrap_v1','patients':len(patients),'n_bootstrap':args.n_bootstrap,'same_patient_cosine_mean':float(pos.mean()),'bootstrap_pair_id_auroc_mean':float(np.mean(aucs)),'bootstrap_pair_id_auroc_95ci':[float(np.quantile(aucs,.025)),float(np.quantile(aucs,.975))],'bootstrap_cosine_gap_mean':float(np.mean(gaps)),'bootstrap_cosine_gap_95ci':[float(np.quantile(gaps,.025)),float(np.quantile(gaps,.975))],'negative_control':'patient_grade_matched derangement'}
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))


if __name__=='__main__': main()
