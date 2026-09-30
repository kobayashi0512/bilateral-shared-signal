#!/usr/bin/env python3
"""Normalize the local CFP-OCT paired manifest for bilateral probe use."""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('input',type=Path); ap.add_argument('--out',type=Path,required=True); args=ap.parse_args()
    df=pd.read_csv(args.input); out=pd.DataFrame({'patient_id':df.patient_id.astype(str),'eye':df.eye.map({'OD':'R','OI':'L'}),'image_id':df.fundus_name,'image_path':df.fundus_path,'dme_oct':df.dme_oct.astype(int),'benchmark_split':df.split})
    if out.eye.isna().any(): raise ValueError('Unknown eye values')
    if not out.image_path.map(Path).map(Path.exists).all(): raise FileNotFoundError('Missing fundus image')
    args.out.parent.mkdir(parents=True,exist_ok=True); out.to_csv(args.out,index=False); print({'rows':len(out),'patients':int(out.patient_id.nunique()),'complete_patients':int(sum(out.groupby('patient_id').eye.nunique()==2))})


if __name__=='__main__': main()
