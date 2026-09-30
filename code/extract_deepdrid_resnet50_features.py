#!/usr/bin/env python3
"""Extract cached ImageNet ResNet50 features for the full audited DeepDRiD split."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--manifest',type=Path,required=True); ap.add_argument('--image-root',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--batch-size',type=int,default=8); args=ap.parse_args()
    df=pd.read_csv(args.manifest)
    w=torchvision.models.ResNet50_Weights.IMAGENET1K_V2; net=torchvision.models.resnet50(weights=w).eval(); net.fc=torch.nn.Identity(); transform=w.transforms(); torch.set_num_threads(2)
    out=np.lib.format.open_memmap(args.out,mode='w+',dtype=np.float32,shape=(len(df),2048))
    with torch.inference_mode():
        for start in range(0,len(df),args.batch_size):
            batch=[]
            for _,row in df.iloc[start:start+args.batch_size].iterrows():
                path=args.image_root/str(row['patient_id'])/(str(row['image_id'])+'.jpg')
                if not path.exists(): raise FileNotFoundError(path)
                with Image.open(path) as im: batch.append(transform(im.convert('RGB')))
            out[start:start+len(batch)]=net(torch.stack(batch)).numpy(); out.flush()
            if start==0 or (start//args.batch_size)%10==0: print(f'{start+len(batch)}/{len(df)}',flush=True)
    print({'rows':len(df),'dim':2048,'out':str(args.out)})


if __name__=='__main__': main()
