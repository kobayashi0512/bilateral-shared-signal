#!/usr/bin/env python3
"""Extract frozen ImageNet ResNet50 features from a manifest image_path column."""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd, torch, torchvision
from PIL import Image


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--manifest',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--batch-size',type=int,default=8); args=ap.parse_args()
    df=pd.read_csv(args.manifest); w=torchvision.models.ResNet50_Weights.IMAGENET1K_V2; net=torchvision.models.resnet50(weights=w).eval(); net.fc=torch.nn.Identity(); transform=w.transforms(); torch.set_num_threads(2); out=np.lib.format.open_memmap(args.out,mode='w+',dtype=np.float32,shape=(len(df),2048))
    with torch.inference_mode():
        for s in range(0,len(df),args.batch_size):
            batch=[]
            for p in df.image_path.iloc[s:s+args.batch_size]:
                with Image.open(p) as im: batch.append(transform(im.convert('RGB')))
            out[s:s+len(batch)]=net(torch.stack(batch)).numpy(); out.flush()
    print({'rows':len(df),'dim':2048,'out':str(args.out)})


if __name__=='__main__': main()
