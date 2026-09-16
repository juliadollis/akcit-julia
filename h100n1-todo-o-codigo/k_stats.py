import os, io, numpy as np
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
from PIL import Image
tok=os.environ["HF_TOKEN"]
ks=[]; s1s=[]; ss=[]
for i in range(1,7):
    p=hf_hub_download("AKCITPixel3/CMiQdveBBzNii",f"data/train_batch_{i:04d}.parquet",repo_type="dataset",token=tok)
    t=pq.read_table(p, columns=["k","s1","calibration_ssim"]).to_pydict()
    ks+=t["k"]; s1s+=t["s1"]; ss+=t["calibration_ssim"]
ks=np.array(ks,dtype=float); s1s=np.array(s1s,dtype=float)
print("rota c k       : ", np.percentile(ks,[0,10,25,50,75,90,100]).round(2))
print("rota c s1      : ", np.percentile(s1s,[0,10,25,50,75,90,100]).round(4))
# magnitude tipica do mapa de defocus no TREINO: k*|D-s1|/100 com D em [0,1]
p=hf_hub_download("AKCITPixel3/CMiQdveBBzNii","data/train_batch_0001.parquet",repo_type="dataset",token=tok)
t=pq.read_table(p, columns=["depth","k","s1"]).to_pydict()
mags=[]
for j in range(30):
    d=np.array(Image.open(io.BytesIO(t["depth"][j]["bytes"])).convert("L"),dtype=np.float32)/255.0
    dm=np.clip(t["k"][j]*np.abs(d-t["s1"][j])/100.0,0,1)
    mags.append((dm.mean(), dm.max(), float(np.mean(dm>0.5))))
mags=np.array(mags)
print("TREINO defocus map -> media=%.3f  max=%.3f  fracao>0.5=%.3f" % (mags[:,0].mean(), mags[:,1].mean(), mags[:,2].mean()))
