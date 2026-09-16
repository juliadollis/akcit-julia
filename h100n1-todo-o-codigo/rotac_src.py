import os, re, collections
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
tok=os.environ.get("HF_TOKEN")
todos=[]
for i in range(1,7):
    p=hf_hub_download("AKCITPixel3/CMiQdveBBzNii",f"data/train_batch_{i:04d}.parquet",repo_type="dataset",token=tok)
    t=pq.read_table(p, columns=["stem","source_aif","source_bokeh","k","calibration_ssim"]).to_pydict()
    for j in range(len(t["stem"])):
        todos.append((t["stem"][j],t["source_aif"][j],t["source_bokeh"][j],t["k"][j],t["calibration_ssim"][j]))
print("total rota c:", len(todos))
sp=collections.Counter()
cenas=set(); ap_aif=collections.Counter(); ap_bok=collections.Counter()
for stem,sa,sb,k,cs in todos:
    m=re.search(r"RealBokeh_3MP/(train|test)/gt/([^/]+)/[^/]*_f([0-9.]+)\.JPG", str(sa) or "")
    if m:
        sp[m.group(1)]+=1; cenas.add(m.group(2)); ap_aif[m.group(3)]+=1
    else: sp["OUTRO:"+str(sa)[:60]]+=1
    m2=re.search(r"_f([0-9.]+)\.JPG", str(sb) or "")
    if m2: ap_bok[m2.group(1)]+=1
print("split de origem:", dict(sp))
print("cenas RealBokeh usadas (train):", len(cenas))
print("abertura do AIF:", dict(sorted(ap_aif.items(), key=lambda x: float(x[0]))))
print("abertura do BOKEH:", dict(sorted(ap_bok.items(), key=lambda x: float(x[0]))))
