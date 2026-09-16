import glob, re
import numpy as np, pandas as pd
from datasets import load_dataset
import os
tok=os.environ["HF_TOKEN"]
df = pd.concat([pd.read_parquet(a) for a in sorted(glob.glob("/host/ri_eq4_gpu*/riemann_por_imagem.parquet"))], ignore_index=True)
bru = load_dataset("juliadollis/bokeh-eval-metricas", split="train", token=tok).to_pandas()
rot = dict(zip(bru["Dataset"].astype(str), bru["Model"].astype(str)))
df["id"] = df["repo"].map(rot)
df = df[df["id"].notna()]
def bench(n):
    for b in ("LFREPRO","EBB","RD","RB"):
        if b in n: return b
    return None
MOD={"rotac-only":"so rota c (a+c)","nosso":"nosso fase 2 (a+b+c)","kfix":"kfix","oficial":"oficial do paper",
     "fase1":"fase 1 (a)","nofilter":"sem filtro SSIM","sem-treino":"sem treino","LINHA-DE-IDENTIDADE":"IDENTIDADE"}
def mod(n):
    for k in sorted(MOD,key=len,reverse=True):
        if n.startswith(k): return MOD[k]
    return None
df["b"]=df["id"].map(bench); df["m"]=df["id"].map(mod)
df=df[df.b.notna() & df.m.notna()]
for b in ["RD","RB"]:
    s=df[df.b==b]
    if s.empty: continue
    print("\n### %s  (n cenas = %d)" % (b, s.groupby("m").size().max()))
    print("%-26s %10s %10s %10s %10s %10s" % ("modelo","bF_desfoc","corr","ssim_bord","ssim_plan","queda"))
    ag=s.groupby("m").agg(bF=("bF_desfoque","mean"), co=("corr_desfoque","mean"),
                          sb=("ssim_borda","mean"), sp=("ssim_plano","mean"), qb=("queda_na_borda","mean"))
    for m,r in ag.sort_values("bF",ascending=False).iterrows():
        print("%-26s %10.4f %+10.4f %10.4f %10.4f %10.4f" % (m, r.bF, r.co, r.sb, r.sp, r.qb))
