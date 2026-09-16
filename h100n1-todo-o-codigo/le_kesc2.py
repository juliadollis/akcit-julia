import os, re, pandas as pd
from datasets import load_dataset
pd.set_option("display.width", 200)
tok = os.environ["HF_TOKEN"]
d = load_dataset("juliadollis/bokeh-eval-metricas", split="train", token=tok,
                 download_mode="force_redownload").to_pandas()
def escala(m):
    if m.endswith("-eq4"): return "3.0"
    if "kesc1" in m: return "1.0"
    if "kesc5" in m: return "5.0"
    return None
def mesa(m):
    for t in ["-RB-", "-EBB-"]:
        if t in m: return t.strip("-")
    return None
d = d.assign(escala=d["Model"].map(escala), mesa=d["Model"].map(mesa),
             modelo=d["Model"].map(lambda m: re.sub(r"-(RB|EBB)-.*$", "", m)))
d = d.dropna(subset=["escala","mesa"])
d = d.drop_duplicates(subset=["modelo","escala","mesa"], keep="last")
for ms in ["RB","EBB"]:
    sub = d[d["mesa"] == ms]
    if sub.empty: continue
    t = sub.pivot(index="modelo", columns="escala", values="LPIPS")
    cols = [c for c in ["1.0","3.0","5.0"] if c in t.columns]
    print("\n### LPIPS na mesa %s" % ms)
    print(t[cols].round(4).to_string())
