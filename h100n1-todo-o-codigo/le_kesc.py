import os, re, pandas as pd
from datasets import load_dataset
pd.set_option("display.width", 200)
tok = os.environ["HF_TOKEN"]
d = load_dataset("juliadollis/bokeh-eval-metricas", split="train", token=tok,
                 download_mode="force_redownload").to_pandas()
d = d[d["Model"].astype(str).str.contains("-RB-")]
def escala(m):
    if m.endswith("-eq4"): return "3.0 (campanha)"
    if "kesc1" in m: return "1.0"
    if "kesc5" in m: return "5.0"
    return "?"
def modelo(m):
    return re.sub(r"-RB-.*$", "", m)
d = d.assign(escala=d["Model"].map(escala), modelo=d["Model"].map(modelo))
d = d[d["escala"] != "?"].drop_duplicates(subset=["modelo","escala"], keep="last")
for met in ["LPIPS", "DISTS", "LVCorr"]:
    t = d.pivot(index="modelo", columns="escala", values=met)
    cols = [c for c in ["1.0","3.0 (campanha)","5.0"] if c in t.columns]
    print("\n### %s (RealBokeh test v2, 217 cenas)" % met)
    print(t[cols].round(4).to_string())
