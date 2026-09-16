import os, re, pandas as pd
from datasets import load_dataset
tok=os.environ["HF_TOKEN"]
d=load_dataset("juliadollis/bokeh-eval-metricas", split="train", token=tok, download_mode="force_redownload").to_pandas()
d=d[d["Dataset"].astype(str).str.contains("bokeh-eq4-")]
def mesa(r):
    m=re.search(r"bokeh-eq4-(rb|rd|ebb|lfrepro)-", r); return m.group(1) if m else "?"
def modelo(r):
    return re.sub(r".*bokeh-eq4-(rb|rd|ebb|lfrepro)-","",r)
d["mesa"]=d["Dataset"].map(mesa); d["modelo"]=d["Dataset"].map(modelo)
t=pd.crosstab(d["modelo"], d["mesa"])
print("### cobertura (modelo x mesa) na campanha Eq.4")
print(t.to_string())
