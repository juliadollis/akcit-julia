import os
from datasets import load_dataset
tok=os.environ["HF_TOKEN"]
d=load_dataset("juliadollis/bokeh-eval-metricas", split="train", token=tok, download_mode="force_redownload").to_pandas()
novo = d[d["Dataset"].astype(str).str.contains("bokeh-eq4-")]
print("total de linhas:", len(d), "| da campanha Eq.4 (prefixo bokeh-eq4-):", len(novo))
print()
print("%-28s %-42s %8s %8s %8s %9s" % ("Model","Dataset","LPIPS","DISTS","CLIP-I","LVCorr"))
for _,r in novo.iterrows():
    print("%-28s %-42s %8.4f %8.4f %8.4f %+9.4f" % (r["Model"], str(r["Dataset"]).split("/")[-1], r["LPIPS"], r["DISTS"], r["CLIP-I"], r["LVCorr"]))
print()
print("colisao de nome com a campanha antiga?", any(d[~d["Dataset"].astype(str).str.contains("bokeh-eq4-")]["Model"].isin(novo["Model"])))
