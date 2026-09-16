import os, numpy as np
from datasets import load_dataset
tok=os.environ["HF_TOKEN"]
ds=load_dataset("parquet", data_files={"v":"hf://datasets/juliadollis/bokeh-bench-realbokeh-test/data/validation-*.parquet"}, token=tok)["v"]
print("total no bench:", len(ds))
s=ds.select(range(40))
r=np.array(s["lv_alvo"])/np.maximum(np.array(s["lv_aif"]),1e-6)
print("40 cenas usadas | razao LV(alvo)/LV(aif): min=%.3f p25=%.3f med=%.3f p75=%.3f max=%.3f" % (
    r.min(), np.percentile(r,25), np.median(r), np.percentile(r,75), r.max()))
print("cenas com alvo MAIS NITIDO que a entrada (razao>1):", int((r>1).sum()))
print("primeiras 5 cenas:", s["cena"][:5])
