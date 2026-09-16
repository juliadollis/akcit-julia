import os, re, numpy as np
from datasets import load_dataset
tok=os.environ["HF_TOKEN"]
ds=load_dataset("parquet", data_files={"v":"hf://datasets/juliadollis/bokeh-bench-realbokeh-test/data/validation-*.parquet"}, token=tok)["v"]
s=ds.select(range(40))
ruim=0
for i,n in enumerate(s["file_name_base"]):
    ok = n.endswith("_aligned")
    razao = s["lv_alvo"][i]/max(s["lv_aif"][i],1e-6)
    flag = "" if (ok and razao<1.0) else "   <<< PROBLEMA"
    if flag: ruim+=1
    print("%2d %-58s razao=%.3f%s" % (i, n, razao, flag))
print("TOTAL com problema nas 40:", ruim)
