import os, io, json, re, collections, numpy as np
from huggingface_hub import list_repo_files, hf_hub_download
from datasets import load_dataset
from PIL import Image
import pyarrow.parquet as pq
tok=os.environ["HF_TOKEN"]

# ---------- 1. Por que 233 linhas para 220 cenas? ----------
ds=load_dataset("parquet", data_files={"v":"hf://datasets/juliadollis/bokeh-bench-realbokeh-test/data/validation-*.parquet"}, token=tok)["v"]
nomes=ds["file_name_base"]; cenas=ds["cena"]
print("linhas:", len(nomes), "| cena unica:", len(set(cenas)))
naomatch=[n for n in nomes if not re.match(r"^(.*_test_f_\d+)_level_(\d+)_aligned$", n)]
print("nomes que NAO casam o padrao level:", len(naomatch), naomatch[:10])
ids=collections.Counter()
for n in nomes:
    m=re.search(r"_test_f_(\d+)", n)
    ids[m.group(1) if m else "SEM_ID"]+=1
print("ids distintos extraidos:", len(ids), "| ids repetidos:", {k:v for k,v in ids.items() if v>1})

# ---------- 2. Pares ruins ----------
lva=np.array(ds["lv_aif"]); lvb=np.array(ds["lv_alvo"]); r=lvb/np.maximum(lva,1e-6)
ruins=[(nomes[i], float(r[i])) for i in range(len(r)) if r[i]>=1.0]
print("pares com alvo NAO mais borrado que a AIF:", len(ruins), ruins)

# ---------- 3. Cobertura dos metadados ----------
fs=list_repo_files("timseizinger/RealBokeh_3MP", repo_type="dataset", token=tok)
metas={re.match(r"^test/metadata/(.+)\.json$",f).group(1): f for f in fs if re.match(r"^test/metadata/.+\.json$",f)}
print("jsons disponiveis:", len(metas))
faltando=[i for i in ids if i not in metas]
print("ids do bench SEM json:", len(faltando), faltando[:10])
