import os
from huggingface_hub import HfApi, hf_hub_download
from datasets import load_dataset, get_dataset_split_names
tok=os.environ.get("HF_TOKEN")
api=HfApi(token=tok)
r="comHannah/bokeh-dataset"
info=api.dataset_info(r)
print("arquivos:", [(s.rfilename) for s in info.siblings])
try:
    p=hf_hub_download(r,"README.md",repo_type="dataset",token=tok); print("\n--- README ---\n", open(p).read()[:1500])
except Exception as e: print("sem README:", str(e)[:80])
try:
    sp=get_dataset_split_names(r, token=tok); print("\nsplits:", sp)
    d=load_dataset(r, split=sp[0], token=tok)
    print("n=", len(d), "colunas:", d.column_names)
    row=d[0]
    for k,v in row.items():
        print("   %-20s %s" % (k, type(v).__name__ if not isinstance(v,(str,int,float)) else str(v)[:60]))
except Exception as e: print("erro ao abrir:", str(e)[:200])
