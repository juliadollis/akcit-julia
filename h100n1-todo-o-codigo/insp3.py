import os, io, json
from huggingface_hub import hf_hub_download, HfApi
tok=os.environ.get("HF_TOKEN")
api=HfApi(token=tok)
for repo in ["comHannah/bokeh-dataset","atfortes/BokehDiffusion"]:
    print("="*70); print(repo)
    try:
        p=hf_hub_download(repo,"README.md",repo_type="dataset",token=tok)
        print(open(p).read()[:2500])
    except Exception as e: print(" sem README", e)
print("#"*70)
import pyarrow.parquet as pq
p=hf_hub_download("atfortes/BokehDiffusion","train.parquet",repo_type="dataset",token=tok)
f=pq.ParquetFile(p); print("BokehDiffusion rows:", f.metadata.num_rows); print(f.schema_arrow)
