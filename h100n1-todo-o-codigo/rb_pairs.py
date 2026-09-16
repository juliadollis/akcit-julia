import os, re, collections
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
tok=os.environ.get("HF_TOKEN")
p=hf_hub_download("akcit-pixel/RealBokeh","data/test-00000-of-00006.parquet",repo_type="dataset",token=tok)
pf=pq.ParquetFile(p); print("rows no shard 0:", pf.metadata.num_rows)
t=pq.read_table(p, columns=["file_name_base"]).to_pydict()["file_name_base"]
print("exemplos file_name_base:", t[:25])
print("n unicos:", len(set(t)))
cen=collections.Counter(str(x).split("_")[0] for x in t)
print("cenas distintas no shard:", len(cen), list(cen.items())[:15])
