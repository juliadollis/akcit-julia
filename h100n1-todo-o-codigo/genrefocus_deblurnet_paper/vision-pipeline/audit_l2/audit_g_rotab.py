#!/usr/bin/env python3
"""Rota b: tem procedencia? Toca no EBB!?"""
import os, collections
import pyarrow.parquet as pq, pyarrow as pa
from huggingface_hub import HfApi
TOK = os.environ.get("HF_TOKEN"); api = HfApi()
REPO = "AKCITPixel3/BKXcuVXCmeRvN"
arqs = sorted(f for f in api.list_repo_files(REPO, repo_type="dataset", token=TOK)
              if f.endswith(".parquet"))
print("parquets:", len(arqs))
esq = pq.read_schema("hf://datasets/" + REPO + "/" + arqs[0])
print("TODAS as colunas:", esq.names)
cols = [c for c in ("source_path", "source_aif", "source_bokeh", "stem", "exif") if c in esq.names]
print("colunas de procedencia:", cols)
sel = list(dict.fromkeys([arqs[0], arqs[len(arqs)//2], arqs[-1]]))
T = pa.concat_tables([pq.read_table("hf://datasets/" + REPO + "/" + f, columns=cols) for f in sel])
print("linhas amostradas:", T.num_rows)
for c in cols:
    print("")
    print("--- " + c + " ---")
    for x in T[c].to_pylist()[:8]:
        print("    " + str(x)[:150])
todos = []
for c in cols:
    if c != "exif":
        todos += [str(x or "") for x in T[c].to_pylist()]
def fonte(p):
    pl = p.lower()
    for k in ("ebb", "everything", "flickr", "itw", "bokeh_diffusion", "realbokeh", "lfdof"):
        if k in pl:
            return k
    return "outro"
print("")
print("fontes detectadas: " + str(collections.Counter(fonte(p) for p in todos).most_common()))
print("mencionam EBB!: " + str(sum(1 for p in todos if "ebb" in p.lower() or "everything" in p.lower())))
