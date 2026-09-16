#!/usr/bin/env python3
"""A rota c (AKCITPixel3/CMiQdveBBzNii) contem amostras do split `test` do
RealBokeh_3MP? Se sim, usar esse split para avaliar seria contaminacao."""
import os, re, collections
from datasets import load_dataset
TOK = os.environ["HF_TOKEN"]
ds = load_dataset("AKCITPixel3/CMiQdveBBzNii", split="train", token=TOK)
print("n =", len(ds), "| colunas:", ds.column_names)
for col in ("source_path", "source_aif", "source_bokeh", "route", "stem"):
    if col not in ds.column_names: continue
    v = ds[col][:12000]
    v = [str(x) for x in v if x is not None]
    print(f"\n--- {col} --- exemplos:")
    for x in v[:6]: print("   ", x)
    pref = collections.Counter()
    for x in v:
        p = x.split("/")
        pref["/".join(p[:4]) if len(p) >= 4 else x] += 1
    print("   prefixos mais comuns:")
    for k, c in pref.most_common(8): print(f"     {c:>6}  {k}")
    for pal in ("test", "val", "train", "RealBokeh", "LFDOF", "3MP"):
        n = sum(1 for x in v if re.search(pal, x, re.I))
        if n: print(f"   contem '{pal}': {n} de {len(v)}")
