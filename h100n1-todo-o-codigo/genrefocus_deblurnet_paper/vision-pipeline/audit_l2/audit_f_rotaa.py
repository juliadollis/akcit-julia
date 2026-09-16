#!/usr/bin/env python3
"""Rapido: o `stem` da rota a permite rastrear a imagem EBB! de origem?"""
import os, collections, re
import pyarrow.parquet as pq
from huggingface_hub import HfApi
TOK = os.environ.get("HF_TOKEN"); api = HfApi()
REPO = "AKCITPixel3/AfONERuvNmglv"
arqs = sorted(f for f in api.list_repo_files(REPO, repo_type="dataset", token=TOK)
              if f.endswith(".parquet"))
print("parquets:", len(arqs))
esq = pq.read_schema("hf://datasets/" + REPO + "/" + arqs[0])
print("TODAS as colunas:", esq.names)
# 3 parquets espalhados
for f in [arqs[0], arqs[len(arqs)//2], arqs[-1]]:
    T = pq.read_table("hf://datasets/" + REPO + "/" + f, columns=["stem"])
    st = T["stem"].to_pylist()
    print("")
    print("--- " + f + "  (" + str(len(st)) + " linhas) ---")
    for s in st[:10]: print("    ", s)
    pre = collections.Counter(re.sub(r"\d+", "#", str(s)) for s in st)
    print("   padroes:", pre.most_common(5))
