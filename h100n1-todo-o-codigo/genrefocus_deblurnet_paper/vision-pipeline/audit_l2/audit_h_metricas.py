#!/usr/bin/env python3
"""Tabela consolidada de metricas ja medidas, para nao repetir trabalho."""
import os
import pyarrow.parquet as pq, pyarrow as pa
from huggingface_hub import HfApi
TOK = os.environ.get("HF_TOKEN"); api = HfApi()
ALVOS = ["juliadollis/bokeh-eval-metricas", "juliadollis/bokeh-bench-metricas",
         "juliadollis/bokeh-eval-rb-metricas", "juliadollis/bokeh-controlabilidade-lv"]
for repo in ALVOS:
    try:
        arqs = [f for f in api.list_repo_files(repo, repo_type="dataset", token=TOK)
                if f.endswith(".parquet")]
    except Exception as e:
        print(repo + ": inacessivel (" + str(e)[:60] + ")")
        continue
    if not arqs:
        print(repo + ": sem parquet")
        continue
    T = pa.concat_tables([pq.read_table("hf://datasets/" + repo + "/" + f) for f in arqs])
    print("")
    print("#" * 110)
    print(repo + "   (" + str(T.num_rows) + " linhas)")
    print("#" * 110)
    cols = T.column_names
    print(" | ".join(c[:18].ljust(18) for c in cols))
    for r in T.to_pylist():
        linha = []
        for c in cols:
            v = r[c]
            s = ("%.4f" % v) if isinstance(v, float) else str(v)
            linha.append(s[:18].ljust(18))
        print(" | ".join(linha))
