#!/usr/bin/env python3
"""Quantos pares do RealDOF estao DESALINHADOS?

Metrica full-reference (LPIPS/DISTS/SSIM) sobre par desalinhado pune o modelo
por um deslocamento que nao e culpa dele. Se a fracao for alta, o benchmark
precisa de filtro e isso muda a leitura da tabela.
"""
import os, re, collections
import pyarrow.parquet as pq, pyarrow as pa
from huggingface_hub import HfApi
TOK = os.environ.get("HF_TOKEN"); api = HfApi()
REPO = "akcit-pixel/RealDOF"
arqs = sorted(f for f in api.list_repo_files(REPO, repo_type="dataset", token=TOK)
              if f.endswith(".parquet"))
T = pa.concat_tables([pq.read_table("hf://datasets/" + REPO + "/" + f,
                                    columns=["file_name_base"]) for f in arqs])
nomes = T["file_name_base"].to_pylist()
def cat(n):
    if "misaligned" in n: return "misaligned"
    if "shift" in n: return "shift"
    if "aligned" in n: return "aligned"
    return "outro"
c = collections.Counter(cat(n) for n in nomes)
print("total: %d" % len(nomes))
for k, v in c.most_common():
    print("  %-12s %3d  (%.0f%%)" % (k, v, 100.0 * v / len(nomes)))
sh = [float(m.group(1)) for n in nomes if (m := re.search(r"shift_([0-9.]+)px", n))]
if sh:
    print("  deslocamentos declarados: %s" % sorted(set(sh)))
lim = [n for n in nomes if cat(n) == "aligned"]
print("")
print("  SUBCONJUNTO LIMPO (so 'aligned'): %d de %d" % (len(lim), len(nomes)))
print("  nomes descartados:", [n for n in nomes if cat(n) != "aligned"][:12])
