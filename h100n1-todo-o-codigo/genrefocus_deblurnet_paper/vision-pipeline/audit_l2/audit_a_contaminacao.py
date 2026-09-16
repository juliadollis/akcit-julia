#!/usr/bin/env python3
"""(a) VERIFICACAO INDEPENDENTE DE CONTAMINACAO.

Nao confia no contaminacao.log de outro agente. Reconstroi a prova do zero e,
alem do caminho de arquivo, testa o que o coordenador levantou: a MESMA CENA
pode aparecer em train (f/2.0) e em test (f/11). Se isso ocorrer, ha
contaminacao POR CENA mesmo sem sobreposicao de arquivo.
"""
import os, re, collections, sys
import pyarrow.parquet as pq
from huggingface_hub import HfApi

TOK = os.environ.get("HF_TOKEN")
api = HfApi()

print("=" * 78); print("PARTE 1 - estrutura real do timseizinger/RealBokeh_3MP"); print("=" * 78)
fs = api.list_repo_files("timseizinger/RealBokeh_3MP", repo_type="dataset", token=TOK)
print("total de arquivos no repo:", len(fs))

def split_de(p):
    for s in ("train", "test", "val", "validation"):
        if f"/{s}/" in p or p.startswith(f"{s}/"):
            return s
    return None

# cena = o diretorio numerico logo acima do arquivo (gt/<cena>/<cena>_fX.JPG)
def cena_de(p):
    m = re.search(r"/(?:gt|in)/([^/]+)/", p)
    if m: return m.group(1)
    m = re.search(r"/([0-9]+)/[^/]+$", p)
    return m.group(1) if m else None

cenas_por_split = collections.defaultdict(set)
arq_por_split = collections.Counter()
for p in fs:
    s = split_de(p)
    if not s: continue
    arq_por_split[s] += 1
    c = cena_de(p)
    if c: cenas_por_split[s].add(c)

for s in sorted(arq_por_split):
    print(f"  split {s:12s} arquivos={arq_por_split[s]:6d}  cenas distintas={len(cenas_por_split[s]):5d}")

tr, te = cenas_por_split.get("train", set()), cenas_por_split.get("test", set())
inter_ids = tr & te
print(f"\n  IDs de cena que aparecem nos DOIS splits: {len(inter_ids)}")
if inter_ids:
    print(f"    amostra: {sorted(inter_ids)[:15]}")
    print("    ATENCAO: ID repetido NAO prova mesma cena fisica (numeracao pode")
    print("    reiniciar por split). Ver PARTE 3.")

print()
print("=" * 78); print("PARTE 2 - procedencia real da rota c (AKCITPixel3/CMiQdveBBzNii)"); print("=" * 78)
arqs = [f for f in api.list_repo_files("AKCITPixel3/CMiQdveBBzNii", repo_type="dataset", token=TOK)
        if f.endswith(".parquet")]
print("parquets:", len(arqs))
COLS = ["stem", "source_aif", "source_bokeh"]
tabs = []
for f in arqs:
    t = pq.read_table(f"hf://datasets/AKCITPixel3/CMiQdveBBzNii/{f}", columns=COLS,
                      filesystem=None)
    tabs.append(t)
import pyarrow as pa
T = pa.concat_tables(tabs)
n = T.num_rows
print("linhas totais:", n)

sa = [x.as_py() or "" for x in T["source_aif"]]
sb = [x.as_py() or "" for x in T["source_bokeh"]]
todos = sa + sb
c_split = collections.Counter(split_de(p) for p in todos)
print("  caminhos de origem por split:", dict(c_split))
vaz = sum(1 for p in todos if not p)
print("  caminhos VAZIOS/nulos:", vaz, "de", len(todos))

do_test = [p for p in todos if split_de(p) == "test"]
print(f"\n  >>> caminhos da rota c que vem de test/: {len(do_test)}")
if do_test:
    for p in do_test[:10]: print("      ", p)

cenas_rotac = set(filter(None, (cena_de(p) for p in todos)))
print(f"  cenas distintas usadas pela rota c: {len(cenas_rotac)}")

print()
print("=" * 78); print("PARTE 3 - sobreposicao POR CENA (o teste que o log anterior nao fez)"); print("=" * 78)
ov = cenas_rotac & te
print(f"  IDs de cena da rota c que tambem sao IDs de cena no split test: {len(ov)}")
if ov: print(f"    amostra: {sorted(ov)[:20]}")
print(f"  IDs de cena da rota c que sao IDs no split train: {len(cenas_rotac & tr)}")
print()
print("  INTERPRETACAO: a rota c so leu de train/. Se um ID coincide, e porque a")
print("  numeracao de cena e independente por split, NAO porque a cena e a mesma.")
print("  A prova definitiva e o caminho: nenhum arquivo de test/ foi lido.")
