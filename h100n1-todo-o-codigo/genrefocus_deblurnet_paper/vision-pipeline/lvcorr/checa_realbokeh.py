#!/usr/bin/env python3
"""Viabilidade do split `test` do RealBokeh_3MP como benchmark de
controlabilidade COM ALVO REAL por abertura. So lista arquivos (nao baixa)."""
import os, re, collections
from huggingface_hub import HfApi
api = HfApi()
TOK = os.environ.get("HF_TOKEN")
fs = api.list_repo_files("timseizinger/RealBokeh_3MP", repo_type="dataset", token=TOK)
print("total de arquivos:", len(fs))
por_split = collections.Counter()
for f in fs:
    m = re.match(r"([^/]+)/", f)
    por_split[m.group(1) if m else f] += 1
print("topo:", por_split.most_common(10))
test = [f for f in fs if "/test/" in f or f.startswith("test/")]
print("\narquivos com 'test':", len(test))
for f in test[:15]: print("  ", f)
cenas = collections.Counter()
aber = collections.Counter()
for f in test:
    p = f.split("/")
    if len(p) >= 3: cenas[p[-2]] += 1
    m = re.search(r"_f([0-9.]+)\.", f, re.I)
    if m: aber[m.group(1)] += 1
print(f"\ncenas distintas no test: {len(cenas)}")
print("arquivos por cena (amostra):", list(cenas.items())[:8])
print("aberturas encontradas:", aber.most_common(20))
js = [f for f in test if f.endswith(".json")][:5]
print("\njsons:", js)
