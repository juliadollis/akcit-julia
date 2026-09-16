#!/usr/bin/env python3
"""(c) O que o benchmark guardou no lugar de focus_plane_distance/target_avs,
e o f-number ainda e recuperavel a partir do que sobrou?"""
import os, collections, re
import pyarrow.parquet as pq, pyarrow as pa
from huggingface_hub import HfApi
TOK = os.environ.get("HF_TOKEN")
BENCH = "juliadollis/bokeh-bench-realbokeh-test"
api = HfApi()
arqs = [f for f in api.list_repo_files(BENCH, repo_type="dataset", token=TOK) if f.endswith(".parquet")]
T = pa.concat_tables([pq.read_table("hf://datasets/" + BENCH + "/" + f,
      columns=["file_name_base", "cena", "nivel_bokeh", "lv_aif", "lv_alvo"]) for f in arqs])
print("linhas:", T.num_rows)
fn = T["file_name_base"].to_pylist()
ce = T["cena"].to_pylist()
nb = T["nivel_bokeh"].to_pylist()
print("")
print("file_name_base (12 exemplos):")
for x in fn[:12]: print("   ", x)
print("")
print("cena: " + str(len(set(ce))) + " distintas, ex:", list(dict.fromkeys(ce))[:10])
print("nivel_bokeh: tipo=" + type(nb[0]).__name__ + ", " + str(len(set(map(str, nb)))) + " distintos")
print("  contagem:", collections.Counter(map(str, nb)).most_common(15))
print("")
print("pares por cena:", collections.Counter(collections.Counter(ce).values()).most_common())
# o f-number sobrevive no nome?
fnum = [re.findall(r"f(\d+\.?\d*)", str(x)) for x in fn]
tem = sum(1 for x in fnum if x)
print("")
print("file_name_base com f-number embutido: " + str(tem) + " de " + str(len(fn)))
if tem: print("  exemplos:", [x for x in fnum if x][:10])
lv_a = T["lv_aif"].to_pylist(); lv_b = T["lv_alvo"].to_pylist()
print("")
print("lv_aif  min=%.1f med=%.1f max=%.1f" % (min(lv_a), sorted(lv_a)[len(lv_a)//2], max(lv_a)))
print("lv_alvo min=%.1f med=%.1f max=%.1f" % (min(lv_b), sorted(lv_b)[len(lv_b)//2], max(lv_b)))
pior = sum(1 for a, b in zip(lv_a, lv_b) if b >= a)
print("linhas em que o ALVO e mais nitido que a AIF: " + str(pior) + " de " + str(len(lv_a)))
