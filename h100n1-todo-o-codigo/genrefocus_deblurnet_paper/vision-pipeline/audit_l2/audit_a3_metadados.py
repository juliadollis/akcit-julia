#!/usr/bin/env python3
"""(a3) Prova quase exaustiva de que train e test sao cenas distintas, e
levantamento do que os metadados do RealBokeh oferecem (focus_plane_distance,
target_avs, focal_length) que o benchmark deveria ter aproveitado."""
import os, json, collections
from huggingface_hub import HfApi, hf_hub_download

TOK = os.environ.get("HF_TOKEN"); R = "timseizinger/RealBokeh_3MP"
api = HfApi()
fs = api.list_repo_files(R, repo_type="dataset", token=TOK)
js = [f for f in fs if f.endswith(".json")]
print("jsons no repo:", len(js))
for f in js[:20]: print("   ", f)

def carrega(f):
    p = hf_hub_download(R, f, repo_type="dataset", token=TOK)
    with open(p) as fh: return json.load(fh)

dados = {}
for f in js:
    try:
        d = carrega(f)
    except Exception as e:
        print("falha", f, e); continue
    split = "test" if "test" in f else ("train" if "train" in f else
            ("validation" if "val" in f else "?"))
    dados.setdefault(split, []).append((f, d))

for split, lst in dados.items():
    for f, d in lst:
        print("")
        print("=== " + f + " (split " + split + ") ===")
        if isinstance(d, dict):
            ks = list(d.keys()); print("  dict, " + str(len(ks)) + " chaves, ex:", ks[:5])
            v = d[ks[0]] if ks else None
        elif isinstance(d, list):
            print("  lista de " + str(len(d)) + " entradas"); v = d[0] if d else None
        else:
            v = None
        print("  exemplo de entrada:", json.dumps(v)[:400] if v is not None else None)

# assinatura por cena = (focal_length, focus_plane_distance) -> compara splits
def assinaturas(lst):
    out = {}
    for _, d in lst:
        it = d.values() if isinstance(d, dict) else d
        for e in it:
            if not isinstance(e, dict): continue
            src = e.get("source_image") or e.get("source") or ""
            cena = src.split("/")[-2] if "/" in src else None
            out[cena] = (e.get("focal_length"), e.get("focus_plane_distance"))
    return out

if "train" in dados and "test" in dados:
    at, ae = assinaturas(dados["train"]), assinaturas(dados["test"])
    comuns = set(at) & set(ae)
    iguais = [c for c in comuns if at[c] == ae[c] and at[c] != (None, None)]
    print("")
    print("=== assinatura (focal_length, focus_plane_distance) por cena ===")
    print("  cenas com id em ambos: " + str(len(comuns)))
    print("  com assinatura IDENTICA: " + str(len(iguais)) + " -> " + str(iguais[:10]))
    print("  assinatura diferente prova cena distinta.")
