#!/usr/bin/env python3
"""(a3b) Amostrado: (1) o metadata do test carrega focus_plane_distance e
target_avs? (2) para os IDs que colidem, train e test tem assinaturas
diferentes (prova adicional de cena distinta)?"""
import os, json
from huggingface_hub import hf_hub_download
TOK = os.environ.get("HF_TOKEN"); R = "timseizinger/RealBokeh_3MP"

IDS = [str(i) for i in [1,2,3,5,8,11,13,21,34,50,55,89,101,110,120,144,150,175,200,220]]

def le(split, cid):
    try:
        p = hf_hub_download(R, split + "/metadata/" + cid + ".json", repo_type="dataset", token=TOK)
        with open(p) as f: return json.load(f)
    except Exception as e:
        return {"__erro__": str(e)[:80]}

print("=== 1. CONTEUDO DO METADATA DO SPLIT TEST ===")
d = le("test", "1")
print(json.dumps(d, indent=1)[:900])
chaves = set(d.keys()) if isinstance(d, dict) else set()
print("")
print("chaves presentes:", sorted(chaves))
for k in ("focus_plane_distance", "target_avs", "source_av", "focal_length", "target_images", "source_image"):
    print("  " + k.ljust(22) + ("PRESENTE" if k in chaves else "ausente"))

print("")
print("=== 2. ASSINATURA train x test para IDs que colidem ===")
print("  id | focal/foco TRAIN         | focal/foco TEST          | veredito")
ig = dif = 0
for cid in IDS:
    a, b = le("train", cid), le("test", cid)
    def sig(x):
        if not isinstance(x, dict) or "__erro__" in x: return None
        return (x.get("focal_length"), x.get("focus_plane_distance"), tuple(x.get("target_avs") or []))
    sa, sb = sig(a), sig(b)
    if sa is None or sb is None:
        print("  " + cid.ljust(4) + " indisponivel"); continue
    v = "IGUAL <<<" if sa == sb else "diferente"
    if sa == sb: ig += 1
    else: dif += 1
    print("  " + cid.ljust(4) + "| " + str(sa[:2]).ljust(24) + "| " + str(sb[:2]).ljust(24) + "| " + v)
print("")
print("  assinaturas diferentes: " + str(dif) + "   iguais: " + str(ig))
print("  (assinatura diferente = cena fisica distinta)")
