#!/usr/bin/env python3
"""Reanalisa a LVCorr JA CALCULADA, a partir dos datasets de inferencia no HF.

Objetivo: mostrar, com os proprios dados que geraram o numero atual, que as 4
imagens do sweep (image_k01..image_k15) sao praticamente a MESMA imagem — que e
a razao de a LVCorr atual ser ruido. So CPU.
"""
import os, io, sys, json
import numpy as np
from PIL import Image
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from controle_lib import lap_resposta, lvcorr, faixa_dinamica

CAMPOS = ["image_k01", "image_k05", "image_k10", "image_k15"]
K = [1.0, 5.0, 10.0, 15.0]
REPOS = sys.argv[1:] or [
    "juliadollis/bokeh-eval-infer-kfix-ddpd",
    "juliadollis/bokeh-eval-infer-nosso-kesc",
    "juliadollis/bokeh-eval-infer-oficial-kesc",
]
TOK = os.environ["HF_TOKEN"]

def pil(d):
    if hasattr(d, "convert"): return d
    if isinstance(d, bytes): return Image.open(io.BytesIO(d))
    if isinstance(d, dict) and d.get("bytes"): return Image.open(io.BytesIO(d["bytes"]))
    if isinstance(d, dict) and d.get("path"): return Image.open(d["path"])
    raise ValueError(type(d))

saida = []
for repo in REPOS:
    try:
        ds = load_dataset(repo, split="validation", token=TOK)
    except Exception as e:
        print(f"[{repo}] falhou: {type(e).__name__} {e}"); continue
    print(f"\n=== {repo}  (n={len(ds)}) ===")
    lvs_all, difs, rhos, drs, bestk = [], [], [], [], []
    for r in ds:
        ims = [pil(r[c]).convert("RGB") for c in CAMPOS]
        lv = [float(lap_resposta(i).var()) for i in ims]
        lvs_all.append(lv)
        a = [np.array(i, np.float32) for i in ims]
        # diferenca de pixel entre o K menor e o K maior do sweep
        difs.append(float(np.abs(a[0] - a[-1]).mean()))
        rhos.append(lvcorr(K, lv))
        drs.append(faixa_dinamica(K, lv))
        if "best_k_value" in r: bestk.append(float(r["best_k_value"]))
    lvs_all = np.array(lvs_all); difs = np.array(difs)
    rhos = np.array([x for x in rhos if np.isfinite(x)])
    drs = np.array([x for x in drs if np.isfinite(x)])
    print(f"  |img(K=1) - img(K=15)| medio por pixel (0-255): "
          f"mediana={np.median(difs):.4f}  max={difs.max():.4f}")
    print(f"  LV por K (mediana sobre as imagens): " +
          "  ".join(f"K{int(k):02d}={np.median(lvs_all[:,j]):.2f}" for j, k in enumerate(K)))
    var_rel = lvs_all.std(axis=1) / np.maximum(lvs_all.mean(axis=1), 1e-9)
    print(f"  desvio relativo da LV ao longo do sweep: mediana={np.median(var_rel)*100:.3f}%")
    print(f"  LVCorr (convencao NOVA, +1=obedece): media={rhos.mean():+.4f} "
          f"mediana={np.median(rhos):+.4f} sd={rhos.std():.4f} min={rhos.min():+.4f} max={rhos.max():+.4f}")
    print(f"  faixa dinamica LV(K1)/LV(K15): mediana={np.median(drs):.4f} (1.0 = nada mudou)")
    if bestk: print(f"  best_k escolhido pela bissecao: mediana={np.median(bestk):.4f} "
                    f"min={min(bestk):.4f} max={max(bestk):.4f}")
    saida.append({"repo": repo, "n": len(ds), "dif_px_mediana": float(np.median(difs)),
                  "desvio_rel_lv_mediana": float(np.median(var_rel)),
                  "lvcorr_media": float(rhos.mean()), "lvcorr_sd": float(rhos.std()),
                  "lvcorr_min": float(rhos.min()), "lvcorr_max": float(rhos.max()),
                  "dr_mediana": float(np.median(drs))})
with open("/workspace/vision-pipeline/lvcorr/reanalise_atual.json", "w") as f:
    json.dump(saida, f, indent=2)
print("\nsalvo em lvcorr/reanalise_atual.json")
