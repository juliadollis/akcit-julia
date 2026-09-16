#!/usr/bin/env python3
"""(a2) PROVA: cena "1" de train e cena "1" de test sao a MESMA CENA FISICA?"""
import os, re, collections
import numpy as np
from PIL import Image
from huggingface_hub import HfApi, hf_hub_download

TOK = os.environ.get("HF_TOKEN"); R = "timseizinger/RealBokeh_3MP"
api = HfApi()
fs = api.list_repo_files(R, repo_type="dataset", token=TOK)

def sp(p):
    for s in ("train", "test", "validation"):
        if f"/{s}/" in p or p.startswith(f"{s}/"): return s
    return None

porcena = collections.defaultdict(list)
for p in fs:
    s = sp(p)
    if s and p.lower().endswith((".jpg", ".jpeg", ".png")):
        m = re.search(r"/(?:gt|in)/([^/]+)/", p) or re.search(r"/([0-9]+)/[^/]+$", p)
        if m: porcena[(s, m.group(1))].append(p)

for cid in ("1", "2", "50"):
    tr = sorted(porcena.get(("train", cid), []))
    te = sorted(porcena.get(("test", cid), []))
    print("")
    print("=== cena id " + cid + " ===")
    print("  train: " + str(len(tr)) + " arquivos, ex: " + str(tr[:3]))
    print("  test : " + str(len(te)) + " arquivos, ex: " + str(te[:3]))
    if not tr or not te: continue
    try:
        a = hf_hub_download(R, tr[0], repo_type="dataset", token=TOK)
        b = hf_hub_download(R, te[0], repo_type="dataset", token=TOK)
    except Exception as e:
        print("  falha no download: " + str(e)); continue
    ia, ib = Image.open(a).convert("RGB"), Image.open(b).convert("RGB")
    print("  train " + os.path.basename(tr[0]) + " " + str(ia.size)
          + "   test " + os.path.basename(te[0]) + " " + str(ib.size))
    s = (256, 256)
    d = float(np.abs(np.asarray(ia.resize(s), float) - np.asarray(ib.resize(s), float)).mean())
    ha = np.histogram(np.asarray(ia.convert("L")), 64, (0, 255))[0].astype(float)
    hb = np.histogram(np.asarray(ib.convert("L")), 64, (0, 255))[0].astype(float)
    cc = float(np.corrcoef(ha, hb)[0, 1])
    print("  diferenca media de pixel = %.1f/255   corr histograma = %.3f" % (d, cc))
    print("  >>> " + ("MESMA CENA (CONTAMINACAO)" if d < 12
                      else "CENAS DIFERENTES (numeracao independente por split)"))
