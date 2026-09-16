#!/usr/bin/env python3
"""Qual candidato tem bokeh FORTE o bastante para discriminar modelos?

Um benchmark so serve se a entrada for bem diferente do alvo. Se a AIF ja e
quase igual ao alvo, a linha de identidade fica imbativel e o numero nao mede
nada. Mede-se por: razao de nitidez (lapvar AIF / lapvar alvo) e diferenca
media de pixel entre entrada e alvo.
"""
import io, os
import numpy as np
from PIL import Image
import pyarrow.parquet as pq, pyarrow as pa
from huggingface_hub import HfApi
from numpy.lib.stride_tricks import sliding_window_view

TOK = os.environ.get("HF_TOKEN"); api = HfApi()
CAND = {
    "RealBokeh test (juliadollis)": ("juliadollis/bokeh-bench-realbokeh-test", "image_focus", "image_blur"),
    "DPDD (akcit-pixel/DDPD)":      ("akcit-pixel/DDPD",     "image_focus", "image_blur"),
    "RealDOF (akcit-pixel)":        ("akcit-pixel/RealDOF",  "image_focus", "image_blur"),
}
K = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], float)

def pil(v):
    if isinstance(v, dict): v = v.get("bytes")
    return Image.open(io.BytesIO(v)).convert("RGB") if isinstance(v, (bytes, bytearray)) else v

def lv(im):
    g = np.asarray(im.convert("L"), float)
    return float((sliding_window_view(g, (3, 3)) * K).sum((-1, -2)).var())

for nome, (repo, cf, cb) in CAND.items():
    print(""); print("=" * 78); print(nome + "  ->  " + repo); print("=" * 78)
    try:
        arqs = [f for f in api.list_repo_files(repo, repo_type="dataset", token=TOK)
                if f.endswith(".parquet")]
    except Exception as e:
        print("  inacessivel: " + str(e)[:110]); continue
    val = [f for f in arqs if "valid" in f or "test" in f] or arqs
    esq = pq.read_schema("hf://datasets/" + repo + "/" + val[0])
    if cf not in esq.names or cb not in esq.names:
        print("  colunas esperadas ausentes. tem: " + str(esq.names)); continue
    T = pq.read_table("hf://datasets/" + repo + "/" + val[0], columns=[cf, cb])
    n = min(16, T.num_rows)
    fo = T[cf].to_pylist()[:n]; bl = T[cb].to_pylist()[:n]
    rz, df = [], []
    for a, b in zip(fo, bl):
        ia, ib = pil(a), pil(b)
        if ia is None or ib is None: continue
        if ia.size != ib.size: ib = ib.resize(ia.size)
        la, lb = lv(ia), lv(ib)
        rz.append(la / max(lb, 1e-9))
        s = (256, 256)
        df.append(float(np.abs(np.asarray(ia.resize(s), float) - np.asarray(ib.resize(s), float)).mean()))
    if not rz: print("  sem amostras"); continue
    print("  amostras: " + str(len(rz)) + " (de " + str(T.num_rows) + " no shard)")
    print("  razao de nitidez AIF/alvo: mediana=%.2fx  min=%.2f  max=%.2f" % (
        float(np.median(rz)), min(rz), max(rz)))
    print("  diferenca media entrada->alvo: %.1f/255" % float(np.mean(df)))
    m = float(np.median(rz))
    print("  >>> " + ("bokeh FORTE, discrimina" if m > 3 else
                      ("bokeh moderado" if m > 1.8 else "bokeh FRACO, nao discrimina")))
