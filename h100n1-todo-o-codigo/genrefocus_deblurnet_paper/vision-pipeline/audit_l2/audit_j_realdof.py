#!/usr/bin/env python3
"""RealDOF como benchmark de bokeh: quantas cenas, pares completos, sentido."""
import io, os
import numpy as np
from PIL import Image
import pyarrow.parquet as pq, pyarrow as pa
from huggingface_hub import HfApi
from numpy.lib.stride_tricks import sliding_window_view
TOK = os.environ.get("HF_TOKEN"); api = HfApi()
REPO = "akcit-pixel/RealDOF"
LAP = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], float)
def pil(v):
    if isinstance(v, dict): v = v.get("bytes")
    return Image.open(io.BytesIO(v)).convert("RGB") if isinstance(v, (bytes, bytearray)) else v
def lv(im):
    g = np.asarray(im.convert("L"), float)
    return float((sliding_window_view(g, (3, 3)) * LAP).sum((-1, -2)).var())

fs = api.list_repo_files(REPO, repo_type="dataset", token=TOK)
arqs = sorted(f for f in fs if f.endswith(".parquet"))
print("arquivos parquet:", arqs)
T = pa.concat_tables([pq.read_table("hf://datasets/" + REPO + "/" + f) for f in arqs])
print("linhas TOTAIS:", T.num_rows)
print("colunas:", T.column_names)
print("")
print("--- preenchimento ---")
for c in T.column_names:
    col = T[c]; vals = col.to_pylist()
    vaz = sum(1 for v in vals if v is None
              or (isinstance(v, (bytes, str)) and len(v) == 0)
              or (isinstance(v, dict) and not v.get("bytes") and not v.get("path")))
    print("  %-22s nulos=%-5d vazios=%-5d de %d %s" % (c, col.null_count, vaz, T.num_rows,
                                                        "<<< PROBLEMA" if vaz else ""))
CF, CB = "image_focus", "image_blur"
if CF not in T.column_names or CB not in T.column_names:
    print("!!! colunas image_focus/image_blur ausentes"); raise SystemExit(1)
fo, bl = T[CF].to_pylist(), T[CB].to_pylist()
nomes = T["file_name_base"].to_pylist() if "file_name_base" in T.column_names else [str(i) for i in range(T.num_rows)]
print("")
print("--- sentido, identidade e dimensoes (TODAS as %d linhas) ---" % T.num_rows)
inv = same = dim = 0; rz = []
for i, (a, b) in enumerate(zip(fo, bl)):
    ia, ib = pil(a), pil(b)
    if ia is None or ib is None: print("  [%d] NULA" % i); continue
    if ia.size != ib.size: dim += 1; print("  [%d] dim %s vs %s" % (i, ia.size, ib.size)); ib = ib.resize(ia.size)
    la, lb = lv(ia), lv(ib); rz.append(la / max(lb, 1e-9))
    if lb > la: inv += 1; print("  [%d] INVERTIDO %s  aif=%.0f alvo=%.0f" % (i, nomes[i], la, lb))
    d = float(np.abs(np.asarray(ia.resize((256,256)), float) - np.asarray(ib.resize((256,256)), float)).mean())
    if d < 0.5: same += 1; print("  [%d] focus==blur" % i)
print("")
print("  razao de nitidez AIF/alvo: mediana=%.2fx  p10=%.2f  p90=%.2f" % (
    float(np.median(rz)), float(np.percentile(rz, 10)), float(np.percentile(rz, 90))))
print("  invertidos=%d  focus==blur=%d  dim divergente=%d  de %d" % (inv, same, dim, len(rz)))
print("  VEREDITO:", "PRONTO PARA USO" if (inv == 0 and same == 0) else ">>> REVISAR <<<")
print("")
print("  nomes (10):", nomes[:10])
