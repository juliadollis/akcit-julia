#!/usr/bin/env python3
"""(b) AUDITORIA ADVERSARIAL DO BENCHMARK + (a2) prova de cena fisica distinta.

Nao pergunta "o esquema esta certo?", pergunta "de que jeito isso esta errado?".
Testa o que produziria numeros PLAUSIVEIS E FALSOS:
  1. campos PREENCHIDOS, nao so presentes;
  2. papeis invertidos (image_focus tem de ser o MAIS NITIDO);
  3. image_focus == image_blur (alvo igual a entrada = identidade disfarcada);
  4. dimensoes divergentes entre entrada e alvo;
  5. focus_plane_distance / target_avs aproveitados ou perdidos.
"""
import io, os, sys, collections
import numpy as np
from PIL import Image
import pyarrow.parquet as pq
import pyarrow as pa
from huggingface_hub import HfApi

TOK = os.environ.get("HF_TOKEN")
BENCH = os.environ.get("BENCH_REPO", "juliadollis/bokeh-bench-realbokeh-test")
api = HfApi()

def pil(v):
    if v is None: return None
    if isinstance(v, dict): v = v.get("bytes")
    if isinstance(v, (bytes, bytearray)): return Image.open(io.BytesIO(v))
    return v

def lapvar(im):
    g = np.asarray(im.convert("L"), dtype=np.float64)
    k = np.array([[0,1,0],[1,-4,1],[0,1,0]], float)
    from numpy.lib.stride_tricks import sliding_window_view
    w = sliding_window_view(g, (3,3))
    return float((w * k).sum((-1,-2)).var())

print("=" * 78); print(f"BENCHMARK AUDITADO: {BENCH}"); print("=" * 78)
try:
    arqs = [f for f in api.list_repo_files(BENCH, repo_type="dataset", token=TOK)
            if f.endswith(".parquet")]
except Exception as e:
    print("ERRO ao listar:", e); sys.exit(1)
print("parquets:", len(arqs))
for f in arqs[:8]: print("   ", f)

T = pa.concat_tables([pq.read_table(f"hf://datasets/{BENCH}/{f}") for f in arqs])
print("\nlinhas:", T.num_rows)
print("colunas:", T.column_names)

print("\n--- 1. PREENCHIMENTO (nao basta existir) ---")
for c in T.column_names:
    col = T[c]
    nulos = col.null_count
    vazios = 0
    try:
        vals = col.to_pylist()
        vazios = sum(1 for v in vals if v is None
                     or (isinstance(v, (bytes, str)) and len(v) == 0)
                     or (isinstance(v, dict) and not v.get("bytes") and not v.get("path")))
    except Exception:
        pass
    flag = "  <<< PROBLEMA" if vazios else ""
    print(f"  {c:26s} nulos={nulos:5d}  vazios/nulos={vazios:5d} de {T.num_rows}{flag}")

cols = set(T.column_names)
print("\n--- 5. CAMPOS QUE JUSTIFICAVAM O REALBOKEH ---")
for c in ("focus_plane_distance", "target_avs", "source_av", "target_av",
          "k_ref", "focal_length", "s1", "focus_point_xy", "focus_mask"):
    print(f"  {c:24s} {'PRESENTE' if c in cols else 'AUSENTE'}")

CF = "image_focus" if "image_focus" in cols else None
CB = "image_blur" if "image_blur" in cols else None
if not (CF and CB):
    print("\n!!! colunas image_focus/image_blur AUSENTES — run_3models.py nao le este df")
    sys.exit(1)

print("\n--- 2/3/4. PAPEIS, IDENTIDADE E DIMENSOES (amostra) ---")
N = min(24, T.num_rows)
idx = np.linspace(0, T.num_rows - 1, N).astype(int)
foc = T[CF].to_pylist(); blu = T[CB].to_pylist()
inv = same = dim = 0
lv_f, lv_b = [], []
for i in idx:
    a, b = pil(foc[i]), pil(blu[i])
    if a is None or b is None:
        print(f"  [{i}] imagem NULA"); continue
    a = a.convert("RGB"); b = b.convert("RGB")
    if a.size != b.size:
        dim += 1; print(f"  [{i}] DIMENSAO DIFERENTE focus={a.size} blur={b.size}")
    va, vb = lapvar(a), lapvar(b)
    lv_f.append(va); lv_b.append(vb)
    if a.size == b.size:
        d = float(np.abs(np.asarray(a, float) - np.asarray(b, float)).mean())
        if d < 0.5:
            same += 1; print(f"  [{i}] focus ~= blur (diff={d:.3f}) IDENTIDADE DISFARCADA")
    if vb > va:
        inv += 1; print(f"  [{i}] INVERTIDO: blur({vb:.0f}) MAIS NITIDO que focus({va:.0f})")

print(f"\n  nitidez media  image_focus={np.mean(lv_f):10.1f}   image_blur={np.mean(lv_b):10.1f}")
print(f"  razao focus/blur = {np.mean(lv_f)/max(np.mean(lv_b),1e-9):.2f}x  (deve ser > 1)")
print(f"  papeis invertidos: {inv}/{len(lv_f)}")
print(f"  focus == blur:     {same}/{len(lv_f)}")
print(f"  dimensoes divergentes: {dim}/{len(lv_f)}")
print("\n  VEREDITO ESQUEMA:", "OK" if (inv == 0 and same == 0 and dim == 0) else ">>> DEFEITO <<<")
