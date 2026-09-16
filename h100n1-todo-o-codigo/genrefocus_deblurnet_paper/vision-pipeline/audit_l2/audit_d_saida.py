#!/usr/bin/env python3
"""(b3) image_best_k e mesmo a imagem GERADA, ou uma copia da entrada?
Se for copia da entrada, todas as metricas viram a linha de identidade
disfarcada e o numero parece bom sem o modelo ter feito nada."""
import io, os, sys
import numpy as np
from PIL import Image
import pyarrow.parquet as pq, pyarrow as pa
from huggingface_hub import HfApi
TOK = os.environ.get("HF_TOKEN"); api = HfApi()

BENCH = "juliadollis/bokeh-bench-realbokeh-test"
SAIDAS = sys.argv[1:] or ["juliadollis/bokeh-eval-rb-piloto",
                          "juliadollis/bokeh-eval-rb-piloto-k300"]

def pil(v):
    if isinstance(v, dict): v = v.get("bytes")
    return Image.open(io.BytesIO(v)).convert("RGB") if isinstance(v, (bytes, bytearray)) else v

def lapvar(im):
    from numpy.lib.stride_tricks import sliding_window_view
    g = np.asarray(im.convert("L"), float)
    k = np.array([[0,1,0],[1,-4,1],[0,1,0]], float)
    return float((sliding_window_view(g,(3,3))*k).sum((-1,-2)).var())

# entradas do benchmark, indexadas por file_name_base
arqs = [f for f in api.list_repo_files(BENCH, repo_type="dataset", token=TOK) if f.endswith(".parquet")]
B = pa.concat_tables([pq.read_table("hf://datasets/"+BENCH+"/"+f,
        columns=["file_name_base","image_focus","image_blur"]) for f in arqs])
ent = {n: (f, b) for n, f, b in zip(B["file_name_base"].to_pylist(),
                                    B["image_focus"].to_pylist(), B["image_blur"].to_pylist())}
print("entradas no benchmark:", len(ent))

for repo in SAIDAS:
    print(""); print("=" * 70); print("SAIDA:", repo); print("=" * 70)
    try:
        fs = [f for f in api.list_repo_files(repo, repo_type="dataset", token=TOK) if f.endswith(".parquet")]
    except Exception as e:
        print("  inacessivel:", str(e)[:120]); continue
    if not fs: print("  sem parquet ainda"); continue
    T = pa.concat_tables([pq.read_table("hf://datasets/"+repo+"/"+f) for f in fs])
    print("  linhas:", T.num_rows, "| colunas:", T.column_names)
    if "best_k_value" in T.column_names:
        bk = [v for v in T["best_k_value"].to_pylist() if v is not None]
        if bk: print("  best_k: min=%.3f med=%.3f max=%.3f" % (min(bk), sorted(bk)[len(bk)//2], max(bk)))
    if "ssim_score" in T.column_names:
        ss = [v for v in T["ssim_score"].to_pylist() if v is not None]
        if ss: print("  ssim  : min=%.3f med=%.3f max=%.3f" % (min(ss), sorted(ss)[len(ss)//2], max(ss)))
    nomes = T["file_name_base"].to_pylist()
    gen = T["image_best_k"].to_pylist(); alvo = T["image_real_bokeh"].to_pylist()
    ks = [c for c in ("image_k01","image_k05","image_k10","image_k15") if c in T.column_names]
    kcols = {c: T[c].to_pylist() for c in ks}
    for i in range(min(4, T.num_rows)):
        g, a = pil(gen[i]), pil(alvo[i])
        nm = nomes[i]
        print("  --- " + str(nm))
        if nm in ent:
            e = pil(ent[nm][0])
            if e.size != g.size: e = e.resize(g.size)
            d_in = float(np.abs(np.asarray(e,float)-np.asarray(g,float)).mean())
            print("      gerada vs ENTRADA(aif): diff=%.2f/255  %s" % (
                d_in, "<<< COPIA DA ENTRADA" if d_in < 1.0 else "ok, difere da entrada"))
        if a.size != g.size: a = a.resize(g.size)
        print("      gerada vs ALVO: diff=%.2f/255 | nitidez gerada=%.0f alvo=%.0f"
              % (float(np.abs(np.asarray(a,float)-np.asarray(g,float)).mean()), lapvar(g), lapvar(a)))
        if len(ks) >= 2:
            lv = [lapvar(pil(kcols[c][i])) for c in ks]
            print("      nitidez por K " + str(ks) + " = " + str([round(x) for x in lv]))
            print("      monotonica decrescente (mais K = mais blur)? " +
                  str(all(lv[j] >= lv[j+1] for j in range(len(lv)-1))))
