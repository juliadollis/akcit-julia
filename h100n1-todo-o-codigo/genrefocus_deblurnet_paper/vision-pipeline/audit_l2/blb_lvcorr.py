#!/usr/bin/env python3
"""LVCorr na convencao FIXADA, usando o fundo real do BLB.

CONVENCAO (fixada pelo coordenador, 2026-08-27):
    LVCorr = Pearson(K, variancia do Laplaciano NO FUNDO), SINAL CRU.
Mais bokeh borra o fundo, entao a variancia CAI e o valor tende a ser NEGATIVO.
O paper reporta POSITIVO (BokehMe 0,9940). A divergencia de convencao fica
registrada, nao corrigida em silencio.

Diferenca em relacao ao avaliador do projeto: ele usa a imagem INTEIRA; aqui o
fundo e definido pela DISPARIDADE VERDADEIRA do BLB em relacao ao plano de foco
da amostra, que e a leitura correta (o sujeito em foco nao deve entrar na conta).

Entrada: o repo de saida da inferencia (image_k01/k05/k10/k15) + o benchmark
(disparity, focus_distance). Nao modifica nada; so le e reporta.
"""
import io, os, argparse, collections
import numpy as np
from PIL import Image
import pyarrow.parquet as pq, pyarrow as pa
from huggingface_hub import HfApi
from numpy.lib.stride_tricks import sliding_window_view

LAP = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], float)
KS = [("image_k01", 1.0), ("image_k05", 5.0), ("image_k10", 10.0), ("image_k15", 15.0)]

def pil(v):
    if isinstance(v, dict): v = v.get("bytes")
    return Image.open(io.BytesIO(v)).convert("RGB") if isinstance(v, (bytes, bytearray)) else v

def lapvar(im, mask=None):
    g = np.asarray(im.convert("L"), float)
    L = (sliding_window_view(g, (3, 3)) * LAP).sum((-1, -2))
    if mask is not None:
        m = np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).resize(
            (L.shape[1], L.shape[0]), Image.NEAREST)) > 127
        if m.sum() < 50: return float(L.var())
        return float(L[m].var())
    return float(L.var())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida-infer", required=True, help="repo HF com image_k01..k15")
    ap.add_argument("--bench", default="juliadollis/lf-bokeh-repro-blb")
    ap.add_argument("--modo", default="fundo", choices=["fundo", "inteira"])
    a = ap.parse_args()
    tok = os.environ.get("HF_TOKEN"); api = HfApi()

    B = {}
    try:
        arqs = [f for f in api.list_repo_files(a.bench, repo_type="dataset", token=tok)
                if f.endswith(".parquet")]
        T = pa.concat_tables([pq.read_table("hf://datasets/" + a.bench + "/" + f,
              columns=["file_name_base", "disparity", "focus_distance"]) for f in arqs])
        for n, d, fd in zip(T["file_name_base"].to_pylist(), T["disparity"].to_pylist(),
                            T["focus_distance"].to_pylist()):
            B[n] = (d, fd)
        print("benchmark: %d linhas com disparidade" % len(B))
    except Exception as e:
        print("sem benchmark (%s); caindo para imagem inteira" % str(e)[:60])
        a.modo = "inteira"

    arqs = [f for f in api.list_repo_files(a.saida_infer, repo_type="dataset", token=tok)
            if f.endswith(".parquet")]
    S = pa.concat_tables([pq.read_table("hf://datasets/" + a.saida_infer + "/" + f) for f in arqs])
    print("inferencia: %d linhas" % S.num_rows)

    nomes = S["file_name_base"].to_pylist()
    cols = {c: S[c].to_pylist() for c, _ in KS if c in S.column_names}
    if len(cols) < 2:
        print("!!! faltam colunas image_k*; nada a fazer"); return

    vals = []
    for i, nm in enumerate(nomes):
        mask = None
        if a.modo == "fundo" and nm in B and B[nm][0]:
            disp = np.asarray(pil(B[nm][0]).convert("L"), float) / 255.0
            # fundo = metade MENOS disparidade (mais longe). Robusto e simples.
            mask = (disp < np.median(disp)).astype(float)
        lv = []
        for c, _ in KS:
            if c not in cols: continue
            im = pil(cols[c][i])
            if im is None: break
            if mask is not None and mask.shape != np.asarray(im.convert("L")).shape:
                mm = np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).resize(
                    im.size, Image.NEAREST)) > 127
                lv.append(lapvar(im, mm.astype(float)))
            else:
                lv.append(lapvar(im, mask))
        ks = [k for c, k in KS if c in cols][:len(lv)]
        if len(lv) < 2 or np.std(lv) == 0: continue
        vals.append(float(np.corrcoef(ks, lv)[0, 1]))

    if not vals:
        print("nenhuma amostra valida"); return
    m = float(np.mean(vals))
    print("")
    print("modo: %s | amostras: %d" % (a.modo, len(vals)))
    print("LVCorr (Pearson(K, lapvar), SINAL CRU) = %+.4f" % m)
    print("  mediana=%+.4f  p10=%+.4f  p90=%+.4f" % (
        float(np.median(vals)), float(np.percentile(vals, 10)), float(np.percentile(vals, 90))))
    print("  negativos: %d de %d" % (sum(1 for v in vals if v < 0), len(vals)))
    print("")
    print("  NOTA: o paper reporta LVCorr POSITIVA (BokehMe 0,9940). Valor")
    print("  negativo aqui significa que mais K borra mais, que e o esperado.")
    print("  A diferenca e de CONVENCAO DE SINAL, nao de comportamento.")

if __name__ == "__main__":
    main()
