#!/usr/bin/env python3
"""Converte o BLB para o esquema que o run_3models.py consome.

run_3models.py -> bokeh_net.py le: image_focus (AIF), image_blur (alvo),
file_name_base. Guardamos ALEM disso o K de referencia e o plano de foco, que
sao a razao de o BLB existir, e que foram PERDIDOS na conversao do RealBokeh.

Nomes de arquivo `data/validation-*.parquet` para bater com o --padrao default.
Verifica o SENTIDO por nitidez: a AIF tem de ser mais nitida que o alvo.
"""
import io, os, json, argparse
import numpy as np
from PIL import Image
import pyarrow as pa, pyarrow.parquet as pq
from numpy.lib.stride_tricks import sliding_window_view

LAP = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], float)

def lv(im):
    g = np.asarray(im.convert("L"), float)
    return float((sliding_window_view(g, (3, 3)) * LAP).sum((-1, -2)).var())

def png_bytes(p, lado):
    im = Image.open(p).convert("RGB")
    if lado:
        w, h = im.size
        if w >= h: nw, nh = lado, max(int(h * lado / w), 1)
        else:      nh, nw = lado, max(int(w * lado / h), 1)
        im = im.resize((nw, nh), Image.LANCZOS)
    b = io.BytesIO(); im.save(b, format="PNG")
    return b.getvalue(), im

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raiz", default="/workspace/blb/data")
    ap.add_argument("--saida", default="/workspace/out_blb")
    ap.add_argument("--lado", type=int, default=1024, help="0 = resolucao original")
    ap.add_argument("--por-lote", type=int, default=100)
    a = ap.parse_args()
    os.makedirs(a.saida, exist_ok=True)

    linhas, invertidos, faltando = [], 0, 0
    for cena in sorted(os.listdir(a.raiz)):
        d = os.path.join(a.raiz, cena)
        pa_aif, pa_inf = os.path.join(d, "image.jpg"), os.path.join(d, "info.json")
        if not (os.path.exists(pa_aif) and os.path.exists(pa_inf)):
            print("  cena %s incompleta, pulada" % cena); faltando += 1; continue
        info = json.load(open(pa_inf))
        ks = info.get("blur_parameters") or []
        fds = info.get("focus_distances") or []
        aif_b, aif_im = png_bytes(pa_aif, a.lado)
        lv_aif = lv(aif_im)
        disp_b = None
        pd_ = os.path.join(d, "disparity.jpg")
        if os.path.exists(pd_): disp_b = png_bytes(pd_, a.lado)[0]
        for kk in range(len(ks)):
            for dd in range(len(fds)):
                p = os.path.join(d, "bokeh_%02d_%02d.jpg" % (kk, dd))
                if not os.path.exists(p): faltando += 1; continue
                alvo_b, alvo_im = png_bytes(p, a.lado)
                lv_alvo = lv(alvo_im)
                if lv_alvo > lv_aif: invertidos += 1
                linhas.append({
                    "file_name_base": "blb_%s_k%02d_d%02d" % (cena, kk, dd),
                    "image_focus": aif_b, "image_blur": alvo_b,
                    "cena": cena, "k_idx": kk, "refocus_idx": dd,
                    "k_ref": float(ks[kk]), "focus_distance": float(fds[dd]),
                    "f_stop": float((info.get("f_stops") or [0])[kk]) if info.get("f_stops") else 0.0,
                    "focal_length": float(info.get("focal_length") or 0.0),
                    "disparity": disp_b, "lv_aif": lv_aif, "lv_alvo": lv_alvo,
                })
    print("linhas: %d | invertidos(alvo mais nitido): %d | faltando: %d"
          % (len(linhas), invertidos, faltando))
    if invertidos:
        print("  ATENCAO: %d pares com sentido invertido" % invertidos)

    esq = pa.schema([
        pa.field("file_name_base", pa.string()), pa.field("image_focus", pa.binary()),
        pa.field("image_blur", pa.binary()), pa.field("cena", pa.string()),
        pa.field("k_idx", pa.int32()), pa.field("refocus_idx", pa.int32()),
        pa.field("k_ref", pa.float32()), pa.field("focus_distance", pa.float32()),
        pa.field("f_stop", pa.float32()), pa.field("focal_length", pa.float32()),
        pa.field("disparity", pa.binary()), pa.field("lv_aif", pa.float32()),
        pa.field("lv_alvo", pa.float32()),
    ])
    meta = {b"huggingface": json.dumps({"info": {"features": {
        "file_name_base": {"dtype": "string", "_type": "Value"},
        "image_focus": {"_type": "Image"}, "image_blur": {"_type": "Image"},
        "cena": {"dtype": "string", "_type": "Value"},
        "k_idx": {"dtype": "int32", "_type": "Value"},
        "refocus_idx": {"dtype": "int32", "_type": "Value"},
        "k_ref": {"dtype": "float32", "_type": "Value"},
        "focus_distance": {"dtype": "float32", "_type": "Value"},
        "f_stop": {"dtype": "float32", "_type": "Value"},
        "focal_length": {"dtype": "float32", "_type": "Value"},
        "disparity": {"_type": "Image"},
        "lv_aif": {"dtype": "float32", "_type": "Value"},
        "lv_alvo": {"dtype": "float32", "_type": "Value"}}}}).encode()}
    n = a.por_lote
    lotes = [linhas[i:i + n] for i in range(0, len(linhas), n)]
    for i, lote in enumerate(lotes):
        t = pa.Table.from_pylist(lote, schema=esq.with_metadata(meta))
        fn = os.path.join(a.saida, "validation-%05d-of-%05d.parquet" % (i, len(lotes)))
        pq.write_table(t, fn)
        print("  escrito", fn, len(lote), "linhas")

if __name__ == "__main__":
    main()
