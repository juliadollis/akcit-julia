#!/usr/bin/env python3
"""TESTE DE SANIDADE da cadeia de medicao de controlabilidade. So CPU.

Antes de gastar GPU, prova que a metrica:
  (P) detecta obediencia quando ela existe  -> oraculo obediente, LVCorr ~ +1
  (N1) da ~0 quando o 'modelo' ignora o comando -> oraculo surdo
  (N2) da ~0 quando o comando e numericamente nulo -> replica o BUG do
       protocolo atual (k_escala=0.01 deixa o mapa em ~5e-4)
  (D) devolve NaN em entrada degenerada (imagem constante)
E mede o PISO DE RUIDO: a distribuicao de LVCorr sob a hipotese nula.
"""
from __future__ import annotations
import io, os, sys, json, glob
import numpy as np
from PIL import Image
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from controle_lib import (mapa_base, mascaras, lap_resposta, lv_mascarado,
                          lvcorr, lvcorr_pearson_log, faixa_dinamica,
                          frac_monotona, bokeh_oraculo, bootstrap_ic)

ALPHAS = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
DIR_MAPAS = "/workspace/vision-pipeline/temp_depth_maps"
N_IMG = int(os.environ.get("N_IMG", "20"))
LADO = 512


def carregar(n):
    tok = os.environ["HF_TOKEN"]
    ds = load_dataset("parquet", data_files={
        "validation": "hf://datasets/akcit-pixel/DDPD/data/validation-*.parquet"},
        token=tok)["validation"]
    out = []
    for i in range(min(n, len(ds))):
        r = ds[i]
        nome = r.get("file_name_base") or f"val_{i:05d}.png"
        d = r["image_focus"]
        if isinstance(d, dict) and d.get("bytes"): img = Image.open(io.BytesIO(d["bytes"]))
        elif isinstance(d, bytes): img = Image.open(io.BytesIO(d))
        else: img = d
        img = img.convert("RGB")
        w, h = img.size
        if w >= h: nw, nh = LADO, int(h * LADO / w)
        else: nh, nw = LADO, int(w * LADO / h)
        img = img.resize((nw, nh), Image.LANCZOS)
        fw, fh = max(16, (nw // 16) * 16), max(16, (nh // 16) * 16)
        l, t = (nw - fw) // 2, (nh - fh) // 2
        img = img.crop((l, t, l + fw, t + fh))
        p = os.path.join(DIR_MAPAS, f"{nome}_depth.npy")
        if not os.path.exists(p):
            continue
        out.append((nome, img, np.load(p).astype(np.float32)))
    return out


def ruido(img, sigma, seed):
    rng = np.random.default_rng(seed)
    a = np.array(img).astype(np.float32) + rng.normal(0, sigma, np.array(img).shape)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def medir(imgs, b_hat):
    mf, mo = mascaras(b_hat)
    tot, fun, foc = [], [], []
    for im in imgs:
        r = lap_resposta(im)
        tot.append(float(r.var()))
        fun.append(lv_mascarado(r, mf))
        foc.append(lv_mascarado(r, mo))
    return tot, fun, foc


def bloco(nome, gerar, dados, alphas=ALPHAS):
    rho, rhop, dr, mono, rho_f, dr_f, rho_o = [], [], [], [], [], [], []
    for (nm, img, depth) in dados:
        w, h = img.size
        bh, _, _ = mapa_base(depth, w, h)
        saidas = [gerar(img, bh, a, i) for i, a in enumerate(alphas)]
        tot, fun, foc = medir(saidas, bh)
        rho.append(lvcorr(alphas, tot)); rhop.append(lvcorr_pearson_log(alphas, tot))
        dr.append(faixa_dinamica(alphas, tot)); mono.append(frac_monotona(alphas, tot))
        rho_f.append(lvcorr(alphas, fun)); dr_f.append(faixa_dinamica(alphas, fun))
        rho_o.append(lvcorr(alphas, foc))
    def s(v):
        v = np.array([x for x in v if np.isfinite(x)], float)
        return (np.nan, np.nan, np.nan, 0) if len(v) == 0 else (
            float(np.mean(v)), float(np.median(v)), float(np.std(v)), len(v))
    m, md, sd, n = s(rho)
    ic = bootstrap_ic(rho)
    mf_, mdf, sdf, nf = s(rho_f)
    print(f"\n--- {nome} (n={n} imagens, {len(alphas)} pontos) ---")
    print(f"  LVCorr total   media={m:+.4f}  mediana={md:+.4f}  sd={sd:.4f}  IC95=[{ic[0]:+.4f},{ic[1]:+.4f}]")
    print(f"  LVCorr fundo   media={mf_:+.4f}  mediana={mdf:+.4f}  sd={sdf:.4f}")
    print(f"  LVCorr foco    media={s(rho_o)[0]:+.4f}  (deve ser BAIXA num bokeh seletivo)")
    print(f"  LVCorr pearson-log total media={s(rhop)[0]:+.4f}")
    print(f"  faixa dinamica total mediana={s(dr)[1]:.3f}  fundo mediana={s(dr_f)[1]:.3f}  (1.0 = nada mudou)")
    print(f"  fracao monotona decrescente mediana={s(mono)[1]:.3f}")
    return {"nome": nome, "lvcorr_media": m, "lvcorr_mediana": md, "lvcorr_sd": sd,
            "ic95_lo": ic[0], "ic95_hi": ic[1], "n": n,
            "lvcorr_fundo_media": mf_, "lvcorr_foco_media": s(rho_o)[0],
            "dr_total_mediana": s(dr)[1], "dr_fundo_mediana": s(dr_f)[1],
            "mono_mediana": s(mono)[1], "por_imagem": [float(x) for x in rho]}


def main():
    dados = carregar(N_IMG)
    print(f"imagens carregadas com mapa de profundidade: {len(dados)}")
    res = []

    # (P) CONTROLE POSITIVO: oraculo obedece por construcao.
    res.append(bloco("P  oraculo OBEDIENTE (bokeh classico, raio ~ alpha)",
                     lambda im, bh, a, i: bokeh_oraculo(im, bh, a), dados))

    # (N1) CONTROLE NEGATIVO: ignora alpha; so ruido fotografico entre pontos.
    res.append(bloco("N1 oraculo SURDO (alpha ignorado, ruido sigma=0.5/255)",
                     lambda im, bh, a, i: ruido(bokeh_oraculo(im, bh, 0.5), 0.5, 100 + i), dados))

    # (N2) COMANDO NUMERICAMENTE NULO: replica o protocolo atual, em que
    # k_escala=0.01 deixa o mapa em ~5e-4 (medido em diag_mapa.py).
    res.append(bloco("N2 comando NULO (alpha*5e-4, como o k_escala=0.01 de hoje)",
                     lambda im, bh, a, i: ruido(bokeh_oraculo(im, bh, a * 5e-4), 0.5, 200 + i), dados))

    # (N3) mesmo comando nulo, SEM ruido — replica exatamente o pipeline atual,
    # que usa seed FIXA. Aqui a variacao residual e so numerica.
    res.append(bloco("N3 comando NULO sem ruido (seed fixa, como hoje)",
                     lambda im, bh, a, i: bokeh_oraculo(im, bh, a * 5e-4), dados))

    # (P2) OBEDIENCIA FRACA: responde, mas pouco. Serve para checar que a faixa
    # dinamica separa 'obedece muito' de 'obedece pouco' quando o rho satura.
    res.append(bloco("P2 oraculo FRACO (raio_max 12 -> 1.5 px)",
                     lambda im, bh, a, i: ruido(bokeh_oraculo(im, bh, a, raio_max=1.5), 0.5, 300 + i), dados))

    # (P3) BLUR GLOBAL: obedece ao comando mas NAO e seletivo (borra tudo).
    # Serve para provar que LVCorr_foco separa 'controla profundidade de campo'
    # de 'passa um blur na imagem inteira'.
    res.append(bloco("P3 blur GLOBAL (obedece, mas sem seletividade)",
                     lambda im, bh, a, i: bokeh_oraculo(im, bh, a, global_=True), dados))

    # (D) DEGENERADO: imagem constante.
    const = Image.fromarray(np.full((336, 512, 3), 128, np.uint8))
    lv = [float(lap_resposta(const).var()) for _ in ALPHAS]
    print(f"\n--- D  degenerado (imagem constante) ---")
    print(f"  LV constante = {lv[0]:.6f}  ->  LVCorr = {lvcorr(ALPHAS, lv)} (NaN esperado)")

    with open("/workspace/vision-pipeline/lvcorr/sanidade_resultados.json", "w") as f:
        json.dump(res, f, indent=2)
    print("\nresultados em lvcorr/sanidade_resultados.json")


if __name__ == "__main__":
    main()
