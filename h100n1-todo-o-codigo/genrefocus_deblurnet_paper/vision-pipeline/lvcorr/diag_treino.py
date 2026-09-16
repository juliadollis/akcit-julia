#!/usr/bin/env python3
"""Distribuicao do mapa de defocus que cada modelo VIU NO TREINO.

Sem baixar imagem: o mapa e monotono em depth_norm, entao o max do mapa esta
sempre num dos extremos z_min/z_max. Para o kfix isso e exato; para o original
(rota b, k=50 fixo, depth normalizado em [0,1]) tambem, porque
map = clip(50*|D-s1|/100) e monotono por ramo.
"""
import os
import numpy as np
from datasets import load_dataset

TOK = os.environ["HF_TOKEN"]

print("=== kfix: tabela juliadollis/rota-b-kfix-eq3 ===")
ds = load_dataset("juliadollis/rota-b-kfix-eq3", split="train", token=TOK)
print("colunas:", ds.column_names, "| n =", len(ds))
k = np.array(ds["k_eq3"], dtype=np.float64)
zf = np.array(ds["z_focus_m"], dtype=np.float64)
zmin = np.array(ds["z_min_m"], dtype=np.float64)
zmax = np.array(ds["z_max_m"], dtype=np.float64)
mc = np.array(ds["max_coc_calibrado"], dtype=np.float64)
print(f"k_eq3           : min={k.min():.1f} p25={np.percentile(k,25):.1f} med={np.median(k):.1f} p75={np.percentile(k,75):.1f} max={k.max():.1f}")
print(f"max_coc_calibrado: min={mc.min():.4f} med={np.median(mc):.4f} max={mc.max():.4f}  (unico? {len(np.unique(mc))} valores)")
print(f"z_focus_m       : min={zf.min():.3f} med={np.median(zf):.3f} max={zf.max():.3f}")

def coc(z_m, kk, zf_m):
    return kk * np.abs(1.0/(z_m*1000.0) - 1.0/(zf_m*1000.0))

c1 = coc(np.clip(zmin,1e-4,None), k, zf)
c2 = coc(np.clip(zmax,1e-4,None), k, zf)
cmax = np.maximum(c1, c2)
mapa_max = np.clip(cmax/np.maximum(mc,1e-6), 0, 1)
print("\n-- max do MAPA por amostra (kfix, rota b) --")
for p in [0,5,25,50,75,95,100]:
    print(f"   p{p:<3}: {np.percentile(mapa_max,p):.4f}")
print(f"   fracao de amostras com mapa saturando (max>=0.999): {(mapa_max>=0.999).mean()*100:.1f}%")
print(f"   fracao com mapa max < 0.1 (praticamente sem bokeh): {(mapa_max<0.1).mean()*100:.1f}%")

print("\n=== ORIGINAL, rota b: k=50 fixo, depth NORMALIZADO em [0,1], max_coc=100 ===")
print("   map = clip(50*|D_norm - s1|/100) = 0.5*|D_norm - s1|  ->  faixa TEORICA [0, 0.5]")
print("   (auditoria de 2026-08-13 mediu map_max mediana 0.490, teto 0.500 — bate)")

print("\n=== rota c (identica nos dois modelos): k do sweep de SSIM, max_coc=100 ===")
try:
    dc = load_dataset("AKCITPixel3/CMiQdveBBzNii", split="train", token=TOK)
    cols = dc.column_names
    print("colunas:", cols[:20])
    kc = np.array([float(v) if v is not None else np.nan for v in dc["k"]])
    s1 = np.array([float(v) if v is not None else np.nan for v in dc["s1"]])
    ok = ~np.isnan(kc)
    kc, s1 = kc[ok], s1[ok]
    print(f"   k  : min={kc.min():.2f} p25={np.percentile(kc,25):.2f} med={np.median(kc):.2f} p75={np.percentile(kc,75):.2f} max={kc.max():.2f}  (n={len(kc)})")
    print(f"   s1 : min={s1.min():.3f} med={np.median(s1):.3f} max={s1.max():.3f}")
    # D_norm em [0,1] => |D - s1| <= max(s1, 1-s1)
    amp = np.maximum(s1, 1.0-s1)
    mm = np.clip(kc*amp/100.0, 0, 1)
    print("   max do MAPA por amostra (cota superior):")
    for p in [0,5,25,50,75,95,100]:
        print(f"      p{p:<3}: {np.percentile(mm,p):.4f}")
    print(f"      fracao saturando: {(mm>=0.999).mean()*100:.1f}%  |  fracao <0.1: {(mm<0.1).mean()*100:.1f}%")
except Exception as e:
    print("   [falhou]", type(e).__name__, e)
