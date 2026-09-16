#!/usr/bin/env python3
"""Diagnostico: qual e a FAIXA REAL do mapa de condicionamento que o BokehNet
recebe, para cada K do sweep de LVCorr atual.

Reproduz exatamente a matematica de inference/src/pipelines/bokeh_net.py:
    disp = 1/depth ; disp_focus = disp[h//2, w//2]
    cond = clip(|K*(disp-disp_focus)| / MAX_COC, 0, 1)
Sem GPU: usa os .npy ja gravados em temp_depth_maps/.
"""
import glob, os
import numpy as np

MAX_COC = 100.0
DIR = "/workspace/vision-pipeline/temp_depth_maps"
K_SWEEP = [1.0, 5.0, 10.0, 15.0]
K_ESCALAS = [1.0, 0.01]

arquivos = sorted(glob.glob(os.path.join(DIR, "*_depth.npy")))
print(f"mapas encontrados: {len(arquivos)}")

stats = {}
base_p99 = []
for f in arquivos:
    d = np.load(f).astype(np.float32)
    safe = np.where(d > 0.0, d, np.finfo(np.float32).max)
    disp = 1.0 / safe
    h, w = disp.shape
    df = float(disp[h // 2, w // 2])
    b = np.abs(disp - np.float32(df))
    base_p99.append((float(np.percentile(b, 99.5)), float(b.max()), float(np.median(b)), df))
    for esc in K_ESCALAS:
        for k in K_SWEEP:
            cond = np.clip(b * (k * esc) / MAX_COC, 0, 1)
            key = (esc, k)
            stats.setdefault(key, []).append(
                (float(cond.max()), float(cond.mean()), float(np.percentile(cond, 99.5)),
                 float((cond >= 0.999).mean()))
            )

bp = np.array(base_p99)
print("\n=== |disp - disp_focus| (1/m) sobre as %d imagens ===" % len(arquivos))
print(f"  p99.5 : min={bp[:,0].min():.4f} mediana={np.median(bp[:,0]):.4f} max={bp[:,0].max():.4f}")
print(f"  max   : min={bp[:,1].min():.4f} mediana={np.median(bp[:,1]):.4f} max={bp[:,1].max():.4f}")
print(f"  mediana: min={bp[:,2].min():.5f} mediana={np.median(bp[:,2]):.5f} max={bp[:,2].max():.5f}")
print(f"  disp_focus: min={bp[:,3].min():.4f} mediana={np.median(bp[:,3]):.4f} max={bp[:,3].max():.4f}")

print("\n=== mapa de condicionamento cond=clip(K*b/100,0,1) que ENTRA no modelo ===")
print(f"{'k_escala':>9} {'K':>7} {'K_efet':>8} | {'max med':>9} {'max p90':>9} {'mean med':>9} {'%sat med':>9}")
for esc in K_ESCALAS:
    for k in K_SWEEP:
        a = np.array(stats[(esc, k)])
        print(f"{esc:>9} {k:>7} {k*esc:>8.3f} | {np.median(a[:,0]):>9.4f} {np.percentile(a[:,0],90):>9.4f} "
              f"{np.median(a[:,1]):>9.5f} {np.median(a[:,3])*100:>8.2f}%")

print("\n=== faixa dinamica do sweep POR IMAGEM (max do mapa em K=15 / em K=1) ===")
for esc in K_ESCALAS:
    a1 = np.array(stats[(esc, 1.0)])[:, 0]
    a15 = np.array(stats[(esc, 15.0)])[:, 0]
    r = a15 / np.maximum(a1, 1e-12)
    print(f"  k_escala={esc}: razao max_k15/max_k1  mediana={np.median(r):.3f}  "
          f"(15.0 seria o ideal, <15 = saturacao por clip)")
    sat = (np.array(stats[(esc, 15.0)])[:, 3] > 0.001).mean()
    print(f"     imagens com >0.1%% de pixels saturados em K=15: {sat*100:.1f}%")

# Qual K faria o mapa cobrir [0,1]?
alvo = 1.0
k_para_saturar = MAX_COC * alvo / bp[:, 0]
print("\n=== K que levaria o p99.5 do mapa a 1.0 (uso pleno da faixa [0,1]) ===")
print(f"  min={k_para_saturar.min():.1f} mediana={np.median(k_para_saturar):.1f} max={k_para_saturar.max():.1f}")
