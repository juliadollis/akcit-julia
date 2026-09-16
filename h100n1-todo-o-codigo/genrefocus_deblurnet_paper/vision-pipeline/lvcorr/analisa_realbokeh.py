#!/usr/bin/env python3
"""Analise do teste com ALVO REAL (RealBokeh_3MP test)."""
import glob, os, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr, wilcoxon
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from controle_lib import lvcorr, faixa_dinamica, bootstrap_ic

DIR = os.path.dirname(os.path.abspath(__file__))
arqs = sorted(glob.glob(os.path.join(DIR, "rb_*.parquet")))
if not arqs: print("nada"); raise SystemExit(1)
d = pd.concat([pd.read_parquet(a) for a in arqs], ignore_index=True)
print(f"linhas {len(d)} | modelos {sorted(d.modelo.unique())} | cenas {d.cena.nunique()}")

def bloco(sub, col_lv, rotulo):
    rho, dr, casa = [], [], []
    for cena, g in sub.groupby("cena"):
        g = g.sort_values("alpha")
        a = g["alpha"].to_numpy(float); v = g[col_lv].to_numpy(float)
        rho.append(lvcorr(a, v)); dr.append(faixa_dinamica(a, v))
        alv = g[col_lv.replace("gerada", "alvo")].to_numpy(float) if "gerada" in col_lv else None
        if alv is not None and np.isfinite(v).all() and np.isfinite(alv).all() and np.std(alv) > 0:
            casa.append(spearmanr(v, alv).statistic)
    rho = np.array([x for x in rho if np.isfinite(x)])
    dr = np.array([x for x in dr if np.isfinite(x)])
    ic = bootstrap_ic(rho)
    s = (f"  {rotulo:<34} LVCorr media={rho.mean():+.4f} mediana={np.median(rho):+.4f} "
         f"IC95=[{ic[0]:+.4f},{ic[1]:+.4f}]  DR mediana={np.median(dr):.3f}  n={len(rho)}")
    if casa:
        c = np.array(casa); s += f"\n  {'':34} concordancia com o alvo real (spearman LV_gerada x LV_alvo): media={c.mean():+.4f}"
    print(s)
    return {"rotulo": rotulo, "lvcorr": float(rho.mean()), "lvcorr_mediana": float(np.median(rho)),
            "ic_lo": ic[0], "ic_hi": ic[1], "dr_mediana": float(np.median(dr)), "n": len(rho),
            "concord_alvo": float(np.mean(casa)) if casa else np.nan}

print("\nCONVENCAO: LVCorr = -spearman(alpha, LV). +1 = mais bokeh comandado -> menos nitidez.")
print("alpha = F_min/F: vem so da Eq. 3 (dentro da cena K ~ 1/F). Sem parametro livre.\n")
regs = []
m0 = sorted(d.modelo.unique())[0]
sub0 = d[d.modelo == m0]
print("### TETO ALCANCAVEL — as FOTOS REAIS medidas pela mesma metrica")
for suf in ("total", "fundo", "foco"):
    regs.append(bloco(sub0, f"lv_alvo_{suf}", f"foto real [{suf}]"))
for mod in sorted(d.modelo.unique()):
    if "lv_gerada_total" not in d.columns: break
    sub = d[(d.modelo == mod) & d["lv_gerada_total"].notna()]
    if not len(sub): continue
    print(f"\n### {mod}")
    for suf in ("total", "fundo", "foco"):
        regs.append(bloco(sub, f"lv_gerada_{suf}", f"{mod} [{suf}]"))

pd.DataFrame(regs).to_parquet(os.path.join(DIR, "resumo_realbokeh.parquet"), index=False)
print("\nsalvo resumo_realbokeh.parquet")
