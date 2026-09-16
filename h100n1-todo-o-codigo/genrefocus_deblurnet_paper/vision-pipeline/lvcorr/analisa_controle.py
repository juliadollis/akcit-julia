#!/usr/bin/env python3
"""Analise da avaliacao de controlabilidade: LVCorr, faixa dinamica, testes.

Le os parquets de escalares gravados por sweep_controle.py e produz:
  - por modelo: LVCorr (total/fundo/foco), faixa dinamica, monotonicidade;
  - o PISO DE RUIDO do proprio modelo (modo `nulo`: alpha fixo, seeds variando);
  - o teste que importa: o sinal do sweep esta ACIMA do piso de ruido?
  - a comparacao PAREADA original x kfix nas mesmas imagens (Wilcoxon).
"""
from __future__ import annotations
import glob, json, os, sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, mannwhitneyu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from controle_lib import lvcorr, lvcorr_pearson_log, faixa_dinamica, frac_monotona, bootstrap_ic

DIR = os.path.dirname(os.path.abspath(__file__))


def por_imagem(d: pd.DataFrame, coluna_eixo: str) -> pd.DataFrame:
    out = []
    for (mod, modo, fn), g in d.groupby(["modelo", "modo", "file_name_base"]):
        g = g.sort_values(coluna_eixo)
        x = g[coluna_eixo].to_numpy(float)
        r = {"modelo": mod, "modo": modo, "file": fn}
        for suf in ("total", "fundo", "foco"):
            v = g[f"lv_{suf}"].to_numpy(float)
            r[f"rho_{suf}"] = lvcorr(x, v)
            r[f"dr_{suf}"] = faixa_dinamica(x, v)
            r[f"mono_{suf}"] = frac_monotona(x, v)
        r["rho_pearsonlog_total"] = lvcorr_pearson_log(x, g["lv_total"].to_numpy(float))
        r["seletividade"] = (r["dr_fundo"] / r["dr_foco"]) if (
            np.isfinite(r["dr_foco"]) and r["dr_foco"] > 0) else np.nan
        out.append(r)
    return pd.DataFrame(out)


def fmt(v, n=4):
    return "  nan  " if (v is None or not np.isfinite(v)) else f"{v:+.{n}f}"


def main():
    arqs = sorted(glob.glob(os.path.join(DIR, "lv_*.parquet")))
    arqs = [a for a in arqs if "SMOKE" not in a]
    if not arqs:
        print("nenhum lv_*.parquet"); return 1
    d = pd.concat([pd.read_parquet(a) for a in arqs], ignore_index=True)
    print(f"linhas: {len(d)}  modelos: {sorted(d.modelo.unique())}  modos: {sorted(d.modo.unique())}")

    sw = por_imagem(d[d.modo == "sweep"], "alpha")
    nu = por_imagem(d[d.modo == "nulo"], "idx_ponto") if (d.modo == "nulo").any() else pd.DataFrame()

    linhas = []
    print("\n" + "=" * 100)
    print("LVCorr = -spearman(alpha, LV).  +1 = obedece perfeitamente, 0 = indiferente.")
    print("faixa dinamica DR = LV(alpha=0)/LV(alpha=1).  1.0 = o comando nao mudou nada.")
    print("=" * 100)
    for mod in sorted(sw.modelo.unique()):
        a = sw[sw.modelo == mod]
        b = nu[nu.modelo == mod] if len(nu) else pd.DataFrame()
        print(f"\n### {mod}   (n={len(a)} imagens no sweep, {len(b)} no controle nulo)")
        reg = {"modelo": mod, "n_sweep": len(a), "n_nulo": len(b)}
        for suf in ("total", "fundo", "foco"):
            v = a[f"rho_{suf}"].dropna().to_numpy()
            ic = bootstrap_ic(v)
            dr = a[f"dr_{suf}"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
            print(f"  LVCorr {suf:<5} media={fmt(v.mean())} mediana={fmt(np.median(v))} "
                  f"sd={np.std(v):.4f} IC95=[{fmt(ic[0])},{fmt(ic[1])}]   "
                  f"DR mediana={np.median(dr):.3f}")
            reg[f"lvcorr_{suf}"] = float(v.mean())
            reg[f"lvcorr_{suf}_mediana"] = float(np.median(v))
            reg[f"lvcorr_{suf}_ic_lo"] = ic[0]; reg[f"lvcorr_{suf}_ic_hi"] = ic[1]
            reg[f"dr_{suf}_mediana"] = float(np.median(dr))
        # com 9 pontos, |rho| >= 0.683 e significativo a p<0.05 (bicaudal).
        for suf in ("total", "fundo"):
            v = a[f"rho_{suf}"].dropna().to_numpy()
            fr = float((v >= 0.683).mean()); fr_neg = float((v <= -0.683).mean())
            print(f"  imagens com obediencia significativa ({suf}, rho>=+0.683): {fr*100:.1f}%"
                  f"   | com resposta INVERTIDA (rho<=-0.683): {fr_neg*100:.1f}%")
            reg[f"frac_signif_{suf}"] = fr; reg[f"frac_invertida_{suf}"] = fr_neg
        sel = a["seletividade"].replace([np.inf, -np.inf], np.nan).dropna()
        reg["seletividade_mediana"] = float(np.median(sel)) if len(sel) else np.nan
        reg["mono_fundo_mediana"] = float(a["mono_fundo"].dropna().median())
        print(f"  seletividade (DR_fundo/DR_foco) mediana={reg['seletividade_mediana']:.3f}"
              f"   monotonicidade do fundo mediana={reg['mono_fundo_mediana']:.3f}")

        if len(b):
            for suf in ("total", "fundo"):
                vs = a[f"rho_{suf}"].dropna().to_numpy()
                vn = b[f"rho_{suf}"].dropna().to_numpy()
                icn = bootstrap_ic(vn)
                try:
                    u, p = mannwhitneyu(vs, vn, alternative="greater")
                except Exception:
                    p = np.nan
                drn = b[f"dr_{suf}"].replace([np.inf, -np.inf], np.nan).dropna()
                print(f"  [piso de ruido {suf}] LVCorr={fmt(vn.mean())} "
                      f"IC95=[{fmt(icn[0])},{fmt(icn[1])}] DR mediana={np.median(drn):.4f}"
                      f"   -> sweep > ruido? p={p:.2e}")
                reg[f"ruido_lvcorr_{suf}"] = float(vn.mean())
                reg[f"ruido_dr_{suf}_mediana"] = float(np.median(drn))
                reg[f"p_sweep_vs_ruido_{suf}"] = float(p)
        linhas.append(reg)

    # comparacao PAREADA entre modelos, nas mesmas imagens
    mods = sorted(sw.modelo.unique())
    print("\n" + "=" * 100)
    print("COMPARACAO PAREADA (mesmas imagens, mesmo mapa, mesma seed) — Wilcoxon")
    print("=" * 100)
    pares = []
    for i in range(len(mods)):
        for j in range(i + 1, len(mods)):
            A, B = mods[i], mods[j]
            for suf in ("fundo", "total"):
                m = sw[sw.modelo == A][["file", f"rho_{suf}", f"dr_{suf}"]].merge(
                    sw[sw.modelo == B][["file", f"rho_{suf}", f"dr_{suf}"]],
                    on="file", suffixes=("_A", "_B")).dropna()
                if len(m) < 6: continue
                dA, dB = m[f"rho_{suf}_A"], m[f"rho_{suf}_B"]
                try: _, p = wilcoxon(dA, dB)
                except Exception: p = np.nan
                drA = np.median(m[f"dr_{suf}_A"]); drB = np.median(m[f"dr_{suf}_B"])
                print(f"  {A} vs {B}  [{suf}]  n={len(m)}  "
                      f"LVCorr {dA.mean():+.4f} vs {dB.mean():+.4f}  (dif={dA.mean()-dB.mean():+.4f}, p={p:.4f})  "
                      f"| DR {drA:.3f} vs {drB:.3f}")
                pares.append({"A": A, "B": B, "metrica": suf, "n": len(m),
                              "lvcorr_A": float(dA.mean()), "lvcorr_B": float(dB.mean()),
                              "dif": float(dA.mean() - dB.mean()), "p_wilcoxon": float(p),
                              "dr_A": float(drA), "dr_B": float(drB)})

    pd.DataFrame(linhas).to_parquet(os.path.join(DIR, "resumo_modelos.parquet"), index=False)
    if pares: pd.DataFrame(pares).to_parquet(os.path.join(DIR, "resumo_pares.parquet"), index=False)
    sw.to_parquet(os.path.join(DIR, "por_imagem_sweep.parquet"), index=False)
    if len(nu): nu.to_parquet(os.path.join(DIR, "por_imagem_nulo.parquet"), index=False)
    print("\nsalvo: resumo_modelos.parquet, resumo_pares.parquet, por_imagem_*.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
