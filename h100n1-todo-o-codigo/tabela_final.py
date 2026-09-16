"""Tabela final da campanha RealBokeh-test, com a identidade como referencia."""
import os
import numpy as np, pandas as pd
from datasets import load_dataset
tok=os.environ["HF_TOKEN"]
d=load_dataset("juliadollis/bokeh-eval-rb-metricas", split="train", token=tok, download_mode="force_redownload")
df=d.to_pandas().drop_duplicates(subset=["Model","Dataset"], keep="last")
pd.set_option("display.width",240)
print("="*100); print("METRICAS (40 cenas do split TEST do RealBokeh, 512 px, plano de foco pela Eq. 4)")
print(df.sort_values("LPIPS").to_string(index=False))

base=df[df.Model.str.contains("IDENTIDADE")]
if len(base):
    b=base.iloc[0]
    print("\nGANHO SOBRE A LINHA DE IDENTIDADE (LPIPS %.4f, menor e melhor):" % b.LPIPS)
    for _,r in df.sort_values("LPIPS").iterrows():
        if "IDENTIDADE" in r.Model: continue
        print("  %-32s LPIPS %.4f  -> %+.1f%%  %s" % (r.Model, r.LPIPS,
              100*(r.LPIPS-b.LPIPS)/b.LPIPS, "MELHOR que nao fazer nada" if r.LPIPS<b.LPIPS else "PIOR que nao fazer nada"))

print("\n"+"="*100); print("DISTRIBUICAO DO K ESCOLHIDO PELA BUSCA BINARIA (checa saturacao nos limites 1 e 100)")
for nome,r in [("oficial","juliadollis/bokeh-eval-rb-oficial"),("sem-treino","juliadollis/bokeh-eval-rb-semtreino"),
               ("nosso","juliadollis/bokeh-eval-rb-nosso"),("oficial-foco-centro","juliadollis/bokeh-eval-rb-oficial-centro")]:
    try:
        ds=load_dataset(r, split="validation", token=tok, download_mode="force_redownload")
        k=np.array(ds["best_k_value"],dtype=float); s=np.array(ds["ssim_score"],dtype=float)
        print("  %-22s n=%3d  best_k: min=%5.1f p25=%5.1f med=%5.1f p75=%5.1f max=%5.1f | no piso=%d no teto=%d | ssim_busca med=%.4f"
              % (nome, len(k), k.min(), np.percentile(k,25), np.median(k), np.percentile(k,75), k.max(),
                 int((k<=1).sum()), int((k>=100).sum()), float(np.median(s))))
    except Exception as e: print("  %-22s indisponivel (%s)" % (nome, type(e).__name__))
