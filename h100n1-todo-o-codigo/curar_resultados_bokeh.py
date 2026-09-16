#!/usr/bin/env python3
"""Publica uma tabela CURADA dos resultados do bokeh num repo HF novo.

Por que existe: `juliadollis/bokeh-eval-metricas` e um log de APPEND. Ele mistura
campanhas medidas com o pipeline ainda bugado (profundidade sem redimensionar,
plano de foco pelo pixel central, k-escala 0.01 matando o mapa) com as campanhas
corrigidas, e tem linhas repetidas de voltas do keeper. Nao da para publicar
aquilo, nem para ler sem saber a historia toda.

Esta tabela e o oposto: uma linha por (modelo, benchmark), so da campanha final,
com o n, a faixa de K, a linha de identidade de cada benchmark e a margem sobre
ela -- que e a unica comparacao valida ENTRE benchmarks, ja que os pisos sao
muito diferentes (identidade da LPIPS 0.2371 no LF-repro e 0.3587 no RealBokeh).

NAO sobrescreve nada: publica num repo novo.
"""
import os, re, sys
import pandas as pd
from datasets import Dataset, load_dataset

FONTE = "juliadollis/bokeh-eval-metricas"
DESTINO = sys.argv[1] if len(sys.argv) > 1 else "juliadollis/genrefocus-resultados-curados"
TOKEN = os.environ["HF_TOKEN"]

BENCH = {
    "RB": ("RealBokeh test v2", 217,
           "mesma distribuicao do treino -- mede qualidade, nao generalizacao"),
    "RD": ("RealDOF", None,
           "FORA do treino dos dois modelos, camera diferente, bokeh mais forte"),
    "LFREPRO": ("LF-Bokeh reproduzido (BLB)", 500,
                "sintetico; reproduz o protocolo do paper, nao a fonte optica"),
}
MODELO = {
    "rotac-only": "so rota c",
    "nosso": "nosso original (fase 2)",
    "kfix": "kfix (K da rota b recalculado pela Eq. 3)",
    "oficial": "oficial do paper",
    "oficial-paper": "oficial do paper",
    "fase1": "fase 1 (so sintetico)",
    "nofilter": "sem filtro de SSIM",
    "sem-treino": "sem treino (FLUX.1-dev cru)",
    "LINHA-DE-IDENTIDADE": "LINHA DE IDENTIDADE (devolve a entrada)",
}

def classifica(nome):
    bench = next((b for b in ("LFREPRO", "RD", "RB") if re.search(rf"[-_]{b}\b|{b}", nome)), None)
    chave = None
    for k in sorted(MODELO, key=len, reverse=True):
        if nome.startswith(k):
            chave = k; break
    if not chave:
        return bench, None, None
    rotulo = MODELO[chave]
    # O passo do checkpoint vem do proprio nome (ex.: rotac-only-45k-RD-full).
    # Sem isto, os pontos da curva de aprendizado colapsam todos no mesmo rotulo
    # -- foi o que aconteceu na primeira versao desta tabela.
    m = re.search(r"-(\d+)k-", nome)
    passo = f"{m.group(1)}000" if m else None
    if passo:
        rotulo = f"{rotulo} (step {int(passo):,})".replace(",", ".")
    elif chave in ("nosso", "kfix", "fase1", "nofilter"):
        rotulo = f"{rotulo} (checkpoint final)"
    return bench, rotulo, passo

df = pd.DataFrame(load_dataset(FONTE, split="train", download_mode="force_redownload", token=TOKEN))
# campanha final: so as linhas medidas com o pipeline corrigido, que sao as que
# carregam 'full' ou 'LFREPRO' no nome do modelo.
df = df[df.Model.str.contains("full|LFREPRO", case=False, na=False)].copy()
df = df[~df.Model.str.contains("FULLRES", case=False, na=False)]   # campanha antiga
df = df.drop_duplicates(subset=["Model", "Dataset"], keep="last")

linhas = []
for _, r in df.iterrows():
    bench, modelo, passo = classifica(r.Model)
    if not bench or not modelo:
        print(f"[pulado] nao classifiquei: {r.Model}"); continue
    nome_b, n, obs = BENCH[bench]
    linhas.append({
        "modelo": modelo, "benchmark": nome_b, "n": n,
        "step_do_checkpoint": passo,
        "SSIM": round(float(r.SSIM), 4), "LPIPS": round(float(r.LPIPS), 4),
        "DISTS": round(float(r.DISTS), 4), "CLIP_I": round(float(r["CLIP-I"]), 4),
        "LVCorr": round(float(r.LVCorr), 4),
        "faixa_K": "3 a 300 (k-escala 3.0)", "resolucao": "lado maior 512 px",
        "pipeline": "corrigido: profundidade redimensionada + plano de foco pela Eq. 4 (BiRefNet)",
        "observacao_benchmark": obs,
        "repo_metricas_bruto": r.Dataset,
        "id_interno": r.Model,
    })

t = pd.DataFrame(linhas)
# margem sobre a identidade, DENTRO de cada benchmark
base = {b: g.loc[g.modelo.str.startswith("LINHA DE IDENTIDADE"), "LPIPS"]
        for b, g in t.groupby("benchmark")}
t["margem_LPIPS_sobre_identidade"] = [
    round(float(base[r.benchmark].iloc[0]) - r.LPIPS, 4) if len(base.get(r.benchmark, [])) else None
    for _, r in t.iterrows()]
t = t.sort_values(["benchmark", "LPIPS"]).reset_index(drop=True)

print(t[["benchmark","modelo","n","SSIM","LPIPS","margem_LPIPS_sobre_identidade"]].to_string(index=False))
print(f"\n{len(t)} linhas -> {DESTINO}")
Dataset.from_pandas(t).push_to_hub(DESTINO, token=TOKEN, private=True)
print("publicado")
