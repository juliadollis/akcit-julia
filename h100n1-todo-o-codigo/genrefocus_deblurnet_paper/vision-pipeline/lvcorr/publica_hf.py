#!/usr/bin/env python3
"""Sobe os resultados da avaliacao de controlabilidade para o HF e escreve os
READMEs (cards) de cada repo. Repos NOVOS; nada existente e sobrescrito."""
import glob, json, os
import pandas as pd
from huggingface_hub import HfApi

DIR = os.path.dirname(os.path.abspath(__file__))
TOK = os.environ["HF_TOKEN"]
api = HfApi()

R_SWEEP = "juliadollis/bokeh-controlabilidade-sweep"
R_LV = "juliadollis/bokeh-controlabilidade-lv"
R_RES = "juliadollis/bokeh-controlabilidade-resumo"

PROTOCOLO = """
## Protocolo (resumo)

Pergunta: o modelo OBEDECE ao comando de intensidade de bokeh?

- Eixo do sweep: `alpha`, a amplitude do mapa de defocus NORMALIZADO que entra
  no modelo, em 9 pontos de 0 a 1. `alpha` e o K do paper reescalado POR IMAGEM
  para que o mapa cubra [0,1]: com um K comum a todas as imagens, o mesmo K
  satura umas e deixa outras em zero (medido na DDPD: o K que leva o mapa a 1.0
  varia de 35 a 3273 entre imagens). Como a LVCorr e calculada por imagem e a
  correlacao de postos e invariante a reescala monotona, `alpha` nao muda a
  metrica: garante que ela seja medida na faixa util.
- Mapa: `B = |1/z - 1/z_foco|` do Depth Pro metrico, normalizado pelo proprio
  p99.5; condicao = `clip(alpha * B, 0, 1)`.
- Seed FIXA em todos os pontos: o unico fator que varia e `alpha`.
- CONVENCAO DE SINAL: a variancia do laplaciano mede NITIDEZ, entao obedecer ao
  comando significa nitidez CAINDO quando o comando SOBE.
  Definimos **LVCorr = -spearman(alpha, LV)**, de modo que **+1 = obediencia
  perfeita**. Sem a negacao, obediencia aparece como numero negativo.
- Metricas por imagem: LVCorr em tres regioes (imagem inteira, fundo `B>=0.5`,
  foco `B<=0.05`); faixa dinamica `DR = LV(alpha=0)/LV(alpha=1)` (1.0 = o
  comando nao fez nada); fracao de degraus monotonos decrescentes.
- CONTROLE NULO por modelo (`modo = nulo`): alpha CONSTANTE e 9 seeds
  diferentes, medido pelo mesmo codigo. E o piso de ruido: um LVCorr so conta
  como sinal se estiver acima dele.
"""


def card(titulo, corpo, privado=True):
    return f"---\nlicense: other\n---\n\n# {titulo}\n\n{corpo}\n{PROTOCOLO}\n"


def sobe_readme(repo, texto):
    p = "/tmp/README.md"
    open(p, "w").write(texto)
    api.upload_file(path_or_fileobj=p, path_in_repo="README.md", repo_id=repo,
                    repo_type="dataset", token=TOK)
    print(f"[hf] README -> {repo}")


def main():
    api.create_repo(R_RES, repo_type="dataset", private=True, exist_ok=True, token=TOK)
    for nome in ("resumo_modelos", "resumo_pares", "por_imagem_sweep", "por_imagem_nulo"):
        p = os.path.join(DIR, f"{nome}.parquet")
        if os.path.exists(p):
            api.upload_file(path_or_fileobj=p, path_in_repo=f"data/{nome}.parquet",
                            repo_id=R_RES, repo_type="dataset", token=TOK)
            print(f"[hf] {nome} -> {R_RES}")
    for nome in ("sanidade_resultados.json", "reanalise_atual.json"):
        p = os.path.join(DIR, nome)
        if os.path.exists(p):
            api.upload_file(path_or_fileobj=p, path_in_repo=f"sanidade/{nome}",
                            repo_id=R_RES, repo_type="dataset", token=TOK)
            print(f"[hf] {nome} -> {R_RES}")
    for nome in ("controle_lib.py", "sweep_controle.py", "analisa_controle.py",
                 "sanidade.py", "diag_mapa.py", "diag_treino.py", "reanalisa_atual.py"):
        p = os.path.join(DIR, nome)
        if os.path.exists(p):
            api.upload_file(path_or_fileobj=p, path_in_repo=f"codigo/{nome}",
                            repo_id=R_RES, repo_type="dataset", token=TOK)
    print("[hf] codigo enviado")

    sobe_readme(R_RES, card(
        "Controlabilidade de bokeh do BokehNet — resumo e testes de sanidade",
        "Tabelas agregadas da avaliacao de controlabilidade (LVCorr) comparando os\n"
        "checkpoints da fase 2. Contem: `data/resumo_modelos.parquet` (uma linha por\n"
        "modelo), `data/resumo_pares.parquet` (comparacoes pareadas com Wilcoxon),\n"
        "`data/por_imagem_*.parquet` (a medida bruta por imagem, sweep e controle\n"
        "nulo), `sanidade/` (os testes que validam a propria medicao) e `codigo/`\n"
        "(todo o codigo que gerou estes numeros).\n\n"
        "PRIVADO: derivado de datasets de terceiros."))
    sobe_readme(R_LV, card(
        "Controlabilidade de bokeh — escalares por imagem e por ponto do sweep",
        "Uma linha por (modelo, modo, imagem, ponto do sweep) com as variancias de\n"
        "laplaciano medidas e os parametros do mapa de condicionamento. Sem imagens.\n\n"
        "PRIVADO."))
    sobe_readme(R_SWEEP, card(
        "Controlabilidade de bokeh — imagens geradas no sweep",
        "Todas as saidas do sweep, uma linha por render, com a imagem PNG. Serve para\n"
        "inspecao visual da progressao de bokeh e para recalcular qualquer metrica.\n\n"
        "PRIVADO: as imagens derivam de `akcit-pixel/DDPD`."))
    print("pronto")


if __name__ == "__main__":
    main()
