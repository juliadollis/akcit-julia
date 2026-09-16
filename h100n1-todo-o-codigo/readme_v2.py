import os, io
from huggingface_hub import HfApi
tok=os.environ["HF_TOKEN"]
txt = """---
license: cc-by-nc-sa-4.0
---
# Benchmark de bokeh — RealBokeh_3MP split TEST, 1 par por cena, COM metadados

Substituto do **LF-Bokeh** do paper GenRefocus (arXiv 2512.16923), que os autores
nao publicaram (verificado: nada na org `nycu-cplab`, nada no HF).

## Origem e por que este dado

`akcit-pixel/RealBokeh`, split `test` — a conversao em parquet do
`timseizinger/RealBokeh_3MP` (dataset *Bokehlicious*, ICCV 2025). Sao pares
reais de bracket de abertura: a mesma cena fotografada com abertura fechada
(all-in-focus) e com abertura aberta (bokeh optico real).

**Nao ha contaminacao.** A rota c de treino (`AKCITPixel3/CMiQdveBBzNii`,
2.932 pares) vem 100% do split **train** do mesmo dataset — verificado lendo a
coluna `source_aif` das 2.932 linhas: 2.932/2.932 sob `RealBokeh_3MP/train/gt/`.
Nenhuma cena do split test entrou em nenhum treino nosso. Os IDs numericos
COLIDEM entre train e test (a numeracao reinicia por split), mas as cenas sao
outras.

## Selecao

1 linha por cena. Entre os niveis de bokeh disponiveis escolhemos o de menor
variancia do Laplaciano, que e o de abertura mais aberta. **Verificado por
conteudo de pixel** (correlacao com o JPG bruto `gt/<id>/<id>_f2.0.JPG`):
216 de 217 confirmam f/2.0; 1 ficou por inferencia. Uma linha por cena evita
pseudo-replicacao — os 5 niveis de uma cena sao a mesma foto.

Descartes: 1 cena so tinha alvos `misaligned`; 2 cenas tinham alvo NAO mais
borrado que a AIF. Restam **217 cenas**.

## Colunas

| coluna | o que e |
|---|---|
| `image_focus` | AIF: a foto com abertura fechada. Entrada do BokehNet. |
| `image_blur` | alvo: bokeh optico real, f/2.0. |
| `file_name_base`, `cena_id`, `nivel_bokeh` | identificacao |
| `alinhamento`, `desloc_px` | 212 `aligned`; 5 com deslocamento medido de 2,4 a 4,8 px. Filtre por `desloc_px == 0` se quiser so os perfeitos. |
| `lv_aif`, `lv_alvo` | variancia do Laplaciano de cada lado. Razao mediana 0,22: o alvo tem bokeh forte de verdade. |
| **`focus_plane_distance`** | **plano de foco ANOTADO, em metros.** E o que a Eq. 4 do paper apenas ESTIMA (mediana da profundidade na mascara do BiRefNet). |
| `focus_plane_uncertainty` | incerteza da anotacao, em metros |
| **`target_av`**, `source_av`, `target_avs` | aberturas. Com `focal_length` permitem calcular o K pela Eq. 3 analiticamente, em vez de so achar por busca binaria. |
| `focal_length` | distancia focal em mm |
| `iso`, `ev` | exposicao |

Os metadados vem de `timseizinger/RealBokeh_3MP` em `test/metadata/<id>.json` e
tinham se perdido na conversao em parquet; aqui foram reanexados por join no ID.

## Ressalva medida sobre a Eq. 4

Comparando a estimativa da Eq. 4 (BiRefNet + Depth Pro) com o
`focus_plane_distance` anotado, nas 40 primeiras cenas:

- Pearson entre disparidade estimada e anotada: **0,330**
- erro relativo de profundidade: mediana **47%**, dentro de 2x em **63%**
- em **5 de 40** o BiRefNet devolveu mascara VAZIA (cena sem objeto saliente)
- a faixa metrica do Depth Pro **nem contem** a profundidade anotada em 40% das
  cenas, ou seja, 40% do erro nenhuma mascara corrige

Ler o plano de foco na regiao nitida do ALVO (oraculo) melhora para Pearson
0,512 e 75% dentro de 2x, mas continua limitado pelo mesmo mapa de profundidade.

**Consequencia:** comparacoes ENTRE modelos neste benchmark sao validas (todos
recebem o mesmo condicionamento), mas numeros ABSOLUTOS carregam o erro do
plano de foco. Prefira condicionar por `focus_plane_distance` quando o objetivo
for medir a qualidade do bokeh em si.

## Privacidade

Repo **privado**: contem GT de dataset de terceiros com licenca nao comercial
(CC BY-NC-SA 4.0). Cite Seizinger et al., *Bokehlicious*, ICCV 2025.
"""
api=HfApi(token=tok)
api.upload_file(path_or_fileobj=io.BytesIO(txt.encode()), path_in_repo="README.md",
                repo_id="juliadollis/bokeh-bench-realbokeh-test-v2", repo_type="dataset")
print("README v2 enviado")
