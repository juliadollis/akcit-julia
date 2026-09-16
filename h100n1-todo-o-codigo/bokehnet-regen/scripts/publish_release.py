#!/usr/bin/env python3
"""Valida um release e publica no Hugging Face.

    # 1. conferir, sem subir nada (é o default)
    python3 scripts/publish_release.py --release-dir output/c_realbokeh

    # 2. subir, depois de ler o relatório
    python3 scripts/publish_release.py --release-dir output/c_realbokeh \\
        --repo-id akcit-pixel/bokehnet-regen-c --yes

## A regra desta ferramenta

**Valida sempre, sobe só com `--yes`.** O upload é irreversível na prática — um
dataset publicado com rótulo errado vira o dataset que alguém treina em cima seis meses
depois, exatamente o que aconteceu com o release anterior. Então a checagem roda
primeiro, imprime, e só então pergunta.

Repositório **privado por default**, e **nunca sobrescreve** um repo que já tem
arquivos: `--allow-existing` é obrigatório para isso, e mesmo assim a ferramenta lista
o que já está lá antes.

## O que a validação exige

Um release publicável tem que responder três perguntas sem depender de ninguém:

1. **O que cada amostra afirma** — `meta/<id>.json` passa em `validate_metadata`, e
   `control_version` é o mesmo em todas.
2. **De onde vieram os pixels** — toda amostra da rota C tem linha no
   `source_images.jsonl` com sha256 da AIF e da bokeh. Sem isso o release é rótulo
   solto apontando para um espelho privado, e ninguém consegue provar que o K foi
   calibrado contra aqueles bytes.
3. **Onde está a fronteira do split** — `split.json` existe, cobre todas as cenas, e
   nenhuma cena aparece dos dois lados (`check_no_leak`, comparando o split gravado em
   cada linha do manifesto contra o materializado).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataio.sample import validate_metadata                      # noqa: E402
from dataio.split import SceneSplit, check_no_leak               # noqa: E402


class Problema(Exception):
    """Impede a publicação. Sempre com o número que a torna verificável."""


# --------------------------------------------------------------------------------
# Validação
# --------------------------------------------------------------------------------

def _linhas(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def valida(release: Path) -> dict:
    problemas: list[str] = []
    avisos: list[str] = []

    manifesto = _linhas(release / "manifest.jsonl")
    if not manifesto:
        raise Problema(f"{release}/manifest.jsonl vazio ou ausente — não há release.")

    split_path = release / "split.json"
    if not split_path.is_file():
        raise Problema(
            f"{split_path} ausente. Split materializado é requisito do release: deixado "
            "para o config do treino, ele diverge entre runs e some no rsync.")
    split = SceneSplit.load(split_path)

    ids = [linha["sample_id"] for linha in manifesto]
    repetidos = [i for i, n in Counter(ids).items() if n > 1]
    if repetidos:
        problemas.append(f"{len(repetidos)} sample_id repetidos no manifesto "
                         f"(ex.: {repetidos[:3]}) — a contagem publicada está inflada.")

    # -- arquivos por amostra --------------------------------------------------
    faltando: dict[str, list[str]] = {}
    for linha in manifesto:
        sid = linha["sample_id"]
        for sub, ext in (("depth", ".png"), ("mask", ".png"), ("meta", ".json")):
            if not (release / sub / f"{sid}{ext}").is_file():
                faltando.setdefault(sub, []).append(sid)
    for sub, quais in faltando.items():
        problemas.append(f"{len(quais)} amostras sem arquivo em {sub}/ "
                         f"(ex.: {quais[:3]})")

    # -- metadados -------------------------------------------------------------
    versoes, backends, invalidos = Counter(), Counter(), []
    # De onde saiu a região que definiu `D_focus` em cada amostra, e quantas passaram
    # pelo refinamento. Vai para o resumo e para o card: quem baixa o release precisa
    # poder montar o treino COM e SEM as amostras refinadas, e para isso precisa saber
    # quantas são antes de baixar o release inteiro.
    #
    # `refinadas` conta `focus_was_refined`, e não `focus_source`, porque é por esse
    # booleano que o card manda filtrar. `validate_metadata` já reprova metadado em que
    # os dois discordem, então contar pelo campo do filtro não perde nada e não deixa o
    # número do card divergir do subconjunto que o filtro devolve.
    fontes_de_foco: Counter = Counter()
    refinadas = censuradas = sem_validador = 0
    for linha in manifesto:
        meta_path = release / "meta" / f"{linha['sample_id']}.json"
        if not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        try:
            validate_metadata(meta)
        except Exception as exc:
            invalidos.append(f"{linha['sample_id']}: {exc}")
        versoes[meta.get("control_version")] += 1
        backends[meta.get("depth_backend")] += 1
        fontes_de_foco[meta.get("focus_source", "AUSENTE")] += 1
        refinadas += bool(meta.get("focus_was_refined"))
        censuradas += bool(meta.get("is_k_censored"))
        sem_validador += meta.get("k_analytic") is None
    if invalidos:
        problemas.append(f"{len(invalidos)} metadados inválidos "
                         f"(ex.: {invalidos[0]})")
    if len(versoes) > 1:
        problemas.append(
            f"control_version divergente no mesmo release: {dict(versoes)}. Foi assim "
            "que duas convenções de normalização entraram no mesmo batch.")
    if len(backends) > 1:
        problemas.append(f"depth_backend divergente: {dict(backends)} — profundidade "
                         "métrica e disparidade normalizada não são a mesma coisa.")

    # -- proveniência dos pixels ------------------------------------------------
    ledger = {l["sample_id"]: l for l in _linhas(release / "source_images.jsonl")}
    rotas_c = [l for l in manifesto if l.get("route") == "c"]
    sem_ledger = [l["sample_id"] for l in rotas_c if l["sample_id"] not in ledger]
    if sem_ledger:
        problemas.append(
            f"{len(sem_ledger)} amostras da rota C sem linha em source_images.jsonl "
            f"(ex.: {sem_ledger[:3]}). Sem o sha256 dos bytes de origem, o release não "
            "prova contra o que o K foi calibrado.")

    autocontido = (release / "source").is_dir() and any((release / "source").iterdir())
    if not autocontido and rotas_c:
        avisos.append(
            f"release NÃO autocontido: as {len(rotas_c)} amostras da rota C apontam "
            "para o espelho de origem. O join é verificável por sha256, mas quem baixar "
            "precisa de acesso ao espelho. Para embutir os pixels, gere com "
            "--store-source-images.")

    # -- split ------------------------------------------------------------------
    cenas_manifesto = {l["scene_id"] for l in manifesto}
    fora = cenas_manifesto - set(split.assignment)
    if fora:
        problemas.append(f"{len(fora)} cenas no manifesto e fora do split.json "
                         f"(ex.: {sorted(fora)[:3]})")
    vazamento = check_no_leak(manifesto, split)
    if not vazamento.clean:
        problemas.append(f"vazamento de split:{vazamento.summary()}")

    if refinadas / max(len(manifesto), 1) > 0.5:
        avisos.append(
            f"{refinadas} de {len(manifesto)} amostras tiveram a região em "
            f"foco REFINADA ({dict(fontes_de_foco)}). O refinamento é o substituto automático "
            "do passo manual do §3.2(c) e está marcado por amostra — mas um lote em que "
            "ele domina precisa do laudo de scripts/validate_focus_refinement.py antes "
            "de virar treino.")

    contagens_split = Counter(l["split"] for l in manifesto)
    if contagens_split.get("val", 0) == 0:
        avisos.append("nenhuma amostra em `val`. Validação vazia é o jeito mais "
                      "silencioso de não ter validação.")

    if censuradas:
        pct = 100 * censuradas / len(manifesto)
        (problemas if pct > 20 else avisos).append(
            f"{censuradas} amostras ({pct:.1f}%) com K censurado no teto da busca. "
            "No release anterior eram 47,0% — acima de 20% o teto está errado, não os "
            "dados.")

    if problemas:
        raise Problema("\n  - ".join(["release reprovado:"] + problemas))

    return {
        "amostras": len(manifesto),
        "cenas": len(cenas_manifesto),
        "por_split": dict(contagens_split),
        "por_rota": dict(Counter(l.get("route") for l in manifesto)),
        "control_version": next(iter(versoes)),
        "depth_backend": next(iter(backends)),
        "k_censurado": censuradas,
        "sem_validador_analitico": sem_validador,
        "focus_sources": dict(fontes_de_foco),
        "refined": refinadas,
        "autocontido": autocontido,
        "avisos": avisos,
    }


# --------------------------------------------------------------------------------
# Card
# --------------------------------------------------------------------------------

def _mil(valor: int) -> str:
    """15423 -> '15.423'. O card é em português; separador trocado engana o leitor."""
    return f"{valor:,}".replace(",", ".")


def _pt(valor: float, casas: int = 1) -> str:
    """4.2 -> '4,2'. Para casar com os números de prosa do card, todos em pt-BR."""
    return f"{valor:.{casas}f}".replace(".", ",")


def escreve_card(release: Path, resumo: dict, repo_id: str) -> Path:
    """O README que vira a página do dataset. Diz o que o dado É, e o que ele NÃO é."""
    cfg_path = release / "run_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}
    prov = cfg.get("provenance_base", {})
    fonte = prov.get("source_dataset", "(não registrada)")
    # Fora da f-string de propósito: `{{}}` dentro de um campo de substituição não é
    # um literal escapado, é o `set` `{ {} }` — e um dict não é hasheável.
    renderer = prov.get("renderer") or {}
    # O bloco do renderizador grava `renderer_commit`; `commit` é só o nome curto que
    # algum run antigo usou. Ler apenas um dos dois imprimia `?` no card de um release
    # cujo commit estava gravado.
    renderer_commit = str(renderer.get("renderer_commit")
                          or renderer.get("commit") or "?")[:12]

    pixels = ("Os pixels estão neste repositório, em `source/`, nos **bytes originais** "
              "da origem — não recomprimidos, então o sha256 do ledger bate com o "
              "arquivo ao lado."
              if resumo["autocontido"] else
              f"Os pixels **não** estão aqui: cada amostra aponta para `{fonte}` pelo "
              "`source_sample_id`, e `source_images.jsonl` traz o sha256 da AIF e da "
              "bokeh para que o join seja verificável byte a byte.")

    n = resumo["amostras"]
    refinadas = resumo["refined"]
    pct_ref = _pt(100 * refinadas / max(n, 1))
    pct_cens = _pt(100 * resumo["k_censurado"] / max(n, 1))

    # Composição da região em foco, contada do release e não escrita à mão. É o que
    # permite decidir, antes de baixar, quanto do lote depende do refinamento.
    composicao = "\n".join(
        f"| `{fonte}` | {_mil(q)} | {_pt(100 * q / max(n, 1))}% |"
        for fonte, q in sorted(resumo["focus_sources"].items(), key=lambda kv: -kv[1]))

    splits = " · ".join(f"{k} {_mil(v)}" for k, v in sorted(resumo["por_split"].items()))
    treino = resumo["por_split"].get("train", 0)

    # O layout publicado é parte do contrato com quem baixa, então ele é descrito a
    # partir do que a publicação vai de fato fazer, nunca fixo no texto.
    por_cena = _agrupa_por_cena(release)
    cena_no_caminho = "<scene_id>/" if por_cena else ""
    agrupamento = ("""
Os arquivos por amostra ficam **dentro da pasta da cena** — `depth/<scene_id>/<id>.png`.
Não é decoração: o Hub recusa o push de qualquer diretório com mais de 10.000 arquivos,
e são {n} amostras. A cena foi o agrupamento escolhido porque `scene_id` já está em
toda linha do `manifest.jsonl` e em todo `meta/`, então montar o caminho é ler um campo:

```python
f"depth/{{linha['scene_id']}}/{{linha['sample_id']}}.png"
```

O nome do arquivo e o conteúdo são os mesmos do release em disco; muda só o nível de
diretório.
""".format(n=_mil(n)) if por_cena else "")

    card = f"""---
license: other
pretty_name: BokehNet regen — rota C
tags: [bokeh, depth-of-field, defocus, image-to-image]
---

# {repo_id}

Dados de treino da BokehNet regerados segundo o **GenRefocus** (arXiv:2512.16923v3),
§3.2(c) — a **rota C**: pares reais (all-in-focus, bokeh), com o sinal de controle
calibrado pelo sweep da Eq. 5.

Este não é um repack do dataset anterior. É uma regeração do zero, motivada por defeitos
medidos no pipeline original — cada um deles está descrito abaixo com o número que o
comprova, porque é isso que permite decidir se este dado é confiável.

## Números

| | |
|---|---|
| amostras | {_mil(n)} |
| **cenas** | {_mil(resumo['cenas'])} |
| amostras por split (a fronteira é POR CENA) | {splits} |
| K censurado no teto da busca | {_mil(resumo['k_censurado'])} (**{pct_cens}%**) |
| região em foco refinada | {_mil(refinadas)} (**{pct_ref}%**) |
| sem validador analítico | {_mil(resumo['sem_validador_analitico'])} |

A contagem em **cenas** aparece junto com a de amostras de propósito: a origem entrega de
2 a 21 aberturas da mesma cena, então número de amostras não é número de unidades
independentes — e é a cena que define o split.

## O que mudou em relação ao dataset anterior

**A imagem de referência agora é all-in-focus de verdade.** Antes, a "AIF" era escolhida
como o maior f-stop dentro da pasta dos *alvos*: mediana **f/14**, e em 12,7% das cenas
f/5.6 ou mais aberto. Ou seja, numa cena a cada oito, a profundidade, a máscara e o sweep
de K saíam de uma foto que já tinha bokeh forte. A AIF verdadeira sempre existiu, em
`train/in/<id>_f22.JPG` — f/22 em 100% das cenas, confirmada byte a byte por sha256.

**O K deixou de bater no teto.** A busca antiga usava um limite escolhido a priori e
produzia `k == 300` exato em **47,0%** das amostras: metade do dataset tinha como rótulo o
teto da busca, não o K da cena. Aqui são **{pct_cens}%**, e o teto vem medido de um
piloto, não de intuição.

**O normalizador parou de variar.** `max_coc` era uma flag por rota; um normalizador que
muda entre amostras é um segundo rótulo escondido dentro do primeiro. Agora é **100,0
fixo e global**, e a validação do release recusa qualquer amostra que diga outra coisa.

**Acabaram as cascatas que mentiam na proveniência.** O pipeline antigo caía de BiRefNet
para RMBG para GrabCut com `except` nu, e continuava gravando `mask_source="automatic"`;
e de Depth Pro para Depth Anything, que devolve **disparidade** e não profundidade
métrica — a amostra saía espelhada sem deixar rastro. Aqui existe **um** modelo por papel;
se ele falha, a amostra é rejeitada com motivo registrado.

**O split passou a ser por cena.** Com split por imagem, a mesma cena aparecia nos dois
lados e a validação media memorização.

## O contrato do sinal de controle

`control_version = {resumo['control_version']}`. Toda amostra segue exatamente isto:

```python
z          = depth_pro(aif)              # METROS
disp       = 1.0 / z                     # 1/m
focus_disp = median(disp[mask])          # mediana NA disparidade, não 1/median(z)
K          = k_eq3 / 1000.0              # a Eq. 3 é em mm; a inferência é em 1/m
defocus    = clip(abs(K * (disp - focus_disp)) / 100.0, 0.0, 1.0)
```

A profundidade é guardada como **disparidade uint16**, lado longo 768, com os extremos
medidos na resolução cheia — `disparity_min`/`disparity_max` no metadado desfazem a
quantização.

## O refinamento da região em foco — leia antes de treinar

{pct_ref}% das amostras tiveram a região em foco **refinada**, e isso precisa de
explicação porque é um desvio declarado do paper.

Antes de qualquer coisa, o que está no disco: **`mask/<id>.png` é a REGIÃO FINAL de foco**
— a que produziu `focus_disparity` —, **não** a saída crua do BiRefNet. Quando a amostra
foi refinada, o arquivo é a região refinada, e `mask_source` diz isso; a validação do
release reprova metadado em que os dois discordem, justamente para ninguém auditar o
rótulo olhando a máscara errada.

Composição medida **neste** release:

| `focus_source` | amostras | |
|---|---|---|
{composicao}

O §3.2(c) usa o BiRefNet para obter a máscara de foco, e reconhece que ela é *"sometimes
unreliable"* nestes datasets. A solução dele é **refinamento manual**, e ele diz
explicitamente que **não descarta** os casos ruins (`paper.txt:359-368`).

Medimos o tamanho do problema num piloto de 162 amostras, sem refino: a máscara do
BiRefNet acerta o plano de foco em apenas **35,2%** das amostras, comparada contra a
distância de foco que a RealBokeh publica **medida a ±1 cm**. A razão mediana é 0,579 — o
plano escolhido fica ~1,7x mais longe do que estava. E **20,6%** dos pares eram
descartados por máscara vazia.

A causa está nas imagens: a RealBokeh é feita de **cenas**, não de fotos de objeto — um
tronco de árvore num parque, um muro de pedra. O BiRefNet segmenta objeto **saliente**, e
fora desse domínio ele devolve probabilidade exatamente 0 em cerca de um terço dos casos.

Substituímos o passo manual por um automático que usa a definição física de estar em
foco: temos a AIF **e** a bokeh da mesma cena, e no plano de foco a bokeh **preservou** o
detalhe da AIF.

```
retencao(x) = media_local(|laplaciano(bokeh)|) / media_local(|laplaciano(aif)|)
```

O denominador normaliza pela textura da própria cena — é o que faz medida de nitidez
absoluta falhar, porque folhagem desfocada tem mais alta frequência que parede lisa em
foco.

**Cada amostra carrega `focus_source` e `focus_was_refined`**, no metadado e no manifesto,
mais `focus_disparity_from_initial_mask`: o rótulo que ela teria tido **sem** refino. Isso
permite treinar com e sem as refinadas e medir a diferença, em vez de acreditar.

Medido no bloco pareado do piloto com refino (302 amostras) — a mesma amostra com e sem
refino, nas 88 refinadas que tinham linha de base: dentro de ±25% do gabarito, **19,3%
antes contra 23,9% depois**, com 59 amostras melhorando e 29 piorando. Nesse mesmo piloto
o descarte por máscara vazia caiu de 20,6% para **zero**.

Os números deste parágrafo e do próximo são **do piloto**, não deste lote: a validação
pareada não foi refeita sobre as {_mil(n)} amostras. Os contadores da tabela acima, sim,
são contados deste release.

**E o achado que muda a prioridade:** **33,1%** das amostras têm o gabarito **fora da
faixa de disparidade da própria cena**. Nenhuma máscara chegaria lá — nem a perfeita. Esse
erro é da **profundidade**, não da escolha de região. Restringindo às alcançáveis, a
concordância sobe de 32,1% para **45,0%**.

## O teto de 4 aberturas por cena

O suplemento descreve séries *"containing 2 to 4 images per set"* (`paper.txt:1001-1003`)
e um total de **13K** imagens novas. Com teto de 4, a enumeração do split `train` dá
**13.799** amostras — e a anotação manual que o paper relata (8 horas a 4-8 s por imagem)
dá 3.600 a 7.200 máscaras, contra as 4.399 cenas da RealBokeh. Duas contas independentes
fecham. Este release entrega **{_mil(treino)}** no `train`: a diferença são as amostras
rejeitadas no run, e cada uma está em `rejections.jsonl` com o motivo.

Sem teto seriam 20.495, 58% acima do publicado, e **244 cenas (6,2%) gerariam 25% das
amostras**. O teto não descarta cena nenhuma — só tira o peso excessivo de umas poucas. Os
níveis mantidos são **espaçados uniformemente** pela faixa de aberturas, preservando a
amplitude de K dentro da cena, que é o sinal que a rede precisa aprender.

## Arquivos

```
depth/{cena_no_caminho}<id>.png
    disparidade uint16, lado longo 768
mask/{cena_no_caminho}<id>.png
    a REGIÃO FINAL de foco — a que definiu `focus_disparity`, já refinada quando
    `focus_was_refined` é true
meta/{cena_no_caminho}<id>.json
    K, focus_disparity, gates medidos, proveniência completa
manifest.jsonl
    uma linha por amostra, para varredura rápida
split.json
    split POR CENA, materializado no release
source_images.jsonl
    sha256 dos bytes de origem, shard e linha
rejections.jsonl
    UM par processado por linha — `status` `ok` ou `rejected`, e o `reason` quando
    recusado. Não é só a lista de recusas: é o desfecho de tudo que entrou, que é o
    que permite calcular a taxa sem regerar
```
{agrupamento}
{pixels}

## O que este dataset NÃO é

- **Não é revisado por humano.** O paper filtra com inspeção manual; aqui isso foi
  substituído por gates automáticos, todos **medidos e gravados** em cada `meta/`.
- **Os gates estão em modo medir.** Nenhum limiar bloqueia: este é o lote **bruto**, com
  toda métrica registrada. O dataset de treino sai daqui por **filtro do manifesto**, sem
  reprocessar, porque o `sample_id` é determinístico. Escolher limiar antes de ver a
  distribuição foi o que produziu os 47% de K censurado no release anterior.
- **`rejections.jsonl` faz parte do release.** O histograma de motivos é o que permite
  recalibrar sem regerar, e é ele que denuncia fallback novo.
- **Amostra com `is_valid_for_control == false` não deve entrar no treino de controle.**
  Ela fica aqui porque escondê-la esconderia a taxa.

## Proveniência

| | |
|---|---|
| pipeline | `{prov.get('pipeline_commit', '?')}` |
| profundidade | `{prov.get('depth_backend', '?')}` · `{str(prov.get('depth_model_sha256', '?'))[:16]}…` |
| máscara | `{prov.get('mask_backend', '?')}` · `{str(prov.get('mask_model_sha256', '?'))[:16]}…` |
| renderer | BokehMe `{renderer_commit}` |
| fator efetivo de K medido | `{prov.get('k_effective_factor', '?')}` |
| origem do split | `{prov.get('split_origin', '?')}` |

O renderizador foi **verificado em GPU** antes de gerar rótulo: raio linear em K, resíduo
relativo do ajuste ~0. Sem esse laudo o pipeline recusa rodar — foi assim que se descobriu,
no pipeline antigo, um fallback gaussiano que ninguém via porque nada media o borrão.

## Limitações conhecidas

- **A escala métrica do Depth Pro é o gargalo atual**, não a máscara. Ver a seção do
  refinamento: um terço dos gabaritos é inalcançável por qualquer máscara.
- **Sem validador analítico.** A largura do sensor da RealBokeh não é publicada, então a
  Eq. 3 não fecha e não há segundo cálculo independente para conferir o K do sweep.
- **O limiar de SSIM da Eq. 5 é `[A]`.** O paper afirma aplicá-lo duas vezes e não publica
  o valor.
- **421 pares do `train` da origem (2,05%)** são anotados por ela como `misaligned` ou
  `shift_<X>px`, em 228 cenas. O campo viaja no metadado e ninguém filtra por ele; o
  efeito do desalinhamento na calibração não foi medido.

## Licença

Os rótulos são deste projeto. **Os pixels seguem a licença da origem (`{fonte}`)** —
verifique-a antes de redistribuir.
"""
    destino = release / "README.md"
    destino.write_text(card, encoding="utf-8")
    return destino


# --------------------------------------------------------------------------------
# Upload
# --------------------------------------------------------------------------------

#: Não é release: o HF cria no `create_repo`, e o `.cache/huggingface` é o diário de
#: bordo que o `upload_large_folder` usa para retomar. Contar qualquer um deles como
#: "o repo já tem arquivos" trancava a retentativa depois de um upload interrompido —
#: justamente quando retomar é o que se quer.
_ANDAIME = frozenset({".gitattributes"})

IGNORAR = ["*.tmp", "__pycache__/*", "_mirror_index.json", ".cache/huggingface/*"]

#: Acima disto o commit único não passa. Os dois limites do Hub são opostos, e um
#: release desta rota fica exatamente entre eles — os dois foram batidos de verdade
#: publicando este release, de 46.274 arquivos e 7,6 GB:
#:
#: 1. **um commit com tudo → 413 Payload Too Large.** Os bytes subiram inteiros (8 GB
#:    pelo Xet) e o commit foi recusado no fim, o que é o pior desfecho possível.
#: 2. **um commit por punhado de arquivos → 429.** O `upload_large_folder` fatia em
#:    lotes de ~75 e precisaria de ~600 commits; o Hub permite **256 commits/hora** e o
#:    corta no meio, e a cada 429 ele reduz o lote, o que piora a conta.
#:
#: Então a fatia certa é grossa mas não única: `LOTE_POR_COMMIT` arquivos por commit dá
#: ~10 commits para este release — longe do 413 e longe do 429. Release pequeno
#: continua indo em **um** commit, porque aí ele é atômico: aparece inteiro ou não
#: aparece.
LIMITE_COMMIT_UNICO_ARQUIVOS = 8_000
LIMITE_COMMIT_UNICO_BYTES = 2 * 1024**3
#: 2.000 dá ~24 commits para este release: uma ordem de grandeza abaixo do 429, e cada
#: commit ainda termina dentro de `TIMEOUT_REQUEST_S`. Com 5.000 o servidor levava mais
#: que o timeout para processar o lote.
LOTE_POR_COMMIT = 2_000


def _arquivos(release: Path) -> list[Path]:
    """O que vai subir. O `.cache/huggingface` de uma tentativa anterior fica fora: ele
    não é release, e inflaria o número impresso antes do upload."""
    return sorted(p for p in release.rglob("*")
                  if p.is_file()
                  and ".cache" not in p.relative_to(release).parts
                  and "__pycache__" not in p.relative_to(release).parts
                  and p.suffix != ".tmp"
                  and p.name != "_mirror_index.json")


def _tamanho(release: Path) -> tuple[int, int]:
    arquivos = _arquivos(release)
    return len(arquivos), sum(p.stat().st_size for p in arquivos)


#: Pastas por amostra: no disco são planas, no repositório precisam de um nível a mais.
_POR_AMOSTRA = ("depth", "mask", "meta")

#: O Hub recusa o push com **400 Bad Request** quando um diretório passa de 10.000
#: arquivos — medido publicando este release, em que `depth/`, `mask/` e `meta/` têm
#: 15.423 cada. Não é limite de commit, é do repositório: nenhum fatiamento resolve, o
#: layout é que não cabe.
MAX_ARQUIVOS_POR_PASTA = 10_000


def _agrupa_por_cena(release: Path) -> bool:
    """Se este release precisa ser reagrupado para caber no Hub.

    Só reagrupa quem não cabe: manter o layout plano onde ele cabe evita mudar o
    contrato de caminho de um release pequeno sem motivo nenhum.
    """
    return any(sum(1 for _ in (release / sub).iterdir()) > MAX_ARQUIVOS_POR_PASTA
               for sub in _POR_AMOSTRA if (release / sub).is_dir())


def _cenas_do_manifesto(release: Path) -> dict[str, str]:
    return {l["sample_id"]: l["scene_id"] for l in _linhas(release / "manifest.jsonl")}


def _caminho_no_repo(relativo: Path, cenas: dict[str, str]) -> str:
    """Onde o arquivo vai morar no repositório.

    Agrupa por **cena** (`depth/<scene_id>/<sample_id>.png`) em vez de deixar os 15.423
    arquivos soltos em `depth/`. A cena é a escolha certa entre os agrupamentos
    possíveis por três motivos: é a unidade que o projeto já usa para o split e para as
    contagens; está gravada em toda linha do manifesto e em todo `meta/`, então a regra
    de recuperar o caminho é ler um campo e não rodar um hash; e uma cena tem de 2 a 21
    amostras, então nenhuma pasta chega perto do limite.

    O nome do arquivo não muda, e o conteúdo não muda. Só entra um nível de diretório.
    """
    partes = relativo.parts
    if len(partes) == 2 and partes[0] in _POR_AMOSTRA:
        cena = cenas.get(Path(partes[1]).stem)
        if cena:
            return f"{partes[0]}/{cena}/{partes[1]}"
    return relativo.as_posix()


#: O 429 do Hub é por hora, então a espera útil é da ordem de minutos, não de segundos.
ESPERA_429_S = (300, 600, 900, 1800, 1800)

#: O default do `huggingface_hub` é **10 s** (`constants.DEFAULT_REQUEST_TIMEOUT`), e o
#: POST de commit de milhares de arquivos leva minutos do lado do servidor: com 10 s ele
#: sempre estoura em `httpx.ReadTimeout`, e o release fica pela metade. O cliente httpx
#: é criado na primeira chamada e lê a constante ali, então elevá-la antes do primeiro
#: request é o único jeito de alcançá-lo — não há env var para isto nesta versão.
TIMEOUT_REQUEST_S = 900


def _retentavel(exc: Exception) -> str:
    """Motivo pelo qual vale tentar o mesmo commit de novo, ou "" se não vale.

    Os dois casos são ambíguos-mas-idempotentes: repetir um commit com **os mesmos
    arquivos e os mesmos caminhos** não muda o repositório, então repetir é mais seguro
    que abortar no meio.
    """
    texto = f"{type(exc).__name__}: {exc}"
    if "429" in texto or "Too Many Requests" in texto:
        return "429 (limite de commits/hora)"
    if "Timeout" in texto or "timed out" in texto:
        return "timeout no POST do commit"
    return ""


def _commita_esperando_o_limite(api, repo_id: str, *, operations, mensagem: str) -> None:
    """Um commit, com espera nos erros que o Hub devolve sob carga.

    Sem isto, bater no limite de 256 commits/hora — ou no timeout de um commit grande —
    abortava o upload no meio e deixava o repositório com metade do release, que é pior
    que não publicar, porque parece publicado. Espera e tenta de novo; desiste alto,
    nunca em silêncio.
    """
    for tentativa, espera in enumerate((*ESPERA_429_S, None), 1):
        try:
            api.create_commit(repo_id=repo_id, repo_type="dataset",
                              operations=operations, commit_message=mensagem)
            return
        except Exception as exc:
            motivo = _retentavel(exc)
            if not motivo:
                raise
            if espera is None:
                raise Problema(
                    f"{motivo} ainda depois de {tentativa} tentativas. O release pode "
                    "estar PELA METADE no repositório: rode de novo mais tarde com "
                    "--allow-existing para completar.") from exc
            print(f"[hf] {motivo}. Esperando {espera // 60} min e tentando de novo "
                  f"(tentativa {tentativa}) …", flush=True)
            time.sleep(espera)


def publica(release: Path, repo_id: str, *, private: bool, allow_existing: bool) -> None:
    import huggingface_hub
    from huggingface_hub import HfApi

    # Antes de qualquer request: ver TIMEOUT_REQUEST_S.
    constantes = getattr(huggingface_hub, "constants", None)
    if constantes is not None and getattr(
            constantes, "DEFAULT_REQUEST_TIMEOUT", 0) < TIMEOUT_REQUEST_S:
        constantes.DEFAULT_REQUEST_TIMEOUT = TIMEOUT_REQUEST_S

    api = HfApi()                       # token vem do login/env; nunca impresso
    existe = True
    try:
        arquivos = api.list_repo_files(repo_id, repo_type="dataset")
    except Exception:
        existe, arquivos = False, []

    conteudo = [f for f in arquivos
                if f not in _ANDAIME and not f.startswith(".cache/")]
    if existe and conteudo and not allow_existing:
        raise Problema(
            f"{repo_id} já tem {len(conteudo)} arquivos de release "
            f"(ex.: {conteudo[:3]}).\n"
            "Publicar por cima sobrescreveria um release existente. Prefira um destino "
            "novo; se a intenção é mesmo atualizar este, passe --allow-existing.")

    if not existe:
        api.create_repo(repo_id, repo_type="dataset", private=private)
        print(f"[hf] repositório criado: {repo_id} (private={private})")

    n_arquivos, n_bytes = _tamanho(release)
    grande = (n_arquivos > LIMITE_COMMIT_UNICO_ARQUIVOS
              or n_bytes > LIMITE_COMMIT_UNICO_BYTES)
    print(f"[hf] enviando {release} → {repo_id} "
          f"({n_arquivos} arquivos, {n_bytes / 1024**3:.2f} GB, "
          f"{'vários commits' if grande else 'commit único'}) …")

    if grande:
        from huggingface_hub import CommitOperationAdd, CommitOperationDelete

        reagrupou = _agrupa_por_cena(release)
        cenas = _cenas_do_manifesto(release) if reagrupou else {}
        destinos = [(_caminho_no_repo(p.relative_to(release), cenas), p)
                    for p in _arquivos(release)]

        # Restos de uma tentativa anterior no layout plano ficariam no repositório para
        # sempre, e quem clonasse veria cada amostra duas vezes.
        planos = sorted({f.split("/")[0] for f in conteudo
                         if f.split("/")[0] in _POR_AMOSTRA and f.count("/") == 1})
        if reagrupou and planos:
            print(f"[hf] apagando {planos} do layout plano de uma tentativa anterior …",
                  flush=True)
            _commita_esperando_o_limite(
                api, repo_id,
                operations=[CommitOperationDelete(path_in_repo=d, is_folder=True)
                            for d in planos],
                mensagem="remove o layout plano que o Hub recusa (10k arquivos/pasta)")

        lotes = [destinos[i:i + LOTE_POR_COMMIT]
                 for i in range(0, len(destinos), LOTE_POR_COMMIT)]
        for i, lote in enumerate(lotes, 1):
            print(f"[hf] commit {i}/{len(lotes)} ({len(lote)} arquivos) …", flush=True)
            _commita_esperando_o_limite(
                api, repo_id,
                operations=[CommitOperationAdd(path_in_repo=destino,
                                               path_or_fileobj=str(p))
                            for destino, p in lote],
                mensagem=("release da rota C, contrato "
                          f"metric_disparity_official_v1 ({i}/{len(lotes)})"))
    else:
        api.upload_folder(
            folder_path=str(release), repo_id=repo_id, repo_type="dataset",
            commit_message="release da rota C, contrato metric_disparity_official_v1",
            ignore_patterns=IGNORAR)
    print(f"[hf] pronto: https://huggingface.co/datasets/{repo_id}")


def main() -> int:
    p = argparse.ArgumentParser(description="Valida um release e publica no HF.")
    p.add_argument("--release-dir", required=True)
    p.add_argument("--repo-id", default="", help="ex.: akcit-pixel/bokehnet-regen-c")
    p.add_argument("--yes", action="store_true",
                   help="sobe de verdade. Sem isto, só valida e escreve o card.")
    p.add_argument("--public", action="store_true", help="default é privado")
    p.add_argument("--allow-existing", action="store_true")
    args = p.parse_args()

    release = Path(args.release_dir)
    try:
        resumo = valida(release)
    except Problema as exc:
        print(f"\n{exc}\n")
        return 1

    print(f"\n{'=' * 66}\n  release: {release}")
    for chave in ("amostras", "cenas", "por_split", "por_rota", "control_version",
                  "depth_backend", "k_censurado", "sem_validador_analitico",
                  "focus_sources", "refined", "autocontido"):
        print(f"    {chave:<26} {resumo[chave]}")
    for aviso in resumo["avisos"]:
        print(f"  [aviso] {aviso}")
    print("=" * 66)

    if args.repo_id:
        card = escreve_card(release, resumo, args.repo_id)
        print(f"\n  card escrito: {card}")

    if not args.yes:
        print("\n  Nada foi enviado (falta --yes). Leia o card e o resumo acima.\n")
        return 0
    if not args.repo_id:
        print("\n  --yes sem --repo-id: para onde?\n")
        return 1

    try:
        publica(release, args.repo_id, private=not args.public,
                allow_existing=args.allow_existing)
    except Problema as exc:
        print(f"\n{exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
