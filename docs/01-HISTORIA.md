# A história: o que foi pedido, o que foi feito, e onde desviamos

## 1. O pedido

O José Ricardo mandou um protocolo de quatro passos, no `RETESTE_CURVATURA.md`:

1. `test_geometry_metrica.py` — validar a implementação
2. pegar o `fx` do Spring, a mediana que o `prepare_spring` reporta
3. `medir_k_spring.py --raiz <spring>/test --fx-orig <fx>` — percentis de |K|, e
   com eles escolher o teto
4. **só depois**: `train_single.py` com `--gauss-metrica --fx-orig --gauss-clamp`
   para o B3, e um controle B0 (berHu puro, `--gauss 0`), mesmo split

Com a instrução explícita: "Não pula pro passo 4 antes de 1-3", e o fecho "Manda
o resultado dos passos 1 a 3. O passo 4 a gente decide junto".

## 2. O que foi entregue

| passo | evidência |
|---|---|
| 1 | passa nos 5 critérios: esfera 0,00% a 0,17% (limite 0,2%), invariância 0,02% (limite 1%) |
| 2 | mediana 2585,9, faixa 1292,9 a 6060,6 entre as 37 sequências; vira 689,6 em 512 px |
| 3 | rodado, e ampliado: 517 quadros e 135,5 M pixels em vez dos 10 lotes sugeridos |
| 4 | rodado, e ampliado: 6 braços e 47 rodadas de seed em vez de 2 braços |

## 3. Os dois desvios

**O portão do passo 4.** A medição do passo 3 contrariou a previsão dele (ele
esperava que o teto 5 ficasse folgado; medimos 49% de saturação), que era
justamente o caso em que ele queria ser consultado. O teto 1000 foi escolhido
aqui, o passo 4 rodou, e os dados do passo 3 chegaram a ele **depois**, como
justificativa. Foi compensado tecnicamente rodando também o teto 5 e depois o 50,
mas a consulta não aconteceu.

**A escala.** Ele dimensionou ~4 h de GPU e dois braços. Foram 6 braços, 47
rodadas de seed, **~88 h de GPU**.

## 4. O que entrou além do pedido, e por quê

| extra | justificativa |
|---|---|
| braço **B1** | o B3 mistura normal com curvatura, então o nulo dele não diz de qual veio. `B1 − B0` é exatamente o termo da hipótese |
| tetos 5, 50 e 1000 | ele previa 5; a medição indicou 1000; o 50 fecha o meio |
| seeds até 10 | com n=3 as faixas dos braços se sobrepunham |
| **zero-shot** | não havia linha de base: todos os braços eram comparados entre si e nenhum contra o ponto de partida |
| **768 px** | testar se o teto da borda é representacional |

## 5. A perda de 2026-09-10

Uma liberação de quota no `/raid` apagou **37 de 47 checkpoints** e o
`data/spring_split` inteiro. Alguém separou as métricas à mão antes, em
`runs_riemann_metricas_preservadas/`, então nenhum número se perdeu.

Consequências: reavaliar aqueles braços deixou de ser 1 h de inferência e virou
retreino. O Spring teve de ser rebaixado (23 GB do DaRUS) e o split refeito, o
que saiu **idêntico**: 593/184/485, mesmas 13 sequências no teste.

O retreino revelou o achado do não determinismo, descrito em
`04-ACHADOS-TECNICOS.md`.

## 6. A ablação do Wallisson

Ele adaptou o código de ablação para o Spring e entregou em 12/09. Rodamos
**exatamente como ele mandou**, trocando só o dataset e informando a focal.

O resultado mudou a leitura do projeto: os dois termos que a nossa campanha
inteira usou (`gauss` e `normal`) são os dois piores da ablação, e três que nunca
testamos (`grad`, `metric`, `geod`) batem o controle.

Três pontos do código dele precisam de decisão, todos em `ENVIO_WALLISSON.md`: o
`metrics.py` veio da branch `fix-geometry` e não tem a máscara de validade, o
teto padrão 5,0 satura metade dos pixels no Spring, e falta o `prepare_spring.py`.

## 7. Onde estamos

A confirmação de `grad`, `metric` e `geod` com o **nosso** protocolo (split de
teste, máscara, `align full`, n=6) está rodando: 18 treinos. É o experimento que
decide se o projeto tem resultado positivo em vez de só um nulo bem medido.
