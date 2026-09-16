# Reteste da curvatura — resultados

Fonte: `resultados/envio_teto5_2026-09-01` (JSONs gravados pelo `train_single.py`).

## Estado das rodadas

| experimento | seeds concluidas | seeds em andamento |
|---|---|---|
| `B0_berhu` | 3 | — |
| `B3_gauss_metrica_teto5` | 3 | — |

## F-score de borda (maior e melhor)

| experimento | n | media | min | max | por seed |
|---|---|---|---|---|---|
| `B0_berhu` | 3 | **0.5944** | 0.5895 | 0.5986 | 0.5986, 0.5895, 0.5952 |
| `B3_gauss_metrica_teto5` | 3 | **0.5654** | 0.5419 | 0.5903 | 0.5641, 0.5419, 0.5903 |

## AbsRel (menor e melhor)

| experimento | n | media | min | max | por seed |
|---|---|---|---|---|---|
| `B0_berhu` | 3 | **0.2507** | 0.2460 | 0.2562 | 0.2460, 0.2562, 0.2497 |
| `B3_gauss_metrica_teto5` | 3 | **0.2806** | 0.2773 | 0.2870 | 0.2870, 0.2775, 0.2773 |

## delta1 (maior e melhor)

| experimento | n | media | min | max | por seed |
|---|---|---|---|---|---|
| `B0_berhu` | 3 | **0.6955** | 0.6909 | 0.6981 | 0.6981, 0.6975, 0.6909 |
| `B3_gauss_metrica_teto5` | 3 | **0.6868** | 0.6820 | 0.6923 | 0.6860, 0.6923, 0.6820 |

## RMSE (menor e melhor)

| experimento | n | media | min | max | por seed |
|---|---|---|---|---|---|
| `B0_berhu` | 3 | **4.2276** | 4.2073 | 4.2536 | 4.2073, 4.2536, 4.2218 |
| `B3_gauss_metrica_teto5` | 3 | **4.4744** | 4.4331 | 4.5345 | 4.5345, 4.4556, 4.4331 |

## `B3_gauss_metrica_teto5` contra o controle `B0_berhu`

| metrica | B3 | B0 (controle) | diferenca | quem vence |
|---|---|---|---|---|
| F-score de borda | 0.5654 | 0.5944 | -0.0290 | **B0** |
| AbsRel | 0.2806 | 0.2507 | +0.0299 | **B0** |
| delta1 | 0.6868 | 0.6955 | -0.0087 | **B0** |
| RMSE | 4.4744 | 4.2276 | +0.2468 | **B0** |

Diferenca no F-score de borda: media -0.0290, desvio 0.0197 (n=3).

> Com 3 seeds nao ha poder estatistico para um teste pareado conclusivo. O que a tabela sustenta e a direcao, nao a significancia.

