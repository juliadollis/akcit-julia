# Reteste da curvatura — resultados

Fonte: `resultados/reteste_curvatura_2026-08-31` (JSONs gravados pelo `train_single.py`).

## Estado das rodadas

| experimento | seeds concluidas | seeds em andamento |
|---|---|---|
| `B0_berhu` | 8 | seed_6 |
| `B1_gauss_metrica_teto1000` | 6 | seed_6, seed_8 |
| `B1_gauss_metrica_teto5` | 5 | seed_3, seed_6, seed_8 |
| `B3_gauss_metrica_teto1000` | 10 | — |
| `B3_gauss_metrica_teto5` | 8 | seed_8 |

## F-score de borda (maior e melhor)

| experimento | n | media | min | max | por seed |
|---|---|---|---|---|---|
| `B0_berhu` | 8 | **0.5943** | 0.5895 | 0.5996 | 0.5986, 0.5895, 0.5952, 0.5996, 0.5933, 0.5964, 0.5902, 0.5915 |
| `B1_gauss_metrica_teto1000` | 6 | **0.5339** | 0.5229 | 0.5491 | 0.5491, 0.5229, 0.5242, 0.5283, 0.5307, 0.5483 |
| `B1_gauss_metrica_teto5` | 5 | **0.5632** | 0.5521 | 0.5784 | 0.5784, 0.5542, 0.5610, 0.5703, 0.5521 |
| `B3_gauss_metrica_teto1000` | 10 | **0.5711** | 0.5468 | 0.5999 | 0.5908, 0.5614, 0.5712, 0.5741, 0.5543, 0.5708, 0.5999, 0.5516, 0.5468, 0.5899 |
| `B3_gauss_metrica_teto5` | 8 | **0.5746** | 0.5419 | 0.6009 | 0.5641, 0.5419, 0.5903, 0.6009, 0.5746, 0.5673, 0.5712, 0.5864 |

## AbsRel (menor e melhor)

| experimento | n | media | min | max | por seed |
|---|---|---|---|---|---|
| `B0_berhu` | 8 | **0.2508** | 0.2460 | 0.2562 | 0.2460, 0.2562, 0.2497, 0.2493, 0.2533, 0.2509, 0.2544, 0.2466 |
| `B1_gauss_metrica_teto1000` | 6 | **0.2758** | 0.2483 | 0.3173 | 0.2589, 0.3173, 0.2570, 0.2483, 0.2877, 0.2858 |
| `B1_gauss_metrica_teto5` | 5 | **0.2608** | 0.2424 | 0.3167 | 0.2424, 0.2588, 0.2440, 0.2424, 0.3167 |
| `B3_gauss_metrica_teto1000` | 10 | **0.2933** | 0.2609 | 0.3098 | 0.2609, 0.2969, 0.2993, 0.2895, 0.3027, 0.3098, 0.2782, 0.3028, 0.2983, 0.2944 |
| `B3_gauss_metrica_teto5` | 8 | **0.2775** | 0.2523 | 0.3024 | 0.2870, 0.2775, 0.2773, 0.2523, 0.2665, 0.3024, 0.2894, 0.2674 |

## delta1 (maior e melhor)

| experimento | n | media | min | max | por seed |
|---|---|---|---|---|---|
| `B0_berhu` | 8 | **0.6940** | 0.6857 | 0.6987 | 0.6981, 0.6975, 0.6909, 0.6913, 0.6937, 0.6960, 0.6857, 0.6987 |
| `B1_gauss_metrica_teto1000` | 6 | **0.6901** | 0.6556 | 0.7105 | 0.7049, 0.6556, 0.7044, 0.7105, 0.6926, 0.6724 |
| `B1_gauss_metrica_teto5` | 5 | **0.6957** | 0.6680 | 0.7061 | 0.7023, 0.7044, 0.7061, 0.6979, 0.6680 |
| `B3_gauss_metrica_teto1000` | 10 | **0.6870** | 0.6749 | 0.7303 | 0.7303, 0.6797, 0.6832, 0.6858, 0.6823, 0.6749, 0.6803, 0.6807, 0.6864, 0.6864 |
| `B3_gauss_metrica_teto5` | 8 | **0.6942** | 0.6820 | 0.7234 | 0.6860, 0.6923, 0.6820, 0.7234, 0.6986, 0.6836, 0.6858, 0.7017 |

## RMSE (menor e melhor)

| experimento | n | media | min | max | por seed |
|---|---|---|---|---|---|
| `B0_berhu` | 8 | **4.2466** | 4.1972 | 4.2951 | 4.2073, 4.2536, 4.2218, 4.2848, 4.2951, 4.1972, 4.2933, 4.2194 |
| `B1_gauss_metrica_teto1000` | 6 | **4.6544** | 4.3257 | 5.3003 | 4.3884, 5.3003, 4.3412, 4.3257, 4.9136, 4.6572 |
| `B1_gauss_metrica_teto5` | 5 | **4.4096** | 4.2047 | 4.8108 | 4.2047, 4.3982, 4.3659, 4.2685, 4.8108 |
| `B3_gauss_metrica_teto1000` | 10 | **4.5602** | 4.2093 | 4.7522 | 4.2093, 4.5556, 4.5994, 4.5916, 4.5996, 4.7522, 4.6127, 4.5811, 4.5608, 4.5401 |
| `B3_gauss_metrica_teto5` | 8 | **4.4347** | 4.2702 | 4.6891 | 4.5345, 4.4556, 4.4331, 4.2702, 4.2815, 4.6891, 4.5242, 4.2894 |

## `B1_gauss_metrica_teto1000` contra o controle `B0_berhu`

> **Comparacao ainda NAO e valida:** B1_gauss_metrica_teto1000 tem 6 seed(s) e B0_berhu tem 8. Os numeros abaixo sao provisorios.

| metrica | B1 | B0 (controle) | diferenca | quem vence |
|---|---|---|---|---|
| F-score de borda | 0.5339 | 0.5943 | -0.0604 | **B0** |
| AbsRel | 0.2758 | 0.2508 | +0.0250 | **B0** |
| delta1 | 0.6901 | 0.6940 | -0.0039 | **B0** |
| RMSE | 4.6544 | 4.2466 | +0.4078 | **B0** |

## `B1_gauss_metrica_teto5` contra o controle `B0_berhu`

> **Comparacao ainda NAO e valida:** B1_gauss_metrica_teto5 tem 5 seed(s) e B0_berhu tem 8. Os numeros abaixo sao provisorios.

| metrica | B1 | B0 (controle) | diferenca | quem vence |
|---|---|---|---|---|
| F-score de borda | 0.5632 | 0.5943 | -0.0311 | **B0** |
| AbsRel | 0.2608 | 0.2508 | +0.0100 | **B0** |
| delta1 | 0.6957 | 0.6940 | +0.0018 | **B1** |
| RMSE | 4.4096 | 4.2466 | +0.1631 | **B0** |

## `B3_gauss_metrica_teto1000` contra o controle `B0_berhu`

> **Comparacao ainda NAO e valida:** B3_gauss_metrica_teto1000 tem 10 seed(s) e B0_berhu tem 8. Os numeros abaixo sao provisorios.

| metrica | B3 | B0 (controle) | diferenca | quem vence |
|---|---|---|---|---|
| F-score de borda | 0.5711 | 0.5943 | -0.0232 | **B0** |
| AbsRel | 0.2933 | 0.2508 | +0.0425 | **B0** |
| delta1 | 0.6870 | 0.6940 | -0.0070 | **B0** |
| RMSE | 4.5602 | 4.2466 | +0.3137 | **B0** |

## `B3_gauss_metrica_teto5` contra o controle `B0_berhu`

| metrica | B3 | B0 (controle) | diferenca | quem vence |
|---|---|---|---|---|
| F-score de borda | 0.5746 | 0.5943 | -0.0197 | **B0** |
| AbsRel | 0.2775 | 0.2508 | +0.0267 | **B0** |
| delta1 | 0.6942 | 0.6940 | +0.0002 | **B3** |
| RMSE | 4.4347 | 4.2466 | +0.1881 | **B0** |

Diferenca no F-score de borda: media -0.0197, desvio 0.0147 (n=8).

> Com 3 seeds nao ha poder estatistico para um teste pareado conclusivo. O que a tabela sustenta e a direcao, nao a significancia.

## Efeito do teto em B1: `B1_gauss_metrica_teto1000` contra `B1_gauss_metrica_teto5`

| metrica | B1_gauss_metrica_teto1000 | B1_gauss_metrica_teto5 | diferenca | quem vence |
|---|---|---|---|---|
| F-score de borda | 0.5339 | 0.5632 | +0.0293 | **B1_gauss_metrica_teto5** |
| AbsRel | 0.2758 | 0.2608 | -0.0150 | **B1_gauss_metrica_teto5** |
| delta1 | 0.6901 | 0.6957 | +0.0057 | **B1_gauss_metrica_teto5** |
| RMSE | 4.6544 | 4.4096 | -0.2448 | **B1_gauss_metrica_teto5** |

## Efeito do teto em B3: `B3_gauss_metrica_teto1000` contra `B3_gauss_metrica_teto5`

| metrica | B3_gauss_metrica_teto1000 | B3_gauss_metrica_teto5 | diferenca | quem vence |
|---|---|---|---|---|
| F-score de borda | 0.5711 | 0.5746 | +0.0035 | **B3_gauss_metrica_teto5** |
| AbsRel | 0.2933 | 0.2775 | -0.0158 | **B3_gauss_metrica_teto5** |
| delta1 | 0.6870 | 0.6942 | +0.0072 | **B3_gauss_metrica_teto5** |
| RMSE | 4.5602 | 4.4347 | -0.1255 | **B3_gauss_metrica_teto5** |

