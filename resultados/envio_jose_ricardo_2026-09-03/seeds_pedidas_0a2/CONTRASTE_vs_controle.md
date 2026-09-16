# Cada braco contra o controle `B0_berhu`, pareado por seed

Gerado por `scripts/contraste_controle.py`. So entram as seeds que os DOIS lados tem; o `n` de cada linha e esse conjunto comum.

`delta` negativo em AbsRel/RMSE e positivo em F-score/delta1 favorece o braco. `seeds a favor` conta em quantas seeds do conjunto comum o braco bateu o controle.


## F-score de borda (maior melhor)

| braco | n comum | seeds | delta medio | desvio | amplitude do delta | seeds a favor |
|---|---|---|---|---|---|---|
| B1_gauss_metrica_teto1000 | 3 | 0,1,2 | -0.0624 | 0.0114 | -0.0711 a -0.0494 | 0/3 |
| B1_gauss_metrica_teto5 | 3 | 0,1,2 | -0.0299 | 0.0084 | -0.0352 a -0.0202 | 0/3 |
| B3_gauss_metrica_teto1000 | 3 | 0,1,2 | -0.0200 | 0.0108 | -0.0281 a -0.0078 | 0/3 |
| B3_gauss_metrica_teto5 | 3 | 0,1,2 | -0.0290 | 0.0218 | -0.0475 a -0.0050 | 0/3 |

## AbsRel (menor melhor)

| braco | n comum | seeds | delta medio | desvio | amplitude do delta | seeds a favor |
|---|---|---|---|---|---|---|
| B1_gauss_metrica_teto1000 | 3 | 0,1,2 | +0.0271 | 0.0296 | +0.0073 a +0.0611 | 0/3 |
| B1_gauss_metrica_teto5 | 3 | 0,1,2 | -0.0023 | 0.0044 | -0.0058 a +0.0026 | 2/3 |
| B3_gauss_metrica_teto1000 | 3 | 0,1,2 | +0.0351 | 0.0180 | +0.0149 a +0.0496 | 0/3 |
| B3_gauss_metrica_teto5 | 3 | 0,1,2 | +0.0299 | 0.0101 | +0.0213 a +0.0410 | 0/3 |

## delta1 (maior melhor)

| braco | n comum | seeds | delta medio | desvio | amplitude do delta | seeds a favor |
|---|---|---|---|---|---|---|
| B1_gauss_metrica_teto1000 | 3 | 0,1,2 | -0.0072 | 0.0302 | -0.0419 a +0.0135 | 2/3 |
| B1_gauss_metrica_teto5 | 3 | 0,1,2 | +0.0088 | 0.0057 | +0.0042 a +0.0152 | 3/3 |
| B3_gauss_metrica_teto1000 | 3 | 0,1,2 | +0.0022 | 0.0264 | -0.0178 a +0.0321 | 1/3 |
| B3_gauss_metrica_teto5 | 3 | 0,1,2 | -0.0087 | 0.0035 | -0.0121 a -0.0052 | 0/3 |

## RMSE (menor melhor)

| braco | n comum | seeds | delta medio | desvio | amplitude do delta | seeds a favor |
|---|---|---|---|---|---|---|
| B1_gauss_metrica_teto1000 | 3 | 0,1,2 | +0.4491 | 0.5184 | +0.1194 a +1.0466 | 0/3 |
| B1_gauss_metrica_teto5 | 3 | 0,1,2 | +0.0953 | 0.0849 | -0.0027 a +0.1446 | 1/3 |
| B3_gauss_metrica_teto1000 | 3 | 0,1,2 | +0.2272 | 0.1987 | +0.0020 a +0.3776 | 0/3 |
| B3_gauss_metrica_teto5 | 3 | 0,1,2 | +0.2468 | 0.0698 | +0.2019 a +0.3272 | 0/3 |
