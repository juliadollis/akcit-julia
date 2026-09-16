# Cada braco contra o controle `B0_berhu`, pareado por seed

Gerado por `scripts/contraste_controle.py`. So entram as seeds que os DOIS lados tem; o `n` de cada linha e esse conjunto comum.

`delta` negativo em AbsRel/RMSE e positivo em F-score/delta1 favorece o braco. `seeds a favor` conta em quantas seeds do conjunto comum o braco bateu o controle.


## F-score de borda (maior melhor)

| braco | n comum | seeds | delta medio | desvio | amplitude do delta | seeds a favor |
|---|---|---|---|---|---|---|
| B1_gauss_metrica_teto1000 | 3 | 3,4,5 | -0.0607 | 0.0117 | -0.0713 a -0.0482 | 0/3 |
| B1_gauss_metrica_teto5 | 2 | 4,5 | -0.0337 | 0.0151 | -0.0444 a -0.0230 | 0/2 |
| B3_gauss_metrica_teto1000 | 5 | 3,4,5,8,9 | -0.0270 | 0.0163 | -0.0435 a -0.0016 | 0/5 |
| B3_gauss_metrica_teto5 | 3 | 3,4,5 | -0.0155 | 0.0155 | -0.0291 a +0.0013 | 1/3 |

## AbsRel (menor melhor)

| braco | n comum | seeds | delta medio | desvio | amplitude do delta | seeds a favor |
|---|---|---|---|---|---|---|
| B1_gauss_metrica_teto1000 | 3 | 3,4,5 | +0.0228 | 0.0206 | -0.0010 a +0.0349 | 1/3 |
| B1_gauss_metrica_teto5 | 2 | 4,5 | +0.0274 | 0.0542 | -0.0109 a +0.0658 | 1/2 |
| B3_gauss_metrica_teto1000 | 5 | 3,4,5,8,9 | +0.0480 | 0.0070 | +0.0402 a +0.0589 | 0/5 |
| B3_gauss_metrica_teto5 | 3 | 3,4,5 | +0.0226 | 0.0256 | +0.0030 a +0.0515 | 0/3 |

## delta1 (maior melhor)

| braco | n comum | seeds | delta medio | desvio | amplitude do delta | seeds a favor |
|---|---|---|---|---|---|---|
| B1_gauss_metrica_teto1000 | 3 | 3,4,5 | -0.0018 | 0.0214 | -0.0236 a +0.0192 | 1/3 |
| B1_gauss_metrica_teto5 | 2 | 4,5 | -0.0119 | 0.0227 | -0.0279 a +0.0042 | 1/2 |
| B3_gauss_metrica_teto1000 | 5 | 3,4,5,8,9 | -0.0099 | 0.0082 | -0.0211 a +0.0007 | 1/5 |
| B3_gauss_metrica_teto5 | 3 | 3,4,5 | +0.0082 | 0.0224 | -0.0123 a +0.0321 | 2/3 |

## RMSE (menor melhor)

| braco | n comum | seeds | delta medio | desvio | amplitude do delta | seeds a favor |
|---|---|---|---|---|---|---|
| B1_gauss_metrica_teto1000 | 3 | 3,4,5 | +0.3732 | 0.2984 | +0.0409 a +0.6185 | 0/3 |
| B1_gauss_metrica_teto5 | 2 | 4,5 | +0.2935 | 0.4527 | -0.0265 a +0.6136 | 1/2 |
| B3_gauss_metrica_teto1000 | 5 | 3,4,5,8,9 | +0.3509 | 0.1158 | +0.2675 a +0.5550 | 0/5 |
| B3_gauss_metrica_teto5 | 3 | 3,4,5 | +0.1546 | 0.2921 | -0.0146 a +0.4919 | 2/3 |
