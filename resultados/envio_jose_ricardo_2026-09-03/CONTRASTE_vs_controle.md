# Cada braco contra o controle `B0_berhu`, pareado por seed

Gerado por `scripts/contraste_controle.py`. So entram as seeds que os DOIS lados tem; o `n` de cada linha e esse conjunto comum.

`delta` negativo em AbsRel/RMSE e positivo em F-score/delta1 favorece o braco. `seeds a favor` conta em quantas seeds do conjunto comum o braco bateu o controle.


## F-score de borda (maior melhor)

| braco | n comum | seeds | delta medio | desvio | amplitude do delta | seeds a favor |
|---|---|---|---|---|---|---|
| B1_gauss_metrica_teto1000 | 6 | 0,1,2,3,4,5 | -0.0615 | 0.0104 | -0.0713 a -0.0482 | 0/6 |
| B1_gauss_metrica_teto5 | 5 | 0,1,2,4,5 | -0.0314 | 0.0098 | -0.0444 a -0.0202 | 0/5 |
| B3_gauss_metrica_teto1000 | 8 | 0,1,2,3,4,5,8,9 | -0.0244 | 0.0141 | -0.0435 a -0.0016 | 0/8 |
| B3_gauss_metrica_teto5 | 6 | 0,1,2,3,4,5 | -0.0222 | 0.0185 | -0.0475 a +0.0013 | 1/6 |

## AbsRel (menor melhor)

| braco | n comum | seeds | delta medio | desvio | amplitude do delta | seeds a favor |
|---|---|---|---|---|---|---|
| B1_gauss_metrica_teto1000 | 6 | 0,1,2,3,4,5 | +0.0249 | 0.0229 | -0.0010 a +0.0611 | 1/6 |
| B1_gauss_metrica_teto5 | 5 | 0,1,2,4,5 | +0.0096 | 0.0318 | -0.0109 a +0.0658 | 3/5 |
| B3_gauss_metrica_teto1000 | 8 | 0,1,2,3,4,5,8,9 | +0.0432 | 0.0129 | +0.0149 a +0.0589 | 0/8 |
| B3_gauss_metrica_teto5 | 6 | 0,1,2,3,4,5 | +0.0263 | 0.0178 | +0.0030 a +0.0515 | 0/6 |

## delta1 (maior melhor)

| braco | n comum | seeds | delta medio | desvio | amplitude do delta | seeds a favor |
|---|---|---|---|---|---|---|
| B1_gauss_metrica_teto1000 | 6 | 0,1,2,3,4,5 | -0.0045 | 0.0236 | -0.0419 a +0.0192 | 3/6 |
| B1_gauss_metrica_teto5 | 5 | 0,1,2,4,5 | +0.0005 | 0.0165 | -0.0279 a +0.0152 | 4/5 |
| B3_gauss_metrica_teto1000 | 8 | 0,1,2,3,4,5,8,9 | -0.0054 | 0.0166 | -0.0211 a +0.0321 | 2/8 |
| B3_gauss_metrica_teto5 | 6 | 0,1,2,3,4,5 | -0.0002 | 0.0171 | -0.0123 a +0.0321 | 2/6 |

## RMSE (menor melhor)

| braco | n comum | seeds | delta medio | desvio | amplitude do delta | seeds a favor |
|---|---|---|---|---|---|---|
| B1_gauss_metrica_teto1000 | 6 | 0,1,2,3,4,5 | +0.4111 | 0.3806 | +0.0409 a +1.0466 | 0/6 |
| B1_gauss_metrica_teto5 | 5 | 0,1,2,4,5 | +0.1746 | 0.2581 | -0.0265 a +0.6136 | 2/5 |
| B3_gauss_metrica_teto1000 | 8 | 0,1,2,3,4,5,8,9 | +0.3045 | 0.1518 | +0.0020 a +0.5550 | 0/8 |
| B3_gauss_metrica_teto5 | 6 | 0,1,2,3,4,5 | +0.2007 | 0.1966 | -0.0146 a +0.4919 | 2/6 |
