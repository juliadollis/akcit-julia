# Cada braco contra o controle `B0_berhu`, pareado por seed

Gerado por `scripts/contraste_controle.py`. So entram as seeds que os DOIS lados tem; o `n` de cada linha e esse conjunto comum.

`delta` negativo em AbsRel/RMSE e positivo em F-score/delta1 favorece o braco. `seeds a favor` conta em quantas seeds do conjunto comum o braco bateu o controle.


## F-score de borda (limiar fixo) (maior melhor)

| braco | n comum | seeds | delta medio | desvio | amplitude do delta | seeds a favor |
|---|---|---|---|---|---|---|
| B0_berhu_size768 | 3 | 0,1,2 | -0.0026 | 0.0094 | -0.0132 a +0.0048 | 2/3 |
| B1_gauss_metrica_teto1000 | 6 | 0,1,2,3,4,5 | -0.0615 | 0.0104 | -0.0713 a -0.0482 | 0/6 |
| B1_gauss_metrica_teto5 | 6 | 0,1,2,3,4,5 | -0.0351 | 0.0126 | -0.0533 a -0.0202 | 0/6 |
| B3_gauss_metrica_teto1000 | 6 | 0,1,2,3,4,5 | -0.0250 | 0.0101 | -0.0390 a -0.0078 | 0/6 |
| B3_gauss_metrica_teto5 | 6 | 0,1,2,3,4,5 | -0.0222 | 0.0185 | -0.0475 a +0.0013 | 1/6 |
| B3_gauss_metrica_teto50 | 6 | 0,1,2,3,4,5 | -0.0190 | 0.0263 | -0.0553 a +0.0095 | 2/6 |

## F-score de borda no melhor limiar (maior melhor)

| braco | n comum | seeds | delta medio | desvio | amplitude do delta | seeds a favor |
|---|---|---|---|---|---|---|
| B0_berhu_size768 | 3 | 0,1,2 | -0.0344 | 0.0077 | -0.0428 a -0.0275 | 0/3 |
| B1_gauss_metrica_teto1000 | 6 | 0,1,2,3,4,5 | -0.0561 | 0.0139 | -0.0758 a -0.0377 | 0/6 |
| B1_gauss_metrica_teto5 | 6 | 0,1,2,3,4,5 | -0.0145 | 0.0145 | -0.0354 a +0.0020 | 1/6 |
| B3_gauss_metrica_teto1000 | 6 | 0,1,2,3,4,5 | -0.0159 | 0.0024 | -0.0202 a -0.0137 | 0/6 |
| B3_gauss_metrica_teto5 | 6 | 0,1,2,3,4,5 | -0.0113 | 0.0204 | -0.0416 a +0.0089 | 2/6 |
| B3_gauss_metrica_teto50 | 6 | 0,1,2,3,4,5 | -0.0078 | 0.0208 | -0.0340 a +0.0177 | 3/6 |

## area sob a varredura de limiar (maior melhor)

| braco | n comum | seeds | delta medio | desvio | amplitude do delta | seeds a favor |
|---|---|---|---|---|---|---|
| B0_berhu_size768 | 3 | 0,1,2 | -0.0197 | 0.0057 | -0.0252 a -0.0138 | 0/3 |
| B1_gauss_metrica_teto1000 | 6 | 0,1,2,3,4,5 | -0.0644 | 0.0111 | -0.0794 a -0.0517 | 0/6 |
| B1_gauss_metrica_teto5 | 6 | 0,1,2,3,4,5 | -0.0320 | 0.0198 | -0.0633 a -0.0107 | 0/6 |
| B3_gauss_metrica_teto1000 | 6 | 0,1,2,3,4,5 | -0.0459 | 0.0120 | -0.0594 a -0.0236 | 0/6 |
| B3_gauss_metrica_teto5 | 6 | 0,1,2,3,4,5 | -0.0370 | 0.0186 | -0.0608 a -0.0167 | 0/6 |
| B3_gauss_metrica_teto50 | 6 | 0,1,2,3,4,5 | -0.0348 | 0.0255 | -0.0690 a -0.0020 | 0/6 |

## AbsRel (menor melhor)

| braco | n comum | seeds | delta medio | desvio | amplitude do delta | seeds a favor |
|---|---|---|---|---|---|---|
| B0_berhu_size768 | 3 | 0,1,2 | -0.0006 | 0.0070 | -0.0053 a +0.0075 | 2/3 |
| B1_gauss_metrica_teto1000 | 6 | 0,1,2,3,4,5 | +0.0249 | 0.0229 | -0.0010 a +0.0611 | 1/6 |
| B1_gauss_metrica_teto5 | 6 | 0,1,2,3,4,5 | +0.0226 | 0.0427 | -0.0109 a +0.0876 | 3/6 |
| B3_gauss_metrica_teto1000 | 6 | 0,1,2,3,4,5 | +0.0423 | 0.0151 | +0.0149 a +0.0589 | 0/6 |
| B3_gauss_metrica_teto5 | 6 | 0,1,2,3,4,5 | +0.0263 | 0.0178 | +0.0030 a +0.0515 | 0/6 |
| B3_gauss_metrica_teto50 | 6 | 0,1,2,3,4,5 | +0.0224 | 0.0277 | -0.0238 a +0.0547 | 1/6 |

## delta1 (maior melhor)

| braco | n comum | seeds | delta medio | desvio | amplitude do delta | seeds a favor |
|---|---|---|---|---|---|---|
| B0_berhu_size768 | 3 | 0,1,2 | +0.0044 | 0.0046 | +0.0009 a +0.0097 | 3/3 |
| B1_gauss_metrica_teto1000 | 6 | 0,1,2,3,4,5 | -0.0045 | 0.0236 | -0.0419 a +0.0192 | 3/6 |
| B1_gauss_metrica_teto5 | 6 | 0,1,2,3,4,5 | -0.0072 | 0.0241 | -0.0460 a +0.0152 | 4/6 |
| B3_gauss_metrica_teto1000 | 6 | 0,1,2,3,4,5 | -0.0052 | 0.0192 | -0.0211 a +0.0321 | 1/6 |
| B3_gauss_metrica_teto5 | 6 | 0,1,2,3,4,5 | -0.0002 | 0.0171 | -0.0123 a +0.0321 | 2/6 |
| B3_gauss_metrica_teto50 | 6 | 0,1,2,3,4,5 | +0.0014 | 0.0172 | -0.0108 a +0.0313 | 2/6 |

## RMSE (menor melhor)

| braco | n comum | seeds | delta medio | desvio | amplitude do delta | seeds a favor |
|---|---|---|---|---|---|---|
| B0_berhu_size768 | 3 | 0,1,2 | -0.1018 | 0.0882 | -0.1847 a -0.0091 | 3/3 |
| B1_gauss_metrica_teto1000 | 6 | 0,1,2,3,4,5 | +0.4111 | 0.3806 | +0.0409 a +1.0466 | 0/6 |
| B1_gauss_metrica_teto5 | 6 | 0,1,2,3,4,5 | +0.3231 | 0.4308 | -0.0265 a +1.0655 | 2/6 |
| B3_gauss_metrica_teto1000 | 6 | 0,1,2,3,4,5 | +0.3080 | 0.1787 | +0.0020 a +0.5550 | 0/6 |
| B3_gauss_metrica_teto5 | 6 | 0,1,2,3,4,5 | +0.2007 | 0.1966 | -0.0146 a +0.4919 | 2/6 |
| B3_gauss_metrica_teto50 | 6 | 0,1,2,3,4,5 | +0.1583 | 0.2152 | -0.1735 a +0.4651 | 1/6 |
