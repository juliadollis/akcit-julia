# Resultados com n fixo em 6 (seeds 0 a 5)

Gerado por `scripts/compara_zeroshot.py`. Zero-shot = DepthPro de prateleira, sem fine-tune, medido pelo mesmo `Trainer.validate()` que gerou o `test_metrics.json` de cada seed treinada.

Recorte de seeds: `0-5`.

`pior seed` e a seed menos favoravel do braco naquela metrica.


## F-score de borda (limiar fixo) (maior melhor)

Zero-shot: **0.5402**

| braco | n | seeds | media | amplitude | pior seed | delta vs zero-shot | bate o zero-shot |
|---|---|---|---|---|---|---|---|
| B0_berhu | 6 | 0,1,2,3,4,5 | 0.5954 | 0.5895 a 0.5996 | 0.5895 | +0.0552 | sim, ate a pior seed |
| B0_berhu_size768 | 3 | 0,1,2 | 0.5918 | 0.5854 a 0.6001 | 0.5854 | +0.0516 | sim, ate a pior seed |
| B1_gauss_metrica_teto1000 | 6 | 0,1,2,3,4,5 | 0.5339 | 0.5229 a 0.5491 | 0.5229 | -0.0063 | **nao** |
| B1_gauss_metrica_teto5 | 6 | 0,1,2,3,4,5 | 0.5604 | 0.5462 a 0.5784 | 0.5462 | +0.0201 | sim, ate a pior seed |
| B3_gauss_metrica_teto1000 | 6 | 0,1,2,3,4,5 | 0.5704 | 0.5543 a 0.5908 | 0.5543 | +0.0302 | sim, ate a pior seed |
| B3_gauss_metrica_teto5 | 6 | 0,1,2,3,4,5 | 0.5732 | 0.5419 a 0.6009 | 0.5419 | +0.0329 | sim, ate a pior seed |
| B3_gauss_metrica_teto50 | 6 | 0,1,2,3,4,5 | 0.5765 | 0.5442 a 0.5990 | 0.5442 | +0.0362 | sim, ate a pior seed |

## F-score de borda no melhor limiar (maior melhor)

Zero-shot: **0.7674**

| braco | n | seeds | media | amplitude | pior seed | delta vs zero-shot | bate o zero-shot |
|---|---|---|---|---|---|---|---|
| B0_berhu | 6 | 0,1,2,3,4,5 | 0.7791 | 0.7710 a 0.7857 | 0.7710 | +0.0117 | sim, ate a pior seed |
| B0_berhu_size768 | 3 | 0,1,2 | 0.7463 | 0.7430 a 0.7499 | 0.7430 | -0.0210 | **nao** |
| B1_gauss_metrica_teto1000 | 6 | 0,1,2,3,4,5 | 0.7230 | 0.7020 a 0.7480 | 0.7020 | -0.0444 | **nao** |
| B1_gauss_metrica_teto5 | 6 | 0,1,2,3,4,5 | 0.7646 | 0.7424 a 0.7878 | 0.7424 | -0.0028 | **nao** |
| B3_gauss_metrica_teto1000 | 6 | 0,1,2,3,4,5 | 0.7632 | 0.7563 a 0.7694 | 0.7563 | -0.0041 | **nao** |
| B3_gauss_metrica_teto5 | 6 | 0,1,2,3,4,5 | 0.7678 | 0.7375 a 0.7863 | 0.7375 | +0.0004 | sim |
| B3_gauss_metrica_teto50 | 6 | 0,1,2,3,4,5 | 0.7713 | 0.7438 a 0.7968 | 0.7438 | +0.0039 | sim |

## area sob a varredura de limiar (maior melhor)

Zero-shot: **0.5641**

| braco | n | seeds | media | amplitude | pior seed | delta vs zero-shot | bate o zero-shot |
|---|---|---|---|---|---|---|---|
| B0_berhu | 6 | 0,1,2,3,4,5 | 0.6254 | 0.6231 a 0.6287 | 0.6231 | +0.0614 | sim, ate a pior seed |
| B0_berhu_size768 | 3 | 0,1,2 | 0.6041 | 0.5997 a 0.6093 | 0.5997 | +0.0400 | sim, ate a pior seed |
| B1_gauss_metrica_teto1000 | 6 | 0,1,2,3,4,5 | 0.5610 | 0.5440 a 0.5770 | 0.5440 | -0.0030 | **nao** |
| B1_gauss_metrica_teto5 | 6 | 0,1,2,3,4,5 | 0.5935 | 0.5648 a 0.6142 | 0.5648 | +0.0294 | sim, ate a pior seed |
| B3_gauss_metrica_teto1000 | 6 | 0,1,2,3,4,5 | 0.5795 | 0.5651 a 0.6014 | 0.5651 | +0.0154 | sim, ate a pior seed |
| B3_gauss_metrica_teto5 | 6 | 0,1,2,3,4,5 | 0.5885 | 0.5626 a 0.6104 | 0.5626 | +0.0244 | sim |
| B3_gauss_metrica_teto50 | 6 | 0,1,2,3,4,5 | 0.5906 | 0.5591 a 0.6214 | 0.5591 | +0.0265 | sim |

## AbsRel (menor melhor)

Zero-shot: **0.3602**

| braco | n | seeds | media | amplitude | pior seed | delta vs zero-shot | bate o zero-shot |
|---|---|---|---|---|---|---|---|
| B0_berhu | 6 | 0,1,2,3,4,5 | 0.2509 | 0.2460 a 0.2562 | 0.2562 | -0.1093 | sim, ate a pior seed |
| B0_berhu_size768 | 3 | 0,1,2 | 0.2500 | 0.2444 a 0.2535 | 0.2535 | -0.1102 | sim, ate a pior seed |
| B1_gauss_metrica_teto1000 | 6 | 0,1,2,3,4,5 | 0.2758 | 0.2483 a 0.3173 | 0.3173 | -0.0844 | sim, ate a pior seed |
| B1_gauss_metrica_teto5 | 6 | 0,1,2,3,4,5 | 0.2735 | 0.2424 a 0.3369 | 0.3369 | -0.0867 | sim, ate a pior seed |
| B3_gauss_metrica_teto1000 | 6 | 0,1,2,3,4,5 | 0.2932 | 0.2609 a 0.3098 | 0.3098 | -0.0671 | sim, ate a pior seed |
| B3_gauss_metrica_teto5 | 6 | 0,1,2,3,4,5 | 0.2772 | 0.2523 a 0.3024 | 0.3024 | -0.0831 | sim, ate a pior seed |
| B3_gauss_metrica_teto50 | 6 | 0,1,2,3,4,5 | 0.2733 | 0.2324 a 0.3044 | 0.3044 | -0.0869 | sim, ate a pior seed |

## delta1 (maior melhor)

Zero-shot: **0.6594**

| braco | n | seeds | media | amplitude | pior seed | delta vs zero-shot | bate o zero-shot |
|---|---|---|---|---|---|---|---|
| B0_berhu | 6 | 0,1,2,3,4,5 | 0.6946 | 0.6909 a 0.6981 | 0.6909 | +0.0351 | sim, ate a pior seed |
| B0_berhu_size768 | 3 | 0,1,2 | 0.7000 | 0.6990 a 0.7006 | 0.6990 | +0.0405 | sim, ate a pior seed |
| B1_gauss_metrica_teto1000 | 6 | 0,1,2,3,4,5 | 0.6901 | 0.6556 a 0.7105 | 0.6556 | +0.0306 | sim |
| B1_gauss_metrica_teto5 | 6 | 0,1,2,3,4,5 | 0.6873 | 0.6453 a 0.7061 | 0.6453 | +0.0279 | sim |
| B3_gauss_metrica_teto1000 | 6 | 0,1,2,3,4,5 | 0.6894 | 0.6749 a 0.7303 | 0.6749 | +0.0299 | sim, ate a pior seed |
| B3_gauss_metrica_teto5 | 6 | 0,1,2,3,4,5 | 0.6943 | 0.6820 a 0.7234 | 0.6820 | +0.0349 | sim, ate a pior seed |
| B3_gauss_metrica_teto50 | 6 | 0,1,2,3,4,5 | 0.6960 | 0.6805 a 0.7288 | 0.6805 | +0.0366 | sim, ate a pior seed |

## RMSE (menor melhor)

Zero-shot: **5.4002**

| braco | n | seeds | media | amplitude | pior seed | delta vs zero-shot | bate o zero-shot |
|---|---|---|---|---|---|---|---|
| B0_berhu | 6 | 0,1,2,3,4,5 | 4.2433 | 4.1972 a 4.2951 | 4.2951 | -1.1569 | sim, ate a pior seed |
| B0_berhu_size768 | 3 | 0,1,2 | 4.1258 | 4.0371 a 4.1983 | 4.1983 | -1.2744 | sim, ate a pior seed |
| B1_gauss_metrica_teto1000 | 6 | 0,1,2,3,4,5 | 4.6544 | 4.3257 a 5.3003 | 5.3003 | -0.7458 | sim, ate a pior seed |
| B1_gauss_metrica_teto5 | 6 | 0,1,2,3,4,5 | 4.5664 | 4.2047 a 5.3503 | 5.3503 | -0.8338 | sim, ate a pior seed |
| B3_gauss_metrica_teto1000 | 6 | 0,1,2,3,4,5 | 4.5513 | 4.2093 a 4.7522 | 4.7522 | -0.8489 | sim, ate a pior seed |
| B3_gauss_metrica_teto5 | 6 | 0,1,2,3,4,5 | 4.4440 | 4.2702 a 4.6891 | 4.6891 | -0.9562 | sim, ate a pior seed |
| B3_gauss_metrica_teto50 | 6 | 0,1,2,3,4,5 | 4.4016 | 4.0801 a 4.6870 | 4.6870 | -0.9986 | sim, ate a pior seed |
