# Somente as seeds que fecharam depois (3 em diante) contra o zero-shot

Gerado por `scripts/compara_zeroshot.py`. Zero-shot = DepthPro de prateleira, sem fine-tune, medido pelo mesmo `Trainer.validate()` que gerou o `test_metrics.json` de cada seed treinada.

`pior seed` e a seed menos favoravel do braco naquela metrica.


## F-score de borda (maior melhor)

Zero-shot: **0.5402**

| braco | n | seeds | media | amplitude | pior seed | delta vs zero-shot | bate o zero-shot |
|---|---|---|---|---|---|---|---|
| B0_berhu | 5 | 3,4,5,8,9 | 0.5942 | 0.5902 a 0.5996 | 0.5902 | +0.0540 | sim, ate a pior seed |
| B1_gauss_metrica_teto1000 | 3 | 3,4,5 | 0.5357 | 0.5283 a 0.5483 | 0.5283 | -0.0045 | **nao** |
| B1_gauss_metrica_teto5 | 2 | 4,5 | 0.5612 | 0.5521 a 0.5703 | 0.5521 | +0.0210 | sim, ate a pior seed |
| B3_gauss_metrica_teto1000 | 7 | 3,4,5,6,7,8,9 | 0.5696 | 0.5468 a 0.5999 | 0.5468 | +0.0294 | sim, ate a pior seed |
| B3_gauss_metrica_teto5 | 5 | 3,4,5,6,7 | 0.5801 | 0.5673 a 0.6009 | 0.5673 | +0.0398 | sim, ate a pior seed |

## AbsRel (menor melhor)

Zero-shot: **0.3602**

| braco | n | seeds | media | amplitude | pior seed | delta vs zero-shot | bate o zero-shot |
|---|---|---|---|---|---|---|---|
| B0_berhu | 5 | 3,4,5,8,9 | 0.2509 | 0.2466 a 0.2544 | 0.2544 | -0.1093 | sim, ate a pior seed |
| B1_gauss_metrica_teto1000 | 3 | 3,4,5 | 0.2739 | 0.2483 a 0.2877 | 0.2877 | -0.0863 | sim, ate a pior seed |
| B1_gauss_metrica_teto5 | 2 | 4,5 | 0.2795 | 0.2424 a 0.3167 | 0.3167 | -0.0807 | sim, ate a pior seed |
| B3_gauss_metrica_teto1000 | 7 | 3,4,5,6,7,8,9 | 0.2965 | 0.2782 a 0.3098 | 0.3098 | -0.0637 | sim, ate a pior seed |
| B3_gauss_metrica_teto5 | 5 | 3,4,5,6,7 | 0.2756 | 0.2523 a 0.3024 | 0.3024 | -0.0846 | sim, ate a pior seed |

## delta1 (maior melhor)

Zero-shot: **0.6594**

| braco | n | seeds | media | amplitude | pior seed | delta vs zero-shot | bate o zero-shot |
|---|---|---|---|---|---|---|---|
| B0_berhu | 5 | 3,4,5,8,9 | 0.6931 | 0.6857 a 0.6987 | 0.6857 | +0.0336 | sim, ate a pior seed |
| B1_gauss_metrica_teto1000 | 3 | 3,4,5 | 0.6918 | 0.6724 a 0.7105 | 0.6724 | +0.0324 | sim, ate a pior seed |
| B1_gauss_metrica_teto5 | 2 | 4,5 | 0.6830 | 0.6680 a 0.6979 | 0.6680 | +0.0235 | sim, ate a pior seed |
| B3_gauss_metrica_teto1000 | 7 | 3,4,5,6,7,8,9 | 0.6824 | 0.6749 a 0.6864 | 0.6749 | +0.0230 | sim, ate a pior seed |
| B3_gauss_metrica_teto5 | 5 | 3,4,5,6,7 | 0.6986 | 0.6836 a 0.7234 | 0.6836 | +0.0392 | sim, ate a pior seed |

## RMSE (menor melhor)

Zero-shot: **5.4002**

| braco | n | seeds | media | amplitude | pior seed | delta vs zero-shot | bate o zero-shot |
|---|---|---|---|---|---|---|---|
| B0_berhu | 5 | 3,4,5,8,9 | 4.2579 | 4.1972 a 4.2951 | 4.2951 | -1.1423 | sim, ate a pior seed |
| B1_gauss_metrica_teto1000 | 3 | 3,4,5 | 4.6322 | 4.3257 a 4.9136 | 4.9136 | -0.7681 | sim, ate a pior seed |
| B1_gauss_metrica_teto5 | 2 | 4,5 | 4.5397 | 4.2685 a 4.8108 | 4.8108 | -0.8606 | sim, ate a pior seed |
| B3_gauss_metrica_teto1000 | 7 | 3,4,5,6,7,8,9 | 4.6054 | 4.5401 a 4.7522 | 4.7522 | -0.7948 | sim, ate a pior seed |
| B3_gauss_metrica_teto5 | 5 | 3,4,5,6,7 | 4.4109 | 4.2702 a 4.6891 | 4.6891 | -0.9893 | sim, ate a pior seed |
