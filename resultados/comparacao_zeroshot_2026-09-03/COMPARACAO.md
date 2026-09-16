# Bracos treinados contra o DepthPro zero-shot (Spring, teste)

Gerado por `scripts/compara_zeroshot.py`. Zero-shot = DepthPro de prateleira, sem fine-tune, medido pelo mesmo `Trainer.validate()` que gerou o `test_metrics.json` de cada seed treinada.

Cada braco tem um numero diferente de seeds concluidas; o `n` esta na tabela. `pior seed` e a seed menos favoravel do braco naquela metrica.


## F-score de borda (maior melhor)

Zero-shot: **0.5402**

| braco | n | media | amplitude | pior seed | delta vs zero-shot | bate o zero-shot |
|---|---|---|---|---|---|---|
| B0_berhu | 8 | 0.5943 | 0.5895 a 0.5996 | 0.5895 | +0.0540 | sim, ate a pior seed |
| B1_gauss_metrica_teto1000 | 6 | 0.5339 | 0.5229 a 0.5491 | 0.5229 | -0.0063 | **nao** |
| B1_gauss_metrica_teto5 | 5 | 0.5632 | 0.5521 a 0.5784 | 0.5521 | +0.0230 | sim, ate a pior seed |
| B3_gauss_metrica_teto1000 | 10 | 0.5711 | 0.5468 a 0.5999 | 0.5468 | +0.0308 | sim, ate a pior seed |
| B3_gauss_metrica_teto5 | 8 | 0.5746 | 0.5419 a 0.6009 | 0.5419 | +0.0343 | sim, ate a pior seed |

## AbsRel (menor melhor)

Zero-shot: **0.3602**

| braco | n | media | amplitude | pior seed | delta vs zero-shot | bate o zero-shot |
|---|---|---|---|---|---|---|
| B0_berhu | 8 | 0.2508 | 0.2460 a 0.2562 | 0.2562 | -0.1094 | sim, ate a pior seed |
| B1_gauss_metrica_teto1000 | 6 | 0.2758 | 0.2483 a 0.3173 | 0.3173 | -0.0844 | sim, ate a pior seed |
| B1_gauss_metrica_teto5 | 5 | 0.2608 | 0.2424 a 0.3167 | 0.3167 | -0.0994 | sim, ate a pior seed |
| B3_gauss_metrica_teto1000 | 10 | 0.2933 | 0.2609 a 0.3098 | 0.3098 | -0.0670 | sim, ate a pior seed |
| B3_gauss_metrica_teto5 | 8 | 0.2775 | 0.2523 a 0.3024 | 0.3024 | -0.0828 | sim, ate a pior seed |

## delta1 (maior melhor)

Zero-shot: **0.6594**

| braco | n | media | amplitude | pior seed | delta vs zero-shot | bate o zero-shot |
|---|---|---|---|---|---|---|
| B0_berhu | 8 | 0.6940 | 0.6857 a 0.6987 | 0.6857 | +0.0345 | sim, ate a pior seed |
| B1_gauss_metrica_teto1000 | 6 | 0.6901 | 0.6556 a 0.7105 | 0.6556 | +0.0306 | sim |
| B1_gauss_metrica_teto5 | 5 | 0.6957 | 0.6680 a 0.7061 | 0.6680 | +0.0363 | sim, ate a pior seed |
| B3_gauss_metrica_teto1000 | 10 | 0.6870 | 0.6749 a 0.7303 | 0.6749 | +0.0276 | sim, ate a pior seed |
| B3_gauss_metrica_teto5 | 8 | 0.6942 | 0.6820 a 0.7234 | 0.6820 | +0.0347 | sim, ate a pior seed |

## RMSE (menor melhor)

Zero-shot: **5.4002**

| braco | n | media | amplitude | pior seed | delta vs zero-shot | bate o zero-shot |
|---|---|---|---|---|---|---|
| B0_berhu | 8 | 4.2466 | 4.1972 a 4.2951 | 4.2951 | -1.1537 | sim, ate a pior seed |
| B1_gauss_metrica_teto1000 | 6 | 4.6544 | 4.3257 a 5.3003 | 5.3003 | -0.7458 | sim, ate a pior seed |
| B1_gauss_metrica_teto5 | 5 | 4.4096 | 4.2047 a 4.8108 | 4.8108 | -0.9906 | sim, ate a pior seed |
| B3_gauss_metrica_teto1000 | 10 | 4.5602 | 4.2093 a 4.7522 | 4.7522 | -0.8400 | sim, ate a pior seed |
| B3_gauss_metrica_teto5 | 8 | 4.4347 | 4.2702 a 4.6891 | 4.6891 | -0.9655 | sim, ate a pior seed |
