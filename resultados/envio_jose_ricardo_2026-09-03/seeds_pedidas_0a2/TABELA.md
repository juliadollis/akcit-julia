# As 3 seeds do envio original (0, 1, 2) contra o zero-shot

Gerado por `scripts/compara_zeroshot.py`. Zero-shot = DepthPro de prateleira, sem fine-tune, medido pelo mesmo `Trainer.validate()` que gerou o `test_metrics.json` de cada seed treinada.

`pior seed` e a seed menos favoravel do braco naquela metrica.


## F-score de borda (maior melhor)

Zero-shot: **0.5402**

| braco | n | seeds | media | amplitude | pior seed | delta vs zero-shot | bate o zero-shot |
|---|---|---|---|---|---|---|---|
| B0_berhu | 3 | 0,1,2 | 0.5944 | 0.5895 a 0.5986 | 0.5895 | +0.0542 | sim, ate a pior seed |
| B1_gauss_metrica_teto1000 | 3 | 0,1,2 | 0.5320 | 0.5229 a 0.5491 | 0.5229 | -0.0082 | **nao** |
| B1_gauss_metrica_teto5 | 3 | 0,1,2 | 0.5645 | 0.5542 a 0.5784 | 0.5542 | +0.0243 | sim, ate a pior seed |
| B3_gauss_metrica_teto1000 | 3 | 0,1,2 | 0.5745 | 0.5614 a 0.5908 | 0.5614 | +0.0342 | sim, ate a pior seed |
| B3_gauss_metrica_teto5 | 3 | 0,1,2 | 0.5654 | 0.5419 a 0.5903 | 0.5419 | +0.0252 | sim, ate a pior seed |

## AbsRel (menor melhor)

Zero-shot: **0.3602**

| braco | n | seeds | media | amplitude | pior seed | delta vs zero-shot | bate o zero-shot |
|---|---|---|---|---|---|---|---|
| B0_berhu | 3 | 0,1,2 | 0.2507 | 0.2460 a 0.2562 | 0.2562 | -0.1096 | sim, ate a pior seed |
| B1_gauss_metrica_teto1000 | 3 | 0,1,2 | 0.2777 | 0.2570 a 0.3173 | 0.3173 | -0.0825 | sim, ate a pior seed |
| B1_gauss_metrica_teto5 | 3 | 0,1,2 | 0.2484 | 0.2424 a 0.2588 | 0.2588 | -0.1119 | sim, ate a pior seed |
| B3_gauss_metrica_teto1000 | 3 | 0,1,2 | 0.2857 | 0.2609 a 0.2993 | 0.2993 | -0.0745 | sim, ate a pior seed |
| B3_gauss_metrica_teto5 | 3 | 0,1,2 | 0.2806 | 0.2773 a 0.2870 | 0.2870 | -0.0796 | sim, ate a pior seed |

## delta1 (maior melhor)

Zero-shot: **0.6594**

| braco | n | seeds | media | amplitude | pior seed | delta vs zero-shot | bate o zero-shot |
|---|---|---|---|---|---|---|---|
| B0_berhu | 3 | 0,1,2 | 0.6955 | 0.6909 a 0.6981 | 0.6909 | +0.0361 | sim, ate a pior seed |
| B1_gauss_metrica_teto1000 | 3 | 0,1,2 | 0.6883 | 0.6556 a 0.7049 | 0.6556 | +0.0289 | sim |
| B1_gauss_metrica_teto5 | 3 | 0,1,2 | 0.7043 | 0.7023 a 0.7061 | 0.7023 | +0.0448 | sim, ate a pior seed |
| B3_gauss_metrica_teto1000 | 3 | 0,1,2 | 0.6977 | 0.6797 a 0.7303 | 0.6797 | +0.0383 | sim, ate a pior seed |
| B3_gauss_metrica_teto5 | 3 | 0,1,2 | 0.6868 | 0.6820 a 0.6923 | 0.6820 | +0.0273 | sim, ate a pior seed |

## RMSE (menor melhor)

Zero-shot: **5.4002**

| braco | n | seeds | media | amplitude | pior seed | delta vs zero-shot | bate o zero-shot |
|---|---|---|---|---|---|---|---|
| B0_berhu | 3 | 0,1,2 | 4.2276 | 4.2073 a 4.2536 | 4.2536 | -1.1726 | sim, ate a pior seed |
| B1_gauss_metrica_teto1000 | 3 | 0,1,2 | 4.6767 | 4.3412 a 5.3003 | 5.3003 | -0.7236 | sim, ate a pior seed |
| B1_gauss_metrica_teto5 | 3 | 0,1,2 | 4.3229 | 4.2047 a 4.3982 | 4.3982 | -1.0773 | sim, ate a pior seed |
| B3_gauss_metrica_teto1000 | 3 | 0,1,2 | 4.4548 | 4.2093 a 4.5994 | 4.5994 | -0.9455 | sim, ate a pior seed |
| B3_gauss_metrica_teto5 | 3 | 0,1,2 | 4.4744 | 4.4331 a 4.5345 | 4.5345 | -0.9258 | sim, ate a pior seed |
