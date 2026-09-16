# Ablacao no Spring: resultado

Gerado do `ablation_results.csv`. Ordenado pelo monitor (`boundary_fscore`).
`delta` e contra o controle `B0_berhu`.

| config | F-borda | fmax | f_auc | AbsRel | delta1 | RMSE | delta F-borda |
|---|---|---|---|---|---|---|---|
| **B1_berhu+grad** | 0.5314 | 0.6912 | 0.5540 | 0.3341 | 0.5194 | 11.4880 | +0.0302 |
| **B1_berhu+metric** | 0.5287 | 0.6593 | 0.5515 | 0.2639 | 0.6341 | 9.7924 | +0.0276 |
| B1_berhu+geod | 0.5163 | 0.6680 | 0.5472 | 0.2893 | 0.5937 | 9.8114 | +0.0152 |
| B0_berhu | 0.5011 | 0.6485 | 0.5326 | 0.2645 | 0.6441 | 9.6464 | +0.0000 |
| B7_gaussheavy | 0.4940 | 0.6181 | 0.5087 | 0.2954 | 0.6184 | 9.7177 | -0.0071 |
| ZERO_SHOT_sem_finetune | 0.4861 | 0.6201 | 0.5132 | 0.3075 | 0.5990 | 9.7290 | -0.0150 |
| **B1_berhu+normal** | 0.4315 | 0.5579 | 0.4518 | 0.3051 | 0.5989 | 9.6466 | -0.0696 |
| **B1_berhu+gauss** | 0.4311 | 0.6205 | 0.4669 | 0.2921 | 0.6267 | 10.0792 | -0.0700 |
