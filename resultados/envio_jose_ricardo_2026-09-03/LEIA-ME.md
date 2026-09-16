# Envio ao José Ricardo, 2026-09-03

Leia primeiro `MENSAGEM.md`. Este arquivo só descreve o layout.

## As duas pastas de seeds

| pasta | seeds | por que separada |
|---|---|---|
| `seeds_pedidas_0a2/` | 0, 1, 2 | são exatamente as três do envio anterior. Recortando só elas, os números da mensagem de 01/09 saem idênticos, então dá para conferir que nada mudou de pipeline no caminho |
| `seeds_novas_3a9/` | 3 em diante | as que fecharam depois. Sozinhas elas não substituem nada; servem para ver se a direção do resultado se manteve fora do primeiro conjunto |

As duas juntas são o que alimenta `TABELA_todas_as_seeds.md` e
`CONTRASTE_vs_controle.md`, na raiz.

Nem todo braço tem todas as seeds. As que faltam estão listadas na seção de
ressalvas da mensagem: sete rodadas ficaram sem `test_metrics.json` e não entram
em conta nenhuma.

## Os braços

| pasta | perda |
|---|---|
| `B0_berhu` | berHu 0,7 puro. É o controle |
| `B1_gauss_metrica_teto5` / `_teto1000` | berHu 0,7 + curvatura métrica 0,45. A curvatura SOZINHA, que é a hipótese isolada |
| `B3_gauss_metrica_teto5` / `_teto1000` | berHu 0,7 + normal 0,9 + curvatura métrica 0,45 |

O sufixo é o `gauss_clamp`, o teto por pixel da curvatura em 1/m².

## Dentro de cada seed

| arquivo | o que é |
|---|---|
| `test_metrics.json` | **o número reportado.** Teste held-out, avaliado uma vez no fim |
| `summary.json` | melhor época e a métrica de validação que a escolheu |
| `history.json` | a curva completa, época a época |

## O zero-shot

`zero_shot/` é o DepthPro sem fine-tune nenhum, no mesmo split de teste. Não tem
seeds: sem treino, com o modelo em `eval()` e o loader sem embaralhar, a passada
é determinística.

`por_imagem.csv` traz as 485 imagens com cena e métricas, para um teste pareado
cena a cena quando os braços treinados forem reavaliados por imagem.

## Como as tabelas foram feitas

Geradas por script a partir dos JSONs, nunca transcritas à mão:

```bash
python3 scripts/compara_zeroshot.py <pasta_dos_bracos> \
    --zero-shot resultados/zero_shot_2026-09-03 [--seeds 0-2]
python3 scripts/contraste_controle.py <pasta_dos_bracos> [--controle B0_berhu]
```
