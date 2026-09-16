# Passo 4 com teto 5,0

Complemento do resultado anterior. Mesmo experimento, mesma configuração, só o
`gauss_clamp` muda: 5,0 no lugar de 1000. Rodou com a curvatura métrica
corrigida, 3 seeds, igual ao anterior.

## Resultado

Conjunto de teste separado, média entre 3 seeds:

| métrica | teto 1000 (enviado antes) | teto 5 | B0 controle |
|---|---|---|---|
| F-score de borda (maior melhor) | 0,5745 | **0,5654** | **0,5944** |
| AbsRel (menor melhor) | 0,2857 | **0,2806** | **0,2507** |
| delta1 (maior melhor) | 0,6977 | **0,6868** | 0,6955 |
| RMSE (menor melhor) | 4,4548 | **4,4744** | **4,2276** |

**O teto 5 não muda a conclusão: perde do controle nas quatro métricas.** Com
teto 1000 o B3 ao menos empatava no delta1 (+0,0022); com teto 5 perde também
nessa, e a diferença no F-score de borda cresce de -0,020 para -0,029.

Por seed, no F-score de borda: teto 5 deu 0,5641 / 0,5419 / 0,5903, contra
0,5908 / 0,5614 / 0,5712 do teto 1000 e 0,5986 / 0,5895 / 0,5952 do controle.

## Por que eu tinha escolhido 1000

O documento previa que 5,0 ficaria folgado. Medi antes de rodar, sobre 517
quadros do Spring e 135,5 milhões de pixels, usando o `fx` de cada sequência:

- percentis de |K| (1/m²): p50 = 4,41 | p90 = 987 | p99 = 68.745 | p99,9 = 3.230.846
- fração de pixels acima do teto: >1 = 61,3% | >5 = 49,0% | >20 = 37,2% | >50 = 29,5%

Com teto 5, metade dos pixels satura. Como o corte é aplicado nos dois lados
antes da subtração, um pixel onde a verdade é K=800 e a predição é K=300 vira
|5 - 5| = 0: erro de 500 registrado como zero, sem gerar gradiente. Foi esse o
motivo da escolha.

Agora as duas pontas do intervalo estão testadas e dão a mesma direção, então o
resultado não depende dessa escolha. Era a dúvida que restava.

## O que isso não diz

- É o Spring, e só o Spring. Os nulos anteriores em Hypersim e DIODE não contam
  como evidência: saíram com o `h` normalizado, ou seja, com um termo que não
  media curvatura.
- Não há teste formal. Com 3 seeds o que a tabela sustenta é a direção, não a
  significância.
- Não varremos o peso do termo, que ficou em 0,45.

## Configuração, para conferência

Spring, split 593 treino / 184 val / 485 teste, seed 42, idêntico nos dois
braços. DepthPro variante `heads`, lr 1e-5, até 100 épocas com early stop no
F-score de borda de validação. Curvatura métrica com `fx = 689,6` e
`fy = 1225,9` px em 512. Pesos berHu 0,7, normal 0,9, curvatura 0,45. Teste
avaliado uma única vez, no fim, por seed.
