# O zero-shot que faltava: existe headroom no Spring

Medido em 2026-09-03. Números em `COMPARACAO.md`, gerado por
`scripts/compara_zeroshot.py` a partir dos `test_metrics.json`, nunca
transcritos à mão. O bruto do zero-shot está em `../zero_shot_2026-09-03/`.

## O que foi medido

O DepthPro de prateleira, sem fine-tune nenhum, no mesmo split de teste do
reteste da curvatura: 485 imagens, 13 cenas, seed 42 do split, 512 px, batch 8.
O agregado sai do **mesmo `Trainer.validate()`** que gerou o `test_metrics.json`
de cada seed treinada, então os números são diretamente comparáveis. Nenhum
`best.pt` foi carregado e nenhuma época foi treinada.

Não há variação por seed a reportar: sem treino, com o modelo em `eval()` e o
loader sem embaralhar, a passada é determinística.

## Por que ele faltava

Os dois passos do reteste (`passo4.sh` e `passo4_teto.sh`) chamam só o
`train_single.py`, que treina e avalia o modelo treinado. O caminho zero-shot
existe no `evaluate_paired.py`, mas nunca foi executado no Spring. Consequência:
todos os braços eram comparados entre si, e nenhum contra o ponto de partida.

## A resposta

**Existe headroom no Spring, e é grande.** Todo braço treinado bate o zero-shot
em AbsRel, delta1 e RMSE, e quase nunca pela média: em **AbsRel e RMSE até a
pior seed de cada braço** bate o zero-shot. Em delta1 a única exceção é a pior
seed do B1 teto 1000 (0,6556 contra 0,6594).

| métrica | zero-shot | melhor braço | ganho |
|---|---|---|---|
| AbsRel (menor melhor) | 0,3602 | 0,2508 (B0 berHu, n=8) | -0,1094 |
| RMSE (menor melhor) | 5,4002 | 4,2466 (B0 berHu, n=8) | -1,1537 |
| delta1 (maior melhor) | 0,6594 | 0,6957 (B1 teto 5, n=5) | +0,0363 |
| F-score de borda (maior melhor) | 0,5402 | 0,5943 (B0 berHu, n=8) | +0,0540 |

Isso é o oposto do que acontecia no Hypersim, onde nenhuma configuração superava
a linha de base e até o berHu puro degradava. A leitura do `GUIA_DIODE.md` fica
confirmada por medição: lá não havia espaço a tomar porque o Hypersim está no
treino do DepthPro; aqui há.

## O que isso muda na leitura da curvatura

Não resgata a hipótese. Ordena melhor o nulo.

No F-score de borda, que é onde o método promete ganho e é o monitor do early
stop, a ordem é:

```
berHu puro (+0,0540)  >  curvatura (+0,023 a +0,034)  >  zero-shot
```

com uma exceção: o **B1 com teto 1000 fica ABAIXO do zero-shot** (-0,0063), ou
seja, aquele braço piora a borda em relação a não treinar nada.

Antes do zero-shot dava para dizer só que a curvatura perdia do controle. Agora
dá para dizer o que ela custa: **a curvatura entrega parte do ganho que o berHu
puro sozinho entrega, e joga fora o resto.** O termo geométrico não é neutro,
ele consome ganho disponível.

## Ressalva de simetria

Os braços têm números diferentes de seeds concluídas (B0 n=8, B1 teto5 n=5,
B1 teto1000 n=6, B3 teto5 n=8, B3 teto1000 n=10). Para a pergunta "o treino
ajuda?" isso não incomoda, porque a conclusão vale até na pior seed de cada
braço. Para comparar **braço contra braço** o `consolida_reteste.py` se recusa a
comparar n diferentes, e com razão: essas diferenças de média são da ordem da
dispersão entre seeds.

## Pendência que isto levanta

O `CONCLUSAO.md` de 2026-09-01 foi escrito com n=3 e avisa, no topo, que as
seeds 3 a 9 estavam rodando e que era para reconsolidar antes de mandar para
fora. **Elas fecharam.** Com o n maior, o B3 teto 5 sai de 0,5654 para 0,5746 e o
B3 teto 1000 sai de 0,5745 para 0,5711, enquanto o B0 fica em 0,5943 contra
0,5944. A conclusão não vira, mas os números da mensagem enviada ao José Ricardo
são de n=3 e já não são os melhores disponíveis.
