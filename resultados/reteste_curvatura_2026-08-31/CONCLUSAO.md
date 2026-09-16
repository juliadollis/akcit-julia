# Reteste da curvatura: o que os dois tetos decidem

Fechado em 2026-09-01. Números em `RESULTADOS.md` (gerado por
`scripts/consolida_reteste.py`, nunca transcrito à mão). Logs brutos em `logs/`.

> **Atenção: os números abaixo são de n=3 por braço.** Desde 2026-09-01 11:00
> estão rodando as seeds 3 a 9 dos três braços (ver `EM_ANDAMENTO.md`). Quando
> fecharem, reconsolidar e revisar esta leitura antes de mandar para fora. A
> seção "O que os dados NÃO sustentam" é a que mais deve mudar.

## A pergunta que faltava responder

O passo 4 rodou com `--gauss-clamp 1000`. Esse teto foi escolha minha, feita a
partir da medição do passo 3, e ia **contra** a expectativa do documento
(`RETESTE_CURVATURA.md` previa que 5,0 ficaria folgado). Com um só teto, o nulo
do B3 tinha duas leituras possíveis:

1. a hipótese da curvatura não ajuda; ou
2. a hipótese não foi testada de verdade, porque eu escolhi o teto errado.

O braço `teto5` existe para separar as duas. Configuração idêntica ao `teto1000`
em tudo (mesmo split train 593 / val 184 / test 485 seed 42, mesmos pesos
berhu 0,7 + normal 0,9 + gauss 0,45, mesma focal `fx=689.6 fy=1225.9` em 512,
mesmas 3 seeds). A única variável é o clamp.

## Resposta

**Teto 5 não resgata a hipótese. Perde do controle nas quatro métricas.**

| métrica | B0 (controle) | B3 teto 1000 | B3 teto 5 |
|---|---|---|---|
| F-score de borda (maior melhor) | **0,5944** | 0,5745 | 0,5654 |
| AbsRel (menor melhor) | **0,2507** | 0,2857 | 0,2806 |
| delta1 (maior melhor) | 0,6955 | **0,6977** | 0,6868 |
| RMSE (menor melhor) | **4,2276** | 4,4548 | 4,4744 |

Com teto 1000 o B3 ao menos empatava no delta1 (+0,0022). Com teto 5 ele perde
também nessa, e a perda no F-score de borda cresce de -0,0200 para -0,0290.

Isso elimina a leitura 2. O teto era o último confundidor que restava depois da
correção da curvatura métrica, e agora as duas pontas do intervalo plausível
dão a mesma direção. **O nulo é da hipótese, não da minha escolha de teto.**

## O que os dados NÃO sustentam

Que o teto 1000 seja melhor que o teto 5. A diferença entre os dois no F-score
de borda é de 0,0090, e a dispersão entre seeds do próprio teto 5 é de 0,0484
(0,5419 a 0,5903), cinco vezes maior. Com n=3 essa comparação é ruído. O que dá
para afirmar é o negativo: **nenhum dos dois tetos vira o resultado**, e é isso
que a pergunta pedia.

## Como ler o teto 5, mecanicamente

Vale registrar por que o teto 5 não era obviamente o certo. O clamp é aplicado
nos dois lados antes da subtração:

```
perda = média( |corta(K_predito) - corta(K_verdadeiro)| )
```

Num pixel onde a verdade é K=800 e o modelo prevê K=300, com teto 5 os dois
viram 5 e a perda registra zero: o pixel erra por 500 e não gera gradiente.

A medição do passo 3 (517 quadros do Spring, 135.528.448 pixels, com o `fx` de
cada sequência) diz quanto isso pesa:

- percentis de |K| (1/m²): p50 = 4,41 | p90 = 987 | p99 = 68.745 | p99,9 = 3.230.846
- fração de pixels acima do teto: >1 = 61,27% | **>5 = 48,97%** | >20 = 37,20% | >50 = 29,52%

Com teto 5, quase metade da imagem sai do treino. Foi por isso que escolhi 1000.
O experimento mostra que a escolha não mudou a conclusão, mas a razão para
tê-la feito continua de pé e deve ir junto na resposta.

## Ressalva que segue valendo

Os três nulos anteriores (Hypersim, DIODE, Spring) foram obtidos com o `h`
normalizado, ou seja, com um termo que não media curvatura. Aqueles resultados
não contam como teste da hipótese. Este conta: fórmula métrica correta, teto
controlado nas duas pontas, mesmo protocolo do controle. O nulo aqui é
defensável.

## Custo

Teto 5: 3 seeds, 12:18Z a 16:49Z de 2026-08-31 (4h31), GPU 2 do dgx-H100-01,
`rc=0`. Early stop nas épocas 42, 42 e 50 (melhores em 17, 17 e 25).
