# Ampliação de seeds: n=3 para n=10

Lançado em 2026-09-01 às 11:00 (14:00Z) no dgx-H100-01.

## Por que

O `CONCLUSAO.md` fecha um resultado NULO com n=3, e nulo com n=3 é a forma mais
fraca de nulo. Pior: as faixas dos braços se sobrepõem no F-score de borda.

- B0 (controle): 0,5986 / 0,5895 / 0,5952, mínimo **0,5895**
- B3 teto 1000: 0,5908 / 0,5614 / 0,5712, máximo **0,5908**

O máximo do B3 é maior que o mínimo do B0. Com três pontos, "o B0 vence" é uma
direção, não um fato, e é exatamente aí que o José Ricardo vai bater. Sete seeds
a mais por braço apertam o intervalo o suficiente para a afirmação parar de pé
ou cair de vez. Nos dois casos a resposta melhora.

## O que está rodando

21 treinos, seeds 3 a 9 de cada um dos três braços, configuração idêntica à das
seeds 0 a 2 (mesmo split, mesmos pesos, mesma focal, mesmo teto por braço).

| GPU | braço | seeds | runs |
|---|---|---|---|
| 0 | `B0_berhu` | 3 a 7 | 5 |
| 1 | `B3_gauss_metrica_teto1000` | 3 a 7 | 5 |
| 2 | `B3_gauss_metrica_teto5` | 3 a 7 | 5 |
| 7 | os três, seeds 8 e 9 | 8 a 9 | 6 |

Previsão: 9 a 12 horas, ou seja, madrugada de 2026-09-02.

## Garantias contra estrago

As seeds novas caem na MESMA pasta das antigas (é o ponto: elas têm de ser o
mesmo experimento), então a proteção não pode ser o nome da pasta. Ela é outra:

1. `train_single.py` pula qualquer seed que já tenha `test_metrics.json`. Uma
   seed concluída nunca é reescrita, nem por engano nem por relançamento.
2. As faixas passadas a cada GPU são disjuntas por construção, então dois
   processos nunca disputam a mesma seed.
3. O resumo passou a ser lido do disco, varrendo `seed_*/test_metrics.json`, em
   vez de acumulado em memória. Cada processo grava o resumo com TODAS as seeds
   prontas naquele momento, então o último a terminar grava o completo.

A fonte de verdade continua sendo os JSONs por seed. O `consolida_reteste.py`
recalcula tudo a partir deles, então o `RESULTADOS.md` não depende de nenhum
processo ter terminado numa ordem específica.

## Mudanças de código que isso exigiu

- `scripts/train_single.py`: argumento `--seed-inicio`, guarda anti-sobrescrita
  por seed, resumo lido do disco com `n_seeds` real e a lista de seeds.
- `passo4_seeds.sh` (no cluster): lançador por faixa, aceita várias tarefas por
  GPU.
- `scripts/consolida_reteste.py`: já corrigido antes para comparar todos os
  braços B3 contra o controle, não só o primeiro.

## Segunda frente: o braço que faltava (GPUs 3, 5 e 6)

Lançado às 11:49 (14:49Z).

O `B3` que testamos é berhu + normal + gauss. O `B0` é berhu puro. Então o nulo
do B3 contra o B0 mede **normal e gauss juntos**, e não diz de qual dos dois
veio. Não existia braço com a curvatura sozinha, ou seja, a hipótese central
nunca foi isolada.

`B1 = berhu + gauss_metrica`. B1 menos B0 é exatamente o termo de curvatura.

| GPU | braço | seeds |
|---|---|---|
| 3 | `B1_gauss_metrica_teto1000` | 0 a 3 |
| 5 | `B1_gauss_metrica_teto5` | 0 a 3 |
| 6 | os dois tetos | 4 e 5 |

n=6 por teto. Se o B1 também empatar ou perder para o B0, a hipótese cai de
forma direta, sem depender do termo de normais. Se o B1 ganhar e o B3 perder, o
problema era a combinação, não a curvatura, e isso muda a resposta inteira.

Confirmei que passa da primeira época antes de escalar. Detalhe observado: com
teto 1000 o `train_raw` começa em ~204 contra ~10 do teto 5, o que é o esperado
(a curvatura crua é enorme) e é absorvido pela normalização por termo, com
`train_total` em ~1,09 nos dois.

## Disco: atenção

A cota é 500 GB soft / 600 GB hard (uid 1089) e o uso estava em **492 GB** no
momento do lançamento. Cada seed grava um `best.pt` de 1,37 GB.

- 33 treinos em voo x 1,3 GB = ~43 GB de checkpoints
- projeção ao fim: **~530 GB**

Isso **passa do soft e fica abaixo do hard**, então os treinos terminam. O que
acontece ao cruzar 500 GB é começar a contar o grace do soft, e o grace é de
dias enquanto os treinos precisam de ~11 horas. Não há risco para esta rodada.

Não apaguei nada. Candidatos a liberar espaço, se for preciso, em ordem de
segurança aparente:

| candidato | tamanho | evidência |
|---|---|---|
| `hf-cache/hub` | 54 GB | sem toque desde 16/jun, substituído pelo `hf-cache-julia` |
| `data/spring_zips` | 23 GB | zips originais, já extraídos em `data/spring` |
| `hf-cache-julia/hub` | 180 GB | último toque 31/ago |
| `hf-cache-julia/datasets` | 108 GB | usado hoje pela fila de avaliação, NÃO mexer |

## GPU 7

Estava presa atrás do keeper da fila de avaliação do paper, que girava em vazio
(fila esgotada, todos os itens saindo em 30 segundos com "já está na tabela").
Usei o mecanismo previsto para isso, a flag `PARAR_KEEPER_7`, que o
`vigia_keepers.sh` respeita justamente para o treino poder tomar a GPU. Para
devolver a GPU à fila de avaliação: apagar a flag e religar o keeper.

As GPUs 3, 5 e 6 receberam o mesmo tratamento às 11:48, pelo mesmo motivo e pelo
mesmo mecanismo. Todos os keepers da fila de avaliação estão parados por flag,
nenhum foi morto e nenhum arquivo foi apagado. Para devolver qualquer GPU à fila
de avaliação: apagar a flag `PARAR_KEEPER_<n>` e religar o keeper.

## Quando terminar

1. `rsync` das seeds novas para cá
2. `python scripts/consolida_reteste.py resultados/reteste_curvatura_2026-08-31`
3. revisar o `CONCLUSAO.md`: os números mudam, a leitura pode mudar junto

## DIODE: destravado tecnicamente, travado por disco

Investiguei o DIODE como segundo dataset. Duas descobertas:

**O bloqueio técnico não existe.** `gauss_loss_metrica` exige `fx > 0` e o
`prepare_diode.py` não extrai intrínseco nenhum, o que parecia inviabilizar a
curvatura métrica lá. Mas o devkit oficial publica um `intrinsics.txt`:

```
fx = 886.81   fy = 927.06   cx = 512   cy = 384      (imagens 1024x768)
```

É uma câmera computacional única, usada para gerar todos os recortes a partir
dos scans, então o mesmo intrínseco vale para o dataset inteiro. Isso é até mais
simples que o Spring, onde o `fx` varia por sequência.

**O bloqueio é disco.** O DIODE Depth train tem 81 GB comprimidos, e indoor e
outdoor vêm misturados no mesmo tarball (a separação é por lista de arquivos).
Com a projeção de ~530 GB ao fim desta rodada e o teto duro em 600 GB, sobram
~70 GB, e não cabe. Treinar no DIODE exige liberar ~100 GB antes.

O `val` sozinho tem 2,6 GB e cabe, mas serve só para avaliação cruzada, não para
treinar os braços.

Fonte dos intrínsecos: https://github.com/diode-dataset/diode-devkit/blob/master/intrinsics.txt
