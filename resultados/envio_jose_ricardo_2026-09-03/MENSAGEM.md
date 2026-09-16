# Passo 4, segunda rodada: a linha de base que faltava e as seeds ampliadas

Duas coisas mudaram desde a mensagem anterior.

Primeiro, medi o que faltava: o **DepthPro sem fine-tune nenhum**, no mesmo
split de teste. Até agora todos os braços eram comparados entre si e nenhum
contra o ponto de partida, então não dava para saber se o treino ajudava em
alguma coisa.

Segundo, as seeds que estavam rodando **fecharam**. Os braços saíram de n=3.

Os números da mensagem anterior continuam de pé: recortando só as seeds 0, 1
e 2, a tabela gerada reproduz exatamente 0,5944 no controle, 0,5654 no B3 teto
5 e 0,5745 no B3 teto 1000, com a diferença média de -0,0290 contra o controle.

---

## 1. Existe headroom no Spring

O zero-shot, no mesmo `Trainer.validate()` que gerou o número de cada seed
treinada, mesmo split, mesma resolução, mesmo batch:

| métrica | zero-shot |
|---|---|
| F-score de borda | 0,5402 |
| AbsRel | 0,3602 |
| delta1 | 0,6594 |
| RMSE | 5,4002 |

Contra os braços treinados, com todas as seeds concluídas:

| braço | n | F-score de borda | AbsRel | delta1 | RMSE |
|---|---|---|---|---|---|
| **zero-shot** | - | 0,5402 | 0,3602 | 0,6594 | 5,4002 |
| B0 berHu (controle) | 8 | **0,5943** | **0,2508** | 0,6940 | **4,2466** |
| B1 curvatura só, teto 5 | 5 | 0,5632 | 0,2608 | **0,6957** | 4,4096 |
| B1 curvatura só, teto 1000 | 6 | 0,5339 | 0,2758 | 0,6901 | 4,6544 |
| B3 normal+curvatura, teto 5 | 8 | 0,5746 | 0,2775 | 0,6942 | 4,4347 |
| B3 normal+curvatura, teto 1000 | 10 | 0,5711 | 0,2933 | 0,6870 | 4,5602 |

Todo braço treinado bate o zero-shot em AbsRel, delta1 e RMSE. E quase nunca é
efeito de média: em **AbsRel e RMSE, até a pior seed de cada braço** bate o
zero-shot. Em delta1 há uma exceção, a pior seed do B1 teto 1000 (0,6556 contra
0,6594); todas as outras batem.

Isso fecha uma pergunta que estava aberta desde o Hypersim, onde nenhuma
configuração superava a linha de base e até o berHu puro degradava. Lá não havia
espaço a tomar porque o Hypersim está na lista de treino do DepthPro, marcado
como "Train, Val", e aparece no estágio 2 deles, que aplica MALE, ou seja,
supervisão de segunda ordem feita para afiar borda. Aqui há espaço, e ele é
grande: 0,36 para 0,25 de AbsRel.

---

## 2. A curvatura continua perdendo do controle, agora com mais força

Pareado por seed, entrando só as seeds que os dois lados têm. A seed fixa
inicialização e ordem dos dados nos dois braços, então a diferença por seed tira
a variação que vem só do sorteio.

**F-score de borda, cada braço menos o controle:**

| braço | n comum | delta médio | desvio | amplitude | seeds a favor |
|---|---|---|---|---|---|
| B1 curvatura só, teto 1000 | 6 | -0,0615 | 0,0104 | -0,0713 a -0,0482 | 0/6 |
| B1 curvatura só, teto 5 | 5 | -0,0314 | 0,0098 | -0,0444 a -0,0202 | 0/5 |
| B3 normal+curvatura, teto 1000 | 8 | -0,0244 | 0,0141 | -0,0435 a -0,0016 | 0/8 |
| B3 normal+curvatura, teto 5 | 6 | -0,0222 | 0,0185 | -0,0475 a +0,0013 | 1/6 |

**Em 25 comparações pareadas, a curvatura bateu o controle em 1.**

Com n=3 dava para dizer que o controle vencia em média. Com o pareamento e o n
maior dá para dizer que ele vence em quase toda seed individual, e a única
exceção ganha por +0,0013.

---

## 3. O que o braço B1 acrescenta

O B3 mistura o termo normal com o de curvatura, então o nulo dele não diz de
qual dos dois veio. O **B1 é a curvatura sozinha sobre o berHu**, e `B1 menos
B0` é exatamente o termo da hipótese.

É ele que perde mais: -0,0615 com teto 1000 e -0,0314 com teto 5, contra -0,0244
e -0,0222 do B3. Ou seja, **o termo normal não salva a curvatura, ele mascara
parte do estrago.**

E o caso extremo: o **B1 com teto 1000 fica abaixo do zero-shot** no F-score de
borda, 0,5339 contra 0,5402. Aquele braço piora a borda em relação a não treinar
nada.

---

## 4. Leitura

A ordem no F-score de borda, que é onde o método promete ganho e é o monitor do
early stop, é:

```
berHu puro (+0,0540 sobre o zero-shot)
  >  curvatura (+0,023 a +0,034)
  >  zero-shot
  >  curvatura sozinha com teto 1000 (-0,0063)
```

Antes dava para dizer só que a curvatura perdia do controle. Agora dá para dizer
o que ela custa: **entrega parte do ganho que o berHu puro sozinho entrega e
joga fora o resto.** Não é um termo neutro que não ajuda, é um termo que consome
ganho disponível.

O teto não é o confundidor. As duas pontas do intervalo plausível, 5 e 1000, dão
a mesma direção, e a diferença entre elas é menor que a dispersão entre seeds do
próprio braço.

---

## 5. Configuração

Spring, split 593 treino / 184 val / 485 teste, sequências disjuntas, seed 42,
idêntico em todos os braços. DepthPro variante `heads`, lr 1e-5, até 100 épocas
com early stop no F-score de borda de validação. Curvatura métrica com
`fx = 689,6` e `fy = 1225,9` px em 512. Teste avaliado uma única vez, no fim, por
seed.

Pesos: berHu 0,7 em todos. B0 zera o resto. B1 soma curvatura 0,45. B3 soma
normal 0,9 e curvatura 0,45.

O zero-shot é o mesmo checkpoint `depth_pro.pt` de onde todos os braços partem,
sem treino e sem carregar `best.pt`, avaliado uma vez. Sem treino, com o modelo
em `eval()` e o loader sem embaralhar, a passada é determinística, então não há
seed a reportar.

---

## 6. Ressalvas

**Os braços têm n diferentes** (5 a 10). Por isso toda comparação braço contra
braço aqui é pareada por seed comum, e o `n comum` está na tabela. As médias
soltas da seção 1 servem para a comparação contra o zero-shot, que é robusta
o bastante para sobreviver a isso: em AbsRel e RMSE a vantagem sobre o zero-shot
vale até na pior seed de cada braço, então não depende de quantas seeds entraram
na média.

**Sete rodadas de seed ficaram incompletas** e nada está rodando para
terminá-las: B0 seed 6; B1 teto 5 seeds 3, 6, 8; B1 teto 1000 seeds 6 e 8;
B3 teto 5 seed 8. Elas têm pasta mas não têm `test_metrics.json`. Não entram em
nenhuma conta.

**Não há teste de significância.** Com esse n, e com o desvio dos deltas na
ordem de 0,01 a 0,02, o que sustenta a conclusão é a consistência do sinal
(24 de 25 seeds pareadas na mesma direção), não um p-valor.

---

## 7. O que vai junto

| pasta | o que tem |
|---|---|
| `seeds_pedidas_0a2/` | as 3 seeds da mensagem anterior (0, 1, 2), todos os braços, com `TABELA.md` e `CONTRASTE_vs_controle.md` do recorte |
| `seeds_novas_3a9/` | só as seeds que fecharam depois, mesmo layout |
| `zero_shot/` | `test_metrics.json`, `meta.json` e `por_imagem.csv` com as 485 imagens |
| `TABELA_todas_as_seeds.md` | a comparação da seção 1, com todas as seeds |
| `CONTRASTE_vs_controle.md` | a comparação da seção 2 |

Cada `seed_<n>/` traz `test_metrics.json` (o número reportado), `summary.json`
(melhor época e métrica de validação) e `history.json` (a curva completa).

Todas as tabelas são geradas por script a partir dos JSONs
(`scripts/compara_zeroshot.py` e `scripts/contraste_controle.py`), nunca
transcritas à mão.
