# Resultados finais: só os modelos que ainda existem

Todo número aqui tem um `best.pt` publicado no Hugging Face, em
`akcit-dephpro/depthpro-riemann-modelos`. Isso é deliberado.

Em 2026-09-10 uma liberação de quota no cluster apagou **37 de 47**
checkpoints. As métricas sobreviveram, os pesos não. Um número cujo peso
não existe mais **não pode ser reavaliado, reauditado nem compartilhado**,
então ele não entra numa tabela de resultado. Os 37 originais aparecem
apenas na seção 5, como nota de concordância.

Split de **teste**: 485 imagens de **13 cenas**. Avaliação pelo mesmo
`Trainer.validate()` que gerou o número de cada seed no fim do treino.

---

## 1. Tabela principal

| braço | perda | fonte | **n** | F-borda | fmax | f_auc | AbsRel | delta1 | RMSE |
|---|---|---|---|---|---|---|---|---|---|
| zero-shot | sem fine-tune | - | - | 0.5402 | 0.7674 | 0.5641 | 0.3602 | 0.6594 | 5.4002 |
| Bgrad — berHu + grad 0,3 | berHu 0,7 + grad 0,3 | novo | **6** | 0.6091 | 0.7882 | 0.6253 | 0.2561 | 0.6937 | 4.3492 |
| Bgeod — berHu + geod 0,1 | berHu 0,7 + geod 0,1 | novo | **6** | 0.5984 | 0.7790 | 0.6308 | 0.2481 | 0.6950 | 4.2336 |
| B0 berHu (controle) | berHu 0,7 | retreino | **8** | 0.5943 | 0.7760 | 0.6244 | 0.2508 | 0.6940 | 4.2466 |
| B0 berHu em 768 px | berHu 0,7, treinado a 768 px | sobrevivente | **3** | 0.5918 | 0.7463 | 0.6041 | 0.2500 | 0.7000 | 4.1258 |
| Bmetric — berHu + metric 0,2 | berHu 0,7 + metric 0,2 | novo | **6** | 0.5904 | 0.7681 | 0.6202 | 0.2489 | 0.6958 | 4.2604 |
| B3 normal+curvatura, teto 50 | berHu 0,7 + normal 0,9 + curvatura 0,45, teto 50 | sobrevivente | **6** | 0.5765 | 0.7713 | 0.5906 | 0.2733 | 0.6960 | 4.4016 |
| B3 normal+curvatura, teto 1000 | berHu 0,7 + normal 0,9 + curvatura 0,45, teto 1000 | retreino | **10** | 0.5721 | 0.7657 | 0.5823 | 0.3021 | 0.6793 | 4.7207 |
| B3 normal+curvatura, teto 5 | berHu 0,7 + normal 0,9 + curvatura 0,45, teto 5 | retreino | **8** | 0.5691 | 0.7663 | 0.5848 | 0.2785 | 0.6912 | 4.4061 |
| B1 curvatura só, teto 5 | berHu 0,7 + curvatura 0,45, teto 5 | retreino | **5** | 0.5535 | 0.7587 | 0.5917 | 0.2550 | 0.7024 | 4.3574 |
| B1 curvatura só, teto 5 | berHu 0,7 + curvatura 0,45, teto 5 | sobrevivente | **1** | 0.5462 | 0.7424 | 0.5648 | 0.3369 | 0.6453 | 5.3503 |
| B1 curvatura só, teto 1000 | berHu 0,7 + curvatura 0,45, teto 1000 | retreino | **6** | 0.5378 | 0.7161 | 0.5697 | 0.2520 | 0.7057 | 4.3756 |

---

## 2. Contraste pareado contra o controle

O controle é o **B0 berHu retreinado**, não o original. Só entram as
seeds que os dois lados têm, e o `n` de cada linha é esse conjunto.

| braço | fonte | n | delta F-borda | desvio | a favor | delta fmax |
|---|---|---|---|---|---|---|
| Bgrad — berHu + grad 0,3 | novo | 6 | +0.0137 | 0.0084 | 6/6 | +0.0091 |
| Bgeod — berHu + geod 0,1 | novo | 6 | +0.0030 | 0.0072 | 4/6 | -0.0000 |
| B0 berHu em 768 px | sobrevivente | 3 | -0.0026 | 0.0094 | 2/3 | -0.0344 |
| Bmetric — berHu + metric 0,2 | novo | 6 | -0.0050 | 0.0038 | 1/6 | -0.0109 |
| B3 normal+curvatura, teto 50 | sobrevivente | 6 | -0.0190 | 0.0263 | 2/6 | -0.0078 |
| B3 normal+curvatura, teto 1000 | retreino | 8 | -0.0208 | 0.0177 | 2/8 | -0.0086 |
| B3 normal+curvatura, teto 5 | retreino | 6 | -0.0231 | 0.0223 | 1/6 | -0.0101 |
| B1 curvatura só, teto 5 | retreino | 5 | -0.0411 | 0.0197 | 0/5 | -0.0207 |
| B1 curvatura só, teto 5 | sobrevivente | 1 | -0.0533 | n/d | 0/1 | -0.0354 |
| B1 curvatura só, teto 1000 | retreino | 6 | -0.0576 | 0.0042 | 0/6 | -0.0630 |

**18 vitórias em 53 comparações pareadas.**

Só os braços com curvatura: **5 em 32**.

`Bgrad`: **6 em 6**.
`Bgeod`: **4 em 6**.
`Bmetric`: **1 em 6**.

---

## 3. A leitura

**Existe headroom no Spring.** Todo braço treinado bate o zero-shot, e em
AbsRel a diferença é grande (0,3602 contra ~0,25). É o oposto do Hypersim,
que está na lista de treino do DepthPro e por isso não tinha espaço a tomar.

**A curvatura piora a borda.** O braço B1, que a isola, perde mais que o B3,
que a mistura com o termo normal. Ou seja, o normal **mascara parte do
estrago** em vez de ajudar.

**O `grad` é o único ganho.** E é a única intervenção de toda a campanha que
moveu o `fmax`, o F-score no melhor limiar de cada modelo. Esse teto de
~0,77 não cedeu a seis configurações de perda nem a 768 px de resolução.

---

## 4. O que estes números NÃO sustentam

**O n efetivo é 13 cenas, não 485 imagens.** Quadros consecutivos da mesma
sequência são quase o mesmo dado. O desvio entre cenas é 0,2147 e o erro
padrão da média é **0,0595**, da mesma ordem do maior efeito medido.

**Duas cenas dominam.** A `seq0020` (77 quadros, 16% do teste) tem
`d1 = 0,05`, e a `seq0043` tem `AbsRel = 2,28`. Excluindo as duas, o AbsRel
do zero-shot cai de 0,3591 para 0,1637.

**Os braços com curvatura não são reprodutíveis.** O backward do
`F.pad(mode="replicate")` na CUDA usa `atomicAdd`. Medido: |delta| médio de
0,0148 no F-borda entre duas execuções da mesma seed, contra 0,0000 no
berHu puro. O n maior é o que absorve isso.

**Não há teste de significância.** O que sustenta a leitura é a
consistência do sinal, não um p-valor.

---

## 5. Nota de concordância: a campanha original

Os 37 checkpoints apagados deixaram métricas. Como o caminho geométrico não
é determinístico, original e retreino são **duas campanhas independentes**
da mesma configuração. Elas concordam:

| braço | delta original | delta retreino |
|---|---|---|
| B1 curvatura só, teto 1000 | -0.0615 (n=6) | -0.0576 (n=6) |
| B1 curvatura só, teto 5 | -0.0314 (n=5) | -0.0411 (n=5) |
| B3 normal+curvatura, teto 1000 | -0.0244 (n=8) | -0.0208 (n=8) |
| B3 normal+curvatura, teto 5 | -0.0222 (n=6) | -0.0231 (n=6) |

Mesma direção e mesma magnitude em duas execuções independentes. Isso é
evidência de que o nulo não é artefato de uma campanha, mas **não é dado a
somar**: os pesos originais não existem mais.

