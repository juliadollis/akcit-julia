# Todos os resultados

Todas as tabelas são **geradas por script** a partir dos JSONs, nunca
transcritas à mão. Os geradores são `compara_zeroshot.py` e
`contraste_controle.py`.

---

## 1. A tabela principal, com n fixo em 6

Split de **teste** (485 imagens, 13 cenas), seeds 0 a 5.

| braço | n | F-borda ↑ | fmax ↑ | AbsRel ↓ | delta1 ↑ | RMSE ↓ |
|---|---|---|---|---|---|---|
| zero-shot (sem fine-tune) | - | 0,5402 | 0,7674 | 0,3602 | 0,6594 | 5,4002 |
| **B0 berHu (controle)** | 6 | **0,5954** | **0,7791** | 0,2509 | 0,6946 | 4,2433 |
| B1 curvatura só, teto 5 | 6 | 0,5604 | 0,7646 | 0,2735 | 0,6873 | 4,5664 |
| B1 curvatura só, teto 1000 | 6 | 0,5339 | 0,7230 | 0,2758 | 0,6901 | 4,6544 |
| B3 normal+curvatura, teto 5 | 6 | 0,5732 | 0,7678 | 0,2772 | 0,6943 | 4,4440 |
| B3 normal+curvatura, teto 50 | 6 | 0,5765 | 0,7713 | 0,2733 | 0,6960 | 4,4016 |
| B3 normal+curvatura, teto 1000 | 6 | 0,5704 | 0,7632 | 0,2932 | 0,6894 | 4,5513 |
| B0 berHu em 768 px | 3 | 0,5918 | 0,7463 | **0,2500** | **0,7000** | **4,1258** |

---

## 2. Contraste pareado contra o controle

Só as seeds que os dois lados têm.

### 2.1 Com n=6

| braço | n | delta F-borda | desvio | a favor | delta fmax |
|---|---|---|---|---|---|
| B1 teto 1000 | 6 | -0,0615 | 0,0104 | 0/6 | -0,0561 |
| B1 teto 5 | 6 | -0,0351 | 0,0126 | 0/6 | -0,0145 |
| B3 teto 1000 | 6 | -0,0250 | 0,0101 | 0/6 | -0,0159 |
| B3 teto 5 | 6 | -0,0222 | 0,0185 | 1/6 | -0,0113 |
| B3 teto 50 | 6 | -0,0190 | 0,0263 | 2/6 | -0,0078 |

**4 vitórias em 30 comparações.**

Na **área sob a varredura de limiar** (`f_auc`), que não depende de escolher um
ponto de operação, os cinco braços perdem em **6 de 6 seeds cada**: 30
comparações, **zero vitórias**.

### 2.2 Com o dobro de execuções

Como os braços com curvatura não reproduzem, cada retreino virou uma amostra
independente e dobrou o n.

| braço | n pares | delta F-borda | desvio | a favor |
|---|---|---|---|---|
| B1 teto 1000 | 12 | -0,0596 | 0,0078 | **0/12** |
| B1 teto 5 | 11 | -0,0378 | 0,0156 | **0/11** |
| B3 teto 1000 | 12 | -0,0254 | 0,0132 | 1/12 |
| B3 teto 5 | 12 | -0,0227 | 0,0195 | 2/12 |

**3 vitórias em 47.** Os deltas mal se mexeram, o que indica que a estimativa já
estava estável. O que mudou foi a confiança nela.

---

## 3. As três leituras

**Existe headroom no Spring.** Todo braço treinado bate o zero-shot. Em AbsRel e
RMSE isso vale até na pior seed. É o oposto do Hypersim, que está na lista de
treino do DepthPro, marcado como "Train, Val", e aparece no estágio 2 deles, que
aplica MALE, supervisão de segunda ordem feita para afiar borda.

**A curvatura não ajuda, e o B1 mostra o que ela custa.** O B1, que a isola,
perde mais (-0,0378 e -0,0596) que o B3 (-0,0227 e -0,0254). O termo normal não
salva a curvatura, **mascara parte do estrago**. No teto 1000 o B1 fica abaixo do
zero-shot no F-borda (0,5339 contra 0,5402): piora a borda em relação a não
treinar nada.

**O teto não é o confundidor.** O B3 nos três tetos:

| teto | saturação medida | delta |
|---|---|---|
| 5 | 49,7% | -0,0222 |
| 50 | 29,4% | -0,0190 |
| 1000 | 9,1% | -0,0250 |

Plano ao longo de **200x** de variação. Vale registrar que o teto 50 deu +0,0043
com n=2, -0,0071 com n=3 e -0,0190 com n=6: um bom lembrete do que n pequeno faz.

---

## 4. A ablação: os termos que nunca testamos

Código do Wallisson, defaults dele, **split de validação** (184 imagens, 6
sequências), `metrics.py` sem a máscara, `align-mode detach`, 30 épocas, n=1.

| config | F-borda | f_auc | AbsRel | delta vs controle |
|---|---|---|---|---|
| **B1 berhu+grad** | 0,5314 | 0,5540 | 0,3341 | **+0,0302** |
| **B1 berhu+metric** | 0,5287 | 0,5515 | 0,2639 | **+0,0276** |
| B1 berhu+geod | 0,5163 | 0,5472 | 0,2893 | +0,0152 |
| B0 berhu (controle) | 0,5011 | 0,5326 | 0,2645 | — |
| B7 gaussheavy | 0,4940 | 0,5087 | 0,2954 | -0,0071 |
| zero-shot | 0,4861 | 0,5132 | 0,3075 | -0,0150 |
| **B1 berhu+normal** | 0,4315 | 0,4518 | 0,3051 | **-0,0696** |
| **B1 berhu+gauss** | 0,4311 | 0,4669 | 0,2921 | **-0,0700** |

**Os dois termos que a nossa campanha inteira usou são os dois piores, e ficam
abaixo do zero-shot. Três que nunca testamos batem o controle.** A `f_auc` dá a
mesma ordem, então não é artefato de limiar.

A conclusão certa não é "geometria não ajuda". É que **escolhemos os dois piores
termos da família**.

**Ressalvas:** n=1 por config, e o ruído de reexecução é 0,0148. A separação
entre topo (0,5314) e fundo (0,4311) é 0,10 e sobrevive; `grad` contra `metric`
contra `geod` estão dentro de 0,015 e **não podem ser ordenados**. E estes
números não se comparam com os da seção 1, por causa do split, da máscara e do
`align-mode`.

---

## 5. Reprodutibilidade do retreino

| | \|Δ\| médio no F-borda | melhor época igual |
|---|---|---|
| B0, sem termo geométrico | **0,0000** | 6/6 |
| B1 e B3, com curvatura | 0,0148 (máx 0,0389) | 0/7 |

O B0 reproduziu **24 de 24 métricas idênticas até a quarta casa**, com a mesma
época de early stop. Detalhe da causa em `04-ACHADOS-TECNICOS.md`.

---

## 6. O que os dados NÃO sustentam

**Que o fine-tuning melhore a borda de verdade.** O `fmax` fica em ~0,77 para
tudo, inclusive para o modelo sem fine-tune.

**Precisão estatística.** O n efetivo do teste é 13 cenas. O erro padrão da média
é 0,0595, da mesma ordem do maior efeito medido.

**Que o teste seja limpo.** Duas cenas, 21% do conjunto, dominam. Excluindo-as, o
AbsRel do zero-shot cai pela metade.

**Que o teste represente o treino.** Mediana de profundidade: 23,0 m no treino,
23,7 m na validação, **10,4 m no teste**. Acidente do sorteio por sequência.

**Significância.** Com n=6 e desvio dos deltas entre 0,010 e 0,026, o que
sustenta a leitura é a consistência do sinal, não um p-valor.

---

## 7. O passo 3: a distribuição de |K| no Spring

37 sequências, 517 quadros, **135,5 milhões de pixels**, com o `fx` de cada
sequência.

| percentil | \|K\| (1/m²) |
|---|---|
| p50 | 4,41 |
| p90 | 987 |
| p99 | 68.745 |
| p99,9 | 3.230.846 |

| teto | % saturado | quanto o top 0,1% domina depois do clamp |
|---|---|---|
| 5 | 49,7% | 0,18% |
| 50 | 29,4% | 0,26% |
| 500 | 12,6% | 0,51% |
| **1000** | **9,1%** | 0,66% |
| 5000 | 4,0% | 1,33% |

Duas hipóteses sobre a cauda, **ambas refutadas**: não são as bordas (pixels de
degrau têm p50 de \|K\| = 0,003, e a cauda vive nas regiões mais planas) e não é
quantização no fundo (\|K\| é maior perto: p50 = 237,7 a 0-5 m contra 0,83 a
40-80 m). Como K = 1/R², p50 = 237 a 0-5 m equivale a raio de 6,5 cm, que é dobra
de pano ou folhagem: **boa parte da cauda é geometria real**.

E o ponto mecânico: sem teto, **0,01% dos pixels carregam 99,71% da soma de
\|K\|**. Com teto baixo, onde os dois lados saturam a diferença é exatamente zero
e o pixel não gera gradiente. Não existe teto escalar bom; é um dilema, não um
ajuste. Uma alternativa medida mas não testada em treino: com `log(1+|K|)` a
amplitude cai de 698.000x para **8,5x**.
