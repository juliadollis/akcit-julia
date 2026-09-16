# Ablação no Spring: rodamos o teu código e deu resultado

Wallisson, rodamos a ablação que você mandou, **exatamente como você entregou**,
trocando só o dataset para o Spring. Segue o que saiu, três pontos do código que
precisam de decisão sua, e um achado que apareceu no meio e que muda o
dimensionamento de qualquer ablação futura.

---

## 1. O resultado, em uma frase

**Os dois termos que a nossa campanha inteira usou (`gauss` e `normal`) são os
dois piores da ablação, e ficam abaixo do zero-shot. Três termos que nunca
testamos (`grad`, `metric`, `geod`) batem o controle.**

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

A `f_auc`, que integra o F-score ao longo de todos os limiares e por isso não
depende de escolher um ponto de operação, dá **a mesma ordem**. Então a
separação não é artefato de limiar.

Falta a 9ª config (`B7_gauss_dom`), ainda rodando. Atualizamos quando fechar.

### Por que isso importa para nós

Nossa campanha no Spring foram **47 treinos** em cima de `gauss` e `normal`, com
a conclusão de que a curvatura não ajuda. Isso segue de pé: com n=6 a n=12 por
braço, são 3 vitórias em 47 comparações pareadas contra o controle.

O que a tua ablação mostra é que a conclusão certa **não** é "geometria não
ajuda". É que escolhemos os dois piores termos da família. Isso é outra história,
e é bem melhor.

---

## 2. Como rodamos

Código teu intocado, em `ablacao-wallisson/` na máquina, inclusive o
`metrics.py`. Nada do nosso repo foi misturado.

```bash
python scripts/run_ablation.py \
  --train-root /data/spring_split/train \
  --val-root   /data/spring_split/val \
  --checkpoint /models/checkpoints/depth_pro.pt \
  --out-dir    /workspace/runs/ablacao_spring \
  --focal 2585.859
```

Todos os teus defaults preservados: `--filter B0 B1 B7`, `--epochs 30`,
`--batch-size 2 --grad-accum 4 --grad-checkpointing`, `--align-mode detach`,
`--gauss-clamp-metrico 5.0`, `--eval-zero-shot`.

O **único** valor que informamos foi a focal, que o teu documento manda informar:
**2585,859**, a mediana que o nosso `prepare_spring.py` reporta para as 37
sequências do Spring.

O teu `Dockerfile` e o `requirements.docker.txt` são idênticos aos nossos, então
a imagem `riemann-depthpro:latest` é exatamente o teu ambiente. O smoke test
(`test_geometry_metrica.py`) passou com o teu código, todos os critérios.

Dados: Spring com split por sequências disjuntas, seed 42, frações
0,50/0,15/0,35, dando **593 treino / 184 validação / 485 teste**, de 18 / 6 / 13
sequências.

---

## 3. Três pontos do código que precisam de decisão sua

### 3.1 O `metrics.py` veio da `fix-geometry`, que está atrás da `main`

O teu `riemann/metrics.py` é **byte a byte igual ao `origin/fix-geometry`**. A
`main` tem uma correção que não está nele: a **máscara de validade** no
`boundary_fscore`.

O problema que ela resolve: o dataset zera os pixels inválidos, e a fronteira
entre região válida e os zeros vira uma **borda falsa no alvo** que a predição
(contínua) não tem. O efeito é precisão alta e recall derrubado. Está medido no
comentário do código da `main`: uma predição quase perfeita cai de bF-max 1,000
para 0,792 com apenas **1%** da imagem zerada.

No Spring isso não é detalhe. Medimos a máscara nas cenas de teste: **seq0045
tem 46% mascarado, seq0020 21%, seq0014 17%**.

Efeito prático visível no teu próprio resultado: o zero-shot dá **0,4861** aqui e
**0,5402** na nossa medição. É o mesmo modelo sem fine-tune; a diferença é toda
de protocolo.

Isso não invalida a tua ablação, porque as 9 configs passam pelas mesmas
condições e a pergunta dela é interna. Mas os números **não se comparam** com os
nossos, e com a máscara o ranking pode mudar, já que as cenas mais mascaradas são
justamente onde a métrica distorce.

### 3.2 O `gauss_clamp_metrico` padrão é 5,0, e no Spring isso satura metade

Medimos a distribuição de \|K\| no Spring completo: 37 sequências, 517 quadros,
**135,5 milhões de pixels**, com o `fx` de cada sequência.

| percentil de \|K\| (1/m²) | valor |
|---|---|
| p50 | 4,41 |
| p90 | 987 |
| p99 | 68.745 |
| p99,9 | 3.230.846 |

| teto | % de pixels saturados |
|---|---|
| **5** | **49,0%** |
| 50 | 29,5% |
| 1000 | 9,1% |

O teu comentário diz que o teto "vale só para estabilidade em descontinuidade;
não deve cortar o corpo da distribuição". No Spring, **5,0 corta metade**. E
onde os dois lados saturam, a diferença é exatamente zero e o pixel não gera
gradiente nenhum.

Vale dizer que testamos 5, 50 e 1000 na nossa campanha e o resultado é **plano**:
-0,0222, -0,0190 e -0,0250 contra o controle. Então o teto não salva a curvatura.
Mas para a ablação é melhor não rodar no pior ponto conhecido.

### 3.3 Falta o que prepara o Spring

A pasta não tem `prepare_spring.py` nem `medir_k_spring.py`, só os do Hypersim.
Nós temos os dois e o `/data/spring_split` já está pronto na máquina, então isso
não travou nada, mas para outra pessoa reproduzir seria bloqueio.

E o documento manda criar `{"fx": 2586.0}` por partição, ou seja **uma focal
só**. No Spring o `fx` varia **4,7x** entre as sequências (1292,9 a 6060,6). O
teu `dataset.py` já devolve `fx`/`fy` por amostra, então a capacidade existe; o
que falta é o `intrinsics.json` ser por sequência e não por partição.

Medimos o impacto disso numa esfera sintética de K conhecido: usar a mediana erra
**8,3x para cima** na sequência mais aberta e **25x para baixo** na mais fechada.
Nos percentis agregados o efeito se cancela (0,91x a 1,19x), mas por sequência
não.

---

## 4. O achado que muda o dimensionamento de qualquer ablação

Descobrimos isso retreinando os 37 checkpoints que perdemos numa limpeza de
quota, e ele afeta o teu trabalho diretamente.

**O caminho da curvatura não é reprodutível na GPU. O berHu puro é.**

Mesma seed, mesmo código, mesmo split, mesma máquina:

| | \|Δ\| médio no F-borda | máximo | melhor época igual |
|---|---|---|---|
| B0, sem termo geométrico | **0,0000** | 0,0000 | **6/6** |
| B1 e B3, com curvatura métrica | **0,0148** | **0,0389** | **0/7** |

A causa é o `F.pad(mode="replicate")` do `geometry.py` (linhas 67, 75, 91, 92),
dentro do `surface_curvatures`, que é chamado sobre o `pred` e portanto tem
backward. **O backward do replication padding na CUDA usa `atomicAdd` e é não
determinístico**, e é um caso documentado do PyTorch. O `metrics.py` também usa
replicate, mas sob `@torch.no_grad()`, então não contamina.

A correção é trocar por fatiamento e `torch.cat`, que dá o **mesmo forward** e um
backward determinístico.

### Por que isso é pré-requisito, e não detalhe

A ablação existe para **ordenar** configurações. Com o ruído atual, a mesma
config rodada duas vezes difere em 0,0148 no F-borda, às vezes 0,0389.

No teu resultado, `grad` (0,5314), `metric` (0,5287) e `geod` (0,5163) estão
dentro de **0,015** um do outro. **Não dá para ordená-los.** O que dá para
afirmar é que os três estão acima do controle e que `gauss` e `normal` estão bem
abaixo, porque essas separações (0,03 e 0,07) sobrevivem ao ruído.

Quantas seeds por config para distinguir uma diferença de 0,02:

| cenário | desvio | seeds/config | 39 configs | GPU em 2 placas |
|---|---|---|---|---|
| como está | 0,018 | 13 | 507 rodadas | ~17 dias |
| com o determinismo corrigido | 0,012 | 6 | 234 rodadas | ~9 dias |

Corrigir o padding corta o n pela metade. Mesmo assim, as 39 não cabem, e é por
isso que o teu default de `B0 B1 B7` foi uma boa escolha.

---

## 5. O que sugerimos

1. **Corrigir o `F.pad(mode="replicate")`** no `geometry.py`. Validar com duas
   execuções da mesma seed de um braço com curvatura: têm que bater exato, como o
   B0 bate.
2. **Rebasear na `main`**, principalmente pelo `metrics.py`.
3. **Rodar `grad`, `metric` e `geod` com n=6, no split de TESTE**, com a máscara
   e `align-mode full`. São 18 treinos, umas 30 h em duas placas. É o experimento
   que decide se o sinal positivo vira resultado.
4. `intrinsics.json` por sequência, e um teto medido em vez de 5,0.

O item 3 é o que interessa. Se `grad` ou `metric` se confirmarem com o nosso
protocolo, o trabalho deixa de ser um nulo e passa a ter um resultado positivo.

---

## 6. O que tem nesta pasta

| caminho | conteúdo |
|---|---|
| `TABELA_ABLACAO.md` | a tabela acima, gerada do CSV |
| `resultados/ablation_results.csv` | o CSV cru que o teu script produziu |
| `resultados/heads__<config>/summary.json` | melhor época e métrica de validação |
| `resultados/heads__<config>/history.json` | a curva completa, época a época |

Os `best.pt` de cada config (11 GB no total) ficaram na máquina e vão para o
Hugging Face. Se quiser algum, é só pedir.

**Ressalva de leitura, para os números não serem usados fora de lugar:** estes
resultados são do split de **validação** (184 imagens, 6 sequências), medidos com
`metrics.py` sem a máscara de validade e `align-mode detach`. O nosso relatório
usa o **teste** (485 imagens, 13 sequências), com máscara e `align full`. As duas
tabelas não se comparam entre si, só internamente.
