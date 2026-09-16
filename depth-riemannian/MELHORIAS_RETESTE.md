# Reteste da curvatura — pontos para discutir com o José Ricardo

Levantados ao executar os passos 1 a 3 do `RETESTE_CURVATURA.md` no Spring completo
(37 sequências, 517 quadros preparados, 135 milhões de pixels).

Nada aqui bloqueou a execução. O passo 4 rodou exatamente como pedido. Esta lista é
o que vale mudar **depois**, para a próxima rodada.

---

## 1. O `medir_k_spring.py` assume uma focal única, e o Spring não tem uma

O script recebe um `--fx-orig` e aplica a todas as imagens. No Spring o `fx` varia
**4,7x** entre as 37 sequências: 1292,9 a 6060,6, com 13 valores distintos.

Medido numa esfera sintética de K conhecido (K = 1/R² = 0,25):

| sequência | fx real a 512 | K com fx certo | K com a mediana | erro |
|---|---|---|---|---|
| mais aberta | 344,8 px | 0,2501 | 2,0765 | 8,30x para cima |
| mais fechada | 1616,2 px | 0,2504 | 0,0099 | 25x para baixo |

**Mas o efeito nos percentis agregados é pequeno** (0,91x a 1,19x): os erros por
sequência se cancelam no conjunto. Ou seja, a escolha do teto é robusta a isso.

Sugestão: aceitar focal por sequência. Deixei `medir_k_spring_porseq.py` como
referência — lê o `fx` do `intrinsics.txt` de cada sequência e reporta as duas
versões lado a lado.

## 2. A expectativa de "5,0 fica folgado" não se confirmou

O documento previa que o teto 5,0 ficaria folgado contra os 40% de saturação
anteriores. Medido: **48,97% dos pixels acima de 5** — pior que antes.

| percentil | \|K\| (1/m²) |
|---|---|
| p50 | 4,41 |
| p90 | 987 |
| p99 | 68.745 |
| p99,9 | 3.230.846 |

A mediana ficou em ordem de unidade, como previsto. A cauda é que não.

## 3. A cauda NÃO é o que se esperaria, e isso muda o diagnóstico

Duas hipóteses testadas e **as duas refutadas, com o resultado invertido**:

- **Não são as bordas.** Pixels com degrau forte de profundidade têm \|K\| minúsculo
  (p50 = 0,003). A cauda vive nas regiões mais planas. Remover bordas *piora*.
- **Não é quantização da disparidade no fundo.** \|K\| é maior perto e menor longe:
  0-5 m dá p50 = 237,7; 40-80 m dá p50 = 0,83.

Como K = 1/R², geometria fina de perto tem K genuinamente alto — p50 = 237 a 0-5 m
equivale a raio de 6,5 cm, que é dobra de pano, folhagem, mão de personagem.
**Boa parte da cauda é geometria real, não artefato.** Um teto de 5 cortaria metade
do sinal verdadeiro.

## 4. O ponto principal: um teto escalar num L1 não tem como funcionar bem

A `gauss_loss_metrica` é `mean(|K_pred − K_alvo|)` com o clamp aplicado **nos dois
lados antes da subtração**. Daí duas consequências mecânicas opostas:

**Sem teto** — a amplitude é de 5,8 ordens de grandeza (p50 = 4,8; p99,9 = 3,4 M).
Medido: **0,01% dos pixels (13.552 de 135 milhões) carregam 99,71% da soma de \|K\|.**
Um L1 sobre isso não é uma perda sobre a imagem, é uma perda sobre treze mil pixels.

**Com teto baixo** — onde predição e alvo saturam do mesmo lado, a diferença é
exatamente zero e o pixel **não gera gradiente nenhum**.

| teto | % saturado | quanto o top 0,1% domina depois do clamp |
|---|---|---|
| 5 | 49,73% | 0,18% |
| 50 | 29,44% | 0,26% |
| 500 | 12,61% | 0,51% |
| **1000** | **9,13%** | **0,66%** |
| 5000 | 4,01% | 1,33% |

Em 1000 não há tensão entre os dois objetivos: 9% de saturação **e** perda ainda bem
distribuída. Foi o teto usado.

**Alternativa estrutural:** com `log(1+|K|)` a amplitude cai de 698.000x para **8,5x**
e o top 1% passa a carregar 4,8% em vez de 99,99%. Muito melhor condicionado, mas é
mudar a forma da perda, não um número — e afasta da comparação com o histórico.

## 5. Acima de 80 m há valores numericamente impossíveis

Nessa faixa (3,6% dos pixels) o p99,9 dá raio de curvatura de **41 micrômetros**.
É pouco pixel, mas é lixo puro entrando na perda. Vale mascarar.

## 6. Confundimento com o histórico

As rodadas antigas usaram teto 50 **com a fórmula quebrada**. A rodada nova troca
fórmula e teto ao mesmo tempo. Se der nulo, não dá para separar "a hipótese falhou"
de "o regime de saturação mudou". Uma rodada extra de B3 com teto 50 isolaria isso.

## 7. Coisas pequenas encontradas na execução

- **`torch.quantile` estoura acima de ~16M elementos.** O `medir_k_spring.py` lê só
  10 lotes (5M pixels) e escapa, mas quebra se alguém aumentar `--n-lotes`. Usar
  `numpy.percentile` resolve.
- **Cobertura da máscara veio 100,0%.** O `GUIA_SPRING.md` diz para esperar alto mas
  não 100%, porque o céu é removido. Vale conferir se a remoção está ativa.
- **Os zips do DaRUS trazem um nível `spring/` a mais.** Descompactando em
  `<raiz>/train` o resultado fica em `<raiz>/train/spring/train/<seq>`. Vale um aviso
  no `GUIA_SPRING.md`, seção 5.1.
- **O Spring não precisa ser baixado inteiro para os passos 1 a 3.** O passo 3 lê
  poucas imagens. Dá para puxar por faixa de bytes só os membros necessários: usamos
  894 MB em 10 minutos em vez de 24 GB em 2 horas (`zip_remoto.py`).

---

## Scripts deixados na máquina

Todos em `depth-riemannian/scripts/`, nenhum altera arquivo existente:

| arquivo | o que faz |
|---|---|
| `medir_k_spring_porseq.py` | passo 3 com focal por sequência, compara com a mediana única |
| `diag_cauda_k.py` | testa a hipótese "a cauda vem das bordas" (refutada) |
| `diag_cauda_k2.py` | testa a hipótese "a cauda vem do fundo quantizado" (refutada) |
| `diag_dominancia.py` | concentração da perda e o dilema do teto |
| `zip_remoto.py` | extrai membros escolhidos de um zip remoto, sem baixar tudo |
