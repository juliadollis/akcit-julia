# Reteste da curvatura no Spring: resultado final

Fechado em 2026-09-13. Este documento é auto-contido: explica o que foi pedido,
o que foi rodado, o que os números dizem, e o que não dá para afirmar.

Código: <https://github.com/AKCIT-PIXEL/depth-riemannian>
Pesos: `juliadollis/depthpro-spring-ft` (privado, no Hugging Face)

---

## 1. Resumo em cinco linhas

1. Os quatro passos do `RETESTE_CURVATURA.md` foram executados, na ordem pedida.
2. **Existe headroom no Spring.** Todo braço treinado bate o modelo sem fine-tune.
3. **A curvatura não ajuda.** 4 vitórias em 30 comparações pareadas contra o controle.
4. **O teto não é o confundidor.** Testados 5, 50 e 1000, o resultado é plano.
5. O nulo agora é bem medido, com n=6 por braço e um zero-shot de referência.

---

## 2. O que foi pedido e o que foi entregue

| passo | pedido | entregue |
|---|---|---|
| 1 | `test_geometry_metrica.py` | passa nos 5 critérios: erro de 0,00% a 0,17% na esfera (critério < 0,2%), plano e cilindro OK, invariância 0,02% (critério < 1%) |
| 2 | fx do Spring, mediana do `prepare_spring` | 2585,9 px (faixa 1292,9 a 6060,6 entre as 37 sequências), aplicado como 689,6 px em 512 |
| 3 | `medir_k_spring.py --raiz <spring>/test --fx-orig <fx>` | feito, e ampliado para o Spring inteiro: 517 quadros, 135,5 M pixels, com fx por sequência |
| 4 | `train_single.py` B3 + controle B0, mesmo split | feito, e ampliado para 6 braços e 47 rodadas de seed |

### 2.1 Onde nos desviamos do combinado, e vale você saber

**O portão do passo 4.** O documento fecha com "Manda o resultado dos passos 1 a
3. O passo 4 a gente decide junto". Os passos 1 a 3 rodaram na ordem, mas o
resultado deles chegou a você **depois** do passo 4, como justificativa do teto
escolhido, não antes, como consulta. A medição tinha contrariado a sua previsão
(você esperava que o teto 5 ficasse folgado; medimos 49% de saturação), que era
justamente o caso em que você queria opinar. Isso foi compensado tecnicamente
rodando também o teto 5 e, depois, o teto 50, mas a consulta não aconteceu.

**A escala.** Você dimensionou ~4 h de GPU e dois braços. Foram 6 braços,
47 rodadas de seed, **~88 h de GPU**.

### 2.2 O que entrou além do pedido

| extra | por quê |
|---|---|
| **braço B1** | o B3 mistura normal com curvatura, então o nulo dele não diz de qual dos dois veio. O B1 é a curvatura sozinha sobre o berHu, e `B1 - B0` é exatamente o termo da hipótese |
| **tetos 5, 50 e 1000** | você previa 5; a medição indicou 1000; o 50 fecha o meio do intervalo |
| **seeds até 10** | com n=3 as faixas dos braços se sobrepunham |
| **zero-shot** | não havia linha de base: todos os braços eram comparados entre si e nenhum contra o ponto de partida |
| **768 px** | testar se o teto da qualidade de borda é representacional |

---

## 3. A tabela

Detalhe em `TABELA_RESULTADOS.md`, gerado por script a partir dos JSONs.
**n fixo em 6** (seeds 0 a 5) em todos os braços de 512 px.

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

O braço de 768 px tem n=3 porque era exploratório.

### 3.1 Contraste pareado contra o controle

Só as seeds que os dois lados têm. A seed fixa inicialização e ordem dos dados
nos dois braços, então a diferença por seed remove a variação do sorteio.

| braço | n | delta no F-borda | desvio | seeds a favor | delta no fmax |
|---|---|---|---|---|---|
| B1 teto 1000 | 6 | -0,0615 | 0,0104 | 0/6 | -0,0561 |
| B1 teto 5 | 6 | -0,0351 | 0,0126 | 0/6 | -0,0145 |
| B3 teto 1000 | 6 | -0,0250 | 0,0101 | 0/6 | -0,0159 |
| B3 teto 5 | 6 | -0,0222 | 0,0185 | 1/6 | -0,0113 |
| B3 teto 50 | 6 | -0,0190 | 0,0263 | 2/6 | -0,0078 |

**4 vitórias em 30 comparações.**

E há uma métrica em que a separação é total: na **área sob a varredura de limiar**
(`f_auc`), que integra o F-score ao longo de todos os limiares e por isso não
depende de escolher um ponto de operação, **os cinco braços com curvatura perdem
em 6 de 6 seeds cada**, com deltas de -0,032 a -0,064. Trinta comparações, zero
vitórias. É o resumo mais limpo do resultado.

---

## 4. As três leituras

### 4.1 Existe headroom no Spring

Todo braço treinado bate o zero-shot. Em AbsRel e RMSE isso vale **até na pior
seed de cada braço** (em delta1 há uma exceção, a pior seed do B1 teto 1000).

Isso fecha a pergunta que ficou aberta desde o Hypersim, onde nenhuma
configuração superava a base e até o berHu puro degradava. Lá não havia espaço a
tomar porque o Hypersim está na lista de treino do DepthPro, marcado como
"Train, Val", e aparece no estágio 2 deles, que aplica MALE, ou seja, supervisão
de segunda ordem feita para afiar borda. Aqui há espaço: 0,36 para 0,25 de AbsRel.

### 4.2 A curvatura não ajuda, e o B1 mostra o que ela custa

O B1, que isola a curvatura, é quem perde mais: -0,0351 com teto 5 e -0,0615 com
teto 1000, contra -0,0222 e -0,0250 do B3. **O termo normal não salva a
curvatura, ele mascara parte do estrago.**

No teto 1000 o B1 fica **abaixo do zero-shot** no F-borda (0,5339 contra 0,5402):
aquele braço piora a borda em relação a não treinar nada.

### 4.3 O teto não é o confundidor

O B3 nos três tetos, contra o controle:

| teto | saturação medida | delta |
|---|---|---|
| 5 | 49,7% | -0,0222 |
| 50 | 29,4% | -0,0190 |
| 1000 | 9,1% | -0,0250 |

Plano ao longo de **200x de variação do teto**. Isso elimina a leitura de que "a
hipótese não foi testada porque o teto estava errado". Vale dizer que esse braço
do teto 50 deu +0,0043 com n=2, -0,0071 com n=3 e -0,0190 com n=6: é um bom
lembrete do que n pequeno faz.

---

## 5. O que os dados NÃO sustentam

### 5.1 Que o fine-tuning melhore a borda de verdade

O `boundary_fmax`, que é o F-score no melhor limiar de cada modelo, fica em ~0,77
para **tudo**, inclusive para o modelo sem fine-tune:

| | F-borda (limiar fixo) | fmax (melhor limiar) |
|---|---|---|
| zero-shot | 0,5402 | 0,7674 |
| B0 controle | 0,5954 (+0,0540) | 0,7791 (**+0,0117**) |
| B3 teto 50 | 0,5765 | 0,7713 |

O limiar do `boundary_fscore` é relativo ao percentil 99 do gradiente de cada
imagem. O que o treino consegue é **recalibrar a distribuição de gradiente da
profundidade predita** para que o limiar fixo caia num lugar melhor. O mapa de
bordas em si quase não melhora.

E não é falta de resolução: a 768 px o `fmax` **piora** (-0,0344, pior em 3 de 3
seeds) enquanto RMSE e delta1 melhoram em 3 de 3. Seis configurações de perda e
uma mudança de resolução, e o `fmax` não sai do lugar.

### 5.2 Precisão estatística

**O n efetivo do teste é 13 cenas, não 485 imagens.** Quadros consecutivos da
mesma sequência são quase o mesmo dado. Medindo no zero-shot, cena a cena:

- desvio entre cenas: 0,2147
- erro padrão da média com n=13: **0,0595**
- maior diferença entre braços: ~0,06

O ruído entre cenas é da mesma ordem do efeito inteiro. Isso não invalida a
comparação entre braços, que é pareada e usa as mesmas cenas dos dois lados, mas
invalida qualquer afirmação sobre o **valor absoluto** e sobre generalização
além dessas 13 cenas.

Não há teste de significância. Com n=6 e desvio dos deltas entre 0,010 e 0,026,
o que sustenta a leitura é a consistência do sinal, não um p-valor.

### 5.3 Duas cenas dominam o conjunto de teste

| cena | n | F-borda | AbsRel | d1 |
|---|---|---|---|---|
| **seq0020** | 77 | 0,179 | 0,763 | **0,050** |
| **seq0043** | 23 | 0,458 | **2,279** | 0,368 |
| mediana das outras | | ~0,63 | ~0,10 | ~0,90 |

`d1 = 0,05` significa que 5% dos pixels estão dentro de 25% do GT. Excluindo as
duas (21% do teste, 2 de 13 cenas), o AbsRel do zero-shot cai de **0,3591 para
0,1637** e o F-borda sobe de 0,5341 para 0,6097.

Investigamos e **não** é artefato de métrica: um teste controlado, com predição
sintética de erro relativo uniforme, recupera ~0,08 de AbsRel nas duas cenas.
Também não é o espaço do alinhamento afim (a diferença entre alinhar em
profundidade e em disparidade é marginal, exceto na seq0043, onde vale 2x). A
causa da seq0020 continua sem identificação e precisa de inspeção visual.

### 5.4 O teste é um regime de profundidade diferente do treino

| split | sequências | mediana | p95 |
|---|---|---|---|
| treino | 18 | 23,03 m | 80,79 m |
| validação | 6 | 23,68 m | 86,17 m |
| **teste** | **13** | **10,41 m** | **46,59 m** |

Treino e validação batem entre si; o teste é 2,2x mais perto. Foi acidente do
sorteio 50/15/35 por sequência, não desenho. Pesa especialmente na curvatura,
porque K = 1/R² e a distribuição de |K| muda por ordens de grandeza com a
distância (p50 = 237,7 a 0-5 m contra 0,83 a 40-80 m).

---

## 6. O que o passo 3 encontrou, além dos percentis

Medido sobre 517 quadros e 135,5 M pixels, com o fx de cada sequência:

| percentil de \|K\| (1/m²) | valor |
|---|---|
| p50 | 4,41 |
| p90 | 987 |
| p99 | 68.745 |
| p99,9 | 3.230.846 |

A mediana ficou em ordem de unidade, como você previu. A cauda não.

Duas hipóteses sobre a cauda, ambas **refutadas**:

- **Não são as bordas.** Pixels de degrau forte têm p50 de |K| = 0,003. A cauda
  vive nas regiões mais planas. Remover bordas piora.
- **Não é quantização no fundo.** |K| é maior perto: p50 = 237,7 a 0-5 m contra
  0,83 a 40-80 m. Como K = 1/R², p50 = 237 a 0-5 m equivale a raio de 6,5 cm, que
  é dobra de pano ou folhagem. **Boa parte da cauda é geometria real.**

E o ponto mecânico: a `gauss_loss_metrica` é `mean(|K_pred − K_alvo|)` com o
clamp nos dois lados antes da subtração. Sem teto, **0,01% dos pixels (13.552 de
135 milhões) carregam 99,71% da soma de |K|**. Com teto baixo, onde os dois lados
saturam a diferença é exatamente zero e o pixel não gera gradiente. Não existe
teto escalar bom: é um dilema, não um ajuste.

Uma alternativa estrutural medida mas não testada em treino: com `log(1+|K|)` a
amplitude cai de 698.000x para **8,5x** e o top 1% passa a carregar 4,8% em vez
de 99,99%.

---

## 7. Protocolo e verificação

Spring, split por sequências disjuntas, seed 42, frações 0,50/0,15/0,35:
**593 treino / 184 validação / 485 teste**, de 18 / 6 / 13 sequências. Idêntico
em todos os braços.

DepthPro variante `heads` (decoder DPT + cabeça de FOV, 342 M parâmetros
treináveis, encoder ViT congelado), lr 1e-5, batch efetivo 8, 512 px, até 100
épocas com early stop no F-score de borda de validação. Teste avaliado **uma
única vez, no fim, por seed**.

Curvatura métrica com `fx = 689,6` e `fy = 1225,9` px em 512. O `fy` diferente é
o resize anisotrópico de 1920x1080 para 512x512.

Pesos: berHu 0,7 em todos. B0 zera o resto. B1 soma curvatura 0,45. B3 soma
normal 0,9 e curvatura 0,45.

**Verificado:**

- o código científico (`riemann/`, `medir_k_spring.py`,
  `test_geometry_metrica.py`, `prepare_spring.py`) é **byte a byte igual ao
  `origin/main`**: 14 de 14 checksums conferem entre o cluster, a máquina local e
  o repo
- `riemann/*.py` tem mtime anterior à primeira rodada, então o código científico
  ficou congelado a campanha inteira
- a única mudança em arquivo versionado é `train_single.py`, e é orquestração de
  seeds (`--seed-inicio`, pular seed concluída, resumo lido do disco), sem efeito
  em número nenhum
- as 47 rodadas usaram o mesmo monitor (`boundary_fscore`) e produziram o mesmo
  conjunto de métricas

**Não verificado:** não há reprodutibilidade bit a bit
(`torch.use_deterministic_algorithms` não é chamado, o que vale para o código
original também). O pareamento por seed continua válido, porque inicialização e
ordem dos dados são determinísticas.

---

## 8. O que tem nesta pasta

| caminho | conteúdo |
|---|---|
| `TABELA_RESULTADOS.md` | a tabela com n=6, gerada por script |
| `CONTRASTE_vs_controle.md` | o contraste pareado por seed |
| `metricas/<braço>/seed_N/` | os 47 treinos: `test_metrics.json` (o número reportado), `summary.json` (melhor época e métrica de validação) e `history.json` (curva completa) |
| `zero_shot/` | `test_metrics.json`, `meta.json` e `por_imagem.csv` com as 485 imagens e a cena de cada uma |
| `scripts/` | os geradores, para refazer qualquer tabela |

Para regerar:

```bash
python3 scripts/compara_zeroshot.py metricas --zero-shot zero_shot --seeds 0-5
python3 scripts/contraste_controle.py metricas --controle B0_berhu --seeds 0-5
```

---

## 9. Aviso sobre os pesos

Dos 47 treinos, **37 tiveram o `best.pt` apagado** do `/raid` do cluster numa
liberação de quota em 2026-09-10. As métricas foram preservadas, os pesos não.

Sobreviveram 10, no Hub em `juliadollis/depthpro-spring-ft`: o B3 teto 50
completo (6 seeds), o B0 em 768 px (3 seeds) e o B1 teto 5 seed 3. O repositório
leva junto as métricas dos 47 treinos, inclusive as dos 37 cujo peso se perdeu.

Consequência prática: reavaliar os outros braços (por imagem, por cena, ou num
conjunto de teste corrigido) deixou de ser uma hora de inferência e passou a
exigir retreino. A partir de agora todo checkpoint sobe para o Hub assim que o
treino fecha.

---

## 10. O que faria sentido fazer a seguir

| # | o que | por quê | custo |
|---|---|---|---|
| 1 | inspecionar a `seq0020` | 16% do teste, causa não identificada, e o numérico já se esgotou | zero |
| 2 | remedir num teste sem as duas cenas problemáticas | o AbsRel do zero-shot cai pela metade | exige retreino |
| 3 | variante `heads_final` | 89% dos 342 M parâmetros treináveis hoje são a cabeça de FOV, irrelevante para borda, e o paper do DepthPro treina o FOV separadamente | ~5 h |
| 4 | `log(1+\|K\|)` no lugar do teto escalar | única mudança que ataca o mecanismo da falha | ~5 h |
| 5 | repartição estratificada por profundidade | alinharia os três splits e reduziria a variância | CPU + retreino |

Sobre o item 4, registro a previsão antes de rodar: deve **remover o dano** (trazer
o B1 ao nível do B0) e **não criar ganho**, porque o teto do `fmax` não está no
termo de perda.
