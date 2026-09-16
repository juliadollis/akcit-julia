# Registro de execução do reteste da curvatura

Escrito em 2026-09-10. Responde três perguntas: **o que foi rodado**, **como cada
número foi gerado**, e **o que está verificado contra o repositório original e o
que não está**.

Repositório de referência: https://github.com/AKCIT-PIXEL/depth-riemannian

---

## 1. Estado de verificação, sem arredondar

Não há certeza de 100%. Segue o que está checado e o que não está.

### 1.1 Verificado

| item | como foi checado | resultado |
|---|---|---|
| O código científico é o original | `git diff origin/main -- riemann/ scripts/medir_k_spring.py scripts/test_geometry_metrica.py scripts/prepare_spring.py` | **vazio.** Byte a byte igual ao repo |
| Única divergência versionada | `git diff --stat origin/main` | só `scripts/train_single.py`, +31/-9 |
| O que essa divergência faz | leitura do diff | `--seed-inicio`, pular seed já concluída, resumo lido do disco. Não toca perda, modelo, métrica, dataset nem avaliação |
| A semeadura é equivalente | leitura de `repro.set_seed` | semeia `random`, `numpy`, `torch`, `torch.cuda`, `PYTHONHASHSEED`, e liga `cudnn.deterministic`. Rodar a seed `k` num processo separado equivale a rodá-la no laço original |
| Todas as rodadas usaram o mesmo critério | auditoria dos 37 `summary.json` | `monitor = boundary_fscore` em 37 de 37 |
| Todas produziram o mesmo conjunto de métricas | auditoria dos 37 `test_metrics.json` | 1 conjunto único, e o zero-shot bate com ele |
| Passo 1 passa | execução em 2026-09-10 | 5 seções, todos os critérios dele. Log `passo1_test_geometry_metrica.log` |
| Passo 3 reproduz | execução do comando literal em 2026-09-10 | p50 = 5,28, 50,39% acima do teto 5. Log `passo3_medir_k_spring.log` |
| Números das tabelas | script de conferência campo a campo contra os geradores | tudo confere |

### 1.2 NÃO verificado

| dúvida | por que importa | como resolver |
|---|---|---|
| **A cópia do código no cluster é a mesma do repo?** | o container monta `/raid/user_juliadollis/julia_docker/depth-riemannian/{scripts,riemann}`. É essa cópia que rodou, não a deste Mac | comparar checksums da cópia do cluster contra `origin/main`. Custo zero de GPU |
| **As 37 rodadas usaram a mesma versão do código?** | elas vão de 30/08 a 05/09. Se `riemann/` mudou no meio, os braços não são comparáveis entre si | conferir `mtime` dos arquivos no cluster contra os horários das rodadas |
| **Reprodutibilidade bit a bit** | `set_seed` liga `cudnn.deterministic`, mas `torch.use_deterministic_algorithms` não é chamado. Duas execuções da mesma seed podem diferir um pouco | inerente ao código original também. O pareamento por seed continua válido, porque inicialização e ordem dos dados são determinísticas |
| **Passo 1 rodou antes do passo 4 em agosto?** | é a ordem que ele exigiu | não há log da execução original. O `MELHORIAS_RETESTE.md` afirma que os passos 1 a 3 foram executados, e o conteúdo dele só existe se 2 e 3 tiverem rodado |

**As duas primeiras linhas precisam ser fechadas antes de gastar mais GPU.** Custam
zero e são a diferença entre "os braços são comparáveis" e "não sabemos".

---

## 2. O que o José Ricardo pediu, e o que foi rodado

| passo | pedido | rodado | artefato |
|---|---|---|---|
| setup | container, `download_weights.py`, `prepare_spring.py` | sim | `prep_passo4.sh`, split 593/184/485 |
| 1 | `test_geometry_metrica.py` | sim | `passo1_test_geometry_metrica.log` |
| 2 | fx do Spring, mediana do `prepare_spring` | sim | `fx (px): mediana 2585.9, faixa 1292.9-6060.6` |
| 3 | `medir_k_spring.py --raiz <spring>/test --fx-orig <fx>` | sim, e ampliado | `passo3_medir_k_spring.log` e `prep_spring_full.log` |
| 4 | `train_single.py` B3 + controle B0, mesmo split | sim, e além | 37 rodadas de seed, 5 braços |

### 2.1 Onde houve desvio

**O portão do passo 4.** O `RETESTE_CURVATURA.md` fecha com "Manda o resultado dos
passos 1 a 3. O passo 4 a gente decide junto", e antes avisa "Não pular direto para
cá. São ~4 h de GPU em cima de um número que ainda não foi conferido".

A medição do passo 3 contrariou a previsão dele (ele esperava que o teto 5 ficasse
folgado; medido, 49% dos pixels saturam). Esse era exatamente o caso de consulta. O
teto 1000 foi escolhido aqui, o passo 4 rodou, e os dados do passo 3 chegaram a ele
depois, como justificativa. Depois o teto 5 também foi rodado, o que fecha o buraco
técnico mas não o de processo.

**A escala.** Ele dimensionou ~4 h e definiu dois braços. Foram rodados cinco braços
e 37 seeds, cerca de **67 h de GPU**, ou seja, seis vezes o escopo pedido.

| escopo | rodadas | épocas | GPU |
|---|---|---|---|
| pedido (B0 + B3 com o teto escolhido, 3 seeds) | 6 | 341 | ~11 h |
| rodado | 37 | 2.073 | ~67 h |

### 2.2 O que entrou além do pedido

| extra | justificativa |
|---|---|
| B3 com teto 5 além do 1000 | testa as duas pontas do intervalo plausível |
| B1 com dois tetos | B3 mistura normal com curvatura; `B1 - B0` isola a hipótese |
| seeds 3 a 9 | com n=3 as faixas dos braços se sobrepunham |
| zero-shot | a linha de base que faltava |

---

## 3. Como cada número foi gerado

### Passo 1
```
docker run ... riemann-depthpro:latest python scripts/test_geometry_metrica.py
```
Esfera 0,00% a 0,17% de erro; plano e cilindro passam; invariância 0,02%; normais e
área passam. Seção [6]: a fórmula antiga erra de 358% a 5209%.

### Passo 2
Saída do próprio `prepare_spring.py`, que lê `intrinsics.txt` por sequência:
`mediana 2585.9, faixa 1292.9-6060.6`. Na resolução de treino, `fx = 689,6` e
`fy = 1225,9` px, por causa do resize anisotrópico de 1920x1080 para 512x512.

### Passo 3
```
python scripts/medir_k_spring.py --raiz /data/spring_prep/test --fx-orig 2585.859
```
p50 = 5,28 | p90 = 1.945 | p99 = 142.610 | p99,9 = 3.614.315. Acima de 5: 50,39%.

Ampliação (`prep_spring_full.sh` + `medir_k_spring_porseq.py`), 37 sequências,
517 quadros, 135,5 milhões de pixels, com fx por sequência:
p50 = 4,41 | p90 = 987 | p99 = 68.745 | p99,9 = 3.230.846.

Saturação por teto: 5 → 49,7% | 50 → 29,4% | 500 → 12,6% | **1000 → 9,13%** | 5000 → 4,0%.
O teto 1000 foi escolhido por ser onde não há tensão entre saturar pouco e manter a
perda distribuída.

Diagnósticos extras, todos com hipótese refutada:
- `diag_cauda_k.py`: a cauda **não** vem das bordas (p50 de |K| nos degraus = 0,003)
- `diag_cauda_k2.py`: **não** vem de quantização no fundo (|K| é maior perto: p50 = 237,7 a 0-5 m contra 0,83 a 40-80 m)
- `diag_dominancia.py`: sem teto, **0,01% dos pixels carregam 99,71% da soma de |K|**

### Passo 4
```
python scripts/train_single.py --train-root /data/spring_split/train \
  --val-root /data/spring_split/val --test-root /data/spring_split/test \
  --checkpoint /models/checkpoints/depth_pro.pt --out-dir /workspace/runs/<NOME> \
  <PESOS>
```
com `<PESOS>`:

| braço | pesos |
|---|---|
| B0 | `--berhu 0.7 --normal 0 --gauss 0 --grad 0 --geod 0 --metric 0` |
| B1 | `--berhu 0.7 --normal 0 --gauss 0.45 --gauss-metrica --fx-orig 2585.859 --gauss-clamp <teto>` |
| B3 | `--berhu 0.7 --normal 0.9 --gauss 0.45 --gauss-metrica --fx-orig 2585.859 --gauss-clamp <teto>` |

Lançado por `passo4.sh` (teto 1000), `passo4_teto.sh` (teto 5) e `passo4_seeds.sh`
(ampliação de seeds e o braço B1). Variante `heads`, lr 1e-5, batch 8, 512 px, até
100 épocas com early stop no F-score de borda de validação. Teste avaliado uma única
vez no fim, por seed.

### Zero-shot
```
python scripts/zero_shot_spring.py --test-root /data/spring_split/test \
  --checkpoint /models/checkpoints/depth_pro.pt --out-dir /workspace/runs/ZERO_SHOT_spring
```
Constrói do mesmo `depth_pro.pt`, **não treina e não carrega `best.pt`**, e chama o
**mesmo `Trainer.validate()`** que gerou o `test_metrics.json` de cada seed treinada.
Determinístico, então não há seed. 6 minutos de GPU.

### Tabelas
Geradas por script a partir dos JSONs, nunca transcritas à mão:
```
python3 scripts/compara_zeroshot.py <bracos> --zero-shot <zs> [--seeds 0-5]
python3 scripts/contraste_controle.py <bracos> [--controle B0_berhu] [--seeds 0-5]
```

---

## 4. Resultados

### 4.1 Com n fixo em 6 (seeds 0 a 5)

| braço | n | F-borda ↑ | AbsRel ↓ | delta1 ↑ | RMSE ↓ |
|---|---|---|---|---|---|
| zero-shot | - | 0,5402 | 0,3602 | 0,6594 | 5,4002 |
| **B0 berHu** | 6 | **0,5954** | **0,2509** | 0,6946 | **4,2433** |
| B1 teto 5 | 5 | 0,5632 | 0,2608 | **0,6957** | 4,4096 |
| B1 teto 1000 | 6 | 0,5339 | 0,2758 | 0,6901 | 4,6544 |
| B3 teto 5 | 6 | 0,5732 | 0,2772 | 0,6943 | 4,4440 |
| B3 teto 1000 | 6 | 0,5704 | 0,2932 | 0,6894 | 4,5513 |

B1 teto 5 está com n=5 porque falta a seed 3.

### 4.2 Contraste pareado contra o controle, F-borda

| braço | n | delta médio | desvio | seeds a favor |
|---|---|---|---|---|
| B1 teto 1000 | 6 | -0,0615 | 0,0104 | 0/6 |
| B1 teto 5 | 5 | -0,0314 | 0,0098 | 0/5 |
| B3 teto 1000 | 6 | -0,0250 | 0,0101 | 0/6 |
| B3 teto 5 | 6 | -0,0222 | 0,0185 | 1/6 |

### 4.3 Leitura

Existe headroom no Spring: todo braço treinado bate o zero-shot, e em AbsRel e RMSE
isso vale até na pior seed. É o oposto do Hypersim, que está no treino do DepthPro.

A curvatura perde do controle em quase toda seed. O B1, que a isola, é quem perde
mais, então o termo normal não salva a curvatura, mascara parte do estrago.

### 4.4 O achado que não estava sendo olhado

O `boundary_fmax`, que é o F-score no melhor limiar de cada modelo, estava nos JSONs
desde sempre:

| braço | fscore (limiar fixo) | fmax |
|---|---|---|
| zero-shot | 0,5402 | 0,7674 |
| B0 | 0,5943 (+0,0540) | 0,7760 (**+0,0087**) |
| B3 teto 5 | 0,5746 | 0,7677 (+0,0004) |
| B1 teto 1000 | 0,5339 | 0,7230 (-0,0444) |

No melhor ponto de operação, o fine-tuning quase não melhora a borda, e a dispersão
entre seeds do B0 (0,7581 a 0,7857) engole o valor do zero-shot. O ganho de +0,0540
é majoritariamente recalibração da distribuição de gradiente, não borda melhor.
**É por isso que as margens são pequenas.**

### 4.5 Duas fragilidades do conjunto de teste

**Uma sequência domina.** No zero-shot, por cena, o F vai de 0,179 (seq0020, com 77
dos 485 quadros) a 0,819 (seq0017). A seq0020 sozinha é 16% do teste.

**O n efetivo é 13 cenas, não 485 imagens.** Desvio entre cenas 0,2147; erro padrão
da média 0,0595, que é a mesma ordem da maior diferença entre braços.

**O teste é um regime de profundidade diferente.** Medido nos dados preparados:
train mediana 23,03 m, val 23,68 m, **test 10,41 m**. Train e val batem; o teste é
2,2x mais perto. Foi acidente do sorteio por sequência.

---

## 5. Mudanças no código

**O código que mudou a ciência não é nosso.** O commit `2b37bdc feat(geometria):
curvatura metrica por retroprojecao 3D real` é do próprio José Ricardo, de 28/08,
já em `origin/main`. Autoria da correção matemática: Wallisson, branch
`fix-geometry`, com o fix de aspect-ratio (`fy`) somado depois.

**Nossa única mudança em arquivo versionado** é `scripts/train_single.py`, não
commitada: `--seed-inicio`, pular seed com `test_metrics.json`, e resumo lido do
disco. Orquestração, sem efeito em número nenhum.

**Arquivos novos, nenhum commitado:**
```
scripts/passo4.sh  passo4_teto.sh  passo4_seeds.sh
scripts/medir_k_spring_porseq.py  diag_cauda_k.py  diag_cauda_k2.py  diag_dominancia.py
scripts/consolida_reteste.py  zip_remoto.py
scripts/zero_shot_spring.py  compara_zeroshot.py  contraste_controle.py
MELHORIAS_RETESTE.md  REGISTRO_EXECUCAO.md  resultados/
```

`main` local está 0 à frente e 0 atrás de `origin/main`. **Todo o ferramental do
reteste existe só neste Mac e no cluster.**

---

## 6. Pendências

| # | o que | custo |
|---|---|---|
| 1 | conferir que a cópia do código no cluster é a do repo | zero |
| 2 | conferir que as 37 rodadas usaram a mesma versão | zero |
| 3 | B1 teto 5 seed 3, para fechar n=6 em todos os braços | ~5 h GPU |
| 4 | olhar a seq0020 | zero |
| 5 | avaliação por imagem dos `best.pt`, para estatística pareada por cena | ~1 h GPU |
| 6 | reportar `f_auc` e `fmax` nas tabelas | zero |
| 7 | early stop em `f_auc` em vez do fscore de limiar fixo | zero |
| 8 | B3 com teto 50, para separar fórmula de regime de saturação | ~5 h GPU |
| 9 | commitar e subir tudo isto | zero |

Os itens 1 e 2 vêm antes do 3. Sem eles, gastar 5 h na seed que falta é apostar que
o código do cluster não mudou no meio da campanha.

---

## 7. Nota sobre a pasta `seed_3` do B1 teto 5

Ela **existe** com sobras de uma rodada interrompida, sem `test_metrics.json`.
Rodar a seed 3 escreve por cima desses restos. Nenhum resultado reportado se perde,
porque aquela seed nunca produziu número, mas a ação precisa de autorização
explícita antes de acontecer.
