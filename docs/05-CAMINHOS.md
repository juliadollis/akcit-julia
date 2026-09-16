# O inventário: onde mora cada coisa e o que ela é

Este é o mapa. Se alguém precisar achar um peso, uma métrica ou um script daqui a
seis meses, começa por aqui.

---

## 1. Hugging Face

### 1.1 `akcit-dephpro/` — a org oficial do projeto

| repo | tipo | o que é |
|---|---|---|
| `akcit-dephpro/depthpro-riemann-modelos` | model | **todos os checkpoints**, organizados por origem e braço |
| `akcit-dephpro/depthpro-riemann-avaliacoes` | dataset | **todas as avaliações em tabela**: agregado e por imagem, um registro por (modelo, mesa) |

A organização dentro do repo de modelos:

```
pesos_originais/<braço>/seed_N/best.pt     os 10 que sobreviveram à limpeza
pesos_retreino/<braço>/seed_N/best.pt      os 37 retreinados depois
pesos_confirma/<braço>/seed_N/best.pt      grad, metric e geod
ablacao_spring/<config>/best.pt            as 9 configs da ablação do Wallisson
metricas_campanha_completa/                as métricas dos 47 treinos originais
```

**Cada pasta de seed traz o `best.pt` junto de `test_metrics.json`,
`summary.json` e `history.json`**, para o peso nunca viajar sem o número dele.

### 1.2 Aviso de leitura que vale ouro

Os pesos em `pesos_retreino/` **não são cópias** dos que se perderam. São
execuções novas da mesma configuração e seed. Nos braços com curvatura elas **não
reproduzem** o original, porque o caminho da curvatura é não determinístico na
GPU (ver `04-ACHADOS-TECNICOS.md`).

Só os de `pesos_originais/` correspondem exatamente aos números da tabela do
relatório. Quem baixar `pesos_retreino/B3_gauss_metrica_teto1000/seed_0/best.pt`
esperando o modelo que deu 0,5908 de F-borda vai receber um que deu 0,5519. O
`test_metrics.json` ao lado de cada peso é sempre a verdade daquele peso.

### 1.3 Repositórios anteriores, mantidos por histórico

| repo | por que ainda existe |
|---|---|
| `juliadollis/depthpro-spring-ft` | onde tudo foi publicado primeiro, antes da org existir |
| `akcit-pixel/depthpro-spring-ft` | tentativa parcial: a org estourou a cota de armazenamento privado |
| `AKCITPixel3/depthpro-spring-ft` | idem, parcial |
| `juliadollis/depth-riemannian-checkpoints` | primeira tentativa, com nome que não pegou |

Os três parciais podem ser removidos quando a migração para `akcit-dephpro`
estiver conferida. **Nenhum foi apagado**, pela regra.

---

## 2. GitHub

| repo | o que tem |
|---|---|
| `juliadollis/akcit-julia` | **este repositório**: todo o código, os resultados e a documentação |
| `AKCIT-PIXEL/depth-riemannian` | o repositório original, do José Ricardo. O nosso `depth-riemannian/riemann/` é byte a byte igual ao `origin/main` dele |

---

## 3. No cluster (dgx-H100-01)

`/raid/user_juliadollis/julia_docker/`. **O `/raid` não é durável.** Tudo o que
importa já está no Hub ou aqui no git.

| caminho | o que é |
|---|---|
| `depth-riemannian/` | o código montado nos containers |
| `ablacao-wallisson/` | a ablação, intocada |
| `data/spring_split/{train,val,test}` | o split preparado, 593/184/485 |
| `data/spring/` | o Spring bruto, 40 GB |
| `runs_riemann/` | os 10 checkpoints que sobreviveram |
| `runs_retreino/` e `runs_b0_retreino/` | os 37 retreinados |
| `runs_confirma/` | grad, metric e geod |
| `runs_ablacao_spring/` | as 9 configs da ablação |
| `runs_riemann_metricas_preservadas/` | **as métricas dos 47 originais**, o que sobrou da limpeza |
| `avaliacoes/<rótulo>/<mesa>/` | as reavaliações, com `test_metrics.json`, `por_imagem.csv` e `meta.json` |

---

## 4. Os modelos, um por um

### 4.1 Os 10 que sobreviveram (`pesos_originais/`)

| braço | seeds | perda |
|---|---|---|
| `B3_gauss_metrica_teto50` | 0 a 5 | berHu 0,7 + normal 0,9 + curvatura 0,45, teto 50 |
| `B0_berhu_size768` | 0 a 2 | controle berHu puro, treinado em 768 px |
| `B1_gauss_metrica_teto5` | 3 | berHu 0,7 + curvatura 0,45, teto 5 |

### 4.2 Os 37 retreinados (`pesos_retreino/`)

| braço | seeds | perda |
|---|---|---|
| `B0_berhu` | 0 a 5, 8, 9 | controle berHu puro, 512 px |
| `B1_gauss_metrica_teto5` | 0, 1, 2, 4, 5 | curvatura isolada, teto 5 |
| `B1_gauss_metrica_teto1000` | 0 a 5 | curvatura isolada, teto 1000 |
| `B3_gauss_metrica_teto5` | 0 a 7 | normal + curvatura, teto 5 |
| `B3_gauss_metrica_teto1000` | 0 a 9 | normal + curvatura, teto 1000 |

### 4.3 A confirmação (`pesos_confirma/`)

Os três termos que bateram o controle na ablação, agora com o **nosso** protocolo
(split de teste, `metrics.py` com a máscara de validade, `align-mode full`, n=6).

| braço | perda |
|---|---|
| `Bgrad` | berHu 0,7 + grad 0,3 |
| `Bmetric` | berHu 0,7 + metric 0,2 |
| `Bgeod` | berHu 0,7 + geod 0,1 |

Os pesos dos termos são **idênticos ao `BASE_W`** da ablação do Wallisson, para
a confirmação testar a mesma coisa que deu sinal.

### 4.4 A ablação (`ablacao_spring/`)

9 configurações, com os defaults dele: `--filter B0 B1 B7`, 30 épocas, batch 2
com acumulação 4, `align-mode detach`, teto 5,0. **Avaliada na validação**, não
no teste.

---

## 5. As avaliações

Cada registro é um par **(modelo, mesa)** e traz três arquivos:

| arquivo | o que é |
|---|---|
| `test_metrics.json` | o **agregado**, do mesmo `Trainer.validate()` que gerou os números do relatório. É o que se compara com o histórico |
| `por_imagem.csv` | uma linha por imagem, com a **cena** de cada uma. É o que permite estatística agrupada por cena |
| `meta.json` | o checkpoint, a mesa, o n de imagens e de cenas, o batch e a resolução |

O rótulo carrega a procedência: `runs_riemann__B3_gauss_metrica_teto50__seed_0`
contra `runs_retreino__B3_...__seed_0`. Misturar os dois sem olhar o rótulo é
erro de leitura, pelo motivo da seção 1.2.

**Por que o `por_imagem.csv` importa tanto.** O teste do Spring tem 485 imagens
mas só **13 cenas**, e quadros consecutivos da mesma sequência são quase o mesmo
dado. O desvio entre cenas é 0,2147 e o erro padrão da média é **0,0595**, da
mesma ordem do maior efeito medido. Toda estatística honesta aqui tem que ser
agrupada por cena, e isso só é possível com o por imagem.

### 5.1 As mesas

| mesa | n | estado |
|---|---|---|
| `spring_test` | 485 imagens, 13 cenas | pronta e usada |
| `spring_val` | 184 imagens, 6 cenas | usada só pela ablação |
| DIODE | ~2,5 GB o val | **cabe na quota**, `prepare_diode.py` existe |
| Hypersim | ~150 GB | **não cabe**, e está no treino do DepthPro, então não mede generalização |

---

## 6. Os scripts, e para que serve cada um

### Em `depth-riemannian/scripts/` (científicos)

| script | o que faz |
|---|---|
| `test_geometry_metrica.py` | valida a curvatura em esfera, plano, cilindro e invariância. **Roda antes de qualquer treino** |
| `prepare_spring.py` | converte o Spring de disparidade para profundidade métrica e faz o split |
| `medir_k_spring.py` | percentis de \|K\| e saturação por teto |
| `medir_k_spring_porseq.py` | o mesmo, com a focal de cada sequência |
| `train_single.py` | o treino. Nossa única alteração: `--seed-inicio` e pular seed concluída |
| `zero_shot_spring.py` | o DepthPro sem fine-tune, o piso de cada mesa |
| `compara_zeroshot.py` | tabela dos braços contra o zero-shot |
| `contraste_controle.py` | contraste pareado por seed contra o controle |
| `diag_cauda_k.py`, `diag_cauda_k2.py`, `diag_dominancia.py` | os diagnósticos da cauda de \|K\| |

### Em `orquestracao-cluster/` (infraestrutura)

| script | o que faz |
|---|---|
| `baixa_spring.sh`, `prep_passo4.sh` | dados do zero |
| `roda_passo4.sh` | um braço, com guarda de memória e retentativa de OOM |
| `fila_retreino.sh` | fila com **reivindicação atômica por `mkdir`**: várias GPUs, nenhum item duplicado |
| `fila_confirma.sh` | a fila do grad/metric/geod |
| `roda_ablacao.sh` | a ablação do Wallisson |
| `avalia_modelos.py` | **N checkpoints × M datasets**, agregado e por imagem |
| `roda_avaliacao.sh` | o lançador da avaliação |
| `sobe_um_seed2.py` | sobe uma seed para o Hub assim que ela fecha |

---

## 7. O que NÃO está aqui, e por quê

| o que | onde está | por quê |
|---|---|---|
| os `.pt` dos checkpoints | só no Hugging Face | são 1,37 GB cada, 60 GB no total |
| o Spring | só no cluster e no DaRUS | 40 GB |
| `.env` com o `HF_TOKEN` | só no cluster | segredo, e o `.gitignore` bloqueia |
| 12 checkpoints de outras linhas (~47 GB) | só no cluster | bokeh e deblur, fora do escopo deste repo. **Continuam sem cópia no Hub** |

O último item é dívida conhecida: as runs `bokehnet-geo-A2linha` e
`bokehnet-geo-Blinha` têm progresso no disco além do último step publicado e não
têm safetensors final. Pela regra da seção 6 do README, são o material mais
exposto que existe hoje.
