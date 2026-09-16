---
license: apache-2.0
tags:
  - depth-estimation
  - monocular-depth
  - spring
  - depthpro
---

# DepthPro fine-tunado no Spring: pesos e métricas do reteste de curvatura

Pesos e métricas de uma campanha que testa se um termo de **curvatura Gaussiana
métrica** na perda melhora a qualidade de borda do DepthPro fora do domínio de
treino dele.

Código: <https://github.com/AKCIT-PIXEL/depth-riemannian>

---

> **Por que aqui e não em `akcit-pixel`.** A `akcit-pixel` estourou o limite de
> armazenamento privado do plano da organização, que é independente do plano
> pessoal. Este repositório fica na conta `juliadollis`, que tem cota.

## Por que este repositório existe

**Checkpoint que não está no Hub não existe.**

Dos 47 treinos desta campanha, **37 tiveram o `best.pt` apagado** do `/raid` do
cluster numa liberação de quota em 2026-09-10. As métricas foram preservadas,
mas os pesos não, e com isso qualquer reavaliação daqueles braços (por imagem,
por cena, num conjunto de teste corrigido) passou a exigir retreino.

Este repositório existe para que isso não se repita. O `/raid` é área de
trabalho, não armazenamento durável.

---

## O que tem aqui

### `pesos/` — os 10 checkpoints que sobreviveram

| braço | seeds | o que é |
|---|---|---|
| `B3_gauss_metrica_teto50` | 0 a 5 | berHu 0,7 + normal 0,9 + curvatura 0,45, teto 50 |
| `B0_berhu_size768` | 0 a 2 | controle berHu puro, treinado em 768 px |
| `B1_gauss_metrica_teto5` | 3 | berHu 0,7 + curvatura 0,45 (curvatura isolada), teto 5 |

Cada pasta de seed traz `best.pt` mais `test_metrics.json`, `summary.json` e
`history.json`, para o peso nunca viajar sem o número dele.

O `best.pt` guarda **apenas os parâmetros treináveis** (o encoder ViT fica
congelado). Carregue com `strict=False` sobre o `depth_pro.pt` oficial.

### `metricas_campanha_completa/` — os 47 treinos

Métricas de todos os braços, inclusive os 37 cujo peso se perdeu. É o que
sustenta as tabelas do relatório.

---

## Como carregar

```python
import torch
from riemann.model import build_model     # do repo depth-riemannian

model = build_model("depth_pro.pt", variant="heads", device="cuda")
sd = torch.load("pesos/B3_gauss_metrica_teto50/seed_0/best.pt", map_location="cuda")
model.load_state_dict(sd, strict=False)   # strict=False: só as cabeças foram treinadas
model.eval()
```

---

## Protocolo

Spring, split por sequências disjuntas, seed 42, frações 0,50/0,15/0,35:
**593 treino / 184 validação / 485 teste**, de 18 / 6 / 13 sequências.

DepthPro variante `heads` (decoder DPT + cabeça de FOV, 342 M parâmetros
treináveis, encoder congelado), lr 1e-5, batch efetivo 8, 512 px (exceto o braço
de 768), até 100 épocas com early stop no F-score de borda de validação. O teste
é avaliado **uma única vez, no fim, por seed**.

Curvatura métrica com `fx = 689,6` e `fy = 1225,9` px em 512, vindos da mediana
de 2585,9 px do `intrinsics.txt` das 37 sequências, reescalada pelo resize
anisotrópico de 1920x1080 para 512x512.

---

## Resultados, com n=6 por braço

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

**Três leituras:**

1. **Existe headroom no Spring.** Todo braço treinado bate o modelo sem
   fine-tune. Em AbsRel e RMSE isso vale até na pior seed. É o oposto do
   Hypersim, que está na lista de treino do DepthPro.
2. **A curvatura não ajuda.** Contra o controle, pareado por seed, são
   **4 vitórias em 30 comparações**. O braço B1, que isola a curvatura, é o que
   perde mais, então o termo normal do B3 mascara parte do estrago.
3. **O teto não é o confundidor.** O B3 dá -0,0222 / -0,0190 / -0,0250 nos tetos
   5, 50 e 1000: praticamente plano ao longo de 200x de variação.

---

## Ressalvas que viajam com estes números

**O `fmax` quase não se mexe.** O F-score no melhor limiar de cada modelo fica em
~0,77 para tudo, inclusive para o modelo sem fine-tune. O ganho aparente do
controle no limiar fixo (+0,054) cai para +0,0117 no melhor limiar, ou seja, é em
boa parte recalibração da distribuição de gradiente, não borda melhor. Aumentar a
resolução para 768 px **piora** o `fmax` (-0,0344, pior em 3 de 3 seeds) enquanto
melhora RMSE e delta1 em 3 de 3.

**O n efetivo do teste é 13 cenas, não 485 imagens.** Quadros consecutivos da
mesma sequência são quase o mesmo dado. O desvio entre cenas é 0,2147 e o erro
padrão da média é 0,0595, da mesma ordem do maior efeito medido.

**Duas cenas dominam.** A `seq0020` (77 quadros, 16% do teste) tem `d1 = 0,05`, e
a `seq0043` tem `AbsRel = 2,28`. Excluindo as duas, o AbsRel do zero-shot cai de
0,3591 para 0,1637. A causa da `seq0020` não foi identificada: não é a métrica,
não é o espaço de alinhamento e não é a máscara.

**O teste é um regime de profundidade diferente do treino.** Mediana de 23,0 m no
treino, 23,7 m na validação e **10,4 m no teste**. Foi acidente do sorteio por
sequência.

**Não há teste de significância.** Com n=6 e desvio dos deltas na ordem de 0,010
a 0,026, o que sustenta a leitura é a consistência do sinal, não um p-valor.

---

## Reprodução

O código científico (`riemann/`, `scripts/medir_k_spring.py`,
`scripts/test_geometry_metrica.py`, `scripts/prepare_spring.py`) é byte a byte
igual ao `origin/main` do repositório. A correção da curvatura métrica é o commit
`2b37bdc`, de autoria do mantenedor do repo.
