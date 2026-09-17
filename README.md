# AKCIT · Julia — reteste da curvatura Riemanniana no DepthPro

Todo o código, os dados de resultado e a documentação da investigação sobre
**se um termo de curvatura Gaussiana métrica na função de perda melhora a
qualidade de borda do DepthPro**.

Este repositório existe para que nada se perca. Em 2026-09-10 uma liberação de
quota no cluster apagou 37 de 47 checkpoints e o dataset preparado. As métricas
sobreviveram porque alguém as separou à mão antes. A regra que saiu disso está
na seção 6, e este repositório é a outra metade dela.

---

## 1. A pergunta e a resposta, em cinco linhas

1. O DepthPro erra bordas. A hipótese é que penalizar a diferença de **curvatura
   Gaussiana** entre predição e verdade afie essas bordas.
2. Um bug de unidade fazia a "curvatura" não ser curvatura. Corrigido pelo
   mantenedor do repositório original (commit `2b37bdc`), com retroprojeção 3D
   real.
3. Com a fórmula correta, **a curvatura não ajuda**: 47 treinos, 3 vitórias em 47
   comparações pareadas contra o controle berHu puro.
4. Mas **existe headroom no Spring**: todo braço treinado bate o modelo sem
   fine-tune, o oposto do que acontecia no Hypersim.
5. E a ablação mostrou que **escolhemos os dois piores termos da família**:
   `gauss` e `normal` são os únicos que perdem; `grad`, `metric` e `geod` batem
   o controle. Essa é a linha aberta.

---

## 2. Onde está cada coisa

| pasta | o que é |
|---|---|
| `depth-riemannian/` | o código científico. É o [AKCIT-PIXEL/depth-riemannian](https://github.com/AKCIT-PIXEL/depth-riemannian) mais os scripts que escrevemos por cima |
| `orquestracao-cluster/` | os scripts que rodam na máquina: filas, guardas de memória, retreino, avaliação, subida para o Hub |
| `ablacao-wallisson/` | a ablação adaptada pelo Wallisson, **intocada**, como ele entregou |
| `resultados/` | os JSONs por seed, as tabelas geradas e os pacotes de envio |
| `docs/` | a documentação longa, listada abaixo |

### A documentação

| documento | responde |
|---|---|
| [`docs/01-HISTORIA.md`](docs/01-HISTORIA.md) | o que foi pedido, o que foi feito, em que ordem, e onde desviamos |
| [`docs/02-RESULTADOS.md`](docs/02-RESULTADOS.md) | todas as tabelas, com as ressalvas que viajam junto |
| [`docs/03-INFRAESTRUTURA.md`](docs/03-INFRAESTRUTURA.md) | a máquina, o docker, as regras, como rodar do zero |
| [`docs/04-ACHADOS-TECNICOS.md`](docs/04-ACHADOS-TECNICOS.md) | os cinco achados que valem além deste projeto |
| [`docs/05-CAMINHOS.md`](docs/05-CAMINHOS.md) | **o inventário: cada modelo, cada avaliação, onde mora e o que é** |
| [`docs/06-TEORIA.md`](docs/06-TEORIA.md) | **a geometria por trás da hipótese, o que cada termo mede, e por que a curvatura falhou** |
| [`docs/07-RESULTADOS-FINAIS.md`](docs/07-RESULTADOS-FINAIS.md) | **a tabela final, só com modelos que ainda existem, n=6 por braço** |
| [`docs/08-ABLACAO.md`](docs/08-ABLACAO.md) | **a ablação dos cinco termos, e o que a confirmação com o nosso protocolo mostrou** |
| [`docs/ENVIO_WALLISSON.md`](docs/ENVIO_WALLISSON.md) | o retorno sobre a ablação, com os três pontos do código dele |

Se você só vai ler um, leia o `05-CAMINHOS.md`. É o mapa.

---

## 3. O protocolo, em um parágrafo

Spring, split por sequências **disjuntas**, seed 42, frações 0,50/0,15/0,35,
dando **593 treino / 184 validação / 485 teste** de 18 / 6 / 13 sequências.
DepthPro variante `heads` (decoder DPT mais cabeça de FOV, 342 M parâmetros
treináveis, encoder ViT congelado), lr 1e-5, batch 8, 512 px, até 100 épocas com
early stop no F-score de borda de validação. O teste é avaliado **uma única vez,
no fim, por seed**. Curvatura métrica com `fx = 689,6` e `fy = 1225,9` px em 512,
vindos da mediana de 2585,9 px das 37 sequências, reescalada pelo resize
anisotrópico de 1920x1080 para 512x512.

---

## 4. Os braços

| braço | perda | o que isola |
|---|---|---|
| **B0** | berHu 0,7 puro | o controle: o piso contra o qual todo ganho é medido |
| **B1** | berHu + curvatura 0,45 | a curvatura **sozinha**. `B1 − B0` é exatamente a hipótese |
| **B3** | berHu + normal 0,9 + curvatura 0,45 | a config "campeã" herdada da ablação anterior |
| sufixo `teto5/50/1000` | o `gauss_clamp`, em 1/m² | o teto por pixel da curvatura |
| `B0_berhu_size768` | controle a 768 px | se o teto da borda é resolução |

---

## 5. Como reproduzir

```bash
# 1. dados
bash orquestracao-cluster/baixa_spring.sh      # 23 GB do DaRUS
bash orquestracao-cluster/prep_passo4.sh       # split 593/184/485, seed 42

# 2. validar a geometria (CPU, 2 min)
python depth-riemannian/scripts/test_geometry_metrica.py

# 3. medir |K| e escolher o teto
python depth-riemannian/scripts/medir_k_spring.py \
    --raiz /data/spring_prep/test --fx-orig 2585.859

# 4. treinar
bash orquestracao-cluster/roda_passo4.sh <gpu> <b0|b1|b3> <teto> <seed> <n>

# 5. avaliar qualquer checkpoint em qualquer dataset
python orquestracao-cluster/avalia_modelos.py \
    --checkpoints "/runs/*/seed_*/best.pt" \
    --dataset spring=/data/spring_split/test \
    --checkpoint-base /models/checkpoints/depth_pro.pt \
    --saida /host/avaliacoes --incluir-zero-shot
```

Detalhe de cada passo em [`docs/03-INFRAESTRUTURA.md`](docs/03-INFRAESTRUTURA.md).

---

## 6. A regra que este repositório serve

> **Checkpoint que não está no Hub não existe.**
>
> Todo checkpoint sobe para o Hugging Face assim que o treino fecha. O `/raid` do
> cluster é área de trabalho, não armazenamento durável. Nunca apagar checkpoint,
> nem para liberar quota: subir primeiro, confirmar que subiu, e só então
> perguntar antes de remover o local.

Em 2026-09-10 perdemos 37 de 47 `best.pt` numa liberação de quota. As métricas
foram preservadas, os pesos não, e com isso reavaliar aqueles braços deixou de
ser uma hora de inferência e virou **88 h de retreino**.

O retreino aconteceu, e revelou um segundo achado: os braços com curvatura **não
são reprodutíveis** (ver [`docs/04-ACHADOS-TECNICOS.md`](docs/04-ACHADOS-TECNICOS.md)).

---

## 7. O que está aberto

| # | o que | custo |
|---|---|---|
| 1 | corrigir o `F.pad(mode="replicate")` do `geometry.py`, que é a fonte do não determinismo | 1 h, sem GPU |
| 2 | confirmar `grad`, `metric` e `geod` com n=6 no split de teste | **em andamento**, 18 treinos |
| 3 | teste pareado cena a cena, agora que existe `por_imagem.csv` para os 47 modelos | análise, sem GPU |
| 4 | investigar a `seq0020`, que é 16% do teste e tem `d1 = 0,05` | precisa de olho humano |
| 5 | avaliar em outros datasets (DIODE cabe; Hypersim são 150 GB) | ver `docs/05-CAMINHOS.md` |

---

## 8. Crédito

A correção matemática da curvatura métrica é do **Wallisson** (branch
`fix-geometry`), integrada pelo **José Ricardo** no commit `2b37bdc` do
repositório original, com o ajuste de aspect-ratio (`fy`) somado depois. A
ablação em `ablacao-wallisson/` é do Wallisson. O código científico em
`depth-riemannian/riemann/` é byte a byte igual ao `origin/main` do repositório
original: não alteramos a matemática, só construímos por cima.
