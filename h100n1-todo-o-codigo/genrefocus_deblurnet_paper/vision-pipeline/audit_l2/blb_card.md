# LF-Bokeh (reproducao aproximada) a partir do BLB

Reproducao APROXIMADA do benchmark **LF-Bokeh** do paper GenRefocus
(arXiv 2512.16923), que os autores **nao publicaram**. Construida a partir do
**BLB** (Blender-generated Bokeh) do BokehMe (Peng et al., CVPR 2022 Oral),
Apache-2.0.

**Nao e o LF-Bokeh.** E um substituto funcional. A lista de divergencias abaixo
existe para que qualquer numero medido aqui seja lido com o desconto correto.

## O que o paper diz sobre o LF-Bokeh

Secao 4.1, item (ii): *"We introduce LF-Bokeh, featuring 200 images with diverse
focus planes and aperture sizes synthesized from light-field captures [14,50]."*
Referencias: [14] Dansereau et al., LiFF, CVPR 2019 (Stanford); [50] Rerabek e
Ebrahimi, New light field image dataset, QoMEX 2016 (EPFL).

## Tabela de divergencias

| Dimensao | LF-Bokeh oficial | Esta reproducao (BLB) | Tamanho da divergencia |
|---|---|---|---|
| Fonte do ground truth | capturas de light field reais | ray tracing, Blender 2.93 | **GRANDE**. Muda a natureza optica do bokeh |
| Realismo | fotografias reais | cenas sinteticas | **GRANDE** |
| Numero de imagens | 200 | **500** (10 cenas x 5 aberturas x 10 planos) | 2,5x mais |
| Numero de cenas | nao informado no paper | **10** (todas) | provavelmente bem menor |
| Planos de foco | "diverse", nao quantificado | 10 por cena, **94 planos distintos** no total | comparavel ou melhor |
| Tamanhos de abertura | "diverse", nao quantificado | 5 por cena, `f_stops` de 0,055 a 135,8 | comparavel ou melhor |
| K de referencia | **AUSENTE** (o paper afirma que a falta de K obriga a busca binaria) | **PRESENTE** (`blur_parameters`, K de 7,4 a 2161,4) | esta reproducao e MAIS RICA |
| Profundidade GT | **AUSENTE** (idem) | **PRESENTE** (`disparity`) | esta reproducao e MAIS RICA |
| Resolucao | nao informada | 1920x1080 (reamostrada na conversao) | desconhecida |
| Licenca | Stanford sem licenca declarada; EPFL research-only | Apache-2.0 | melhor |
| Fora do treino dos modelos | sim | sim (nem o paper nem nos treinamos em BLB) | equivalente |

## Consequencia para o protocolo de avaliacao

O paper faz **busca binaria por K** maximizando SSIM justamente porque nao tem
K nem profundidade de referencia (*"the absence of ground-truth depth maps and
reference bokeh levels K precludes the computation of accurate defocus maps"*).

Aqui esses dados EXISTEM. Para manter a comparabilidade, a avaliacao principal
**mantem a busca binaria**, identica ao paper. O K verdadeiro fica guardado nas
colunas para permitir, em separado, uma medida de **K-oraculo** que o paper nao
podia fazer. Sao duas leituras diferentes e nao devem ser misturadas.

Para a **LVCorr**, a estrutura do BLB e nativamente a que o paper descreve
("same all-in-focus input with a fixed focus plane across varying bokeh levels
K"): basta fixar `refocus_idx` e variar `k_idx`. A diferenca e que aqui existe
**alvo real para cada K**, o que o LF-Bokeh nao oferecia.

Convencao adotada: `LVCorr = Pearson(K, variancia do Laplaciano no fundo)`, com
**sinal cru, sem inverter**. Mais blur significa menos variancia, entao o valor
tende a ser NEGATIVO. O paper reporta positivo (BokehMe 0,9940). Essa diferenca
de convencao esta registrada de proposito e nao foi "corrigida" em silencio.

## Esquema

| coluna | conteudo |
|---|---|
| `file_name_base` | `blb_<cena>_k<K>_d<refoco>` |
| `image_focus` | all-in-focus (entrada) |
| `image_blur` | alvo com bokeh |
| `cena`, `k_idx`, `refocus_idx` | indices da grade |
| `k_ref` | K verdadeiro (`blur_parameters`) |
| `focus_distance` | plano de foco verdadeiro |
| `f_stop`, `focal_length` | optica |
| `disparity` | mapa de disparidade |
| `lv_aif`, `lv_alvo` | variancia do Laplaciano, para auditar o sentido do par |

`image_focus` e `image_blur` e `file_name_base` sao os nomes que o
`run_3models.py` do projeto consome, e os parquets seguem
`data/validation-*.parquet` para casar com o `--padrao` default.

## Credito

Dados originais: BokehMe, Peng et al., CVPR 2022, Apache-2.0,
https://github.com/JuewenPeng/BokehMe

## Por que nao usamos light field de verdade

A maior divergencia acima e a fonte do ground truth. A rota literal foi
investigada e MEDIDA antes de ser descartada. O registro:

**Disponibilidade (verificada por HTTP, nao por suposicao).** As duas fontes que
o paper cita estao vivas. Stanford Lytro Light Field Archive: arquivos ESLF
individuais em `lightfields.stanford.edu/images/<categoria>/raw/<nome>_eslf.png`,
HTTP 200, ~180 MB cada, 353 cenas, **sem licenca declarada**. EPFL: o FTP oficial
nao respondeu, mas o espelho do JPEG Pleno (`plenodb.jpeg.org/lf/epfl/`) serve por
cena, HTTP 200, ~42 MB de LFR por cena, 118 cenas, **research-only**. O Stanford
MVLF do LiFF tambem esta vivo (207 GiB em tar unico, com range requests).

**O metodo funciona.** Implementamos e rodamos: o ESLF e um arranjo lenslet
(`LF[v,u] = img[v::14, u::14]` para Lytro Illum, 14x14 views de 540x375, medido).
A AIF e a sub-aperture central (pinhole), e o bokeh sai de shift-and-add sobre um
subconjunto circular das views. Custo medido: 1,5 s para carregar, 0,9 s por
imagem, em CPU.

**O problema e fisico, e foi medido.** O CoC maximo do bokeh sintetizado e a
faixa de disparidade da cena vezes a extensao angular. Medindo a paralaxe real
por correlacao entre views extremas:

| cena | disparidade | CoC maximo | % da largura |
|---|---|---|---|
| `flowers_plants_1` (close-up) | 0,89 px/view | 11,6 px | **2,1%** |
| `flowers_plants_5` | 0,00 px/view | 0 px | **0,0%** |
| `people_1` | 0,11 px/view | 1,4 px | **0,3%** |
| `bamboo` (MVLF) | < 0,25 px/view | ~3 px | ~0,6% |

Confirmacao independente: no plano de disparidade zero, variar a abertura
sintetica de r=1,5 a r=6,5 muda a imagem em menos de 1/255 e **nao altera a
nitidez** (23,3 contra 22,7). A baseline angular da Lytro Illum e de ~1 cm, e so
cenas com sujeito muito proximo geram bokeh utilizavel.

**Consequencia.** Uma reproducao literal teria bokeh de 1 a 3% da largura, contra
5 a 10% de uma foto f/1,8, e a maior parte das 353 cenas do Stanford e das 118 da
EPFL seria descartada por um filtro de paralaxe. Um benchmark com bokeh fraco nao
discrimina modelo: premia quem nao mexe na imagem. O BLB, por ser sintetico, nao
tem esse teto e entrega a variacao de abertura e de plano de foco que o nome
"LF-Bokeh" promete.

Ou seja: trocamos **fidelidade de fonte** por **poder discriminativo**. Essa e a
escolha, explicitada, e nao um efeito colateral.

## Quatro ressalvas que mudam como o numero deve ser lido

Levantadas na auditoria do dataset. Nenhuma invalida o conjunto; todas mudam a
leitura. Estao aqui porque publicar o numero sem elas seria enganoso.

### 1. Parte do benchmark testa EXTRAPOLACAO, nao interpolacao

Medido nas 500 linhas: `k_ref` vai de **7,4 a 2161,4**, mediana 119,2.

| faixa de K | linhas | o que significa |
|---|---|---|
| K <= 100 | 220 (44%) | dentro da faixa de busca do paper |
| 100 < K <= 300 | 140 (28%) | acima do paper, dentro do nosso treino |
| K > 300 | 140 (28%) | **acima de tudo que qualquer modelo viu em treino** |

Os 28% com K > 300 penalizam todos os modelos igualmente por uma capacidade que
nenhum foi treinado a ter. Por isso a metrica deve ser reportada **global E
estratificada por faixa de K**. A estratificada e a que responde se o modelo
aprendeu a lei optica ou decorou a faixa de treino. Isso e propriedade do BLB,
nao defeito da conversao.

### 2. 56% das linhas tem abertura fisicamente impossivel

**280 das 500 linhas tem `f_stop` < 0,7**, com minimo em **f/0,055** (o maximo e f/135,8, ou seja, a faixa e enorme nos dois extremos). Nenhuma
lente real chega perto: as mais rapidas ja construidas param em torno de f/0,7.
Com as focais aqui (24,8 a 50 mm), f/0,055 implica pupila de entrada de quase um
metro.

Consequencia direta, e e o oposto do que o LF-Bokeh oficial mede: **na maioria das
linhas este benchmark nao mede fidelidade a optica real**. Ele mede consistencia
com um renderizador sem restricao fisica. Para a pergunta "o modelo reproduz
bokeh de camera de verdade", esta parte do conjunto nao serve.

### 3. O 65504 no plano de foco e foco no INFINITO, nao saturacao. Verificado.

Trinta e cinco linhas tem `focus_distance` exatamente 65504, que e o maximo do float16 e a
assinatura classica de saturacao. **Checamos na fonte e nao e saturacao.** Prova:

- Os 10 planos de foco de cada cena sao exatamente uniformes em **disparidade**
  (1/distancia), com passo constante ate desvio relativo de ~5e-16, que e
  precisao de maquina. Vale em todas as cenas.
- Extrapolando o passo dos indices 1 a 9 para o indice 0, o valor previsto bate
  com o medido em todas as cenas (cena 277: previsto 0,000015, medido 0,000015).
- E decisivo: **nem toda cena tem 65504 no indice 0**. A cena 279 comeca em 21,45
  e a 280 em 1154,0. Se fosse sentinela cega, apareceria em todas.
- 1/65504 = 1,53e-05, exatamente a disparidade que a grade uniforme exige ali.

Ou seja, o renderizador usou a maior distancia representavel em float16 para
expressar disparidade zero. O plano de foco **nao** e inventado nessas 35 linhas.

Efeito colateral util: a grade de foco do BLB e uniforme em disparidade, que e
exatamente o espaco em que a Eq. 2 do paper opera (`defocus = |K(disp - disp_s1)|`).

### 4. O que este dataset NAO resolve

Ele e **sintetico**. Portanto **nao** e resposta para a pergunta de contaminacao
e de vies de distribuicao. As duas perguntas sao diferentes e misturar seria o
erro mais facil de cometer aqui:

| pergunta | conjunto certo |
|---|---|
| o numero e comparavel com o protocolo do paper? | **esta reproducao do LF-Bokeh** |
| o numero esta livre de vies de distribuicao? | **RealDOF** (captura real, fora do treino dos dois modelos) |

O RealBokeh test, por sua vez, esta livre de contaminacao literal (verificado:
5864 de 5864 caminhos da rota c vem de `train/`), mas o nosso modelo treinou no
split `train` do mesmo dataset, mesma camera e mesmo pipeline de alinhamento.
Isso e vies de distribuicao, e nenhuma reproducao sintetica corrige isso.
