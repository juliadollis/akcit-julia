# A teoria: por que curvatura, e por que não funcionou

Este documento explica a geometria por trás da hipótese, o que cada termo da
perda mede de fato, por que o termo de curvatura falhou por um motivo que a
própria geometria prevê, e o que o trabalho produziu apesar disso.

---

## 1. O problema concreto

O **DepthPro** estima profundidade a partir de uma imagem só. É bom no geral e
**erra nas bordas**: a transição entre objeto e fundo sai borrada, deslocada ou
inventada.

Isso importa para o uso final do grupo, o **bokeh sintético**. O mapa de desfoque
é `D_def = |D − S₁| · K`, e depende diretamente da profundidade. Borda errada no
mapa, desfoque vazando do fundo para o objeto, halo na foto.

---

## 2. Um mapa de profundidade é uma superfície

Uma imagem RGB é `(u,v) → cor`. Um mapa de profundidade é `(u,v) → D`, e isso
define uma **superfície mergulhada em ℝ³**. Cada pixel vira um ponto físico:

```
S(u,v) = ( (u−cx)·D/fx ,  (v−cy)·D/fy ,  D )
```

Essa superfície é um objeto geométrico de verdade: tem área, curvatura e
distâncias sobre ela. **A ideia do projeto é supervisionar propriedades
geométricas dela, em vez de só os valores de D.**

---

## 3. Primeira forma fundamental: como medir sobre a superfície

Andando na superfície variando `(u,v)`, os vetores tangentes são `S_u` e `S_v`.
O produto interno entre eles define o **tensor métrico**:

```
g = [ E  F ]     E = ⟨S_u,S_u⟩    F = ⟨S_u,S_v⟩    G = ⟨S_v,S_v⟩
    [ F  G ]
```

É a **primeira forma fundamental**, e ela responde perguntas **intrínsecas**: as
que uma formiga andando na superfície conseguiria responder sem sair dela. Qual o
comprimento deste caminho, qual a área desta região, qual o ângulo entre estas
direções.

Com a parametrização simplificada `(x, y, z(x,y))` ela vira:

```
g = [[1 + z_x²,  z_x·z_y],
     [z_x·z_y,   1 + z_y²]]
```

---

## 4. Segunda forma fundamental: como ela se curva no espaço

A primeira forma ainda não vê curvatura. Um plano e um cilindro têm **a mesma**
primeira forma, porque você enrola papel sem esticar.

O que os distingue é como a normal `n` gira quando você anda. Isso é a **segunda
forma fundamental**:

```
II = [ L  M ]     L = ⟨S_uu, n⟩    M = ⟨S_uv, n⟩    N = ⟨S_vv, n⟩
     [ M  N ]
```

Dela saem as curvaturas principais `k₁` e `k₂`, a máxima e a mínima no ponto:

```
K = k₁·k₂ = (LN − M²)/(EG − F²)      curvatura GAUSSIANA
H = (k₁+k₂)/2                        curvatura MÉDIA
```

É isso que `riemann/geometry.py::surface_curvatures` computa, e por isso ele
devolve `E, F, G, L, M, Nn, K, H, k1, k2`.

---

## 5. O Teorema Egregium, que é o motivo de tudo

Gauss provou que **K pode ser calculada só com a primeira forma fundamental**.
Embora K seja definida usando a normal, que é informação de como a superfície
está no espaço, o resultado **não depende disso**. Ele chamou de *Theorema
Egregium*, teorema notável.

A consequência: **K é intrínseca**. Dobrar a superfície sem esticar não muda K.

| superfície | K |
|---|---|
| plano, mesmo inclinado | 0 |
| cilindro de qualquer raio | 0 |
| esfera de raio R | 1/R² |
| sela | negativa |

É por isso que **não existe mapa plano perfeito da Terra**: a esfera tem K > 0, o
papel tem K = 0, e nenhuma deformação sem esticar converte um no outro.

**Por que isso seduz num modelo de profundidade.** K é uma assinatura da forma que
não se deixa enganar por inclinação nem por distância. Um modelo pode acertar a
profundidade média de uma parede e errar a curvatura de uma dobra de tecido.
Supervisionar K deveria pegar isso.

E não é ideia solta: o próprio paper do DepthPro usa MALE, um erro de Laplaciano,
no estágio 2, declaradamente para afiar borda. A diferença é que o Laplaciano é
uma quantidade da **imagem**, e K é uma quantidade da **superfície física**.

---

## 6. O bug que invalidava tudo

O código original calculava K sobre o **gráfico** `(u, v, D)` usando
`h = 1/max(H,W)` como espaçamento, misturando **coordenada de imagem normalizada
em [0,1]** com **profundidade em metros**.

O problema não é de escala, é de **tipo**. `z_x` deveria ser "metros de altura por
metro de deslocamento lateral", adimensional. Ali virava "metros por
unidade-de-imagem". A fórmula continua rodando e cuspindo número, mas o número não
é curvatura de nada.

| cena | K verdadeiro | fórmula antiga | erro |
|---|---|---|---|
| esfera R=2 m a 10 m | 0,25 | 10,19 | 3.974% |
| esfera R=0,5 m a 3 m | 4,00 | 18,30 | 358% |
| esfera R=5 m a 30 m | 0,04 | 2,12 | 5.209% |

O teste antigo não pegava porque montava a esfera com x, y e z nas mesmas
unidades. **Só quebra com dado real.**

A correção não mexeu na fórmula da curvatura: trocou a **parametrização**. E é por
isso que ela precisa de `fx`. No código:

```python
Su = [ D/fx + a·Du ,  b·Du ,  Du ]
```

O termo `D/fx` é quanto o ponto se desloca lateralmente, em metros, ao andar um
pixel. Ele **cresce com a distância**, porque um pixel cobre mais mundo quando o
objeto está longe. A fórmula antiga tratava isso como constante.

Consequência: os três resultados negativos anteriores (Hypersim, DIODE, Spring)
foram obtidos com um termo que não media curvatura. Não validavam nem invalidavam
a hipótese. Ela **nunca tinha sido testada**.

---

## 7. Os cinco termos, e o que cada um realmente mede

| termo | o que supervisiona | ordem |
|---|---|---|
| `grad` | `z_x`, `z_y` diretamente | **1ª** |
| `metric` | `g = [[1+z_x², z_xz_y],[·, 1+z_y²]]`, norma de Frobenius | 1ª (quadrática em z') |
| `geod` | `ds = √(1+z_x²+z_y²)`, o elemento de arco | 1ª (escalar de z') |
| `normal` | `n = (−z_x, −z_y, 1)/‖·‖`, via `1 − cos θ` | 1ª (**normalizada**) |
| `gauss` | `K = (LN−M²)/(EG−F²)` | **2ª** |

A ablação mediu os cinco isolados e o resultado se alinha com essa tabela:

```
grad, metric, geod   →  BATEM o controle
gauss, normal        →  os dois PIORES, abaixo do zero-shot
```

**Os três que funcionaram são todos de primeira ordem**, e carregam essencialmente
a mesma informação embalada de formas diferentes: o campo de gradiente da
profundidade.

**Os dois que falharam são os dois "especiais":** `gauss` é o único de segunda
ordem, e `normal` é o único que **normaliza**, jogando fora a magnitude do
gradiente e guardando só a direção.

---

## 8. Por que a segunda ordem falhou, mecanicamente

Aqui a teoria e o dado se encontram.

Medimos a distribuição de |K| no Spring inteiro: 37 sequências, 517 quadros,
**135,5 milhões de pixels**, com o `fx` de cada sequência.

| percentil | \|K\| (1/m²) |
|---|---|
| p50 | 4,41 |
| p90 | 987 |
| p99 | 68.745 |
| p99,9 | 3.230.846 |

Cinco ordens e meia de grandeza. E a razão é **geométrica, não numérica**:
`K = 1/R²`. Uma dobra de tecido de raio 6 cm dá `K = 278`. Um fio, um pelo, uma
quina viva dão K astronômico. **A geometria fina é genuinamente de curvatura
altíssima.**

Confirmamos que a cauda é real, refutando duas hipóteses alternativas:

- **não são as bordas**: pixels de degrau forte têm p50 de |K| = 0,003, e a cauda
  vive nas regiões mais **planas**
- **não é quantização no fundo**: |K| é maior **perto** (p50 = 237,7 a 0-5 m
  contra 0,83 a 40-80 m)

O efeito disso num L1 sobre `|K_pred − K_alvo|`:

**Sem teto**, 0,01% dos pixels (13.552 de 135 milhões) carregam **99,71% da soma**.
Não é uma perda sobre a imagem, é uma perda sobre treze mil pixels.

**Com teto**, onde predição e alvo saturam do mesmo lado a diferença é exatamente
zero e o pixel **não gera gradiente**. Um pixel onde a verdade é K=800 e a
predição é 300 vira `|5 − 5| = 0`: erro de 500 registrado como zero.

| teto | % saturado | quanto o top 0,1% domina depois do clamp |
|---|---|---|
| 5 | 49,7% | 0,18% |
| 50 | 29,4% | 0,26% |
| 1000 | 9,1% | 0,66% |

**Não existe teto escalar bom.** É um dilema estrutural do L1 sobre uma quantidade
com cauda de lei de potência, e foi por isso que testar 5, 50 e 1000 deu resultado
plano: -0,0222, -0,0190, -0,0250 contra o controle.

A saída seria comprimir antes de comparar: com `log(1+|K|)` a amplitude cai de
698.000x para **8,5x** e o top 1% passa a carregar 4,8% em vez de 99,99%. Medimos,
nunca testamos em treino.

---

## 9. Por que o `normal` falhou

Mais sutil. A normal é `(−z_x, −z_y, 1)` **normalizada**, e a perda é `1 − cos θ`.

**Normalizar joga fora a magnitude.** Numa borda real, o que distingue objeto de
fundo é justamente um gradiente **grande**; a direção dele pode ser parecida com a
de uma superfície suavemente inclinada. O termo trata as duas igual.

E `1 − cos θ` é **quadrático** perto de θ = 0: gradiente quase nulo quando a
normal já está quase certa, e saturado quando está muito errada. Pouco informativo
nos dois extremos.

O `grad`, que não normaliza, mantém a magnitude e ganha.

---

## 10. O que este trabalho produziu

A **hipótese** falhou. O **trabalho** não. Vale separar.

### 10.1 Antes desta campanha

Três resultados negativos obtidos com um termo que não calculava curvatura, e uma
configuração "campeã" herdada de uma ablação rodada naquele código quebrado. Nada
disso dizia se a hipótese funciona ou não, e ninguém sabia.

### 10.2 O que existe agora

**A hipótese foi testada.** 47 treinos originais mais 37 retreinos, n de 10 a 20
por braço, duas campanhas independentes chegando ao mesmo lugar. A resposta é não,
e é firme.

**E há um mecanismo.** Não é "tentamos e não deu": é a cauda de lei de potência de
`K = 1/R²` e o dilema do teto, documentado com 135 milhões de pixels.

**O `grad` funciona.** 5 de 5 seeds contra o controle, e é a **única intervenção
de toda a campanha que moveu o `fmax`**, a métrica que não se mexeu com seis
perdas diferentes nem com resolução a 768 px.

**A explicação do fracasso anterior.** O Hypersim está na lista de treino do
DepthPro, com supervisão de segunda ordem no estágio 2 deles, feita para afiar
borda. **Não havia headroom.** Isso explica de uma vez os negativos anteriores e
justifica a mudança para o Spring.

**A configuração campeã era ruim.** `gauss` e `normal`, os dois termos dela, são
os **dois piores dos cinco**, ambos abaixo do modelo sem fine-tune.

### 10.3 Três achados que valem fora deste projeto

**O `boundary_fscore` de limiar fixo confunde calibração com qualidade de borda.**
O ganho de +0,054 do controle vira +0,012 no melhor limiar de cada modelo. Quem
reporta melhoria de borda com essa métrica pode estar reportando recalibração.

**O caminho geométrico não é reprodutível na GPU**, por causa do `atomicAdd` no
backward do `F.pad(mode="replicate")`. Quem rodar uma seed por configuração está
ordenando ruído.

**O n efetivo do Spring é 13 cenas, não 485 imagens.** Erro padrão de 0,0595, da
ordem do maior efeito medido.

### 10.4 A frase que o trabalho sustenta

> Mostramos que um termo de curvatura Gaussiana métrica na perda **degrada** a
> qualidade de borda do DepthPro no Spring, e identificamos a causa: a
> distribuição de |K| em cenas reais tem cauda de lei de potência que torna
> qualquer L1 com teto escalar ou dominado por outliers ou sem gradiente. Em
> contrapartida, um termo de gradiente de primeira ordem melhora, e é a única
> intervenção que eleva o F-score de borda no ponto de operação ótimo.

### 10.5 O que não vale maquiar

Custou ~88 h de GPU na campanha original mais o retreino, contra as ~4 h que o
desenho inicial previa. Parte foi decisão de ampliar, parte foi retrabalho
evitável: os checkpoints apagados numa liberação de quota, um `PermissionError`
que custou 6 h de fila, e o defeito de só publicar no Hub ao fim de cada faixa.

O maior custo foi de **sequência**: o zero-shot e a ablação dos cinco termos
deviam ter vindo primeiro. Se `grad`, `metric` e `geod` tivessem sido testados no
começo, boa parte dos 47 treinos em `gauss` e `normal` não teria acontecido.

É lição para o próximo experimento, não invalidação deste.
