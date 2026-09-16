---
license: other
---

# Controlabilidade de bokeh do BokehNet — original x kfix

Responde a UMA pergunta: o modelo obedece ao comando de intensidade de bokeh?
Ela nao e a mesma que "a imagem ficou parecida com o alvo".

PRIVADO (derivado de datasets de terceiros).

## 1. Por que a LVCorr antiga era ruido

Medido em 2026-08-26 sobre os proprios datasets de inferencia ja publicados:

| dataset de inferencia | desvio relativo da LV ao longo do sweep | LV(K=1)/LV(K=15) | LVCorr por imagem |
|---|---|---|---|
| `bokeh-eval-infer-kfix-ddpd` | 0,45% | 1,0006 | -1,00 a +1,00 (sd 0,59) |
| `bokeh-eval-infer-nosso-kesc` | 0,32% | 1,0004 | -1,00 a +0,80 (sd 0,64) |
| `bokeh-eval-infer-oficial-kesc` | 0,39% | 1,0008 | -0,80 a +1,00 (sd 0,58) |

As quatro imagens do sweep sao a MESMA imagem: a diferenca media por pixel entre
K=1 e K=15 e de 0,26 em 255. Causa medida: com `--k-escala 0.01`, o mapa de
condicionamento entra no modelo com maximo mediano de **0,0005** em uma escala
[0,1]. O modelo nao ignorou o comando — nao houve comando.

Dois defeitos menores tambem confirmados: o eixo x usava `K_VALUES = [0,5,10,15]`
para imagens geradas com K = 1,5,10,15 (ja corrigido no repositorio); e, com
`long_side=512`, o mapa de profundidade era usado na resolucao ORIGINAL
(1120x1680) enquanto a imagem gerada tinha 336x512, e o "pixel central" do plano
de foco era indexado com as coordenadas de 512 dentro do array de 1680 — caindo a
~15% da imagem, nao no centro. Corrigir o eixo NAO conserta a metrica; o mapa nulo
e a causa dominante.

## 2. Desenho da medicao

- Eixo: `alpha`, amplitude do mapa NORMALIZADO, 9 pontos de 0 a 1. `alpha` e o K
  do paper reescalado POR IMAGEM para cobrir [0,1]. Um K comum a todas as imagens
  nao serve: o K que leva o mapa a 1,0 varia de **35 a 3273** entre imagens da
  DDPD, entao o mesmo K satura umas e zera outras. Como a LVCorr e por imagem e a
  correlacao de postos e invariante a reescala monotona, `alpha` nao muda a
  metrica, so garante que ela seja medida na faixa util.
- Faixa justificada pelo treino: o mapa que o kfix viu na rota b tem p25=0,25,
  p50=0,47, p75=0,84 e satura em 19% das amostras; o do original tem teto fixo em
  0,50; o da rota c (comum aos dois) satura em 61%. [0,1] esta no dominio dos dois.
- Seed FIXA em todos os pontos: o unico fator que varia e `alpha`.
- **CONVENCAO DE SINAL: `LVCorr = -spearman(alpha, LV)`. +1 = obediencia
  perfeita.** A variancia do laplaciano mede NITIDEZ, entao obedecer significa
  nitidez caindo quando o comando sobe; sem a negacao a obediencia apareceria como
  numero negativo. (O avaliador do repositorio usa a convencao NAO negada.)
- Metricas por imagem: LVCorr em 3 regioes (inteira, fundo `B>=0,5`, foco
  `B<=0,05`); faixa dinamica `DR = LV(alpha=0)/LV(alpha=1)`, onde 1,0 significa que
  o comando nao mudou nada; fracao de degraus monotonos.
- 40 imagens da DDPD validation, comparacao PAREADA (mesmas imagens, mesmo mapa,
  mesma seed) com Wilcoxon.

## 3. A medicao foi validada antes de ser usada

| controle | LVCorr total | DR fundo | passou? |
|---|---|---|---|
| **P** oraculo obediente (bokeh classico) | **+1,0000** (sd 0) | 417,8 | sim |
| **P2** oraculo fraco (raio max 1,5 px) | +0,9583 | 2,80 | sim (rho alto, DR baixa) |
| **P3** blur global (obedece, sem seletividade) | +1,0000 | 602,7 | sim |
| **N1** oraculo surdo (ignora alpha) | -0,1125 IC95 [-0,25; +0,05] | 0,993 | sim (contem zero) |
| **N2** comando nulo (5e-4, replica o bug atual) | -0,0750 IC95 [-0,20; +0,07] | 1,000 | sim |
| **D** imagem constante | NaN | - | sim |
| **PISO DE RUIDO DO PROPRIO MODELO** (alpha fixo, 9 seeds) | -0,21 / -0,13 | 1,04-1,11 | sim |

E o controle positivo definitivo, com dados reais: as **fotos reais** do
RealBokeh_3MP test, medidas por este mesmo codigo, dao **LVCorr = +1,0000 em 30
de 30 cenas** e DR de fundo **11,93**. O teto da metrica e +1,0.

## 4. Resultado

40 imagens DDPD, 9 pontos de alpha, seed fixa, comparacao pareada.

| | kfix | original | dif | p (Wilcoxon) |
|---|---|---|---|---|
| **LVCorr fundo** (onde o bokeh deve agir) | **+0,520** | +0,275 | **+0,245** | **0,0065** |
| LVCorr imagem inteira (estilo paper) | +0,605 | +0,545 | +0,059 | 0,48 |
| imagens com resposta INVERTIDA no fundo | 5,0% | 12,5% | | |
| monotonicidade do fundo | 0,750 | 0,625 | | |
| DR fundo (1,0 = nada mudou) | 1,133 | 1,106 | | |
| seletividade DR_fundo/DR_foco | 0,823 | 0,756 | | |
| acima do piso de ruido? | p=1,2e-06 | p=2,3e-02 | | |

**Duas conclusoes, e a segunda importa mais que a primeira.**

1. O kfix melhorou o controle de intensidade **na regiao certa**, de forma
   estatisticamente significativa e pareada. O ganho aparece so na LV do FUNDO; na
   imagem inteira os dois sao indistinguiveis (p=0,48). A metrica agregada do paper
   nao teria visto essa diferenca.

2. **Os dois modelos estao muito longe de controlar bokeh.** Varrendo o comando de
   0 a 1, a nitidez do fundo muda 13% (kfix) e 11% (original). Nas fotos reais a
   mesma varredura muda **1093%** (DR 11,93). E a seletividade e menor que 1 nos
   dois, isto e, o efeito e MAIOR na regiao em foco que no fundo — o oposto de
   profundidade de campo. O que os modelos fazem e uma degradacao global leve na
   direcao certa, nao bokeh.

## 5. Limitacoes declaradas

- O plano de foco vem da mediana de um patch central, que e o FALLBACK da
  inferencia oficial, nao a Eq. 4 (mediana na mascara do BiRefNet). Quando o centro
  cai no fundo, os rotulos "fundo" e "foco" trocam de lugar. Isso NAO invalida a
  comparacao pareada (os dois modelos recebem mapa e mascaras identicos), mas torna
  fragil a leitura ABSOLUTA de cada regiao. Refazer com a mascara do BiRefNet e a
  proxima correcao.
- 40 imagens: o intervalo de confianca da LVCorr de fundo ainda tem ~0,15 de
  meia-largura.
- 512 px, nao resolucao original com tiling.
- A rota c usa como alvo sempre a foto de f/2.0, entao nem ela ensina multiplos
  niveis de bokeh para a mesma cena. O split `test` do RealBokeh_3MP (222 cenas com
  5 aberturas reais cada, sem contaminacao com o treino) e o dado certo para o
  proximo passo.

## Conteudo

- `data/resumo_modelos.parquet` — uma linha por modelo
- `data/resumo_pares.parquet` — comparacoes pareadas
- `data/por_imagem_sweep.parquet` / `data/por_imagem_nulo.parquet` — medida bruta
- `data/resumo_realbokeh.parquet` — o teto medido nas fotos reais
- `sanidade/` — os testes que validam a propria medicao
- `codigo/` — todo o codigo que gerou estes numeros

Repos irmaos: `juliadollis/bokeh-controlabilidade-sweep` (imagens geradas),
`juliadollis/bokeh-controlabilidade-lv` (escalares por ponto),
`juliadollis/bokeh-controlabilidade-realbokeh` (alvo real).
