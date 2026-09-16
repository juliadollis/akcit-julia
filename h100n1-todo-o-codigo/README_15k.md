---
license: other
license_name: flux-1-dev-non-commercial-license
license_link: https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md
base_model: black-forest-labs/FLUX.1-dev
library_name: diffusers
tags:
  - lora
  - flux
  - defocus-deblurring
  - image-restoration
  - genrefocus
language:
  - en
---

# GenRefocus DeblurNet — reprodução (AKCIT-PIXEL) · step 15.500 (mínimo da val_loss)

LoRA sobre **FLUX.1-dev** que reproduz o **Stage 1 (DeblurNet)** do
*Generative Refocusing: Flexible Defocus Control from a Single Image*
([arXiv:2512.16923v3](https://arxiv.org/abs/2512.16923)).

Recebe uma imagem com desfoque de foco (defocus blur) e devolve a versão
all-in-focus. Não é o peso dos autores — é um treino independente nosso a
partir da descrição do paper. O peso oficial deles é
[`nycu-cplab/Genfocus-Model`](https://huggingface.co/nycu-cplab/Genfocus-Model).

> ⚠️ **Este NÃO é o checkpoint recomendado.** Use
> [`akcit-pixel/genrefocus-deblurnet`](https://huggingface.co/akcit-pixel/genrefocus-deblurnet)
> (step 60.000), que é **10% melhor em LPIPS** no DPDD.
>
> Este aqui é o **mínimo da `val_loss`** (step 15.500). Está publicado porque
> ele documenta um achado que vale registrar: **a loss de validação escolheu o
> checkpoint errado**. Ver "Por que este existe".

---

## Arquivos

| arquivo | o que é |
|---|---|
| `deblurNet.safetensors` | o peso, step 15.500 (mínimo da `val_loss`) |
| `procedencia.json` | config, step, sha256 e o run do wandb que o gerou |

## Por que este existe

Durante o treino, a validação roda a cada 500 steps com **σ e ruído fixos** —
determinística, comparável entre steps. Ela mede a *loss de flow matching*.

```
step 15.500   val 0.2703   ← mínimo
step 60.000   val 0.2815   (+4,1%)
```

A leitura óbvia seria overfit a partir de 15.5k. **Está errada.** Medido no
benchmark perceptual:

```
DPDD LPIPS   step 15.500 → 0.1942
             step 60.000 → 0.1744   (10% MELHOR)
```

Ou seja: a loss de denoise **não é proxy de qualidade perceptual** neste
regime, e selecionar checkpoint por ela degrada o resultado. Este peso fica
publicado como evidência disso — e como alerta para quem for usar `val_loss`
para *early stopping* em difusão.

---

## Como usar

Este LoRA segue o esquema de condicionamento do
[OminiControl](https://arxiv.org/abs/2411.15098): a imagem de entrada entra como
**tokens de condição** concatenados, não como `img2img`. Ele **não funciona**
com um `FluxPipeline` comum — precisa do `generate` do repositório dos autores
([nycu-cplab/Genfocus](https://github.com/nycu-cplab/Genfocus)).

```python
import torch
from diffusers import FluxPipeline
from Genfocus.pipeline.flux import Condition, generate, seed_everything

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev",
                                    torch_dtype=torch.bfloat16).to("cuda")
pipe.load_lora_weights(".", weight_name="deblurNet.safetensors", adapter_name="deblurring")
pipe.set_adapters(["deblurring"])

img = ...  # PIL RGB, lados múltiplos de 16
seed_everything(42)
out = generate(
    pipe, height=img.height, width=img.width,
    prompt="a sharp photo with everything in focus",   # o prompt é load-bearing
    num_inference_steps=28,
    conditions=[Condition(img, "deblurring", [0, 0], 1.0)],
    # main_adapter NÃO é passado -> None -> LoRA só no branch de condição
).images[0]
```

Três detalhes que mudam o resultado se você errar:

- **`prompt`** tem que ser exatamente `"a sharp photo with everything in focus"`. Foi o único usado no treino.
- **`guidance_scale`** fica no default do `generate`, que é **3.5**. É o valor usado no treino deste peso.
- **`main_adapter=None`** (não passe o parâmetro). Este LoRA é **cond-only**: age só nos tokens de condição. Passar `main_adapter="deblurring"` liga a LoRA num branch que nunca foi treinado.

Resolução: o `generate` processa em resolução nativa com *tiling* de 512×512.
Não redimensione a entrada; só alinhe os lados a múltiplos de 16.

---

## Detalhes do treino

| | |
|---|---|
| Backbone | `black-forest-labs/FLUX.1-dev` (congelado) |
| Adaptação | LoRA rank **128**, alpha 128 (`scaling = 1`) |
| Módulos LoRA | **343** — idêntico ao artefato oficial |
| Variante | **cond-only** (`adapters = [texto=None, main=None, cond=LoRA]`) |
| Steps | **60.000** |
| Batch efetivo | **32** (`batch 1 × accum 8 × 4 GPUs`) |
| Precisão | bf16 |
| Guidance (treino) | 3.5 no main/texto, 1.0 na condição |
| Objetivo | flow matching, `x_t = (1−σ)x₀ + σε`, alvo `ε − x₀` |
| σ | logit-normal com shift de `calculate_shift(seq_len)` |
| Otimizador | AdamW, lr 1e-4, wd 1e-4, cosseno com warmup 500 |
| Hardware | 4× H100 80GB, ~4,3 s/step, ~3 dias |

Rank, steps, batch e a decomposição `1 × 8 × 4` vêm do §4.1 do paper, literais.
Otimizador, learning rate, scheduler, warmup, resolução de treino e guidance
**não são especificados pelo paper** — são escolha nossa, documentada.

### Módulos LoRA: 343, não 344

Uma armadilha que custou uma rodada: o PEFT casa `target_modules` por sufixo, e
a projeção final do `FluxTransformer2DModel` chama-se `proj_out` — o mesmo nome
dos `proj_out` dos blocos single. Uma lista de strings captura os dois, e a
projeção final roda **fora** do controle por branch do `specify_lora`, ficando
ativa mesmo com `main_adapter=None`. O resultado seria um LoRA que não é
cond-only de verdade. Este peso usa regex ancorado por tipo de bloco e tem
**343** módulos, batendo com o checkpoint oficial.

---

## Dados

| fonte | split | usadas |
|---|---|---|
| `akcit-pixel/DDPD` | train | 344 (split inteiro) |
| `akcit-pixel/RealBokeh` | train | 3.000, filtradas por variância do Laplaciano |
| | | **3.344 pares** |

Corresponde ao §B.1 do paper (*"all images from the official training split of
the DPDD dataset"* + *"top 3,000 sharpest images"* do RealBokeh_3MP).

**Sem vazamento.** Verificado por interseção de `file_name_base`:

```
DDPD train ∩ DDPD test              = 0
DDPD train ∩ DDPD validation        = 0
DDPD validation ∩ DDPD test         = 0
RealBokeh(train) ∩ RealDOF(eval)    = 0
DDPD(train) ∩ RealDOF(eval)         = 0
```

**Ressalva conhecida:** o filtro top-3000 ranqueia **linhas**, e o RealBokeh tem
~5 linhas por cena (mesma imagem nítida, aberturas diferentes). As 3.000 linhas
são ~580 cenas distintas, não 3.000 imagens distintas. O paper diz *"top 3,000
sharpest images"*, e a leitura correta é ambígua. Isso reduz a diversidade de
alvos em ~5× e é o candidato mais provável para a diferença de desempenho no
DPDD.

---

## Resultados

Todas as linhas `medido` saíram do **mesmo pipeline**, em resolução nativa,
`guidance_scale=3.5`, `steps=28`, `main_adapter=None`, zero falhas.
`publicado` é a Tabela 2 do arXiv v3.

### DPDD — `akcit-pixel/DDPD:test`, 75 imagens

| modelo | origem | LPIPS ↓ | DISTS ↓ |
|---|---|---|---|
| Input (identidade) | publicado | 0.3485 | 0.1827 |
| Input (identidade) | medido | 0.3440 | 0.1830 |
| GenRefocus (autores) | publicado | **0.1440** | **0.0772** |
| oficial `nycu-cplab` | medido | 0.1596 | 0.0844 |
| **este modelo (60k)** | medido | **0.1744** | **0.1021** |
| este modelo (best 15.5k) | medido | 0.1942 | 0.1055 |

### RealDOF — `akcit-pixel/RealDOF:validation`, 50 imagens

| modelo | origem | LPIPS ↓ | DISTS ↓ |
|---|---|---|---|
| Input (identidade) | publicado | 0.5241 | 0.2865 |
| Input (identidade) | medido | 0.5280 | 0.2865 |
| GenRefocus (autores) | publicado | **0.2408** | **0.1126** |
| este modelo (best 15.5k) | medido | **0.2410** | **0.1104** |

### Como ler isso

**A linha `Input` valida o protocolo.** Ela não depende de modelo nenhum. Ela
reproduz o publicado em LPIPS (0,7–1,3%) e DISTS (idêntico até a 4ª casa no
RealDOF). Portanto a comparação de LPIPS e DISTS com o paper **tem significado**.

**Existe uma lacuna de protocolo de ~11%.** O peso **oficial**, rodado neste
pipeline, dá 0.1596 contra 0.1440 publicado. Mesmo modelo — logo a diferença é
protocolo: o paper não informa a resolução de avaliação nem a variante exata de
cada métrica.

**Este modelo fica atrás do oficial no DPDD** (+9,3% LPIPS, +21% DISTS) e
**empata com o publicado no RealDOF** (0.2410 vs 0.2408, DISTS melhor). A
assimetria entre as duas mesas é real e ainda não tem explicação confirmada.

**CLIP-IQA, MANIQA e MUSIQ não estão aqui de propósito.** A linha `Input`
mostrou que nossas variantes (`clipiqa+`, `maniqa-kadid`) erram 14–39% contra as
publicadas, então comparar seria enganoso. Elas continuam válidas *entre* nossas
próprias linhas.

---

## Desvios conhecidos em relação ao paper

| item | paper | aqui | por quê |
|---|---|---|---|
| `gradient_checkpointing` | não diz | desligado | ~30% mais rápido; não muda a matemática |
| guidance de treino | **não menciona** | 3.5 | é o default do `generate` na inferência oficial de deblur |
| otimizador / lr / scheduler | **não menciona** | AdamW 1e-4, cosseno | escolha nossa |
| resolução de treino | **não menciona** | crop 512² do lado-menor-512 | escolha nossa |
| seleção do RealBokeh | *"top 3,000 images"* | top-3000 **linhas** (~580 cenas) | leitura ambígua, ver "Dados" |
| DPDD test | 76 imagens (canônico) | 75 no repo | falta 1 |
| rank do LoRA | §4.1 diz **128** | 128 | o artefato oficial deles tem rank **64** — o texto e o peso dos autores divergem entre si |

---

## Citação

```bibtex
@article{tuanmu2026genrefocus,
  title   = {Generative Refocusing: Flexible Defocus Control from a Single Image},
  author  = {Tuan Mu, Chun-Wei and Fan, Cheng-De and Huang, Jia-Bin and Liu, Yu-Lun},
  journal = {arXiv preprint arXiv:2512.16923},
  year    = {2026}
}
```

Licença herdada do FLUX.1-dev (não comercial).
