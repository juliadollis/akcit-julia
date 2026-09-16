# Tabela 2 reproduzida — defocus deblurring

Valores `publicado` são do arXiv 2512.16923v3, Tabela 2.
Valores `medido` saíram deste pipeline.

**Antes de tirar conclusão de um `medido` contra um `publicado`, leia o
`PROTOCOLO.md`**: o paper não informa a resolução de avaliação nem a
variante de cada métrica, e as duas coisas mudam os números.

## DPDD — `akcit-pixel/DDPD:test`

split de teste, o benchmark canônico do DPDD. Esperado: 75 imagens.

| linha | origem | LPIPS ↓ | DISTS ↓ | CLIP-IQA ↑ | MANIQA ↑ | MUSIQ ↑ |
|---|---|---|---|---|---|---|
| Input | publicado | 0.3485 | 0.1827 | 0.4337 | 0.3325 | 45.5376 |
| GenRefocus (paper) | publicado | 0.1440 | 0.0772 | 0.4755 | 0.3452 | 49.4122 |
| nosso final (60k) | medido | 0.1744 | 0.1021 | 0.5723 | 0.4291 | 58.2761 |

## RealDOF — `akcit-pixel/RealDOF:validation`

único split do repo; corresponde ao conjunto de teste de 50 imagens do RealDOF. Esperado: 50 imagens.

| linha | origem | LPIPS ↓ | DISTS ↓ | CLIP-IQA ↑ | MANIQA ↑ | MUSIQ ↑ |
|---|---|---|---|---|---|---|
| Input | publicado | 0.5241 | 0.2865 | 0.3562 | 0.2213 | 28.7087 |
| GenRefocus (paper) | publicado | 0.2408 | 0.1126 | 0.4595 | 0.2884 | 43.5222 |
| nosso final (60k) | medido | 0.2291 | 0.1089 | 0.4430 | 0.3358 | 35.1719 |

## Protocolo de cada linha medida

### DPDD · nosso final (60k)

- dataset: `akcit-pixel/DDPD:test` — 75 amostras
- resoluções da métrica: 1680×1120
- `long_side`=0 · GT com resize: True (`resize_and_pad_image(long_side=0)`)
- modo: modelo · adapter: cond-only · texto: igual_ao_main
- `guidance_scale`=3.5 · `steps`=28 · seed=42
- variantes de métrica: LPIPS=lpips+, DISTS=dists, CLIP-IQA=clipiqa+, MANIQA=maniqa-kadid, MUSIQ=musiq

- peso: `deblur.safetensors` (1854533784 bytes, sha256 `a1ed05e48f7a3469…`)

### RealDOF · nosso final (60k)

- dataset: `akcit-pixel/RealDOF:validation` — 50 amostras
- resoluções da métrica: 2304×1536, 2320×1520, 2320×1536, 2336×1536, 2336×1552
- `long_side`=0 · GT com resize: True (`resize_and_pad_image(long_side=0)`)
- modo: modelo · adapter: cond-only · texto: igual_ao_main
- `guidance_scale`=3.5 · `steps`=28 · seed=42
- variantes de métrica: LPIPS=lpips+, DISTS=dists, CLIP-IQA=clipiqa+, MANIQA=maniqa-kadid, MUSIQ=musiq

- peso: `deblur.safetensors` (1854533784 bytes, sha256 `a1ed05e48f7a3469…`)
