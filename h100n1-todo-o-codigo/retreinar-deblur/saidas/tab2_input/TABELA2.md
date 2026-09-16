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

## RealDOF — `akcit-pixel/RealDOF:validation`

único split do repo; corresponde ao conjunto de teste de 50 imagens do RealDOF. Esperado: 50 imagens.

| linha | origem | LPIPS ↓ | DISTS ↓ | CLIP-IQA ↑ | MANIQA ↑ | MUSIQ ↑ |
|---|---|---|---|---|---|---|
| Input | publicado | 0.5241 | 0.2865 | 0.3562 | 0.2213 | 28.7087 |
| GenRefocus (paper) | publicado | 0.2408 | 0.1126 | 0.4595 | 0.2884 | 43.5222 |

## Protocolo de cada linha medida
