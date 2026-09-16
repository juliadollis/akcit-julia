#!/usr/bin/env python3
"""Biblioteca da avaliacao de CONTROLABILIDADE de bokeh (LVCorr).

Convencao de sinal (declarada, ver relatorio):
    LV = variancia do laplaciano = NITIDEZ. Obedecer ao comando de bokeh
    significa NITIDEZ CAINDO quando o comando SOBE. Definimos

        LVCorr = - spearman(alpha, LV)

    de modo que +1 = obediencia perfeita, 0 = indiferenca, -1 = obediencia
    invertida. O sinal negado e o que torna o numero comparavel com o do paper
    (Tab. 3, valores positivos). O avaliador antigo correlacionava K com LV sem
    negar, e por isso a obediencia aparecia como numero NEGATIVO.
"""
from __future__ import annotations

import io
import numpy as np
import cv2
from PIL import Image
from scipy.stats import spearmanr, pearsonr

# ---------------------------------------------------------------- laplaciano

def to_gray(img: Image.Image) -> np.ndarray:
    a = np.array(img.convert("RGB"))
    return cv2.cvtColor(cv2.cvtColor(a, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)


def lap_resposta(img: Image.Image) -> np.ndarray:
    return cv2.Laplacian(to_gray(img), cv2.CV_64F)


def lv_total(img: Image.Image) -> float:
    return float(lap_resposta(img).var())


def lv_mascarado(resp: np.ndarray, mask: np.ndarray) -> float:
    """Variancia do laplaciano restrita a uma mascara ja erodida."""
    if mask.sum() < 64:
        return float("nan")
    return float(resp[mask].var())


def erode(mask: np.ndarray, px: int = 5) -> np.ndarray:
    if px <= 0:
        return mask
    k = np.ones((px, px), np.uint8)
    return cv2.erode(mask.astype(np.uint8), k, iterations=1).astype(bool)


# ------------------------------------------------------------- mapa de base

def mapa_base(depth_arr: np.ndarray, w: int, h: int, patch_frac: float = 0.06):
    """B_hat em [0,1]: |disp - disp_foco| normalizado pelo proprio p99.5.

    CORRECOES em relacao a inference/src/pipelines/bokeh_net.py:
      (1) o depth e reamostrado para (w,h) da imagem GERADA. No pipeline antigo
          o .npy ficava na resolucao original (1120x1680) enquanto a imagem ia a
          512 — o mapa de condicionamento entrava com resolucao diferente da
          imagem, desalinhado em escala.
      (2) o plano de foco e a MEDIANA de um patch central (6% do lado), nao um
          unico pixel. No pipeline antigo o indice do "centro" era calculado com
          as dimensoes de 512 e aplicado no array de 1680 de largura, ou seja o
          ponto amostrado nem era o centro: caia a ~15% da imagem.

    Devolve (B_hat, p995, disp_foco).
    """
    d = cv2.resize(depth_arr.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    safe = np.where(d > 0.0, d, np.finfo(np.float32).max)
    disp = 1.0 / safe
    ph = max(8, int(h * patch_frac)) // 2
    pw = max(8, int(w * patch_frac)) // 2
    cy, cx = h // 2, w // 2
    patch = disp[max(0, cy - ph):cy + ph + 1, max(0, cx - pw):cx + pw + 1]
    disp_foco = float(np.median(patch))
    b = np.abs(disp - np.float32(disp_foco))
    p995 = float(np.percentile(b, 99.5))
    if p995 <= 0:
        p995 = float(max(b.max(), 1e-8))
    return np.clip(b / p995, 0.0, 1.0).astype(np.float32), p995, disp_foco


def mascaras(b_hat: np.ndarray, lim_fundo: float = 0.5, lim_foco: float = 0.05):
    return erode(b_hat >= lim_fundo), erode(b_hat <= lim_foco)


# ------------------------------------------------------ agregacao / metricas

def lvcorr(alphas, lvs) -> float:
    """LVCorr = -spearman(alpha, LV). NaN se degenerado."""
    a = np.asarray(alphas, float)
    v = np.asarray(lvs, float)
    ok = np.isfinite(v)
    if ok.sum() < 3 or np.std(v[ok]) == 0 or np.std(a[ok]) == 0:
        return float("nan")
    rho, _ = spearmanr(a[ok], v[ok])
    return float(-rho)


def lvcorr_pearson_log(alphas, lvs) -> float:
    a = np.asarray(alphas, float)
    v = np.asarray(lvs, float)
    ok = np.isfinite(v) & (v > 0)
    if ok.sum() < 3 or np.std(np.log(v[ok])) == 0:
        return float("nan")
    r, _ = pearsonr(a[ok], np.log(v[ok]))
    return float(-r)


def faixa_dinamica(alphas, lvs) -> float:
    """LV no menor alpha / LV no maior alpha. 1.0 = o comando nao fez nada."""
    a = np.asarray(alphas, float)
    v = np.asarray(lvs, float)
    ok = np.isfinite(v)
    if ok.sum() < 2:
        return float("nan")
    a, v = a[ok], v[ok]
    i0, i1 = int(np.argmin(a)), int(np.argmax(a))
    if v[i1] <= 0:
        return float("nan")
    return float(v[i0] / v[i1])


def frac_monotona(alphas, lvs) -> float:
    """Fracao de degraus consecutivos em que a nitidez CAIU (ideal = 1.0)."""
    a = np.asarray(alphas, float)
    v = np.asarray(lvs, float)
    ok = np.isfinite(v)
    a, v = a[ok], v[ok]
    o = np.argsort(a)
    v = v[o]
    if len(v) < 2:
        return float("nan")
    return float((np.diff(v) < 0).mean())


def bootstrap_ic(vals, n=5000, seed=0):
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    if len(v) < 3:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    m = rng.choice(v, size=(n, len(v)), replace=True).mean(axis=1)
    return (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


# ------------------------------------------- oraculo de bokeh (CPU, sanidade)

def _pilha_blur(a: np.ndarray, niveis):
    saida = []
    for rr in niveis:
        if rr < 0.3:
            saida.append(a)
        else:
            sg = rr / 2.0
            k = int(2 * round(2.5 * sg) + 1)
            saida.append(cv2.GaussianBlur(a, (k, k), sg))
    return saida


def bokeh_oraculo(img: Image.Image, b_hat: np.ndarray, alpha: float,
                  raio_max: float = 12.0, global_: bool = False) -> Image.Image:
    """Bokeh classico com raio CONTINUO: raio(x) = raio_max * alpha * B_hat(x).

    CONTROLE POSITIVO: um 'modelo' que obedece ao comando por construcao.
    Se a cadeia de medicao estiver correta, LVCorr(oraculo) tem de dar ~ +1.

    A composicao entre niveis de blur e por interpolacao LINEAR (peso
    triangular), nao por mascara dura. A versao com mascara dura criava bordas
    de camada que AUMENTAVAM a resposta laplaciana e faziam o proprio controle
    positivo reprovar — foi o teste de sanidade que pegou isso.
    Alem disso, um raio abaixo de 0.3 px NAO aplica blur nenhum: com o piso de
    kernel 3x3 da versao anterior, um comando numericamente nulo ainda borrava.
    """
    a = np.array(img.convert("RGB")).astype(np.float32)
    campo = np.ones_like(b_hat) if global_ else b_hat
    r = raio_max * float(alpha) * campo
    r_top = float(r.max())
    if r_top < 0.3:
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    niveis = np.linspace(0.0, r_top, 9)
    passo = float(niveis[1] - niveis[0])
    pilha = _pilha_blur(a, niveis)
    acc = np.zeros_like(a)
    wsum = np.zeros(r.shape, np.float32)
    for jj, rr in enumerate(niveis):
        w = np.clip(1.0 - np.abs(r - rr) / passo, 0.0, 1.0).astype(np.float32)
        acc += pilha[jj] * w[..., None]
        wsum += w
    acc /= np.maximum(wsum, 1e-6)[..., None]
    return Image.fromarray(np.clip(acc, 0, 255).astype(np.uint8))


def bokeh_oraculo_surdo(img: Image.Image, b_hat: np.ndarray, alpha: float,
                        raio_max: float = 12.0) -> Image.Image:
    """CONTROLE NEGATIVO: ignora alpha (usa sempre 0.5)."""
    return bokeh_oraculo(img, b_hat, 0.5, raio_max)


def pil_bytes(img: Image.Image) -> bytes:
    b = io.BytesIO()
    img.save(b, format="PNG")
    return b.getvalue()
