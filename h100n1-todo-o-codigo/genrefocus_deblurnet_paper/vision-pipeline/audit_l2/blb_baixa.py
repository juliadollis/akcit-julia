#!/usr/bin/env python3
"""Baixa SO o necessario do BLB, com retomada e recuo.

Usa o endpoint direto drive.usercontent.google.com, que aguenta melhor que o
gdown por arquivo. Ignora os 670 EXR e as disparidades corrompidas de proposito.
Por cena: info.json (K e planos de refoco), image.jpg (AIF), disparity.jpg,
e os 50 bokeh_<K>_<refoco>.jpg.
"""
import os, re, time, random, urllib.request

LOG = "/workspace/ids/blb_download.log"
DEST = "/workspace/blb/data"
URL = "https://drive.usercontent.google.com/download?id=%s&export=download&confirm=t"

cenas, atual, itens = [], None, []
for ln in open(LOG, errors="ignore"):
    m = re.match(r"Retrieving folder \S+ (\d+)\s*$", ln)
    if m:
        atual = m.group(1); continue
    m = re.match(r"Processing file (\S+) (\S+)\s*$", ln)
    if m and atual:
        itens.append((atual, m.group(1), m.group(2)))

def pri(nome):
    if nome == "info.json": return 0
    if nome == "image.jpg": return 1
    if nome == "disparity.jpg": return 2
    if nome.startswith("bokeh_") and nome.endswith(".jpg"): return 3
    return None

alvo = sorted([(p, c, i, n) for c, i, n in itens if (p := pri(n)) is not None],
              key=lambda x: (x[0], x[1], x[3]))
print("desejados: %d de %d" % (len(alvo), len(itens)), flush=True)

ok_tot = falhas = 0
for _, cena, fid, nome in alvo:
    d = os.path.join(DEST, cena); os.makedirs(d, exist_ok=True)
    out = os.path.join(d, nome)
    if os.path.exists(out) and os.path.getsize(out) > 1000:
        ok_tot += 1; continue
    feito = False
    for tent in range(4):
        try:
            req = urllib.request.Request(URL % fid, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as r, open(out, "wb") as f:
                f.write(r.read())
        except Exception as e:
            print("  http %s/%s: %s" % (cena, nome, str(e)[:60]), flush=True)
        if os.path.exists(out) and os.path.getsize(out) > 1000:
            with open(out, "rb") as f:
                cab = f.read(16)
            if not cab.startswith(b"<!DOCTYPE") and not cab.startswith(b"<html"):
                feito = True; break
        time.sleep(15 * (tent + 1))
    if feito:
        ok_tot += 1; falhas = 0
        if ok_tot % 50 == 0:
            print("  %d/%d baixados" % (ok_tot, len(alvo)), flush=True)
        time.sleep(random.uniform(0.4, 1.2))
    else:
        falhas += 1
        if os.path.exists(out): os.remove(out)
        print("FALHOU %s/%s (seguidas=%d)" % (cena, nome, falhas), flush=True)
        if falhas >= 6:
            print("cota atingida, pausa de 15 min", flush=True)
            time.sleep(900); falhas = 0

print("FIM. baixados=%d de %d" % (ok_tot, len(alvo)), flush=True)
