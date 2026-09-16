#!/usr/bin/env python3
"""Publica a reproducao do LF-Bokeh num repo HF NOVO e PRIVADO.

PRIVADO por default: as imagens sao de terceiros (BokehMe, Apache-2.0). Apache
permite redistribuir, mas repo publico e dificil de despublicar depois, entao a
escolha segura e privado e abrir de proposito se for o caso.
"""
import argparse, os, glob, sys
from huggingface_hub import HfApi

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="/workspace/out_blb")
ap.add_argument("--repo", required=True)
ap.add_argument("--card", default="/workspace/vision-pipeline/audit_l2/blb_card.md")
ap.add_argument("--publico", action="store_true")
a = ap.parse_args()

tok = os.environ.get("HF_TOKEN")
api = HfApi()
arqs = sorted(glob.glob(os.path.join(a.dir, "*.parquet")))
if not arqs:
    print("nenhum parquet em " + a.dir); sys.exit(1)
print("parquets a subir: %d" % len(arqs))

info = None
try:
    info = api.repo_info(a.repo, repo_type="dataset", token=tok)
except Exception:
    pass
if info is not None:
    print("!!! repo JA EXISTE: " + a.repo)
    print("    abortando para nao sobrescrever. escolha outro nome.")
    sys.exit(2)

api.create_repo(a.repo, repo_type="dataset", private=not a.publico, token=tok)
print("repo criado (privado=%s)" % (not a.publico))

for f in arqs:
    api.upload_file(path_or_fileobj=f, path_in_repo="data/" + os.path.basename(f),
                    repo_id=a.repo, repo_type="dataset", token=tok)
    print("  subiu " + os.path.basename(f))

if os.path.exists(a.card):
    api.upload_file(path_or_fileobj=a.card, path_in_repo="README.md",
                    repo_id=a.repo, repo_type="dataset", token=tok)
    print("  subiu README.md (documento de divergencias)")

print("PRONTO: https://huggingface.co/datasets/" + a.repo)
