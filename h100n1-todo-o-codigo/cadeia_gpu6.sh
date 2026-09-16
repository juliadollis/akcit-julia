#!/usr/bin/env bash
# GPU 6: fecha o n=6 do teto 50 e depois ataca o teto do fmax com 768 px.
set -uo pipefail
cd /raid/user_juliadollis/julia_docker
bash roda_passo4.sh 6 b3 50 5 1
echo "[cadeia6] teto50 seed 5 fechou, indo para o 768 $(date -u +%FT%H:%M:%SZ)"
SIZE=768 BATCH=4 ACUM=2 GRADCKPT=1 bash roda_passo4.sh 6 b0 - 0 3
echo "[cadeia6] FIM $(date -u +%FT%H:%M:%SZ)"
