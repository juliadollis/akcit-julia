#!/usr/bin/env bash
# Espera a cadeia de uma GPU terminar e emenda a proxima. Nao mata nada.
set -uo pipefail
GPU="$1"; LOGESPERA="$2"; shift 2
B=/raid/user_juliadollis/julia_docker
until grep -q "CADEIA COMPLETA" "$LOGESPERA" 2>/dev/null; do sleep 60; done
echo "[depois gpu$GPU] cadeia anterior terminou; emendando: $*"
bash $B/cadeia_rb.sh "$GPU" 3.0 40 "$@"
