#!/usr/bin/env bash
# Roda um item da campanha do benchmark RealBokeh-test numa GPU.
#   uso: roda.sh <gpu> <nome_container> <script> <argumentos...>
set -uo pipefail
GPU="$1"; NOME="$2"; SCRIPT="$3"; shift 3
B=/raid/user_juliadollis/julia_docker
PROJ="$B/genrefocus_deblurnet_paper"
HF_TOKEN="$(grep -h "^HF_TOKEN" "$PROJ/.env" | cut -d= -f2)"
docker run --rm --name "julia_rb_${NOME}" --gpus "\"device=${GPU}\"" \
    --user "$(id -u):$(id -g)" --shm-size=16g \
    -v "$PROJ/vision-pipeline":/workspace/vision-pipeline \
    -v "$B/hf-cache-julia":/workspace/hf-cache \
    -v "$B/pylibs_extra":/workspace/pylibs_extra \
    -e HF_HOME=/workspace/hf-cache -e TORCH_HOME=/workspace/hf-cache/torch \
    -e XDG_CACHE_HOME=/workspace/hf-cache/xdg -e HOME=/workspace/hf-cache/home \
    -e PYTHONPATH=/workspace/pylibs_extra \
    -e HF_TOKEN="$HF_TOKEN" -e HUGGINGFACE_HUB_TOKEN="$HF_TOKEN" \
    -e BOKEH_METRICS_REPO="${BOKEH_METRICS_REPO:-juliadollis/bokeh-eval-rb-metricas}" \
    -e PLANO_FOCO="${PLANO_FOCO:-mascara}" \
    -w /workspace/vision-pipeline julia-genrefocus-eval:3.0 \
    python3 "$SCRIPT" "$@"
