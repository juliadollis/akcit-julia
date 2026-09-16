#!/usr/bin/env bash
# Recalcula metricas POR IMAGEM (com IC95) numa GPU, para a lista dada.
# Nao refaz inferencia: le as imagens ja geradas nos repos bokeh-eval-*.
set -uo pipefail
GPU="${1:?uso: roda_porimg.sh <gpu>}"
B=/raid/user_juliadollis/julia_docker
P=$B/genrefocus_deblurnet_paper
HF_TOKEN="$(grep -h "^HF_TOKEN" $P/.env | cut -d= -f2)"
docker run --rm --name "julia_porimg_gpu${GPU}" --gpus "\"device=${GPU}\"" \
  --user "$(id -u):$(id -g)" --shm-size=16g \
  -v "$P/vision-pipeline":/workspace/vision-pipeline \
  -v "$B/hf-cache-julia":/workspace/hf-cache -v "$B":/host \
  -e HF_HOME=/workspace/hf-cache -e TORCH_HOME=/workspace/hf-cache/torch \
  -e XDG_CACHE_HOME=/workspace/hf-cache/xdg -e HOME=/workspace/hf-cache/home \
  -e HF_TOKEN="$HF_TOKEN" -e HUGGINGFACE_HUB_TOKEN="$HF_TOKEN" \
  -w /workspace/vision-pipeline julia-genrefocus-eval:3.0 \
  python3 metricas_por_imagem.py --lista /host/lista_porimg_gpu${GPU}.txt \
     --saida-local /host/por_imagem_gpu${GPU}
echo "[porimg gpu$GPU] rc=$? $(date -u +%H:%M:%SZ)"
