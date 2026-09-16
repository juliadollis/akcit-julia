#!/usr/bin/env bash
# Espera a passada de metricas por imagem terminar e ja monta a tabela final,
# empurrando tudo para o Hub. Existe para a etapa seguinte nao ficar esperando
# alguem notar que a anterior acabou.
set -uo pipefail
B=/raid/user_juliadollis/julia_docker
P=$B/genrefocus_deblurnet_paper
HF_TOKEN="$(grep -h "^HF_TOKEN" $P/.env | cut -d= -f2)"

echo "[encadeia] aguardando os containers julia_porimg_* $(date -u +%H:%M:%SZ)"
while [ "$(docker ps --format {{.Names}} | grep -c ^julia_porimg_)" -gt 0 ]; do sleep 120; done
echo "[encadeia] passada de metricas terminou $(date -u +%H:%M:%SZ)"

# junta os parquets num so lugar e empurra as linhas por imagem para o Hub
docker run --rm --name julia_tabela --gpus '"device=0"' \
  --user "$(id -u):$(id -g)" --shm-size=16g \
  -v "$P/vision-pipeline":/workspace/vision-pipeline \
  -v "$B/hf-cache-julia":/workspace/hf-cache -v "$B":/host \
  -e HF_HOME=/workspace/hf-cache -e TORCH_HOME=/workspace/hf-cache/torch \
  -e XDG_CACHE_HOME=/workspace/hf-cache/xdg -e HOME=/workspace/hf-cache/home \
  -e HF_TOKEN="$HF_TOKEN" -e HUGGINGFACE_HUB_TOKEN="$HF_TOKEN" \
  -w /workspace/vision-pipeline julia-genrefocus-eval:3.0 \
  python3 tabela_estilo_paper.py \
     --entrada "/host/por_imagem_gpu*/por_imagem.parquet" \
     --saida-md /host/TABELA_FINAL.md \
     --saida-hf juliadollis/genrefocus-tabela-final
echo "[encadeia] tabela rc=$? $(date -u +%H:%M:%SZ)"
