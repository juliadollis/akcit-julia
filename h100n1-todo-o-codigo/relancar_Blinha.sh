#!/usr/bin/env bash
# Relanca o treino da condicao B' (controle de capacidade do condicionamento
# geometrico), parado em 2026-09-10 no step ~5500 para liberar as GPUs 5 e 6
# para a avaliacao do B, a pedido da usuaria.
#
# O config tem resume: true e salva a cada 250 steps, entao ele RETOMA do
# step_5500.pt. Nada foi perdido.
#
# B' e o controle que troca G por ruido de mesma estatistica. Sem ele, um ganho
# do B nao e interpretavel: pode ser capacidade extra em vez de geometria.
set -euo pipefail
B=/raid/user_juliadollis/julia_docker
docker run -d --rm --name julia_geo_Blinha --gpus '"device=5,6"' \
  --user "$(id -u):$(id -g)" --shm-size=32g \
  -v "$B":/workspace \
  -v "$B/hf-cache-julia":/workspace/hf-cache \
  -v "$B/f0b_out":/saida \
  -e TRAIN_CONFIG=configs/train_bokeh_geo_Blinha.yaml \
  -e NUM_GPUS=2 \
  -e INIT_LORA=outputs/bokehnet_synth_2gpu \
  -e HF_REPO_ID=bokehnet-geo-Blinha \
  -e MPLCONFIGDIR=/tmp/mpl \
  -w /workspace/genrefocus_deblurnet_paper \
  julia-genrefocus:1.0 bash scripts/docker_bokeh_bootstrap.sh
echo "B' relancado; retoma do ultimo checkpoint em outputs/bokehnet-geo-Blinha"
