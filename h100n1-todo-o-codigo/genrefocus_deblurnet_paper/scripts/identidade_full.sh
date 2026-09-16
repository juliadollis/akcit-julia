#!/usr/bin/env bash
# LINHA DE IDENTIDADE em amostra COMPLETA nos dois benchmarks.
# Ela nao e um modelo: e a regua. Um modelo que pontua pior que a identidade
# esta piorando a imagem em vez de aplicar bokeh. Foi ela que expos os 3 bugs
# do pipeline (depth sem redimensionar, plano de foco pelo pixel central,
# k-escala 0.01 matando o mapa). Sem ela em amostra completa a tabela nao fecha.
set -uo pipefail
B=/raid/user_juliadollis/julia_docker
P="$B/genrefocus_deblurnet_paper"
T="$(grep -h '^HF_TOKEN' "$P/.env" | cut -d= -f2)"

roda () {
  local tag="$1" entrada="$2" saida="$3" nome="$4"
  echo "[identidade] >>> $tag $(date -u +%H:%M:%SZ)"
  docker run --rm --name "julia_ident_${tag}" --gpus '"device=7"' \
    --user "$(id -u):$(id -g)" --shm-size=16g \
    -v "$P/vision-pipeline":/workspace/vision-pipeline \
    -v "$B/hf-cache-julia":/workspace/hf-cache \
    -e HF_HOME=/workspace/hf-cache -e TORCH_HOME=/workspace/hf-cache/torch \
    -e XDG_CACHE_HOME=/workspace/hf-cache/xdg -e HOME=/workspace/hf-cache/home \
    -e HF_TOKEN="$T" -e HUGGINGFACE_HUB_TOKEN="$T" \
    -w /workspace/vision-pipeline julia-genrefocus-eval:2.0 \
    python3 baseline_identidade.py --dataset-entrada "$entrada" --saida "$saida" \
      --nome "$nome" --limite 0 --long-side 512 --lote 20 \
    > "$B/identidade_${tag}.log" 2>&1
  echo "[identidade] <<< $tag rc=$? $(date -u +%H:%M:%SZ)"
}

roda rb juliadollis/bokeh-bench-realbokeh-test-v2 juliadollis/bokeh-eval-rb-identidade-full LINHA-DE-IDENTIDADE-RB-full
roda rd akcit-pixel/RealDOF                       juliadollis/bokeh-eval-rd-identidade-full LINHA-DE-IDENTIDADE-RD-full
echo "[identidade] FIM $(date -u +%H:%M:%SZ)"
