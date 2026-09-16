#!/usr/bin/env bash
# Cadeia da campanha RealBokeh-test numa GPU. Sequencial, um container por vez.
#   uso: cadeia_rb.sh <gpu> <k_escala> <limite> <item1> [item2 ...]
# itens validos: oficial | semtreino | nosso | oficial-centro
set -uo pipefail
GPU="$1"; KESC="$2"; LIM="$3"; shift 3
B=/raid/user_juliadollis/julia_docker
BENCH=juliadollis/bokeh-bench-realbokeh-test
for item in "$@"; do
  case "$item" in
    oficial)        ARGS="--lora-repo nycu-cplab/Genfocus-Model --lora-file bokehNet.safetensors --nome oficial-paper-RB --saida juliadollis/bokeh-eval-rb-oficial"; FOCO=mascara;;
    semtreino)      ARGS="--lora-repo none --nome sem-treino-RB --saida juliadollis/bokeh-eval-rb-semtreino"; FOCO=mascara;;
    nosso)          ARGS="--lora-repo juliadollis/genrefocus-bokehnet-fase2-real --lora-file bokeh.safetensors --nome nosso-fase2-RB --saida juliadollis/bokeh-eval-rb-nosso"; FOCO=mascara;;
    oficial-centro) ARGS="--lora-repo nycu-cplab/Genfocus-Model --lora-file bokehNet.safetensors --nome oficial-paper-RB-FOCOCENTRO --saida juliadollis/bokeh-eval-rb-oficial-centro"; FOCO=centro;;
    oficial-focoalvo) ARGS="--lora-repo nycu-cplab/Genfocus-Model --lora-file bokehNet.safetensors --nome oficial-paper-RB-FOCOALVO --saida juliadollis/bokeh-eval-rb-oficial-focoalvo"; FOCO=alvo_nitido;;
    nosso-focoalvo) ARGS="--lora-repo juliadollis/genrefocus-bokehnet-fase2-real --lora-file bokeh.safetensors --nome nosso-fase2-RB-FOCOALVO --saida juliadollis/bokeh-eval-rb-nosso-focoalvo"; FOCO=alvo_nitido;;
    oficial-k300)   ARGS="--lora-repo nycu-cplab/Genfocus-Model --lora-file bokehNet.safetensors --nome oficial-paper-RB-k300 --saida juliadollis/bokeh-eval-rb-oficial-k300"; FOCO=mascara;;
    semtreino-k300) ARGS="--lora-repo none --nome sem-treino-RB-k300 --saida juliadollis/bokeh-eval-rb-semtreino-k300"; FOCO=mascara;;
    nosso-k300)     ARGS="--lora-repo juliadollis/genrefocus-bokehnet-fase2-real --lora-file bokeh.safetensors --nome nosso-fase2-RB-k300 --saida juliadollis/bokeh-eval-rb-nosso-k300"; FOCO=mascara;;
    *) echo "item desconhecido: $item"; continue;;
  esac
  echo "[cadeia gpu$GPU] >>> $item (foco=$FOCO, k_escala=$KESC) $(date -u +%H:%M:%SZ)"
  PLANO_FOCO=$FOCO bash $B/roda.sh "$GPU" "$item" run_3models.py $ARGS \
      --dataset-entrada "$BENCH" --limite "$LIM" --long-side 512 --lote 10 \
      --k-escala "$KESC" > "$B/rb_${item}.log" 2>&1
  echo "[cadeia gpu$GPU] <<< $item rc=$? $(date -u +%H:%M:%SZ)"
done
echo "[cadeia gpu$GPU] CADEIA COMPLETA $(date -u +%H:%M:%SZ)"
