#!/bin/bash
#SBATCH --job-name=bokeh-a-full
#SBATCH --partition=h100n3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=20-00:00:00
#SBATCH --output=/raid/user_danielpedrozo/projects/julia/bokehnet-regen/logs/%x-%j.out
#SBATCH --error=/raid/user_danielpedrozo/projects/julia/bokehnet-regen/logs/%x-%j.err

set -euo pipefail

# =============================================================================
# Rota A completa — ~1,7K imagens x ~41 variantes = ~70K pares sintéticos
# (paper.txt:527-528, 998). O alvo de cada par é RENDERIZADO pelo BokehMe [43].
#
# TRÊS PRÉ-REQUISITOS DUROS, e nenhum é formalidade:
#
#  1. **B e C regeradas**, e `manifests/k_distribution_bc.json` construído a
#     partir dos manifestos delas. O entrypoint recusa distribuição degenerada.
#  2. **O piloto rodou** e os limiares foram ESCOLHIDOS por alguém que olhou
#     output/thresholds_rota_a_proposta.json.
#  3. **O orçamento de disco foi conferido.** O entrypoint imprime a conta antes
#     de gerar: a folga medida é ~84 GB (115 GB menos os 31 GB de B+C,
#     REGISTRO.md:585). A rota A cabe a <=1 MP por imagem e NÃO cabe a >=2 MP.
#     Se a conta não fechar, reduza --samples-per-image ANTES de rodar 70 mil
#     renders — quatro coisas desta rota não têm conserto retroativo: a fonte, a
#     multiplicidade, o scene_id e a escala de K.
#
# --time é 20 dias de propósito: o custo por render do BokehMe não foi medido
# (é o que o piloto mede), e `--time` curto num job de 70 mil renders é a forma
# mais fácil de perder trabalho feito. A retomada por sample_id existe, mas
# relançar custa a fila.
#
# NÃO cancela job nenhum. Nada é apagado: a rota retoma por sample_id, então
# relançar continua de onde parou em vez de reescrever.
# =============================================================================

PROJECT_ROOT=/raid/user_danielpedrozo/projects/julia/bokehnet-regen
SIF=/raid/user_danielpedrozo/images/transformers-pytorch-gpu.sif
BIND_ROOT=/raid/user_danielpedrozo

DATA_ROOT=${DATA_ROOT:-/raid/user_danielpedrozo/projects/julia/data}
GENPHOTO_DIR=${GENPHOTO_DIR:-${DATA_ROOT}/GenerativePhotography}
EBB_DIR=${EBB_DIR:-${DATA_ROOT}/EBB_sharp}
MODELS_DIR=${MODELS_DIR:-/raid/user_danielpedrozo/projects/julia/models}
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT_ROOT}/output/a_synthetic}
K_DIST=${K_DIST:-${PROJECT_ROOT}/manifests/k_distribution_bc.json}
SAMPLES_PER_IMAGE=${SAMPLES_PER_IMAGE:-41}
TOP_N_GENPHOTO=${TOP_N_GENPHOTO:-850}
TOP_N_EBB=${TOP_N_EBB:-850}

# --- os limiares congelados, vindos do piloto --------------------------------
# Preencha a partir de output/thresholds_rota_a_proposta.json. Cada linha vazia
# é um gate que MEDE e não bloqueia — escolha legítima, desde que seja escolha.
# Registre os valores usados em reference/ACHADOS.md.
THRESHOLDS=${THRESHOLDS:-""}
# exemplo:
# THRESHOLDS="--min-rendered-coc-p99-px 1.0 --max-defocus-saturation-ratio 0.30 \
#             --max-bokeh-over-aif-sharpness 0.90 --min-mask-area-ratio 0.005"

if [[ -z "${THRESHOLDS}" ]]; then
    echo "ERRO: THRESHOLDS vazio."
    echo "Rode o piloto, leia output/thresholds_rota_a_proposta.json, escolha os"
    echo "números e relance com:"
    echo "  sbatch --export=ALL,THRESHOLDS='--min-rendered-coc-p99-px 1.0 ...' $0"
    echo "Se a intenção for MESMO gerar 70 mil amostras sem filtro para calibrar"
    echo "depois, passe THRESHOLDS='--seed 0' explicitamente."
    exit 1
fi

mkdir -p "${PROJECT_ROOT}/logs" "${OUTPUT_DIR}"

echo "=========================================="
echo " Job:      ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo " Host:     $(hostname)"
echo " Saída:    ${OUTPUT_DIR}"
echo " K dist:   ${K_DIST}"
echo " N/img:    ${SAMPLES_PER_IMAGE}"
echo " Limiares: ${THRESHOLDS}"
echo " Data:     $(date)"
echo "=========================================="

for caminho in "${SIF}" "${PROJECT_ROOT}/third_party/BokehMe/demo.py" \
               "${PROJECT_ROOT}/output/renderer_verification.json" "${K_DIST}"; do
    [[ -e "${caminho}" ]] || { echo "ERRO: não encontrado: ${caminho}"; exit 1; }
done
for dir in "${GENPHOTO_DIR}" "${EBB_DIR}" "${MODELS_DIR}"; do
    [[ -d "${dir}" ]] || { echo "ERRO: diretório não encontrado: ${dir}"; exit 1; }
done

APP_CMD='
set -euo pipefail
cd '"${PROJECT_ROOT}"'

free_gpu=$(nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits 2>/dev/null \
           | grep -E "^[0-9]+," | sort -t, -k3 -n | head -1)
if [ -z "$free_gpu" ]; then echo "[erro] nenhuma GPU visivel."; exit 1; fi
gpu_uuid=$(echo "$free_gpu" | cut -d, -f2 | tr -d " ")
gpu_mem=$(echo "$free_gpu"  | cut -d, -f3 | tr -d " ")
echo "[gpu] mais livre: memory.used=${gpu_mem} MiB"
if [ "$gpu_mem" -gt 2000 ]; then
  echo "[erro] GPU mais livre com ${gpu_mem} MiB em uso. Abortando em vez de tomar OOM."
  exit 1
fi
export CUDA_VISIBLE_DEVICES=$gpu_uuid

export PYTHONUNBUFFERED=1
export PYTHONPATH='"${PROJECT_ROOT}"'/.pydeps-clean:'"${PROJECT_ROOT}"'/src:'"${PROJECT_ROOT}"'/third_party/BokehMe${PYTHONPATH:+:$PYTHONPATH}

python3 -c "import torch, numpy, cupy; print(\"[env] torch\", torch.__version__, \"| numpy\", numpy.__version__, \"| cupy\", cupy.__version__, \"|\", torch.cuda.get_device_name(0))"

echo
echo ">>> suíte local <<<"
python3 -m unittest discover -s tests -p "test_*.py" 2>&1 | tail -4

echo
echo ">>> rota A completa <<<"
python3 scripts/run_route_a.py \
    --output-dir '"${OUTPUT_DIR}"' \
    --samples-per-image '"${SAMPLES_PER_IMAGE}"' \
    --k-distribution '"${K_DIST}"' \
    --source GenerativePhotography='"${GENPHOTO_DIR}"' \
    --source "EBB!"='"${EBB_DIR}"' \
    --top-n-per-source GenerativePhotography='"${TOP_N_GENPHOTO}"' \
    --top-n-per-source "EBB!"='"${TOP_N_EBB}"' \
    --models-dir '"${MODELS_DIR}"' \
    --bokehme-dir '"${PROJECT_ROOT}"'/third_party/BokehMe \
    --renderer-report '"${PROJECT_ROOT}"'/output/renderer_verification.json \
    --device cuda --seed 0 '"${THRESHOLDS}"'

echo
echo ">>> validação do release (NÃO sobe nada; --yes é manual) <<<"
python3 scripts/publish_release.py --release-dir '"${OUTPUT_DIR}"'
'

srun singularity exec --nv \
  --bind "${BIND_ROOT}":"${BIND_ROOT}" \
  "${SIF}" bash -lc "${APP_CMD}"

echo "=========================================="
echo " FIM (rc=$?) — $(date)"
echo "=========================================="
