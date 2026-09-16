#!/bin/bash
# 4o modelo da Tabela 2: o DeblurNet ANTIGO, variante main+cond.
#
# ADAPTADO de proposito, e por isso nao e "o mesmo pipeline" que os outros tres:
#   --main-adapter deblurring  -> o treino dele PUNHA LoRA no branch principal
#   --text-adapter none        -> mas NUNCA no texto. O `generate` oficial monta
#                                 adapters = [main_adapter]*2 + c_adapters, e o
#                                 indice 0 e o TEXTO; sem este override o texto
#                                 receberia LoRA que o treino nunca viu (C6).
set -euo pipefail
export PYTHONPATH=/workspace/retreinar-deblur:/workspace/retreinar-deblur/third_party/Genfocus
cd /workspace/retreinar-deblur

echo ">>> baixando o peso antigo do HF"
W=$(python3 - <<'PY'
import os
from huggingface_hub import hf_hub_download
print(hf_hub_download("juliadollis/genrefocus-deblurnet-paper-4gpu",
                      "deblur.safetensors", token=os.environ["HF_TOKEN"]))
PY
)
echo ">>> peso: $W"

echo ">>> TABELA 2 — ANTIGO (main+cond), adaptado"
python3 inferencia/rodar_tabela2.py \
  --lora "$W" \
  --main-adapter deblurring \
  --text-adapter none \
  --rotulo "antigo main+cond (60k)" \
  --sem-identidade \
  --out saidas/tab2_antigo
echo ">>> FIM"
