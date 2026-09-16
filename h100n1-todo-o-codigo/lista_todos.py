from huggingface_hub import HfApi
import os
api = HfApi(token=os.environ.get("HF_TOKEN"))
print("### MODELOS juliadollis ###")
for m in api.list_models(author="juliadollis"):
    print("  ", m.id)
print("### DATASETS juliadollis (bench/metricas) ###")
for d in api.list_datasets(author="juliadollis"):
    if any(k in d.id for k in ["bench","metric","curad","repro","por-imagem"]): print("  ", d.id)
