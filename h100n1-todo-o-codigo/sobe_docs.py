import os
from huggingface_hub import HfApi
api=HfApi(token=os.environ["HF_TOKEN"])
for arq in ["TABELA_FINAL.md","AUDITORIA_AVALIACAO.md"]:
    api.upload_file(path_or_fileobj=f"/host/{arq}", path_in_repo=arq,
                    repo_id="juliadollis/genrefocus-tabela-final", repo_type="dataset")
    print("enviado:", arq)
