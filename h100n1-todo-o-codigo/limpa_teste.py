import os

from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
alvo = "akcit-dephpro/_teste_de_escrita"
try:
    api.delete_repo(alvo, repo_type="model")
    print(f"removido: {alvo}")
except Exception as e:
    print(f"nao removi {alvo}: {str(e)[:140]}")
