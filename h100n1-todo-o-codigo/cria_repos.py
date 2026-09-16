import os
from huggingface_hub import HfApi
tok=os.environ["HF_TOKEN"]; api=HfApi(token=tok)
privados=["juliadollis/bokeh-eval-rb-oficial","juliadollis/bokeh-eval-rb-semtreino",
          "juliadollis/bokeh-eval-rb-nosso","juliadollis/bokeh-eval-rb-identidade",
          "juliadollis/bokeh-eval-rb-oficial-centro","juliadollis/bokeh-eval-rb-piloto"]
for r in privados:
    api.create_repo(r, repo_type="dataset", private=True, exist_ok=True); print("privado:", r)
api.create_repo("juliadollis/bokeh-eval-rb-metricas", repo_type="dataset", private=False, exist_ok=True)
print("publico: juliadollis/bokeh-eval-rb-metricas")
