import os
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
ALVOS = [
    ("juliadollis/bokehnet-geo-B", "smoke.safetensors"),
    ("juliadollis/genrefocus-bokehnet-fase2-nofilter", "bokeh_step35000.safetensors"),
    ("juliadollis/deblurnet-ft", "checkpoints/best.pt"),
]
for repo, arq in ALVOS:
    try:
        info = api.repo_info(repo, repo_type="model", files_metadata=True)
        fs = {s.rfilename: s.size for s in info.siblings}
        tem = arq in fs
        vis = "PUBLICO" if not info.private else "privado"
        print(f"{repo}")
        print(f"   visibilidade : {vis}")
        print(f"   {arq}: {'presente' if tem else 'AUSENTE'}"
              + (f" ({fs[arq]/1e9:.2f} GB)" if tem and fs[arq] else ""))
        print(f"   ultima modificacao: {info.lastModified}")
    except Exception as e:
        print(f"{repo} -> erro: {str(e)[:150]}")
    print()
