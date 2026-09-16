import os
from huggingface_hub import HfApi
api=HfApi(token=os.environ.get("HF_TOKEN"))
vistos=set()
for termo in ["EBB bokeh","everything is better with bokeh","bokeh dataset","AIM 2019 bokeh","bokeh effect synthesis","EBB!"]:
    for d in api.list_datasets(search=termo, limit=25):
        if d.id in vistos: continue
        vistos.add(d.id)
        try:
            info=api.dataset_info(d.id)
            n=len(info.siblings or [])
            print("%-52s downloads=%-8s arquivos=%-5s %s" % (d.id, getattr(d,"downloads",0), n, str(info.lastModified)[:10]))
        except Exception as e:
            print("%-52s (erro: %s)" % (d.id, str(e)[:50]))
