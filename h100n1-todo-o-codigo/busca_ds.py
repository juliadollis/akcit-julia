import os
from huggingface_hub import HfApi
tok=os.environ.get("HF_TOKEN")
api=HfApi(token=tok)
for q in ["bokeh","EBB","aperture","defocus","refocus","shallow depth of field","AIM bokeh"]:
    print("="*70); print("QUERY:", q)
    try:
        res=list(api.list_datasets(search=q, limit=40))
        for d in res: print("   ", d.id, "| downloads:", getattr(d,"downloads",None))
    except Exception as e: print("  ERRO", e)
