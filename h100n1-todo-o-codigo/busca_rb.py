import os
from huggingface_hub import HfApi
tok=os.environ.get("HF_TOKEN")
api=HfApi(token=tok)
for q in ["RealBokeh","Real Bokeh","3MP","bokeh 3MP","aperture bracket","LFDOF","EBB"]:
    print("="*60); print("Q:",q)
    try:
        for d in api.list_datasets(search=q, limit=30): print("  DS ", d.id)
    except Exception as e: print(" err",e)
# tambem lista tudo da org akcit-pixel e AKCITPixel3
for org in ["akcit-pixel","AKCITPixel3","AkcitPixel2","AKCITPixel","akcit"]:
    print("="*60); print("ORG:",org)
    try:
        for d in api.list_datasets(author=org, limit=200): print("  ", d.id)
    except Exception as e: print(" err",e)
