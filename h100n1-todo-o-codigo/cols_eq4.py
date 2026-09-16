import os
from datasets import load_dataset, get_dataset_split_names
tok=os.environ["HF_TOKEN"]
for r in ["juliadollis/bokeh-eq4-rb-nosso","juliadollis/bokeh-eq4-lfrepro-nosso"]:
    try:
        sp=get_dataset_split_names(r, token=tok)
        d=load_dataset(r, split=sp[0], token=tok)
        print("###", r, "| splits:", sp, "| n=", len(d))
        print("   colunas:", d.column_names)
    except Exception as e:
        print("###", r, "ERRO:", type(e).__name__, str(e)[:100])
