import pandas as pd
from datasets import load_dataset
pd.set_option("display.width",250); pd.set_option("display.max_columns",40); pd.set_option("display.max_rows",100)
for r in ["juliadollis/deblur-metrics"]:
    try:
        d = load_dataset(r, split="train").to_pandas()
        print("###", r, len(d), "linhas"); print(d.to_string())
    except Exception as e:
        print("ERRO", r, e)
