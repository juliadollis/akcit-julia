import os, pandas as pd
from datasets import load_dataset
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50); pd.set_option("display.max_rows", 200)
for repo in ["juliadollis/genrefocus-resultados-curados"]:
    try:
        d = load_dataset(repo, split="train")
        df = d.to_pandas()
        print("### %s: %d linhas, colunas: %s" % (repo, len(df), list(df.columns)))
        print(df.to_string())
    except Exception as e:
        print("ERRO em %s: %s" % (repo, e))
