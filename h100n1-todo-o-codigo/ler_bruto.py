import pandas as pd
from datasets import load_dataset
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 300)
d = load_dataset("juliadollis/bokeh-eval-metricas", split="train").to_pandas()
print("linhas:", len(d))
print("colunas:", list(d.columns))
c = [x for x in d.columns if x.lower() in ("nome","modelo","experimento","run","id")]
print("\n### todos os nomes distintos:")
for col in c[:1]:
    for v in sorted(d[col].astype(str).unique()): print("  ", v)
