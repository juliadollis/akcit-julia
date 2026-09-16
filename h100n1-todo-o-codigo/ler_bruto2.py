import pandas as pd
from datasets import load_dataset
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 300)
d = load_dataset("juliadollis/bokeh-eval-metricas", split="train").to_pandas()
print("### linhas com FULLRES no nome do modelo:")
f = d[d["Model"].astype(str).str.contains("FULLRES|fullres", case=False, na=False)]
print(f.to_string() if len(f) else "  (nenhuma)")
print("\n### todos os 76 Model x Dataset:")
print(d[["Model","Dataset","LPIPS","SSIM"]].to_string())
