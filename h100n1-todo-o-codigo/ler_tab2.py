import pandas as pd
from datasets import load_dataset
pd.set_option("display.width",250); pd.set_option("display.max_columns",40)
d = load_dataset("juliadollis/tab2-metricas", split="train").to_pandas()
print("### tab2-metricas:", len(d), "linhas"); print(d.to_string())
