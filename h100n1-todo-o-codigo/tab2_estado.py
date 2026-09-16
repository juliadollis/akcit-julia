import os
import pandas as pd
from datasets import load_dataset
pd.set_option("display.width",250)
tok=os.environ["HF_TOKEN"]
d=load_dataset("juliadollis/tab2-metricas", split="train", token=tok).to_pandas()
print("### o que temos hoje da Tabela 2")
print(d[["Model","Dataset","LPIPS","DISTS","CLIP-IQA","MANIQA","MUSIQ"]].to_string(index=False))
