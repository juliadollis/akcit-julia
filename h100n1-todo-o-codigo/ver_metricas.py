import os, pandas as pd
from datasets import load_dataset
tok=os.environ["HF_TOKEN"]
d=load_dataset("juliadollis/bokeh-eval-rb-metricas", split="train", token=tok, download_mode="force_redownload")
df=d.to_pandas(); pd.set_option("display.width",220)
print(df.to_string())
