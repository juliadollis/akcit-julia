import glob, os
import pandas as pd
from datasets import Dataset
arqs = sorted(glob.glob("/host/por_imagem_gpu*/por_imagem.parquet"))
df = pd.concat([pd.read_parquet(a) for a in arqs], ignore_index=True)
print("linhas:", len(df), "| repos:", df.repo.nunique(), "| colunas:", list(df.columns))
Dataset.from_pandas(df).push_to_hub("juliadollis/bokeh-metricas-por-imagem", token=os.environ["HF_TOKEN"], private=True)
print("enviado para juliadollis/bokeh-metricas-por-imagem")
