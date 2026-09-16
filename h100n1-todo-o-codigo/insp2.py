import os, json
from huggingface_hub import HfApi, list_repo_files
import pyarrow.parquet as pq
tok=os.environ.get("HF_TOKEN")
api=HfApi(token=tok)
for repo in ["comHannah/bokeh-dataset","atfortes/BokehDiffusion"]:
    print("="*70); print(repo)
    try:
        info=api.dataset_info(repo, token=tok)
        print("  tags:", info.tags)
        fs=list_repo_files(repo, repo_type="dataset", token=tok)
        print("  n:", len(fs)); [print("   ",f) for f in sorted(fs)[:25]]
    except Exception as e: print("  ERRO", type(e).__name__, e)
print("#"*70); print("TABELA DE METRICAS ATUAL")
from datasets import load_dataset
try:
    d=load_dataset("juliadollis/bokeh-eval-metricas", split="train", token=tok, download_mode="force_redownload")
    import pandas as pd
    df=d.to_pandas(); pd.set_option("display.width",250); pd.set_option("display.max_rows",200)
    print(df.to_string())
except Exception as e: print("ERRO", e)
