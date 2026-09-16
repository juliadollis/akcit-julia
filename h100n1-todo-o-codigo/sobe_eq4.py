import glob, os
import pandas as pd
from datasets import Dataset
tok = os.environ["HF_TOKEN"]
pi = sorted(glob.glob("/host/pi_eq4_gpu*/por_imagem.parquet"))
if pi:
    df = pd.concat([pd.read_parquet(a) for a in pi], ignore_index=True)
    print("linhas por imagem:", len(df), "| repos:", df.repo.nunique())
    Dataset.from_pandas(df).push_to_hub("juliadollis/bokeh-eq4-por-imagem", token=tok, private=True)
    print("enviado: juliadollis/bokeh-eq4-por-imagem")
ri = sorted(glob.glob("/host/ri_eq4_gpu*/riemann_por_imagem.parquet"))
if ri:
    dr = pd.concat([pd.read_parquet(a) for a in ri], ignore_index=True)
    print("linhas riemannianas:", len(dr), "| repos:", dr.repo.nunique())
    Dataset.from_pandas(dr).push_to_hub("juliadollis/bokeh-eq4-riemann", token=tok, private=True)
    print("enviado: juliadollis/bokeh-eq4-riemann")
