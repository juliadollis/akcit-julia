import os
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
tok=os.environ.get("HF_TOKEN")
for repo,f in [("AKCITPixel3/CMiQdveBBzNii","data/train_batch_0001.parquet"),
               ("AKCITPixel3/BKXcuVXCmeRvN","data/train_batch_0001.parquet"),
               ("akcit-pixel/DDPD","data/validation-00000-of-00003.parquet"),
               ("akcit-pixel/RealDOF","data/validation-00000-of-00004.parquet")]:
    print("="*70); print(repo, f)
    try:
        p=hf_hub_download(repo,f,repo_type="dataset",token=tok)
        pf=pq.ParquetFile(p)
        print("  rows:", pf.metadata.num_rows)
        for fld in pf.schema_arrow: print("   ", fld.name, "|", fld.type)
    except Exception as e: print("  ERRO", type(e).__name__, e)
