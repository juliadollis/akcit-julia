import os
from datasets import load_dataset
tok=os.environ["HF_TOKEN"]
for nome,r in [("K 1..100","juliadollis/bokeh-eval-rb-piloto"),("K 3..300","juliadollis/bokeh-eval-rb-piloto-k300")]:
    try:
        ds=load_dataset(r, split="validation", token=tok, download_mode="force_redownload")
        print(f"--- {nome} ({r}) n={len(ds)}")
        for row in ds:
            print("    %-52s best_k=%7.1f  ssim=%.4f" % (row["file_name_base"][-50:], row["best_k_value"], row["ssim_score"]))
    except Exception as e: print(nome,"ERRO",type(e).__name__,e)
