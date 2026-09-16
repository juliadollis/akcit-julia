import os
from datasets import load_dataset
from huggingface_hub import list_repo_files
tok = os.environ["HF_TOKEN"]
for m in ["rotac60k","nosso","oficial","fase1","kfix","nofilter","semtreino"]:
    r = "juliadollis/bokeh-eq4-lfrepro-" + m
    try:
        fs = [f for f in list_repo_files(r, repo_type="dataset", token=tok) if f.endswith(".parquet")]
        d = load_dataset(r, split="validation", token=tok)
        ult = sorted(fs)[-1] if fs else "-"
        print("%-10s n=%4d  parquets=%3d  ultimo=%s" % (m, len(d), len(fs), ult))
    except Exception as e:
        print("%-10s ERRO %s: %s" % (m, type(e).__name__, str(e)[:70]))
