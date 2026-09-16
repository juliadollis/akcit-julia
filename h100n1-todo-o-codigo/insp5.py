import os, json, collections
from huggingface_hub import hf_hub_download, list_repo_files
import pyarrow.parquet as pq
tok=os.environ.get("HF_TOKEN")
cols=["stem","route","s1","k","exif","qc","calibration_ssim","shape_kernel","source_path","source_aif","source_bokeh"]
for repo,nb in [("AKCITPixel3/CMiQdveBBzNii",6),("AKCITPixel3/BKXcuVXCmeRvN",3)]:
    print("#"*72); print(repo)
    stems=[]; srcs=collections.Counter(); rows=0; exemplos=[]
    for i in range(1,nb+1):
        p=hf_hub_download(repo,f"data/train_batch_{i:04d}.parquet",repo_type="dataset",token=tok)
        t=pq.read_table(p, columns=cols).to_pydict()
        rows+=len(t["stem"])
        for j in range(len(t["stem"])):
            sp=t["source_path"][j] or ""
            srcs[ "/".join(str(sp).split("/")[:6]) ]+=1
            if len(exemplos)<6:
                exemplos.append({c:(str(t[c][j])[:220]) for c in cols})
        stems+= [str(s) for s in t["stem"]]
    print("total linhas:", rows, "| stems unicos:", len(set(stems)))
    print("prefixos de source_path:")
    for k,v in srcs.most_common(15): print("   ", v, "->", k)
    for e in exemplos:
        print("-"*60)
        for c in cols: print(f"  {c}: {e[c]}")
