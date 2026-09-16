import os, collections
from huggingface_hub import list_repo_files, hf_hub_download, HfApi
tok=os.environ.get("HF_TOKEN")
api=HfApi(token=tok)
for repo in ["akcit-pixel/RealBokeh","timseizinger/RealBokeh_3MP","akcit-pixel/LFDOF"]:
    print("#"*72); print(repo)
    try:
        fs=list_repo_files(repo, repo_type="dataset", token=tok)
        print("  n arquivos:", len(fs))
        pref=collections.Counter("/".join(f.split("/")[:3]) for f in fs)
        for k,v in pref.most_common(40): print("   ", v, "->", k)
        print("  amostra:", sorted(fs)[:8])
        try:
            p=hf_hub_download(repo,"README.md",repo_type="dataset",token=tok); print("  README:\n", open(p).read()[:1800])
        except Exception as e: print("  sem README")
    except Exception as e: print("  ERRO", type(e).__name__, e)
