import os
from huggingface_hub import HfApi, list_repo_files
tok=os.environ.get("HF_TOKEN")
for repo in ["AKCITPixel3/CMiQdveBBzNii","AKCITPixel3/BKXcuVXCmeRvN","akcit-pixel/DDPD","akcit-pixel/RealDOF"]:
    print("="*70); print(repo)
    try:
        fs=list_repo_files(repo, repo_type="dataset", token=tok)
        print("  n_arquivos:", len(fs))
        for f in sorted(fs)[:30]: print("   ", f)
        if len(fs)>30: print("    ... mais", len(fs)-30)
    except Exception as e:
        print("  ERRO:", type(e).__name__, e)
