import os
from datasets import load_dataset
from huggingface_hub import HfApi
tok = os.environ["HF_TOKEN"]
repo = "juliadollis/bokeh-bench-ebb400"
d = load_dataset(repo, split="test", token=tok)
print("lido:", len(d), "cenas")
d.push_to_hub(repo, split="validation", token=tok, private=True)
print("republicado como split 'validation'")
api = HfApi(token=tok)
arqs = [s.rfilename for s in api.dataset_info(repo).siblings]
print("arquivos agora:", arqs)
for a in arqs:
    if a.startswith("data/test-"):
        api.delete_file(path_in_repo=a, repo_id=repo, repo_type="dataset")
        print("removido o duplicado:", a)
