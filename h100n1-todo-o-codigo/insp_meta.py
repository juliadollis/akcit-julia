import os, json, re, collections
from huggingface_hub import list_repo_files, hf_hub_download
tok=os.environ["HF_TOKEN"]
fs=list_repo_files("timseizinger/RealBokeh_3MP", repo_type="dataset", token=tok)
metas=[f for f in fs if re.match(r"^test/metadata/.*\.json$", f)]
print("jsons no test/metadata:", len(metas), metas[:5])
outros=collections.Counter(f.split("/")[1] for f in fs if f.startswith("test/"))
print("subpastas de test/:", dict(outros))
for f in metas[:3]:
    p=hf_hub_download("timseizinger/RealBokeh_3MP", f, repo_type="dataset", token=tok)
    print("---", f); print(json.dumps(json.load(open(p)), indent=1)[:900])
