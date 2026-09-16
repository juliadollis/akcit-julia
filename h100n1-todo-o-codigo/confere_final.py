import os, hashlib, json
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
H = "/host"
alvos = [
    ("juliadollis/deblurnet-ft", "checkpoints/best.pt",
     f"{H}/retreinar-deblur/outputs/deblur_docker_4gpu/deblur/checkpoints/best.pt"),
    ("juliadollis/bokehnet-geo-B", "smoke.safetensors",
     f"{H}/genrefocus_deblurnet_paper/outputs/bokehnet-geo-B/smoke.safetensors"),
    ("juliadollis/genrefocus-bokehnet-fase2-nofilter", "bokeh_step35000.safetensors",
     f"{H}/genrefocus_deblurnet_paper/outputs/bokehnet_fase2_nofilter/bokeh/.upload_bokeh_step35000.safetensors"),
]
locais = {}
for line in open(f"{H}/hash_local.out"):
    sha, p = line.split()
    locais[p.replace("/raid/user_juliadollis/julia_docker", H)] = sha

for repo, path, local in alvos:
    info = api.repo_info(repo, repo_type="model", files_metadata=True)
    fs = {s.rfilename: s for s in info.siblings}
    print(f"\n=== {repo} (privado={info.private}) arquivos={len(fs)}")
    s = fs.get(path)
    if s is None:
        print(f"  FALTA {path}")
        continue
    lfs = s.lfs
    sha_hub = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
    n_loc = os.path.getsize(local)
    sha_loc = locais.get(local, "?")
    print(f"  {path}")
    print(f"    bytes hub={s.size} local={n_loc} iguais={s.size == n_loc}")
    print(f"    sha256 hub={sha_hub}")
    print(f"    sha256 loc={sha_loc}")
    print(f"    CONFERE={sha_hub == sha_loc and s.size == n_loc}")
    for extra in sorted(fs):
        if extra != path:
            print(f"    (tambem no repo) {fs[extra].size:>12} {extra}")
print("\nFIM_CONFERENCIA")
