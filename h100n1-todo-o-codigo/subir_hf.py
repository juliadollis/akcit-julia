import hashlib, json, os, shutil, tempfile
from pathlib import Path
from huggingface_hub import HfApi

API = HfApi(token=os.environ["HF_TOKEN"])
OUT = Path("/workspace/retreinar-deblur/outputs/deblur_docker_4gpu/deblur")

ALVOS = [
    ("akcit-pixel/genrefocus-deblurnet",     OUT/"deblur.safetensors",
     OUT/"deblur.json", "/workspace/README_60k.md", 60000),
    ("akcit-pixel/genrefocus-deblurnet-15k", OUT/"deblur_best_step15500.safetensors",
     OUT/"deblur_best_step15500.json", "/workspace/README_15k.md", 15500),
]

for repo, peso, meta, card, step in ALVOS:
    print(f"\n=== {repo} (step {step}) ===")
    sha = hashlib.sha256(peso.read_bytes()).hexdigest()
    print(f"  sha256 {sha[:16]}…  {peso.stat().st_size/1e6:.1f} MB")
    API.create_repo(repo_id=repo, repo_type="model", private=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir="/workspace") as td:
        td = Path(td)
        shutil.copy(peso, td/"deblurNet.safetensors")
        shutil.copy(card, td/"README.md")
        proc = json.loads(meta.read_text()) if meta.exists() else {}
        proc.update({"sha256_deblurNet_safetensors": sha,
                     "arquivo_origem": peso.name, "step": step,
                     "wandb": "juliadollis-federal-university-of-goi-s/genrefocus-deblurnet-docker/runs/70eea203"})
        (td/"procedencia.json").write_text(json.dumps(proc, indent=2, ensure_ascii=False))
        API.upload_folder(folder_path=str(td), repo_id=repo, repo_type="model",
                          commit_message=f"DeblurNet reproducao AKCIT-PIXEL, step {step}")
    print(f"  https://huggingface.co/{repo}  (PRIVADO)")
