import os, json
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
out = {}
for kind, lister in (("model", api.list_models), ("dataset", api.list_datasets)):
    try:
        repos = list(lister(author="juliadollis"))
    except Exception as e:
        print("ERR list", kind, e); continue
    for r in repos:
        rid = r.id
        rec = {"kind": kind, "private": getattr(r, "private", None), "files": []}
        try:
            info = api.repo_info(rid, repo_type=kind, files_metadata=True)
            rec["private"] = info.private
            for s in info.siblings or []:
                lfs = s.lfs
                sha = None
                if lfs is not None:
                    sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
                rec["files"].append({"path": s.rfilename, "size": s.size, "sha": sha})
        except Exception as e:
            rec["error"] = str(e)
        out[rid] = rec
with open("/host/inv_hub.json", "w") as f:
    json.dump(out, f, indent=1, default=str)
print("REPOS", len(out))
for k, v in sorted(out.items()):
    big = [f for f in v["files"] if (f.get("size") or 0) > 1_000_000]
    print(f"{k} [{v['kind']}] private={v.get('private')} files={len(v['files'])} big={len(big)}")
