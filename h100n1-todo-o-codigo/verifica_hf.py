from huggingface_hub import HfApi
import os
api = HfApi(token=os.environ["HF_TOKEN"])
for r in ["juliadollis/genrefocus-deblurnet", "juliadollis/genrefocus-deblurnet-15k"]:
    info = api.model_info(r, files_metadata=True)
    print("### %s  (privado=%s)" % (r, info.private))
    for s in sorted(info.siblings, key=lambda x: -(x.size or 0)):
        if s.rfilename.startswith("."):
            continue
        mb = (s.size or 0) / 1e6
        sha = (s.lfs.get("sha256") if isinstance(s.lfs, dict) else getattr(s.lfs, "sha256", None)) or ""
        print("   %-26s %8.1f MB  %s" % (s.rfilename, mb, sha[:16]))
