from huggingface_hub import HfApi, hf_hub_download
import os
tok=os.environ.get("HF_TOKEN")
api=HfApi(token=tok)
for r in ["juliadollis/lf-bokeh-repro-blb","juliadollis/bokeh-bench-realbokeh-test-v2"]:
    print("###", r)
    try:
        info=api.dataset_info(r)
        print("   arquivos:", [s.rfilename for s in info.siblings][:8])
        p=hf_hub_download(r,"README.md",repo_type="dataset",token=tok)
        print(open(p).read()[:2500])
    except Exception as e:
        print("   sem README:", str(e)[:150])
