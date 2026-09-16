from huggingface_hub import HfApi
import os
api = HfApi(token=os.environ.get("HF_TOKEN"))
modelos = ["juliadollis/genrefocus-bokehnet-synth-2gpu","juliadollis/genrefocus-bokehnet-fase2-real",
  "juliadollis/genrefocus-bokehnet-fase2-rotac-only","juliadollis/genrefocus-bokehnet-fase2-kfix",
  "juliadollis/genrefocus-bokehnet-fase2-nofilter","nycu-cplab/Genfocus-Model"]
print("########## MODELOS ##########")
for r in modelos:
    try:
        info = api.model_info(r, files_metadata=False)
        fs = [s.rfilename for s in info.siblings if s.rfilename.endswith(".safetensors")]
        print("\n%s  (private=%s, atualizado=%s)" % (r, info.private, str(info.lastModified)[:19]))
        for f in sorted(fs): print("    ", f)
    except Exception as e:
        print("\n%s -> ERRO: %s" % (r, str(e)[:120]))
print("\n########## DATASETS (benchmarks) ##########")
for r in ["juliadollis/bokeh-bench-realbokeh-test-v2","juliadollis/lf-bokeh-repro-blb","akcit-pixel/RealDOF","akcit-pixel/DDPD"]:
    try:
        info = api.dataset_info(r)
        print("%-50s private=%s  atualizado=%s" % (r, info.private, str(info.lastModified)[:19]))
    except Exception as e:
        print("%-50s ERRO: %s" % (r, str(e)[:100]))
