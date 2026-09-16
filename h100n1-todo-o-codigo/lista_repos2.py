from huggingface_hub import HfApi
import os, re
api = HfApi(token=os.environ.get("HF_TOKEN"))
modelos = ["juliadollis/genrefocus-bokehnet-synth-2gpu","juliadollis/genrefocus-bokehnet-fase2-real",
  "juliadollis/genrefocus-bokehnet-fase2-rotac-only","juliadollis/genrefocus-bokehnet-fase2-kfix",
  "juliadollis/genrefocus-bokehnet-fase2-nofilter","nycu-cplab/Genfocus-Model"]
for r in modelos:
    try:
        info = api.model_info(r)
        fs = sorted(s.rfilename for s in info.siblings if s.rfilename.endswith(".safetensors"))
        steps = sorted(int(m.group(1)) for f in fs if (m:=re.search(r"step(\d+)", f)))
        semstep = [f for f in fs if "step" not in f]
        print("%-48s private=%-5s  n_ckpt=%3d  maior_step=%s  sem_step=%s" % (
            r, info.private, len(fs), (max(steps) if steps else "-"), semstep))
    except Exception as e:
        print("%-48s ERRO %s" % (r, str(e)[:80]))
