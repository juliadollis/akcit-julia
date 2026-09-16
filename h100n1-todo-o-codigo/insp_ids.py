import os, re
from collections import Counter
from datasets import load_dataset, get_dataset_split_names
tok=os.environ.get("HF_TOKEN")
for r in ["juliadollis/bokeh-eval-lfrepro-rotac","juliadollis/bokeh-eval-rb-rotac-full","juliadollis/bokeh-eval-rd-rotac-full"]:
    sp=get_dataset_split_names(r, token=tok)[0]
    d=load_dataset(r, split=sp, token=tok)
    ids=[str(x) for x in d["file_name_base"]]
    print("###", r.split("/")[-1], "n=", len(ids))
    for x in ids[:4]: print("    ", x)
    # candidatos de cena
    c1=Counter(re.sub(r"_k\d+_d\d+$","",i) for i in ids)
    c2=Counter(re.sub(r"_level_\d+.*$","",i) for i in ids)
    print("    cenas se tirar _kXX_dXX:", len(c1), "| se tirar _level_N...:", len(c2))
