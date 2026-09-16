import os
from collections import Counter
from datasets import load_dataset, get_dataset_split_names
tok=os.environ.get("HF_TOKEN")
for r,esperado in [("juliadollis/bokeh-eval-lfrepro-oficial",500),
                   ("juliadollis/bokeh-eval-rb-nosso-full",217),
                   ("juliadollis/bokeh-eval-rb-kfix-full",217),
                   ("juliadollis/bokeh-eval-rb-rotac-full",217)]:
    sp = get_dataset_split_names(r, token=tok)[0]
    d = load_dataset(r, split=sp, token=tok)
    cols = d.column_names
    chave = "image_id" if "image_id" in cols else cols[0]
    ids = [str(x) for x in d[chave]]
    c = Counter(ids)
    rep = {k:v for k,v in c.items() if v>1}
    print("%-46s n=%4d esperado=%4d unicos=%4d repetidos=%3d max_rep=%d" % (
        r.split("/")[-1], len(d), esperado, len(c), len(rep), max(c.values())))
    if rep: print("     exemplos:", list(rep.items())[:4])
    print("     colunas:", cols[:8])
