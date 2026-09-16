import os
from datasets import load_dataset, get_dataset_split_names
tok=os.environ.get("HF_TOKEN")
r="juliadollis/lf-bokeh-repro-blb"
sp=get_dataset_split_names(r, token=tok); print("splits:", sp)
d=load_dataset(r, split=sp[0], token=tok)
print("n=", len(d)); print("colunas:", d.column_names)
row=d[0]
for k,v in row.items():
    t=type(v).__name__
    if isinstance(v,(str,int,float)) or v is None: print("   %-26s %-8s %s" % (k,t,str(v)[:70]))
    elif isinstance(v,dict): print("   %-26s %-8s chaves=%s" % (k,t,list(v)[:4]))
    else: print("   %-26s %-8s" % (k,t))
