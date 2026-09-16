import os
from datasets import load_dataset, get_dataset_split_names
tok=os.environ["HF_TOKEN"]
def ids(r):
    sp=get_dataset_split_names(r, token=tok)[0]
    d=load_dataset(r, split=sp, token=tok)
    return set(str(x) for x in d["file_name_base"]), len(d)
a,na = ids("juliadollis/bokeh-eq4-rb-rotac60k")
b,nb = ids("juliadollis/bokeh-eq4-rb-oficial")
print("rotac60k: %d linhas, %d ids unicos" % (na, len(a)))
print("oficial : %d linhas, %d ids unicos" % (nb, len(b)))
falta = sorted(b - a)
print("\nfaltam no rotac60k (%d):" % len(falta))
for x in falta: print("   ", x)
