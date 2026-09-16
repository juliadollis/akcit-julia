import os, re, collections
from huggingface_hub import list_repo_files
tok=os.environ.get("HF_TOKEN")
fs=list_repo_files("akcit-pixel/RealBokeh", repo_type="dataset", token=tok)
print("total arquivos:", len(fs))
top=collections.Counter(f.split("/")[0] for f in fs); print("topo:", top)
cenas=collections.defaultdict(list)
for f in fs:
    m=re.match(r"^(train|test)/gt/([^/]+)/([^/]+)$", f)
    if m: cenas[(m.group(1),m.group(2))].append(m.group(3))
for split in ["train","test"]:
    ks=[k for k in cenas if k[0]==split]
    print(f"--- {split}: {len(ks)} cenas")
    napert=collections.Counter(len(cenas[k]) for k in ks); print("   arquivos por cena:", dict(napert))
    ap=collections.Counter()
    for k in ks:
        for fn in cenas[k]:
            mm=re.search(r"_f([0-9.]+)\.JPG$", fn)
            if mm: ap[mm.group(1)]+=1
    print("   aberturas:", dict(sorted(ap.items(), key=lambda x:-x[1])))
    ex=sorted(ks)[:3]
    for e in ex: print("   ex", e, sorted(cenas[e]))
