import os, re, collections
from huggingface_hub import list_repo_files, hf_hub_download
import pyarrow.parquet as pq
tok=os.environ.get("HF_TOKEN")
print("@"*72,"\ntimseizinger/RealBokeh_3MP")
fs=list_repo_files("timseizinger/RealBokeh_3MP", repo_type="dataset", token=tok)
print("total arquivos:", len(fs))
cenas=collections.defaultdict(list)
for f in fs:
    m=re.match(r"^(train|test)/gt/([^/]+)/([^/]+)$", f)
    if m: cenas[(m.group(1),m.group(2))].append(m.group(3))
for split in ["train","test"]:
    ks=[k for k in cenas if k[0]==split]
    print(f"--- {split}: {len(ks)} cenas")
    print("   arqs/cena:", dict(collections.Counter(len(cenas[k]) for k in ks)))
    ap=collections.Counter()
    for k in ks:
        for fn in cenas[k]:
            mm=re.search(r"_f([0-9.]+)\.JPG$", fn)
            if mm: ap[mm.group(1)]+=1
    print("   aberturas:", dict(sorted(ap.items(), key=lambda x: float(x[0]))))
    for e in sorted(ks)[:2]: print("   ex", e[1], sorted(cenas[e]))
print("@"*72,"\nakcit-pixel/RealBokeh (parquet)")
fs2=list_repo_files("akcit-pixel/RealBokeh", repo_type="dataset", token=tok)
print([f for f in fs2 if not f.startswith("data/")])
print("data files:", sorted(f for f in fs2 if f.startswith("data/"))[:10], "...")
p=hf_hub_download("akcit-pixel/RealBokeh","README.md",repo_type="dataset",token=tok)
print(open(p).read()[:1500])
