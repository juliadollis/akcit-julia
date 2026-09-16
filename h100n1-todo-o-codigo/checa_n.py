import json, glob, os
from datasets import load_dataset
tok=os.environ.get("HF_TOKEN")
cur = load_dataset("juliadollis/genrefocus-resultados-curados", split="train", token=tok).to_pandas()
regs={}
for f in sorted(glob.glob("/host/por_imagem_gpu*/resumo.json")):
    for r in json.load(open(f)): regs[r["repo"]]=r
print("%-46s %-28s %8s %8s %s" % ("repo","benchmark","n_curado","n_real","conferencia"))
print("-"*120)
ruim=0
for _,row in cur.iterrows():
    rp=str(row["repo_metricas_bruto"]); r=regs.get(rp,{})
    nc = row["n"]; nr = r.get("n")
    nc_s = "-" if nc is None or (isinstance(nc,float) and nc!=nc) else str(int(nc))
    flag = "" if (nc_s!="-" and nr==int(float(nc))) else "  <== DIFERE"
    if flag: ruim+=1
    print("%-46s %-28s %8s %8s %-9s%s" % (rp.split("/")[-1], str(row["benchmark"])[:28], nc_s, nr, r.get("conferencia","?"), flag))
print("\nlinhas com n divergente ou ausente:", ruim, "de", len(cur))
