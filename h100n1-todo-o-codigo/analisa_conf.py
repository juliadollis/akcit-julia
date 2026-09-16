import json, glob, os
from datasets import load_dataset
tok=os.environ.get("HF_TOKEN")
cur = load_dataset("juliadollis/genrefocus-resultados-curados", split="train", token=tok).to_pandas()
curados = set(cur["repo_metricas_bruto"].astype(str))
regs=[]
for f in sorted(glob.glob("/host/por_imagem_gpu*/resumo.json")):
    regs += json.load(open(f))
conf=[r for r in regs if r.get("conferencia")=="CONFERE"]
div =[r for r in regs if r.get("conferencia")=="DIVERGE"]
sem =[r for r in regs if r.get("conferencia") not in ("CONFERE","DIVERGE")]
print("total processado:", len(regs), "| confere:", len(conf), "| diverge:", len(div), "| sem linha antiga:", len(sem))
print("\n### DIVERGENTES que estao na tabela CURADA (seria grave):")
g=[r for r in div if r["repo"] in curados]
print("  ", len(g), "de", len(curados), "curados")
for r in g: print("   !!", r["repo"], "pior=%.6f"%r["pior_diferenca"])
print("\n### CURADOS que conferem:")
print("  ", len([r for r in conf if r["repo"] in curados]), "de", len(curados))
print("\n### divergentes FORA da curada (campanha antiga):", len([r for r in div if r["repo"] not in curados]))
print("\n### soma de imagens processadas:", sum(r.get("n",0) for r in regs))
