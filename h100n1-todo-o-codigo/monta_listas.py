from datasets import load_dataset
import os
tok=os.environ.get("HF_TOKEN")
cur = load_dataset("juliadollis/genrefocus-resultados-curados", split="train", token=tok).to_pandas()
bru = load_dataset("juliadollis/bokeh-eval-metricas", split="train", token=tok).to_pandas()
curados = list(dict.fromkeys(cur["repo_metricas_bruto"].astype(str)))
todos   = list(dict.fromkeys(bru["Dataset"].astype(str)))
# prioridade: os do pipeline corrigido (ja curados) e os novos ainda nao curados
novos = [r for r in todos if r not in curados and ("-full" in r or "lfrepro" in r or "-rb-" in r or "-rd-" in r)]
resto = [r for r in todos if r not in curados and r not in novos]
ordem = curados + novos + resto
print("curados:", len(curados), "| novos relevantes:", len(novos), "| resto:", len(resto), "| total:", len(ordem))
N=7
for g,i in zip([0,1,2,3,5,6,7], range(N)):
    faixa = ordem[i::N]
    with open("/host/lista_porimg_gpu%d.txt" % g, "w") as f:
        f.write("\n".join(faixa)+"\n")
    print("gpu%d: %d repos" % (g, len(faixa)))
