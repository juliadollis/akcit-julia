import os
from datasets import load_dataset
tok = os.environ["HF_TOKEN"]
alvos = [("kesc1","rb",217),("kesc5","rb",217),("kesc1","ebb",400),("kesc5","ebb",400)]
for esc, mesa, esperado in alvos:
    for m in ["oficial","nosso","rotac60k","fase1"]:
        r = "juliadollis/bokeh-%s-%s-%s" % (esc, mesa, m)
        try:
            d = load_dataset(r, split="validation", token=tok)
            ok = "completo" if len(d) == esperado else "FALTAM %d" % (esperado - len(d))
            print("%-34s n=%4d/%4d  %s" % (r.split("/")[1], len(d), esperado, ok))
        except Exception as e:
            print("%-34s nao existe" % r.split("/")[1])
