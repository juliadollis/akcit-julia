import os
from datasets import load_dataset
tok = os.environ["HF_TOKEN"]
for esc in ["1","5"]:
    for m in ["oficial","nosso","rotac60k","fase1"]:
        r = "juliadollis/bokeh-kesc%s-rb-%s" % (esc, m)
        try:
            d = load_dataset(r, split="validation", token=tok)
            print("%-38s n=%4d" % (r.split("/")[1], len(d)))
        except Exception as e:
            print("%-38s -- %s" % (r.split("/")[1], type(e).__name__))
