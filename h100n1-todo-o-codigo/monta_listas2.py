import os
from datasets import load_dataset
tok = os.environ["HF_TOKEN"]
bru = load_dataset("juliadollis/bokeh-eval-metricas", split="train", token=tok,
                   download_mode="force_redownload").to_pandas()
todos = list(dict.fromkeys(bru["Dataset"].astype(str)))
gpus = [0, 1, 2, 3, 5, 6, 7]
for i, g in enumerate(gpus):
    with open("/host/lista_porimg_gpu%d.txt" % g, "w") as f:
        f.write("\n".join(todos[i::len(gpus)]) + "\n")
print("repos no total:", len(todos))
