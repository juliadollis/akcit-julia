"""Compara a nitidez da saida gerada com a do alvo real, por imagem."""
import os, io, sys, numpy as np, cv2
from datasets import load_dataset
from PIL import Image
tok=os.environ["HF_TOKEN"]
def pil(d):
    if hasattr(d,"convert"): return d.convert("RGB")
    if isinstance(d,bytes): return Image.open(io.BytesIO(d)).convert("RGB")
    return Image.open(io.BytesIO(d["bytes"])).convert("RGB")
def lv(im):
    return float(cv2.Laplacian(cv2.cvtColor(np.array(im),cv2.COLOR_RGB2GRAY),cv2.CV_64F).var())
for nome,r in [(a.split("=")[0],a.split("=")[1]) for a in sys.argv[1:]]:
    try:
        ds=load_dataset(r, split="validation", token=tok, download_mode="force_redownload")
    except Exception as e:
        print(nome,"indisponivel",type(e).__name__); continue
    ra=[]
    for row in ds:
        la=lv(pil(row["image_real_bokeh"])); lg=lv(pil(row["image_best_k"]))
        ra.append((lg/max(la,1e-6), la, lg, float(row["best_k_value"])))
    ra=np.array(ra)
    print("%-24s n=%3d | nitidez gerada/alvo: mediana=%.2f p10=%.2f p90=%.2f | %d de %d abaixo de 0.5 (borrou demais)" % (
        nome, len(ra), np.median(ra[:,0]), np.percentile(ra[:,0],10), np.percentile(ra[:,0],90),
        int((ra[:,0]<0.5).sum()), len(ra)))
    print("      best_k: med=%.1f min=%.1f max=%.1f | LV alvo med=%.0f  LV gerada med=%.0f" % (
        np.median(ra[:,3]), ra[:,3].min(), ra[:,3].max(), np.median(ra[:,1]), np.median(ra[:,2])))
