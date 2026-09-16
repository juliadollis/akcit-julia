import os, numpy as np, cv2
from datasets import load_dataset
tok=os.environ.get("HF_TOKEN")
d=load_dataset("comHannah/bokeh-dataset", split="train", token=tok)
print("n=", len(d))
def lv(im):
    g=cv2.cvtColor(np.array(im.convert("RGB")), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())
print("%-6s %-16s %-16s %10s %10s %8s" % ("i","nitida","bokeh","lv_nitida","lv_bokeh","razao"))
razoes=[]
for i in [0,1,2,500,1500,3000,4399]:
    r=d[i]; a=r["original_image"]; b=r["bokeh_image"]
    la, lb = lv(a), lv(b)
    razoes.append(lb/la if la>0 else 0)
    print("%-6d %-16s %-16s %10.1f %10.1f %8.3f" % (i, str(a.size), str(b.size), la, lb, razoes[-1]))
print("\nrazao lv_bokeh/lv_nitida < 1 significa que o alvo esta MAIS BORRADO que a entrada, que e o esperado")
