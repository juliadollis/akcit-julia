import os, io
from datasets import load_dataset
from PIL import Image
tok=os.environ.get("HF_TOKEN")
OUT="/workspace/b/diag_out"; os.makedirs(OUT, exist_ok=True)
repos={"oficial":"juliadollis/bokeh-eval-infer-oficial-kesc",
       "nosso":"juliadollis/bokeh-eval-infer-nosso-kesc",
       "semtreino":"juliadollis/bokeh-eval-infer-semtreino-kesc"}
def pil(d):
    if hasattr(d,"convert"): return d
    if isinstance(d,bytes): return Image.open(io.BytesIO(d))
    return Image.open(io.BytesIO(d["bytes"]))
linhas={}
for nome,r in repos.items():
    ds=load_dataset(r, split="validation", token=tok)
    print(nome, r, "n=", len(ds), "cols=", ds.column_names)
    linhas[nome]=ds
    print("   best_k:", [round(float(x),3) for x in ds["best_k_value"]][:20])
    print("   ssim  :", [round(float(x),3) for x in ds["ssim_score"]][:20])
# monta mosaico para 3 indices
for i in [0,1,2]:
    cols=[]
    gt=pil(linhas["oficial"][i]["image_real_bokeh"]).convert("RGB")
    cols.append(("GT(image_blur)",gt))
    for nome in ["oficial","nosso","semtreino"]:
        cols.append((nome, pil(linhas[nome][i]["image_best_k"]).convert("RGB")))
    # tambem k15 do oficial
    cols.append(("oficial_k15", pil(linhas["oficial"][i]["image_k15"]).convert("RGB")))
    W=384
    ims=[im.resize((W,int(im.height*W/im.width))) for _,im in cols]
    H=max(im.height for im in ims)
    canvas=Image.new("RGB",(W*len(ims),H),(20,20,20))
    for j,im in enumerate(ims): canvas.paste(im,(j*W,0))
    p=f"{OUT}/cmp_{i}.png"; canvas.save(p); print("salvo",p, [c[0] for c in cols])
