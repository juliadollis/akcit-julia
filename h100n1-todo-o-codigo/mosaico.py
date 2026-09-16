"""Mosaico comparativo: AIF de entrada | alvo real | saidas dos modelos."""
import os, io, sys
from datasets import load_dataset
from PIL import Image, ImageDraw
tok=os.environ["HF_TOKEN"]
BENCH="juliadollis/bokeh-bench-realbokeh-test"
repos=[("oficial","juliadollis/bokeh-eval-rb-oficial"),
       ("sem-treino","juliadollis/bokeh-eval-rb-semtreino"),
       ("nosso","juliadollis/bokeh-eval-rb-nosso")]
if len(sys.argv)>1: repos=[(a.split("=")[0],a.split("=")[1]) for a in sys.argv[1:]]
def pil(d):
    if hasattr(d,"convert"): return d.convert("RGB")
    if isinstance(d,bytes): return Image.open(io.BytesIO(d)).convert("RGB")
    return Image.open(io.BytesIO(d["bytes"])).convert("RGB")
ent=load_dataset("parquet", data_files={"v":f"hf://datasets/{BENCH}/data/validation-*.parquet"}, token=tok)["v"]
por_nome={r["file_name_base"]: r for r in ent.select(range(60))}
saidas={}
for nome,r in repos:
    try:
        ds=load_dataset(r, split="validation", token=tok, download_mode="force_redownload")
        saidas[nome]={row["file_name_base"]: row for row in ds}
        print(nome, len(saidas[nome]), "linhas | best_k:", [round(float(x["best_k_value"]),1) for x in list(saidas[nome].values())[:12]])
    except Exception as e: print("falhou", nome, e)
comuns=[n for n in por_nome if all(n in s for s in saidas.values())][:6]
os.makedirs("/workspace/b/mos", exist_ok=True)
W=340
for i,n in enumerate(comuns):
    cols=[("AIF entrada", pil(por_nome[n]["image_focus"])), ("ALVO real f/2.0", pil(list(saidas.values())[0][n]["image_real_bokeh"]))]
    for nome,_ in repos:
        if nome in saidas: cols.append((nome, pil(saidas[nome][n]["image_best_k"])))
    ims=[im.resize((W,int(im.height*W/im.width))) for _,im in cols]
    H=max(im.height for im in ims)
    c=Image.new("RGB",(W*len(ims),H+26),(18,18,18)); d=ImageDraw.Draw(c)
    for j,im in enumerate(ims):
        c.paste(im,(j*W,26)); d.text((j*W+6,7), cols[j][0], fill=(240,240,240))
    p=f"/workspace/b/mos/mos_{i}.png"; c.save(p); print("salvo",p,n)
