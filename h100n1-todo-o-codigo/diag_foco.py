"""Separa as duas fontes de erro do plano de foco: mascara errada x escala do Depth Pro.

Tambem mede uma 3a opcao: ler o plano de foco na REGIAO NITIDA DO ALVO, que e o
foco verdadeiro por definicao e, por ser lido no proprio mapa do Depth Pro, nao
depende da escala metrica dele.
"""
import os, io, re, json, glob, numpy as np, cv2
from huggingface_hub import hf_hub_download
from datasets import load_dataset
from PIL import Image
tok=os.environ["HF_TOKEN"]
D="/workspace/vp/temp_depth_maps"
ds=load_dataset("parquet", data_files={"v":"hf://datasets/juliadollis/bokeh-bench-realbokeh-test/data/validation-*.parquet"}, token=tok)["v"].select(range(40))

def prep(img, lado=512):
    w,h=img.size; s=lado/max(w,h); nw,nh=int(w*s),int(h*s)
    img=img.resize((nw,nh), Image.LANCZOS)
    fw,fh=max((nw//16)*16,16), max((nh//16)*16,16)
    l,t=(nw-fw)//2,(nh-fh)//2
    return img.crop((l,t,l+fw,t+fh))

res=[]
for row in ds:
    n=row["file_name_base"]
    dp=os.path.join(D, n+"_depth.npy"); mk=os.path.join(D, n+"_mask.npy")
    if not os.path.exists(dp): continue
    _b=row["image_blur"]; _im=_b if hasattr(_b,"convert") else Image.open(io.BytesIO(_b if isinstance(_b,bytes) else _b["bytes"])); alvo=prep(_im.convert("RGB")); w,h=alvo.size
    depth=cv2.resize(np.load(dp).astype(np.float32),(w,h),interpolation=cv2.INTER_LINEAR)
    disp=1.0/np.where(depth>0,depth,np.finfo(np.float32).max)
    cid=re.search(r"_test_f_(\d+)_level", n).group(1)
    md=json.load(open(hf_hub_download("timseizinger/RealBokeh_3MP", f"test/metadata/{cid}.json", repo_type="dataset", token=tok)))
    fpd=md.get("focus_plane_distance")
    # (a) mascara BiRefNet
    fm=np.nan
    if os.path.exists(mk):
        m=cv2.resize(np.load(mk).astype(np.float32),(w,h),interpolation=cv2.INTER_LINEAR)
        if (m>127).any(): fm=float(np.median(disp[m>127]))
    # (b) regiao NITIDA DO ALVO: 10% dos pixels com maior nitidez local
    g=cv2.cvtColor(np.array(alvo),cv2.COLOR_RGB2GRAY).astype(np.float32)
    lap=np.abs(cv2.Laplacian(g.astype(np.float64),cv2.CV_64F))
    nit=cv2.GaussianBlur(lap.astype(np.float32),(0,0),7)
    lim=np.percentile(nit,90)
    f_alvo=float(np.median(disp[nit>=lim]))
    res.append(dict(n=n, fpd=fpd, dmin=float(depth.min()), dmax=float(np.percentile(depth,99)),
                    dmed=float(np.median(depth)), f_mask=fm, f_alvo=f_alvo,
                    dentro=bool(fpd and depth.min()<=fpd<=np.percentile(depth,99))))
print("n analisadas:", len(res))
d=[r for r in res if r["fpd"]]
print("\nA ESCALA METRICA DO DEPTH PRO CONTEM O PLANO DE FOCO ANOTADO?")
print("  sim em %d de %d (%.0f%%)" % (sum(r["dentro"] for r in d), len(d), 100*np.mean([r["dentro"] for r in d])))
print("  prof. anotada: mediana=%.2f m | Depth Pro min: mediana=%.2f m" % (
    np.median([r["fpd"] for r in d]), np.median([r["dmin"] for r in d])))
for chave,rot in [("f_mask","Eq.4 BiRefNet"),("f_alvo","regiao nitida do ALVO")]:
    v=np.array([r[chave] for r in d], dtype=float); g=np.array([1.0/r["fpd"] for r in d])
    ok=~np.isnan(v)
    pe=1.0/v[ok]; pg=1.0/g[ok]
    print("\n%s (n=%d)" % (rot, ok.sum()))
    print("  correlacao de Pearson com a disparidade anotada: %.3f" % np.corrcoef(v[ok],g[ok])[0,1])
    print("  erro relativo de profundidade: mediana=%.0f%%  | dentro de 2x: %.0f%%" % (
        100*np.median(np.abs(pe-pg)/pg), 100*np.mean((pe/pg>0.5)&(pe/pg<2))))
vm=np.array([r["f_mask"] for r in d],dtype=float); va=np.array([r["f_alvo"] for r in d],dtype=float)
ok=~np.isnan(vm)
print("\ncorrelacao ENTRE as duas estimativas (mascara x regiao nitida do alvo): %.3f" % np.corrcoef(vm[ok],va[ok])[0,1])
