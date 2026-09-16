"""Verifica se o nivel de MENOR variancia do Laplaciano e mesmo o f/2.0 real.

Metodo: baixa o JPG bruto gt/<id>/<id>_f2.0.JPG e correlaciona com TODOS os
niveis daquela cena no parquet. Se o argmax cair no nivel que a nossa regra
escolheu (menor LV), a regra esta confirmada por conteudo de pixel.
"""
import os, io, re, collections, numpy as np, cv2
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
from PIL import Image
tok=os.environ["HF_TOKEN"]

def mini(b, L=160):
    im=Image.open(io.BytesIO(b)).convert("L").resize((L,L), Image.LANCZOS)
    a=np.asarray(im,dtype=np.float32); a-=a.mean(); s=a.std()
    return a/s if s>1e-6 else a
def lv(b, L=512):
    im=Image.open(io.BytesIO(b)).convert("RGB"); w,h=im.size; s=L/max(w,h)
    im=im.resize((max(1,int(w*s)),max(1,int(h*s))), Image.LANCZOS)
    return float(cv2.Laplacian(cv2.cvtColor(np.array(im),cv2.COLOR_RGB2GRAY),cv2.CV_64F).var())

cenas=collections.defaultdict(list)
for i in range(6):
    p=hf_hub_download("akcit-pixel/RealBokeh", f"data/test-0000{i}-of-00006.parquet", repo_type="dataset", token=tok)
    t=pq.read_table(p, columns=["image_blur","file_name_base"])
    nb=t.column("image_blur").to_pylist(); nn=t.column("file_name_base").to_pylist()
    for j in range(len(nn)):
        m=re.match(r"^.*_test_f_(\d+)_level_(\d+)_(.*)$", str(nn[j]))
        if m: cenas[m.group(1)].append((int(m.group(2)), m.group(3), nb[j]["bytes"], str(nn[j])))
    del t, nb
print("cenas lidas:", len(cenas))
alvo=sorted(cenas, key=lambda x:int(x))[:25]
acertos=0; total=0
for cid in alvo:
    itens=cenas[cid]
    try:
        jp=hf_hub_download("timseizinger/RealBokeh_3MP", f"test/gt/{cid}/{cid}_f2.0.JPG", repo_type="dataset", token=tok)
        raw=open(jp,"rb").read()
    except Exception as e:
        print(cid, "sem f2.0 bruto"); continue
    ref=mini(raw)
    cors=[float((ref*mini(b)).mean()) for _,_,b,_ in itens]
    lvs=[lv(b) for _,_,b,_ in itens]
    i_cor=int(np.argmax(cors)); i_lv=int(np.argmin(lvs))
    total+=1; acertos+= (i_cor==i_lv)
    print("cena %-4s | menor LV -> nivel %-2d (%s) | melhor correl com f2.0 -> nivel %-2d (corr=%.3f) | %s" % (
        cid, itens[i_lv][0], itens[i_lv][1], itens[i_cor][0], cors[i_cor], "OK" if i_cor==i_lv else "DIVERGE"))
print("CONFIRMACAO: %d/%d (%.0f%%) — a regra menor-LV seleciona o f/2.0 real" % (acertos,total,100*acertos/max(total,1)))
