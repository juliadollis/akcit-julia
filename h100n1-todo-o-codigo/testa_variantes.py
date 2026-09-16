"""Descobre QUAL variante de MANIQA/MUSIQ/CLIP-IQA reproduz a Tab. 2 do paper.

Roda na linha Input do RealDOF, que nao tem modelo: qualquer diferenca ali e da
metrica, nao do modelo. Alvos publicados: MANIQA 0.2213, MUSIQ 28.7087,
CLIP-IQA 0.3562.
"""
import os
import numpy as np, torch, pyiqa
from datasets import load_dataset
from torchvision import transforms

tok=os.environ["HF_TOKEN"]
ALVO={"maniqa":0.2213,"musiq":28.7087,"clipiqa":0.3562}
disp=[m for m in pyiqa.list_models() if any(m.startswith(k) for k in ALVO)]
print("variantes disponiveis:", disp, flush=True)

d=load_dataset("juliadollis/tab2-infer-realdof-input", split="validation", token=tok)
print("n =", len(d), flush=True)
tt=transforms.ToTensor()
imgs=[tt(r["image_generated"].convert("RGB")).unsqueeze(0) for r in d]

for nome in disp:
    try:
        m=pyiqa.create_metric(nome, device="cuda")
        with torch.no_grad():
            v=float(np.mean([m(x.cuda()).item() for x in imgs]))
        fam=next(k for k in ALVO if nome.startswith(k))
        alvo=ALVO[fam]
        erro=abs(v-alvo)/alvo*100
        marca="  <<< BATE" if erro<3 else ""
        print("%-22s media=%9.4f   publicado=%9.4f   erro=%6.1f%%%s" % (nome, v, alvo, erro, marca), flush=True)
        del m; torch.cuda.empty_cache()
    except Exception as e:
        print("%-22s ERRO: %s" % (nome, str(e)[:90]), flush=True)
