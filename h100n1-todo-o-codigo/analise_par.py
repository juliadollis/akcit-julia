"""A vantagem do nosso modelo e QUALIDADE ou CONSERVADORISMO?

Teste: se a vantagem vier de obedecer menos a um condicionamento ERRADO, ela
deve CRESCER nas imagens em que a Eq. 4 erra mais o plano de foco. Se vier de
qualidade, deve ser indiferente ao erro do plano de foco.
"""
import os, sys, io, re, json
sys.path.insert(0,"/workspace/vision-pipeline"); sys.path.insert(0,"/workspace/vision-pipeline/inference")
import numpy as np, cv2, torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from evaluation.eval_bokeh_synthesis import CloudBokehEvaluator, get_pil_image
tok=os.environ["HF_TOKEN"]
ev=CloudBokehEvaluator(device="cpu")

# erro do plano de foco por imagem (log da corrida oficial x metadado anotado)
log=open("/workspace/b/rb_oficial.log", errors="ignore").read().replace("\r","\n")
est={m.group(1):(m.group(2),float(m.group(3))) for m in re.finditer(r"\[foco\] (\S+): origem=(\S+) disp_focus=([0-9.]+)", log)}
erro={}
for nome,(orig,d) in est.items():
    cid=re.search(r"_test_f_(\d+)_level", nome)
    if not cid: continue
    md=json.load(open(hf_hub_download("timseizinger/RealBokeh_3MP", f"test/metadata/{cid.group(1)}.json", repo_type="dataset", token=tok)))
    fpd=md.get("focus_plane_distance")
    if fpd: erro[nome]=(abs(1.0/d - fpd)/fpd, orig)   # erro relativo de profundidade

def lv(im): return float(cv2.Laplacian(cv2.cvtColor(np.array(im),cv2.COLOR_RGB2GRAY),cv2.CV_64F).var())
def por_imagem(repo):
    ds=load_dataset(repo, split="validation", token=tok, download_mode="force_redownload")
    out={}
    for row in ds:
        a=get_pil_image(row["image_real_bokeh"]).convert("RGB"); g=get_pil_image(row["image_best_k"]).convert("RGB")
        ta=ev.to_tensor(a).unsqueeze(0); tg=ev.to_tensor(g).unsqueeze(0)
        if tg.shape!=ta.shape:
            import torch.nn.functional as F
            tg=torch.clamp(F.interpolate(tg,size=ta.shape[2:],mode="bicubic",align_corners=False),0,1)
        with torch.no_grad(): l=ev.lpips(tg,ta).item()
        out[row["file_name_base"]]=(l, lv(g)/max(lv(a),1e-6), float(row["best_k_value"]))
    return out
of=por_imagem(os.environ.get("REPO_OF","juliadollis/bokeh-eval-rb-oficial"))
no=por_imagem(os.environ.get("REPO_NO","juliadollis/bokeh-eval-rb-nosso"))
idt=por_imagem("juliadollis/bokeh-eval-rb-identidade")
comuns=[n for n in of if n in no and n in idt and n in erro]
print("n comum:", len(comuns))
e=np.array([erro[n][0] for n in comuns])
dl=np.array([no[n][0]-of[n][0] for n in comuns])          # negativo = nosso melhor
gan_no=np.array([idt[n][0]-no[n][0] for n in comuns])     # ganho do nosso sobre identidade
gan_of=np.array([idt[n][0]-of[n][0] for n in comuns])
rn=np.array([no[n][1] for n in comuns]); ro=np.array([of[n][1] for n in comuns]); ri=np.array([idt[n][1] for n in comuns])
print("\nerro relativo do plano de foco: mediana=%.0f%%" % (100*np.median(e)))
print("vantagem do nosso (LPIPS nosso - oficial, negativo = nosso melhor): mediana=%+.4f | nosso ganha em %d de %d" % (
    np.median(dl), int((dl<0).sum()), len(dl)))
print("\nTESTE 1 — a vantagem do nosso cresce onde a Eq.4 erra mais?")
print("  Pearson(erro do foco, vantagem do nosso) = %+.3f  (negativo = sim, cresce)" % np.corrcoef(e,dl)[0,1])
lim=np.median(e); baixo=e<=lim; alto=e>lim
print("  metade com MENOR erro de foco: vantagem mediana %+.4f (n=%d)" % (np.median(dl[baixo]), baixo.sum()))
print("  metade com MAIOR  erro de foco: vantagem mediana %+.4f (n=%d)" % (np.median(dl[alto]), alto.sum()))
print("\nTESTE 2 — quanto cada modelo se afasta da identidade (nitidez gerada/alvo; alvo=1,0)")
print("  identidade %.2f | oficial %.2f | nosso %.2f" % (np.median(ri), np.median(ro), np.median(rn)))
print("  |log da razao| (0 = perfeito): oficial %.2f | nosso %.2f" % (
    np.median(np.abs(np.log(ro))), np.median(np.abs(np.log(rn)))))
print("\nTESTE 3 — ganho sobre a identidade por metade de erro de foco (positivo = melhor que nao fazer nada)")
for rot,m in [("MENOR erro",baixo),("MAIOR  erro",alto)]:
    print("  %s: nosso %+.4f | oficial %+.4f" % (rot, np.median(gan_no[m]), np.median(gan_of[m])))
