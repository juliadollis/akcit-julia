"""Recalcula as metricas EXCLUINDO linhas problematicas, sem re-inferir nada.

Motivo: 1 das 40 cenas avaliadas (`..._f_131_level_1_shift_4.4px`) e uma cena
DUPLICADA (a 131 ja entra alinhada) e vem com desalinhamento de 4,4 px, que
penaliza qualquer metrica de pixel independentemente do modelo. Como afeta os
tres modelos igual, o ranking nao muda; este recalculo mede o quanto.
"""
import os, sys, io
sys.path.insert(0,"/workspace/vision-pipeline"); sys.path.insert(0,"/workspace/vision-pipeline/inference")
import numpy as np, cv2, torch
from datasets import load_dataset
from evaluation.eval_bokeh_synthesis import CloudBokehEvaluator, get_pil_image, img_to_gray_cv2, calculate_laplacian_variance, K_VALUES, K_IMAGE_FIELDS
from skimage.metrics import structural_similarity as ssim
from scipy.stats import pearsonr
tok=os.environ["HF_TOKEN"]
ev=CloudBokehEvaluator(device="cpu")
for arg in sys.argv[1:]:
    nome,repo=arg.split("=")
    try: ds=load_dataset(repo, split="validation", token=tok, download_mode="force_redownload")
    except Exception as e: print(nome,"indisponivel",type(e).__name__); continue
    for rot,filtro in [("TODAS", lambda n: True), ("SO ALINHADAS", lambda n: n.endswith("_aligned"))]:
        acc={"SSIM":0,"LPIPS":0,"DISTS":0,"CLIP-I":0,"LVCorr":0}; n=0
        for row in ds:
            if not filtro(row["file_name_base"]): continue
            a=get_pil_image(row["image_real_bokeh"]).convert("RGB"); g=get_pil_image(row["image_best_k"]).convert("RGB")
            ga,gg=img_to_gray_cv2(a),img_to_gray_cv2(g)
            if gg.shape!=ga.shape: gg=cv2.resize(gg,(ga.shape[1],ga.shape[0]))
            acc["SSIM"]+=ssim(ga,gg)
            ta=ev.to_tensor(a).unsqueeze(0); tg=ev.to_tensor(g).unsqueeze(0)
            if tg.shape!=ta.shape:
                import torch.nn.functional as F
                tg=torch.clamp(F.interpolate(tg,size=ta.shape[2:],mode="bicubic",align_corners=False),0,1)
            with torch.no_grad():
                acc["LPIPS"]+=ev.lpips(tg,ta).item(); acc["DISTS"]+=ev.dists(tg,ta).item()
                acc["CLIP-I"]+=torch.cosine_similarity(ev.get_clip_embedding(g),ev.get_clip_embedding(a)).item()
            lvs=[calculate_laplacian_variance(get_pil_image(row[f])) for f in K_IMAGE_FIELDS]
            acc["LVCorr"]+= 0.0 if np.std(lvs)==0 else pearsonr(K_VALUES,lvs)[0]
            n+=1
        if n: print("%-22s %-13s n=%2d | SSIM %.4f  LPIPS %.4f  DISTS %.4f  CLIP-I %.4f  LVCorr %+.4f" % (
            nome, rot, n, acc["SSIM"]/n, acc["LPIPS"]/n, acc["DISTS"]/n, acc["CLIP-I"]/n, acc["LVCorr"]/n), flush=True)
