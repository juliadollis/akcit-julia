"""Valida o harness do BokehMe reproduzindo a Tab. 1 do paper deles no BLB.

Se o PSNR do nivel 1 sair perto de 43.30 (SSIM 0.9932), o harness esta certo e
os numeros que ele produzir para a nossa tabela sao confiaveis. Se nao sair, o
harness esta errado e nao se publica nada dele.
"""
import subprocess, sys, os, json
import numpy as np, cv2

BASE="/host/blb_laptop/data"
OUT="/host/concorrentes/valida"
os.makedirs(OUT, exist_ok=True)

def psnr(a,b):
    a=a.astype(np.float64)/255; b=b.astype(np.float64)/255
    m=np.mean((a-b)**2)
    return 10*np.log10(1.0/m) if m>0 else 99.0

cena=sys.argv[1] if len(sys.argv)>1 else "277"
info=json.load(open(f"{BASE}/{cena}/info.json"))
print("blur_parameters do BLB:", [round(x,2) for x in info["blur_parameters"]])

alvo = cv2.imread(f"{BASE}/{cena}/bokeh_00_00.jpg")   # k_idx 0, refocus 0
disp = cv2.imread(f"{BASE}/{cena}/disparity.jpg", 0)
print("disparidade: min=%d max=%d | alvo %s" % (disp.min(), disp.max(), alvo.shape))

for rotulo, K in [("blur_parameter do info.json", info["blur_parameters"][0]),
                  ("K=10 (o que a Tab.1 do paper diz)", 10.0)]:
    d=f"{OUT}/{cena}_K{K:.0f}"
    r=subprocess.run([sys.executable,"demo.py","--image_path",f"{BASE}/{cena}/image.jpg",
        "--disp_path",f"{BASE}/{cena}/disparity.jpg","--save_dir",d,
        "--K",str(K),"--disp_focus","0.0","--gamma","2.2"],
        cwd="/host/concorrentes/BokehMe", capture_output=True, text=True)
    p=f"{d}/image/bokeh_pred.jpg"
    if not os.path.exists(p):
        print(f"  {rotulo}: FALHOU", r.stderr[-300:]); continue
    pred=cv2.imread(p)
    if pred.shape!=alvo.shape: pred=cv2.resize(pred,(alvo.shape[1],alvo.shape[0]))
    print("  %-38s K=%8.2f  PSNR=%6.2f" % (rotulo, K, psnr(pred,alvo)))
print("\nreferencia publicada (BokehMe Tab.1, nivel 1): PSNR 43.30, SSIM 0.9932")
