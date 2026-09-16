"""Valida a Eq. 4 (BiRefNet + Depth Pro) contra o plano de foco ANOTADO.

O log da inferencia grava, por imagem, o `disp_focus` que a Eq. 4 estimou.
O metadado do RealBokeh traz `focus_plane_distance` em METROS. Como a
disparidade e 1/profundidade, os dois sao diretamente comparaveis.
"""
import os, re, json, numpy as np
from huggingface_hub import hf_hub_download
log=open("/workspace/b/rb_oficial.log", errors="ignore").read().replace("\r","\n")
est={}
for m in re.finditer(r"\[foco\] (\S+): origem=(\S+) disp_focus=([0-9.]+)", log):
    est[m.group(1)]=(m.group(2), float(m.group(3)))
print("imagens com foco estimado no log:", len(est))
tok=os.environ["HF_TOKEN"]
lin=[]
for nome,(orig,d) in est.items():
    cid=re.search(r"_test_f_(\d+)_level", nome)
    if not cid: continue
    md=json.load(open(hf_hub_download("timseizinger/RealBokeh_3MP", f"test/metadata/{cid.group(1)}.json", repo_type="dataset", token=tok)))
    fpd=md.get("focus_plane_distance"); unc=md.get("focus_plane_uncertainty") or 0.0
    if not fpd: continue
    lin.append((nome, orig, d, 1.0/fpd, fpd, 1.0/d, unc))
print("%-46s %-9s %8s %8s %8s %8s" % ("cena","origem","disp_est","disp_gt","prof_est","prof_gt"))
for nome,orig,de,dg,fpd,pe,unc in lin:
    print("%-46s %-9s %8.4f %8.4f %8.2f %8.2f" % (nome[-44:], orig, de, dg, pe, fpd))
if lin:
    de=np.array([x[2] for x in lin]); dg=np.array([x[3] for x in lin])
    pe=np.array([x[5] for x in lin]); pg=np.array([x[4] for x in lin])
    print("\nn=%d" % len(lin))
    print("erro ABSOLUTO da profundidade do plano de foco (m): mediana=%.3f  p90=%.3f" % (
        np.median(np.abs(pe-pg)), np.percentile(np.abs(pe-pg),90)))
    print("erro RELATIVO: mediana=%.1f%%" % (100*np.median(np.abs(pe-pg)/pg)))
    if len(lin)>2:
        print("correlacao de Pearson entre disparidade estimada e anotada: %.3f" % np.corrcoef(de,dg)[0,1])
        print("fracao dentro de 2x: %.0f%%" % (100*np.mean((pe/pg>0.5)&(pe/pg<2.0))))
