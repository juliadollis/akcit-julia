#!/usr/bin/env python3
"""Controlabilidade COM ALVO REAL: split `test` do RealBokeh_3MP.

Por que este e o teste forte: cada cena tem a MESMA vista fotografada em varias
aberturas reais (f/2.0 ate f/22), com `focus_plane_distance` medido. Entao para
cada ponto do sweep existe a foto que o modelo DEVERIA ter produzido — coisa que
a DDPD nao oferece.

CONTAMINACAO: verificado em 2026-08-26 que as 2.932 amostras da rota c vem
100% de `RealBokeh_3MP/train/gt` (coluna source_aif/source_bokeh). O split
`test` nao foi tocado pelo treino.

Eixo do sweep, direto da fisica e sem parametro livre: na Eq. 3 do paper,
dentro de UMA cena f, D_focus e pixel_ratio sao constantes, logo K ∝ 1/F.
Normalizando pela abertura mais aberta da cena, alpha_j = F_min / F_j.
O mapa e clip(alpha_j * B_hat, 0, 1), com B_hat = |1/z - 1/D_focus| / p99.5.

CONTROLE POSITIVO DE VERDADE: a mesma metrica aplicada as FOTOS REAIS da o teto
alcancavel. Se a LVCorr do alvo real nao for alta, a metrica (ou o dado) nao
suporta a conclusao — e isso tem de aparecer no relatorio.
"""
from __future__ import annotations
import argparse, gc, json, os, sys, time
import numpy as np
import pandas as pd
import torch
from PIL import Image
from huggingface_hub import hf_hub_download, HfApi, list_repo_files

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "inference"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from controle_lib import mapa_base, mascaras, lap_resposta, lv_mascarado, pil_bytes

REPO = "timseizinger/RealBokeh_3MP"
MODEL_ID = "black-forest-labs/FLUX.1-dev"
PROMPT = "an excellent photo with a large aperture"
STEPS = 28
DIR_MAPAS = "/workspace/vision-pipeline/temp_depth_rb"


def prep(img, lado):
    w, h = img.size
    if lado and lado > 0:
        if w >= h: nw, nh = lado, int(h * lado / w)
        else: nh, nw = lado, int(w * lado / h)
        img = img.resize((nw, nh), Image.LANCZOS)
    else:
        nw, nh = w, h
    fw, fh = max(16, (nw // 16) * 16), max(16, (nh // 16) * 16)
    l, t = (nw - fw) // 2, (nh - fh) // 2
    return img.crop((l, t, l + fw, t + fh))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora-repo", required=True)
    ap.add_argument("--lora-file", default="bokeh.safetensors")
    ap.add_argument("--nome", required=True)
    ap.add_argument("--repo-metricas", default="juliadollis/bokeh-controlabilidade-realbokeh")
    ap.add_argument("--repo-imgs", default="juliadollis/bokeh-controlabilidade-realbokeh-imgs")
    ap.add_argument("--cenas", type=int, default=30)
    ap.add_argument("--long-side", type=int, default=512)
    ap.add_argument("--so-alvos", action="store_true",
                    help="nao roda o FLUX; so mede as FOTOS REAIS (controle positivo)")
    args = ap.parse_args()

    tok = os.environ.get("HF_TOKEN")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(DIR_MAPAS, exist_ok=True)

    fs = list_repo_files(REPO, repo_type="dataset", token=tok)
    metas = sorted([f for f in fs if f.startswith("test/metadata/") and f.endswith(".json")],
                   key=lambda x: int(x.split("/")[-1].split(".")[0]))
    cenas = []
    for m in metas:
        d = json.load(open(hf_hub_download(REPO, m, repo_type="dataset", token=tok)))
        if len(d.get("target_avs", [])) >= 4 and d.get("focus_plane_distance"):
            cenas.append(d)
        if len(cenas) >= args.cenas:
            break
    print(f"[dados] {len(cenas)} cenas do split test", flush=True)

    # baixa AIF (a foto mais fechada) + alvos
    dados = []
    for d in cenas:
        cid = d["id"]
        src = d["source_image"]
        cam_src = None
        for cand in (f"test/{src}", f"test/gt/{cid}/{cid}_f22.JPG"):
            try:
                cam_src = hf_hub_download(REPO, cand, repo_type="dataset", token=tok); break
            except Exception:
                continue
        if cam_src is None:
            print(f"  [pula] cena {cid}: sem imagem fonte", flush=True); continue
        alvos = []
        for tp, av in zip(d["target_images"], d["target_avs"]):
            try:
                alvos.append((float(av), hf_hub_download(REPO, f"test/{tp}", repo_type="dataset", token=tok)))
            except Exception:
                pass
        if len(alvos) < 4:
            print(f"  [pula] cena {cid}: {len(alvos)} alvos", flush=True); continue
        dados.append((cid, cam_src, sorted(alvos), float(d["focus_plane_distance"])))
    print(f"[dados] {len(dados)} cenas utilizaveis", flush=True)

    # Depth Pro nas AIFs
    faltam = [(c, s) for (c, s, a, z) in dados if not os.path.exists(f"{DIR_MAPAS}/{c}.npy")]
    if faltam:
        import depth_pro
        print(f"[depth] {len(faltam)} mapas", flush=True)
        mdl, tr = depth_pro.create_model_and_transforms(); mdl.eval().to(device)
        for c, s in faltam:
            base = prep(Image.open(s).convert("RGB"), 1024)
            with torch.no_grad():
                pr = mdl.infer(tr(base).to(device), f_px=None)
            np.save(f"{DIR_MAPAS}/{c}.npy", pr["depth"].cpu().numpy().squeeze())
        del mdl, tr; gc.collect(); torch.cuda.empty_cache()

    pipe = ADAPTER = None
    if not args.so_alvos:
        from diffusers import FluxPipeline
        from Genfocus.pipeline.flux import Condition, generate, seed_everything
        pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, token=tok)
        pipe.to("cuda"); pipe.enable_attention_slicing(); pipe.vae.enable_tiling()
        if str(args.lora_repo).lower() != "none":
            cam = hf_hub_download(args.lora_repo, args.lora_file, token=tok)
            pipe.load_lora_weights(os.path.dirname(cam), weight_name=os.path.basename(cam),
                                   adapter_name="bokeh")
            pipe.set_adapters(["bokeh"]); ADAPTER = "bokeh"
        print(f"[flux] {args.lora_repo}/{args.lora_file}", flush=True)

    linhas = []
    t0 = time.time()
    for (cid, cam_src, alvos, z_foco) in dados:
        aif = prep(Image.open(cam_src).convert("RGB"), args.long_side)
        w, h = aif.size
        z = np.load(f"{DIR_MAPAS}/{cid}.npy").astype(np.float32)
        # aqui o plano de foco vem MEDIDO no dataset, nao estimado
        zz = np.clip(np.array(Image.fromarray(z).resize((w, h), Image.BILINEAR)), 1e-3, None)
        b = np.abs(1.0 / zz - 1.0 / max(z_foco, 1e-3))
        p995 = float(np.percentile(b, 99.5)) or 1e-8
        b_hat = np.clip(b / p995, 0, 1).astype(np.float32)
        m_fun, m_foc = mascaras(b_hat)
        f_min = min(a for a, _ in alvos)
        for (av, cam_alvo) in alvos:
            alpha = float(f_min / av)
            alvo = prep(Image.open(cam_alvo).convert("RGB"), args.long_side)
            r_alvo = lap_resposta(alvo)
            reg = {"modelo": args.nome, "cena": int(cid), "f_number": float(av),
                   "alpha": alpha, "z_foco_m": z_foco,
                   "lv_alvo_total": float(r_alvo.var()),
                   "lv_alvo_fundo": lv_mascarado(r_alvo, m_fun),
                   "lv_alvo_foco": lv_mascarado(r_alvo, m_foc)}
            if not args.so_alvos:
                cond = np.clip(alpha * b_hat, 0, 1).astype(np.float32)
                t = torch.from_numpy(cond).unsqueeze(0).float().repeat(3, 1, 1).unsqueeze(0)
                c1 = Condition(aif, ADAPTER)
                c2 = Condition(t, ADAPTER, [0, 0], 1.0, No_preprocess=True)
                seed_everything(42)
                g = torch.Generator(device=device).manual_seed(1234)
                with torch.no_grad():
                    out = generate(pipe, height=h, width=w, prompt=PROMPT,
                                   num_inference_steps=STEPS, conditions=[c1, c2],
                                   guidance_scale=1.0, kv_cache=False, generator=g,
                                   NO_TILED_DENOISE=min(w, h) < 512).images[0]
                r = lap_resposta(out)
                reg.update({"lv_gerada_total": float(r.var()),
                            "lv_gerada_fundo": lv_mascarado(r, m_fun),
                            "lv_gerada_foco": lv_mascarado(r, m_foc),
                            "imagem": pil_bytes(out)})
            linhas.append(reg)
        print(f"[cena] {cid} ({len(alvos)} aberturas)  {(time.time()-t0)/60:.1f} min", flush=True)
        torch.cuda.empty_cache()

    df = pd.DataFrame(linhas)
    api = HfApi()
    api.create_repo(args.repo_metricas, repo_type="dataset", private=True, exist_ok=True, token=tok)
    esc = df.drop(columns=["imagem"], errors="ignore")
    loc = f"/tmp/rb_{args.nome}.parquet"; esc.to_parquet(loc, index=False)
    api.upload_file(path_or_fileobj=loc, path_in_repo=f"data/{args.nome}.parquet",
                    repo_id=args.repo_metricas, repo_type="dataset", token=tok)
    esc.to_parquet(f"/workspace/vision-pipeline/lvcorr/rb_{args.nome}.parquet", index=False)
    print(f"[hf] -> {args.repo_metricas}/data/{args.nome}.parquet", flush=True)
    print(f"[fim] {len(df)} linhas em {(time.time()-t0)/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
