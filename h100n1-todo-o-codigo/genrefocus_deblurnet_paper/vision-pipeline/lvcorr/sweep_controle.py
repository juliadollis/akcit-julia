#!/usr/bin/env python3
"""Sweep de controlabilidade de bokeh (LVCorr) para UM checkpoint do BokehNet.

DESENHO (ver relatorio):
  - o eixo do sweep e ALPHA, a amplitude do MAPA NORMALIZADO que entra no
    modelo, nao o K fisico. Motivo: alpha e exatamente o K do paper reescalado
    POR IMAGEM para que o mapa cubra [0,1]. Com um K comum a todas as imagens,
    o mesmo K satura umas e deixa outras em zero (medido: o K que leva o mapa a
    1.0 varia de 35 a 3273 entre as imagens da DDPD), e o sweep deixa de medir
    a mesma coisa em cada imagem. Como a LVCorr e calculada POR IMAGEM e a
    correlacao de postos e invariante a reescala monotona, alpha nao muda a
    metrica: so garante que ela seja medida na faixa util.
  - 9 pontos de alpha, de 0 a 1. E a faixa em que os dois modelos foram
    treinados (medido: mapa de treino do kfix p25=0.25 p50=0.47 p75=0.84;
    do original, teto fixo em 0.5; da rota c, 61% saturando em 1.0).
  - seed FIXA em todos os pontos: o unico fator que varia e alpha.
  - modo `nulo`: alpha CONSTANTE e 9 seeds diferentes. E o piso de ruido do
    proprio modelo, medido pelo mesmo codigo. Sem ele nao da para dizer se um
    LVCorr e sinal.

CORRECOES em relacao ao pipeline de inferencia existente (ver controle_lib):
  - depth reamostrado para a resolucao da imagem gerada;
  - plano de foco pela mediana de um patch central de verdade.
"""
from __future__ import annotations
import argparse, gc, io, os, sys, time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image
from datasets import load_dataset
from huggingface_hub import HfApi

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "inference"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from controle_lib import (mapa_base, mascaras, lap_resposta, lv_mascarado, pil_bytes)

MODEL_ID = "black-forest-labs/FLUX.1-dev"
DIR_MAPAS = "/workspace/vision-pipeline/temp_depth_maps"
PROMPT = "an excellent photo with a large aperture"
STEPS = 28
ALPHAS = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
SEEDS_NULO = [1234, 11, 22, 33, 44, 55, 66, 77, 88]
ALPHA_NULO = 0.5


def prep(img: Image.Image, lado: int) -> Image.Image:
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


def pil_de(d):
    if hasattr(d, "convert"): return d
    if isinstance(d, bytes): return Image.open(io.BytesIO(d))
    if isinstance(d, dict) and d.get("bytes"): return Image.open(io.BytesIO(d["bytes"]))
    if isinstance(d, dict) and d.get("path"): return Image.open(d["path"])
    raise ValueError(type(d))


def garantir_depth(itens, device):
    faltam = [(n, im) for (n, im, p) in itens if not os.path.exists(p)]
    if not faltam:
        print(f"[depth] todos os {len(itens)} mapas ja existem — Depth Pro nao sera carregado.", flush=True)
        return
    print(f"[depth] faltam {len(faltam)} mapas; carregando Depth Pro.", flush=True)
    import depth_pro
    m, tr = depth_pro.create_model_and_transforms()
    m.eval().to(device)
    for (n, im, p) in itens:
        if os.path.exists(p): continue
        base = prep(im, 0)
        w, h = base.size
        with torch.no_grad():
            pred = m.infer(tr(base).to(device), f_px=None)
        d = pred["depth"].cpu().numpy().squeeze()
        d = np.array(Image.fromarray(d).resize((w, h), Image.BILINEAR))
        np.save(p, d)
    del m, tr; gc.collect(); torch.cuda.empty_cache()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora-repo", required=True, help="'none' = FLUX cru")
    ap.add_argument("--lora-file", default="bokeh.safetensors")
    ap.add_argument("--nome", required=True)
    ap.add_argument("--repo-imgs", required=True)
    ap.add_argument("--repo-metricas", required=True)
    ap.add_argument("--dataset", default="akcit-pixel/DDPD")
    ap.add_argument("--padrao", default="data/validation-*.parquet")
    ap.add_argument("--limite", type=int, default=30)
    ap.add_argument("--limite-nulo", type=int, default=12)
    ap.add_argument("--long-side", type=int, default=512)
    ap.add_argument("--sem-nulo", action="store_true")
    ap.add_argument("--sem-imagens", action="store_true", help="nao sobe PNGs")
    args = ap.parse_args()

    tok = os.environ.get("HF_TOKEN")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    os.makedirs(DIR_MAPAS, exist_ok=True)

    ds = load_dataset("parquet", data_files={
        "validation": f"hf://datasets/{args.dataset}/{args.padrao}"}, token=tok)["validation"]
    n = min(args.limite, len(ds)) if args.limite else len(ds)
    print(f"[dados] {args.dataset}: {len(ds)} imagens, usando {n}", flush=True)

    itens = []
    for i in range(n):
        r = ds[i]
        nome = r.get("file_name_base") or f"val_{i:05d}.png"
        img = pil_de(r["image_focus"]).convert("RGB")
        itens.append((nome, img, os.path.join(DIR_MAPAS, f"{nome}_depth.npy")))
    garantir_depth(itens, device)

    from diffusers import FluxPipeline
    from Genfocus.pipeline.flux import Condition, generate, seed_everything
    print("[flux] carregando pipeline", flush=True)
    pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype, token=tok)
    if device == "cuda":
        pipe.to("cuda"); pipe.enable_attention_slicing(); pipe.vae.enable_tiling()

    sem_lora = str(args.lora_repo).lower() in ("", "none")
    if sem_lora:
        ADAPTER = None
        print("[flux] SEM LoRA (FLUX.1-dev cru) — controle negativo estrutural", flush=True)
    else:
        from huggingface_hub import hf_hub_download
        cam = hf_hub_download(args.lora_repo, args.lora_file, token=tok)
        pipe.load_lora_weights(os.path.dirname(cam), weight_name=os.path.basename(cam), adapter_name="bokeh")
        pipe.set_adapters(["bokeh"])
        ADAPTER = "bokeh"
        print(f"[flux] LoRA {args.lora_repo}/{args.lora_file}", flush=True)

    def render(aif, b_hat, alpha, w, h, seed):
        cond = np.clip(alpha * b_hat, 0.0, 1.0).astype(np.float32)
        t = torch.from_numpy(cond).unsqueeze(0).float().repeat(3, 1, 1).unsqueeze(0)
        c_img = Condition(aif, ADAPTER)
        c_map = Condition(t, ADAPTER, [0, 0], 1.0, No_preprocess=True)
        seed_everything(42)
        g = torch.Generator(device=device).manual_seed(int(seed))
        with torch.no_grad():
            return generate(pipe, height=h, width=w, prompt=PROMPT,
                            num_inference_steps=STEPS, conditions=[c_img, c_map],
                            guidance_scale=1.0, kv_cache=False, generator=g,
                            NO_TILED_DENOISE=min(w, h) < 512).images[0]

    linhas = []
    t0 = time.time()
    for modo, lim in (("sweep", n), ("nulo", 0 if args.sem_nulo else min(args.limite_nulo, n))):
        for idx in range(lim):
            nome, img_raw, cam_npy = itens[idx]
            aif = prep(img_raw, args.long_side)
            w, h = aif.size
            depth = np.load(cam_npy).astype(np.float32)
            b_hat, p995, disp_foco = mapa_base(depth, w, h)
            m_fun, m_foc = mascaras(b_hat)
            pontos = list(zip(ALPHAS, [1234] * len(ALPHAS))) if modo == "sweep" \
                else [(ALPHA_NULO, s) for s in SEEDS_NULO]
            for j, (al, sd) in enumerate(pontos):
                out = render(aif, b_hat, al, w, h, sd)
                resp = lap_resposta(out)
                linhas.append({
                    "modelo": args.nome, "modo": modo, "file_name_base": str(nome),
                    "idx_img": int(idx), "idx_ponto": int(j),
                    "alpha": float(al), "seed": int(sd),
                    "lv_total": float(resp.var()),
                    "lv_fundo": float(lv_mascarado(resp, m_fun)),
                    "lv_foco": float(lv_mascarado(resp, m_foc)),
                    "frac_fundo": float(m_fun.mean()), "frac_foco": float(m_foc.mean()),
                    "mapa_max": float(np.clip(al * b_hat, 0, 1).max()),
                    "mapa_media": float(np.clip(al * b_hat, 0, 1).mean()),
                    "p995_disp": float(p995), "disp_foco": float(disp_foco),
                    "w": int(w), "h": int(h),
                    "imagem": b"" if args.sem_imagens else pil_bytes(out),
                })
            el = time.time() - t0
            print(f"[{modo}] {idx+1}/{lim} {nome} ({el/60:.1f} min)", flush=True)
            torch.cuda.empty_cache()

    df = pd.DataFrame(linhas)
    api = HfApi()
    # --- imagens: repo PRIVADO (derivam de dataset de terceiros)
    if not args.sem_imagens:
        api.create_repo(args.repo_imgs, repo_type="dataset", private=True, exist_ok=True, token=tok)
        esq = pa.schema([
            pa.field("modelo", pa.string()), pa.field("modo", pa.string()),
            pa.field("file_name_base", pa.string()), pa.field("idx_img", pa.int32()),
            pa.field("idx_ponto", pa.int32()), pa.field("alpha", pa.float32()),
            pa.field("seed", pa.int32()), pa.field("lv_total", pa.float64()),
            pa.field("lv_fundo", pa.float64()), pa.field("lv_foco", pa.float64()),
            pa.field("imagem", pa.binary()),
        ]).with_metadata({b"huggingface": b'{"info": {"features": {'
            b'"modelo":{"dtype":"string","_type":"Value"},"modo":{"dtype":"string","_type":"Value"},'
            b'"file_name_base":{"dtype":"string","_type":"Value"},"idx_img":{"dtype":"int32","_type":"Value"},'
            b'"idx_ponto":{"dtype":"int32","_type":"Value"},"alpha":{"dtype":"float32","_type":"Value"},'
            b'"seed":{"dtype":"int32","_type":"Value"},"lv_total":{"dtype":"float64","_type":"Value"},'
            b'"lv_fundo":{"dtype":"float64","_type":"Value"},"lv_foco":{"dtype":"float64","_type":"Value"},'
            b'"imagem":{"_type":"Image"}}}}'})
        sub = df[[f.name for f in esq]]
        loc = f"/tmp/sweep_{args.nome}.parquet"
        pq.write_table(pa.Table.from_pandas(sub, schema=esq, preserve_index=False), loc)
        api.upload_file(path_or_fileobj=loc, path_in_repo=f"data/{args.nome}.parquet",
                        repo_id=args.repo_imgs, repo_type="dataset", token=tok)
        print(f"[hf] imagens -> {args.repo_imgs}/data/{args.nome}.parquet", flush=True)

    # --- escalares: repo separado, um parquet por modelo (nada e sobrescrito)
    api.create_repo(args.repo_metricas, repo_type="dataset", private=True, exist_ok=True, token=tok)
    esc = df.drop(columns=["imagem"])
    loc2 = f"/tmp/lv_{args.nome}.parquet"
    esc.to_parquet(loc2, index=False)
    api.upload_file(path_or_fileobj=loc2, path_in_repo=f"data/{args.nome}.parquet",
                    repo_id=args.repo_metricas, repo_type="dataset", token=tok)
    print(f"[hf] escalares -> {args.repo_metricas}/data/{args.nome}.parquet", flush=True)
    esc.to_parquet(f"/workspace/vision-pipeline/lvcorr/lv_{args.nome}.parquet", index=False)
    print(f"[fim] {len(df)} renders em {(time.time()-t0)/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
