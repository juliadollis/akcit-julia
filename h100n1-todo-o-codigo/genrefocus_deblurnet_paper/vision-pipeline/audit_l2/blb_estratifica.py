#!/usr/bin/env python3
"""Metricas GLOBAIS e ESTRATIFICADAS POR FAIXA DE K, de UMA unica inferencia.

Motivo (auditoria 2026-08-27): 15% do benchmark tem K > 300, acima de tudo que
qualquer modelo viu em treino, e 30% tem K > 100, acima da faixa do paper. A
media global mistura interpolacao com extrapolacao. A estratificada responde se
o modelo aprendeu a lei optica ou decorou a faixa de treino.

Usa as MESMAS variantes de metrica do avaliador do projeto (lpips+, dists) para
que os numeros sejam comparaveis. Nao modifica nenhum arquivo protegido.
Grava por linha e por faixa, e sobe o resultado para o HF.
"""
import io, os, argparse
import numpy as np
import pandas as pd
from PIL import Image
import pyarrow.parquet as pq, pyarrow as pa
import torch
from huggingface_hub import HfApi
from skimage.metrics import structural_similarity as ssim

FAIXAS = [("K<=100", 0.0, 100.0), ("100<K<=300", 100.0, 300.0), ("K>300", 300.0, 1e9)]

def pil(v):
    if isinstance(v, dict): v = v.get("bytes")
    return Image.open(io.BytesIO(v)).convert("RGB") if isinstance(v, (bytes, bytearray)) else v

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida-infer", required=True)
    ap.add_argument("--bench", default="juliadollis/lf-bokeh-repro-blb")
    ap.add_argument("--nome", required=True)
    ap.add_argument("--repo-metricas", default="juliadollis/lfrepro-metricas-estratificadas")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    tok = os.environ.get("HF_TOKEN"); api = HfApi()

    import pyiqa, clip
    from torchvision import transforms
    dev = torch.device(a.device)
    m_lpips = pyiqa.create_metric("lpips+", device=dev)
    m_dists = pyiqa.create_metric("dists", device=dev)
    clip_model, _ = clip.load("ViT-B/32", device=dev); clip_model.eval()
    prep = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711))])
    tt = transforms.ToTensor()

    arqs = [f for f in api.list_repo_files(a.bench, repo_type="dataset", token=tok) if f.endswith(".parquet")]
    B = pa.concat_tables([pq.read_table("hf://datasets/" + a.bench + "/" + f,
          columns=["file_name_base", "k_ref", "f_stop", "focus_distance"]) for f in arqs])
    meta = {n: (k, fs, fd) for n, k, fs, fd in zip(
        B["file_name_base"].to_pylist(), B["k_ref"].to_pylist(),
        B["f_stop"].to_pylist(), B["focus_distance"].to_pylist())}

    arqs = [f for f in api.list_repo_files(a.saida_infer, repo_type="dataset", token=tok) if f.endswith(".parquet")]
    S = pa.concat_tables([pq.read_table("hf://datasets/" + a.saida_infer + "/" + f) for f in arqs])
    print("inferencia: %d linhas | benchmark: %d" % (S.num_rows, len(meta)))

    nomes = S["file_name_base"].to_pylist()
    ger = S["image_best_k"].to_pylist(); alvo = S["image_real_bokeh"].to_pylist()
    bk = S["best_k_value"].to_pylist() if "best_k_value" in S.column_names else [None]*S.num_rows

    linhas = []
    for i, nm in enumerate(nomes):
        if nm not in meta: continue
        g, t = pil(ger[i]), pil(alvo[i])
        if g is None or t is None: continue
        if g.size != t.size: g = g.resize(t.size, Image.BICUBIC)
        gg = np.array(g.convert("L")); tg = np.array(t.convert("L"))
        s = float(ssim(tg, gg))
        gt_ = tt(g).unsqueeze(0).to(dev); tt_ = tt(t).unsqueeze(0).to(dev)
        with torch.no_grad():
            lp = float(m_lpips(gt_, tt_).item()); ds = float(m_dists(gt_, tt_).item())
            eg = clip_model.encode_image(prep(g).unsqueeze(0).to(dev))
            et = clip_model.encode_image(prep(t).unsqueeze(0).to(dev))
            eg = eg/eg.norm(dim=-1, keepdim=True); et = et/et.norm(dim=-1, keepdim=True)
            ci = float(torch.cosine_similarity(eg, et).item())
        k, fs, fd = meta[nm]
        linhas.append({"file_name_base": nm, "k_ref": k, "f_stop": fs, "focus_distance": fd,
                       "best_k": bk[i], "SSIM": s, "LPIPS": lp, "DISTS": ds, "CLIP-I": ci})
        if len(linhas) % 25 == 0: print("  %d..." % len(linhas), flush=True)

    df = pd.DataFrame(linhas)
    if df.empty: print("nada casou"); return
    print("")
    print("=" * 92)
    print("MODELO: " + a.nome)
    print("=" * 92)
    print("%-14s %5s %8s %8s %8s %8s %9s" % ("faixa", "n", "SSIM", "LPIPS", "DISTS", "CLIP-I", "best_k med"))
    def linha(rot, d):
        if d.empty: return
        print("%-14s %5d %8.4f %8.4f %8.4f %8.4f %9.1f" % (rot, len(d),
              d["SSIM"].mean(), d["LPIPS"].mean(), d["DISTS"].mean(), d["CLIP-I"].mean(),
              d["best_k"].median() if d["best_k"].notna().any() else float("nan")))
    linha("GLOBAL", df)
    for rot, lo, hi in FAIXAS:
        linha(rot, df[(df["k_ref"] > lo) & (df["k_ref"] <= hi)])
    print("")
    print("%-14s %5s %8s %8s" % ("abertura", "n", "LPIPS", "DISTS"))
    linha2 = lambda rot, d: print("%-14s %5d %8.4f %8.4f" % (rot, len(d), d["LPIPS"].mean(), d["DISTS"].mean())) if not d.empty else None
    linha2("f>=0.7 (real)", df[df["f_stop"] >= 0.7])
    linha2("f<0.7 (irreal)", df[df["f_stop"] < 0.7])

    df["modelo"] = a.nome
    loc = "/tmp/estrat_%s.parquet" % a.nome.replace("/", "_")
    df.to_parquet(loc, index=False)
    api.create_repo(a.repo_metricas, repo_type="dataset", private=True, exist_ok=True, token=tok)
    api.upload_file(path_or_fileobj=loc, path_in_repo="por_linha/%s.parquet" % a.nome.replace("/", "_"),
                    repo_id=a.repo_metricas, repo_type="dataset", token=tok)
    print("")
    print("por-linha salvo em https://huggingface.co/datasets/" + a.repo_metricas)

if __name__ == "__main__":
    main()
