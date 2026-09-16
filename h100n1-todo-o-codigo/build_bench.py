"""Monta o benchmark de bokeh a partir do split TEST do RealBokeh_3MP.

POR QUE ESTE DADO
O LF-Bokeh do paper nao e publico. A rota c (AKCITPixel3/CMiQdveBBzNii), que e o
dado mais proximo que temos, foi rastreada e vem 100% do split TRAIN do
RealBokeh_3MP (2932/2932 pares, verificado por source_aif). Ou seja: o split
TEST do mesmo dataset e o unico conjunto que (a) tem o MESMO protocolo de
aquisicao (bracket de abertura real, AIF = abertura fechada, alvo = f/2.0) e
(b) NUNCA foi visto por nenhum dos nossos treinos.

REGRA DE SELECAO: 1 par por CENA (evita pseudo-replicacao: os 5 niveis de uma
mesma cena sao a mesma foto), escolhendo o nivel de bokeh MAIS FORTE, medido
pela menor variancia do Laplaciano do alvo. Aplicado igual para todo modelo.
"""
import os, io, re, collections, numpy as np, cv2
from huggingface_hub import hf_hub_download
import pyarrow as pa, pyarrow.parquet as pq
from PIL import Image

tok=os.environ["HF_TOKEN"]
REPO_IN="akcit-pixel/RealBokeh"
REPO_OUT="juliadollis/bokeh-bench-realbokeh-test"
SHARDS=[f"data/test-0000{i}-of-00006.parquet" for i in range(6)]

def lv_de_bytes(b, lado=512):
    im=Image.open(io.BytesIO(b)).convert("RGB")
    w,h=im.size; s=lado/max(w,h)
    im=im.resize((max(1,int(w*s)),max(1,int(h*s))), Image.LANCZOS)
    g=cv2.cvtColor(np.array(im), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(g,cv2.CV_64F).var())

melhores={}   # cena -> dict
for sh in SHARDS:
    p=hf_hub_download(REPO_IN, sh, repo_type="dataset", token=tok)
    t=pq.read_table(p, columns=["image_blur","image_focus","file_name_base"])
    n=t.num_rows; print(f"[{sh}] {n} linhas", flush=True)
    col_b=t.column("image_blur").to_pylist()
    col_f=t.column("image_focus").to_pylist()
    col_n=t.column("file_name_base").to_pylist()
    for i in range(n):
        nome=str(col_n[i])
        m=re.match(r"^(.*_test_f_\d+)_level_(\d+)_aligned$", nome)
        cena = m.group(1) if m else nome
        nivel = int(m.group(2)) if m else 0
        lvb=lv_de_bytes(col_b[i]["bytes"])
        if cena not in melhores or lvb < melhores[cena]["lv_alvo"]:
            melhores[cena]={"cena":cena,"nivel":nivel,"file_name_base":nome,
                            "image_blur":col_b[i]["bytes"],"image_focus":col_f[i]["bytes"],
                            "lv_alvo":lvb,"lv_aif":lv_de_bytes(col_f[i]["bytes"])}
    del t, col_b, col_f
print("cenas distintas:", len(melhores), flush=True)

linhas=sorted(melhores.values(), key=lambda r: r["cena"])
print("razao LV(alvo)/LV(aif): min=%.3f mediana=%.3f max=%.3f" % tuple(
    np.percentile([r["lv_alvo"]/max(r["lv_aif"],1e-6) for r in linhas],[0,50,100])), flush=True)

os.makedirs("/workspace/b/bench_out", exist_ok=True)
schema=pa.schema([
    pa.field("file_name_base", pa.string()),
    pa.field("image_focus", pa.binary()),
    pa.field("image_blur", pa.binary()),
    pa.field("cena", pa.string()),
    pa.field("nivel_bokeh", pa.int32()),
    pa.field("lv_aif", pa.float32()),
    pa.field("lv_alvo", pa.float32()),
]).with_metadata({b"huggingface": b"""{"info": {"features": {"file_name_base": {"dtype": "string", "_type": "Value"}, "image_focus": {"_type": "Image"}, "image_blur": {"_type": "Image"}, "cena": {"dtype": "string", "_type": "Value"}, "nivel_bokeh": {"dtype": "int32", "_type": "Value"}, "lv_aif": {"dtype": "float32", "_type": "Value"}, "lv_alvo": {"dtype": "float32", "_type": "Value"}}}}"""})

from huggingface_hub import HfApi
api=HfApi(token=tok)
api.create_repo(REPO_OUT, repo_type="dataset", private=True, exist_ok=True)
POR=40
nsh=(len(linhas)+POR-1)//POR
for si in range(nsh):
    bloco=linhas[si*POR:(si+1)*POR]
    tb=pa.Table.from_pydict({
        "file_name_base":[r["file_name_base"] for r in bloco],
        "image_focus":[r["image_focus"] for r in bloco],
        "image_blur":[r["image_blur"] for r in bloco],
        "cena":[r["cena"] for r in bloco],
        "nivel_bokeh":[r["nivel"] for r in bloco],
        "lv_aif":[r["lv_aif"] for r in bloco],
        "lv_alvo":[r["lv_alvo"] for r in bloco],
    }, schema=schema)
    nome=f"data/validation-{si:05d}-of-{nsh:05d}.parquet"
    loc=f"/workspace/b/bench_out/{os.path.basename(nome)}"
    pq.write_table(tb, loc)
    api.upload_file(path_or_fileobj=loc, path_in_repo=nome, repo_id=REPO_OUT, repo_type="dataset")
    print("subiu", nome, len(bloco), flush=True)
api.upload_file(path_or_fileobj=io.BytesIO(b"""---
license: cc-by-nc-sa-4.0
---
# Benchmark de bokeh: RealBokeh_3MP split TEST (1 par por cena)

Substituto do **LF-Bokeh** do paper GenRefocus (arXiv 2512.16923), que nao e publico.

Origem: `akcit-pixel/RealBokeh`, split `test` (= `timseizinger/RealBokeh_3MP` test,
dataset *Bokehlicious*, ICCV 2025), pares alinhados de bracket de abertura real.

- `image_focus`: abertura fechada (all-in-focus), entrada do BokehNet
- `image_blur` : alvo de bokeh real, o nivel MAIS FORTE de cada cena
- 1 linha por cena (evita pseudo-replicacao entre niveis da mesma foto)

**Nao contaminado**: a rota c de treino (`AKCITPixel3/CMiQdveBBzNii`) vem 100%
do split TRAIN do mesmo dataset (2932/2932 verificado por `source_aif`).
Nenhuma cena do TEST entrou em nenhum treino nosso.

PRIVADO: contem GT de dataset de terceiros com licenca nao comercial.
"""), path_in_repo="README.md", repo_id=REPO_OUT, repo_type="dataset")
print("PRONTO:", REPO_OUT, len(linhas), "linhas em", nsh, "shards", flush=True)
