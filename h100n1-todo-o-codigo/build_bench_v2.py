"""Benchmark de bokeh v2: RealBokeh_3MP split TEST, agora COM os metadados.

O QUE MUDA EM RELACAO A v1
1. Agrupamento pelo ID REAL da cena. A v1 usava regex exigindo sufixo
   `_aligned`, e as 15 linhas com sufixo `_misaligned` / `_shift_X.Xpx` viraram
   pseudo-cenas: 233 linhas para 220 cenas de verdade.
2. Descarta `misaligned` (desalinhamento NAO quantificado, veneno para metrica
   de pixel). Mantem `shift_X.Xpx`, que e quantificado e pequeno (2-5 px em
   2000), mas grava a coluna `alinhamento` para quem quiser filtrar.
3. Descarta par cujo alvo nao e mais borrado que a AIF.
4. ANEXA OS METADADOS DO RealBokeh (test/metadata/<id>.json), que a conversao
   em parquet do time tinha perdido:
     - `focus_plane_distance` (m): o PLANO DE FOCO ANOTADO. E o que a Eq. 4 do
       paper apenas ESTIMA via BiRefNet + Depth Pro. Com ele da para validar a
       estimativa e, se quisermos, condicionar pelo valor verdadeiro.
     - `source_av`, `target_avs`, `focal_length`: permitem calcular o K pela
       Eq. 3 analiticamente, em vez de so achar por busca binaria.
     - `ISO`, `EV`, `focus_plane_uncertainty`.
5. `target_av`: a abertura do ALVO desta linha, VERIFICADA por conteudo de
   pixel (correlacao com o JPG bruto gt/<id>/<id>_f2.0.JPG), nao inferida.
"""
import os, io, re, json, collections, numpy as np, cv2
from huggingface_hub import hf_hub_download, HfApi
import pyarrow as pa, pyarrow.parquet as pq
from PIL import Image

tok=os.environ["HF_TOKEN"]
REPO_OUT="juliadollis/bokeh-bench-realbokeh-test-v2"

def mini(b, L=160):
    a=np.asarray(Image.open(io.BytesIO(b)).convert("L").resize((L,L), Image.LANCZOS),dtype=np.float32)
    a-=a.mean(); s=a.std(); return a/s if s>1e-6 else a
def lv(b, L=512):
    im=Image.open(io.BytesIO(b)).convert("RGB"); w,h=im.size; s=L/max(w,h)
    im=im.resize((max(1,int(w*s)),max(1,int(h*s))), Image.LANCZOS)
    return float(cv2.Laplacian(cv2.cvtColor(np.array(im),cv2.COLOR_RGB2GRAY),cv2.CV_64F).var())

print("lendo o parquet de origem...", flush=True)
cenas=collections.defaultdict(list)
for i in range(6):
    p=hf_hub_download("akcit-pixel/RealBokeh", f"data/test-0000{i}-of-00006.parquet", repo_type="dataset", token=tok)
    t=pq.read_table(p, columns=["image_blur","image_focus","file_name_base"])
    cb=t.column("image_blur").to_pylist(); cf=t.column("image_focus").to_pylist(); cn=t.column("file_name_base").to_pylist()
    for j in range(len(cn)):
        m=re.match(r"^.*_test_f_(\d+)_level_(\d+)_(.*)$", str(cn[j]))
        if not m: print("  [aviso] nome fora do padrao:", cn[j]); continue
        cenas[m.group(1)].append(dict(nivel=int(m.group(2)), alinh=m.group(3), blur=cb[j]["bytes"],
                                      focus=cf[j]["bytes"], nome=str(cn[j])))
    del t, cb, cf
print("cenas:", len(cenas), flush=True)

linhas=[]; desc=collections.Counter()
for cid in sorted(cenas, key=lambda x:int(x)):
    cand=[c for c in cenas[cid] if c["alinh"]!="misaligned"]
    if not cand: desc["so_misaligned"]+=1; continue
    for c in cand: c["lv"]=lv(c["blur"])
    esc=min(cand, key=lambda c:c["lv"])
    lva=lv(esc["focus"])
    if esc["lv"]>=lva: desc["alvo_nao_mais_borrado"]+=1; continue
    # metadados
    mp=hf_hub_download("timseizinger/RealBokeh_3MP", f"test/metadata/{cid}.json", repo_type="dataset", token=tok)
    md=json.load(open(mp))
    # verificacao por pixel: o alvo escolhido e mesmo o f/2.0 bruto?
    av=None; conf="nao_verificado"
    try:
        jp=hf_hub_download("timseizinger/RealBokeh_3MP", f"test/gt/{cid}/{cid}_f2.0.JPG", repo_type="dataset", token=tok)
        ref=mini(open(jp,"rb").read())
        cors=[float((ref*mini(c["blur"])).mean()) for c in cand]
        if cand[int(np.argmax(cors))] is esc: av=2.0; conf="verificado_pixel"
        else: conf="divergiu"
    except Exception: conf="sem_jpg_f2.0"
    if av is None:
        avs=sorted(md.get("target_avs") or []); ordem=sorted(cand,key=lambda c:c["lv"])
        idx=ordem.index(esc); av=float(avs[idx]) if idx<len(avs) else float("nan"); conf+="+inferido_lv"
    desc[conf]+=1
    sh=re.match(r"^shift_([0-9.]+)px$", esc["alinh"])
    linhas.append(dict(file_name_base=esc["nome"], image_focus=esc["focus"], image_blur=esc["blur"],
        cena_id=str(cid), nivel_bokeh=esc["nivel"], alinhamento=esc["alinh"],
        desloc_px=float(sh.group(1)) if sh else 0.0, lv_aif=lva, lv_alvo=esc["lv"],
        target_av=float(av), source_av=float(md.get("source_av") or float("nan")),
        target_avs=json.dumps(md.get("target_avs")), focal_length=float(md.get("focal_length") or float("nan")),
        focus_plane_distance=float(md.get("focus_plane_distance") or float("nan")),
        focus_plane_uncertainty=float(md.get("focus_plane_uncertainty") or float("nan")),
        iso=float(md.get("ISO") or float("nan")), ev=float(md.get("EV") or float("nan"))))
    if len(linhas)%25==0: print("  ...", len(linhas), flush=True)
print("linhas finais:", len(linhas), "| descartes/verificacao:", dict(desc), flush=True)
print("target_av distintos:", collections.Counter(r["target_av"] for r in linhas), flush=True)
print("alinhamento:", collections.Counter(r["alinhamento"] for r in linhas), flush=True)

campos=[("file_name_base",pa.string()),("image_focus",pa.binary()),("image_blur",pa.binary()),
        ("cena_id",pa.string()),("nivel_bokeh",pa.int32()),("alinhamento",pa.string()),("desloc_px",pa.float32()),
        ("lv_aif",pa.float32()),("lv_alvo",pa.float32()),("target_av",pa.float32()),("source_av",pa.float32()),
        ("target_avs",pa.string()),("focal_length",pa.float32()),("focus_plane_distance",pa.float32()),
        ("focus_plane_uncertainty",pa.float32()),("iso",pa.float32()),("ev",pa.float32())]
feat={"file_name_base":{"dtype":"string","_type":"Value"},"image_focus":{"_type":"Image"},"image_blur":{"_type":"Image"},
      "cena_id":{"dtype":"string","_type":"Value"},"nivel_bokeh":{"dtype":"int32","_type":"Value"},
      "alinhamento":{"dtype":"string","_type":"Value"},"target_avs":{"dtype":"string","_type":"Value"}}
for n,t_ in campos:
    if n not in feat: feat[n]={"dtype":"float32","_type":"Value"}
schema=pa.schema([pa.field(n,t_) for n,t_ in campos]).with_metadata(
    {b"huggingface": json.dumps({"info":{"features":feat}}).encode()})

api=HfApi(token=tok); api.create_repo(REPO_OUT, repo_type="dataset", private=True, exist_ok=True)
os.makedirs("/workspace/b/bench_v2", exist_ok=True)
POR=40; nsh=(len(linhas)+POR-1)//POR
for si in range(nsh):
    bloco=linhas[si*POR:(si+1)*POR]
    tb=pa.Table.from_pylist(bloco, schema=schema)
    nome=f"data/validation-{si:05d}-of-{nsh:05d}.parquet"
    loc=f"/workspace/b/bench_v2/{os.path.basename(nome)}"
    pq.write_table(tb, loc)
    api.upload_file(path_or_fileobj=loc, path_in_repo=nome, repo_id=REPO_OUT, repo_type="dataset")
    print("subiu", nome, len(bloco), flush=True)
print("PRONTO:", REPO_OUT, len(linhas), flush=True)
