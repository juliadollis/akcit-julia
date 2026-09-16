#!/usr/bin/env python3
"""(1) O EBB! esta dentro do treino? Checa as 3 rotas pela PROCEDENCIA real.
A rota a declara usar EBB! como fonte de AIF. A pergunta que decide tudo:
qual SUBCONJUNTO do EBB!? Se for o train, o Val294 pode continuar limpo.
"""
import os, re, collections
import pyarrow.parquet as pq, pyarrow as pa
from huggingface_hub import HfApi

TOK = os.environ.get("HF_TOKEN"); api = HfApi()
ROTAS = {"a (sintetica, EBB!+GenPhoto)": "AKCITPixel3/AfONERuvNmglv",
         "b (Flickr/ITW + EXIF)":        "AKCITPixel3/BKXcuVXCmeRvN",
         "c (RealBokeh_3MP/LFDOF)":      "AKCITPixel3/CMiQdveBBzNii"}
CAND = ["source_path", "source_aif", "source_bokeh", "stem"]

for nome, repo in ROTAS.items():
    print(""); print("=" * 78); print("ROTA " + nome + "  ->  " + repo); print("=" * 78)
    try:
        arqs = [f for f in api.list_repo_files(repo, repo_type="dataset", token=TOK)
                if f.endswith(".parquet")]
    except Exception as e:
        print("  inacessivel:", str(e)[:150]); continue
    print("  parquets:", len(arqs))
    esq = pq.read_schema("hf://datasets/" + repo + "/" + arqs[0])
    cols = [c for c in CAND if c in esq.names]
    print("  colunas de procedencia disponiveis:", cols)
    if not cols:
        print("  >>> SEM coluna de procedencia: nao da para provar origem por caminho")
        print("      colunas existentes:", esq.names); continue
    tabs = []
    for f in arqs:
        tabs.append(pq.read_table("hf://datasets/" + repo + "/" + f, columns=cols))
    T = pa.concat_tables(tabs)
    print("  linhas:", T.num_rows)
    todos = []
    for c in cols:
        if c == "stem": continue
        todos += [x or "" for x in T[c].to_pylist()]
    if not todos:
        todos = [x or "" for x in T["stem"].to_pylist()]
    print("  caminhos analisados:", len(todos), "| vazios:", sum(1 for p in todos if not p))
    # de que dataset vem cada caminho
    def fonte(p):
        pl = p.lower()
        for chave in ("ebb", "everything", "bokeh_free", "generative_photo", "genphoto",
                      "realbokeh", "lfdof", "flickr", "itw", "dpdd", "pointlight"):
            if chave in pl: return chave
        return "outro"
    print("  fontes detectadas:", collections.Counter(fonte(p) for p in todos).most_common())
    # split dentro da fonte
    def split(p):
        pl = p.lower()
        for s in ("train", "test", "val"):
            if "/" + s in pl or pl.startswith(s): return s
        return "?"
    print("  splits detectados:", collections.Counter(split(p) for p in todos).most_common())
    ebb = [p for p in todos if "ebb" in p.lower() or "everything" in p.lower()]
    print("  >>> caminhos que mencionam EBB!: " + str(len(ebb)))
    for p in ebb[:12]: print("       ", p)
    prefixos = collections.Counter("/".join(p.split("/")[:4]) for p in todos if p)
    print("  prefixos mais comuns:")
    for k, v in prefixos.most_common(8): print("       %7d  %s" % (v, k))
