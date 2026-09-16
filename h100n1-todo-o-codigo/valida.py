import json, struct, zipfile, os
A = "/host/genrefocus_deblurnet_paper/outputs/bokehnet-geo-B/smoke.safetensors"
B = "/host/genrefocus_deblurnet_paper/outputs/bokehnet_fase2_nofilter/bokeh/.upload_bokeh_step35000.safetensors"
C = "/host/retreinar-deblur/outputs/deblur_docker_4gpu/deblur/checkpoints/best.pt"

def st(p):
    sz = os.path.getsize(p)
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    keys = [k for k in hdr if k != "__metadata__"]
    # maior offset declarado precisa caber no arquivo
    end = max(v["data_offsets"][1] for k, v in hdr.items() if k != "__metadata__")
    print(f"{p}\n  bytes={sz} header={n} tensores={len(keys)} fim_declarado={8+n+end} ok={8+n+end==sz}")
    print(f"  metadata={hdr.get('__metadata__')}")
    print(f"  exemplo_chaves={keys[:3]}")

def pt(p):
    sz = os.path.getsize(p)
    z = zipfile.ZipFile(p)
    names = z.namelist()
    print(f"{p}\n  bytes={sz} zip_ok=True entradas={len(names)}")
    print(f"  raiz={sorted({n.split('/')[1] for n in names if len(n.split('/'))>1})[:20]}")
    bad = z.testzip()
    print(f"  testzip(primeiro corrompido)={bad}")

for p, fn in ((A, st), (B, st), (C, pt)):
    try:
        fn(p)
    except Exception as e:
        print(f"{p}\n  ERRO: {type(e).__name__}: {e}")
    print()
