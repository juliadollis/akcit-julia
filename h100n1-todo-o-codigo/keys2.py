import zipfile, pickletools, io, re
p = "/host/retreinar-deblur/outputs/deblur_docker_4gpu/deblur/checkpoints/best.pt"
z = zipfile.ZipFile(p)
raw = z.read([n for n in z.namelist() if n.endswith("data.pkl")][0])
out = io.StringIO(); pickletools.dis(raw, out); txt = out.getvalue()
flat = [a or b for a, b in re.findall(r"SHORT_BINUNICODE '([^']*)'|BINUNICODE '([^']*)'", txt)]
# chaves de topo: strings que nao parecem nome de tensor nem numero
cand = [s for s in dict.fromkeys(flat)
        if not re.search(r"lora_[AB]|transformer_blocks|^\d+$|^x_embedder|^cpu$|^storage$|^torch|^collections|^__|weight$|^exp_avg", s)]
print(len(cand)); print(cand[:80])
