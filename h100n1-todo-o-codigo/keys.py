import zipfile, pickletools, io, re
p = "/host/retreinar-deblur/outputs/deblur_docker_4gpu/deblur/checkpoints/best.pt"
z = zipfile.ZipFile(p)
name = [n for n in z.namelist() if n.endswith("data.pkl")][0]
raw = z.read(name)
out = io.StringIO()
try:
    pickletools.dis(raw, out)
except Exception as e:
    out.write(f"\n(dis parou: {e})\n")
txt = out.getvalue()
strs = re.findall(r"SHORT_BINUNICODE '([^']*)'|BINUNICODE '([^']*)'", txt)
flat = [a or b for a, b in strs]
seen, uniq = set(), []
for s in flat:
    if s not in seen:
        seen.add(s); uniq.append(s)
print("total strings unicas:", len(uniq))
print("primeiras 60:", uniq[:60])
