import os, io, re, collections, numpy as np, cv2
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
from PIL import Image
tok=os.environ.get("HF_TOKEN")
p=hf_hub_download("akcit-pixel/RealBokeh","data/test-00000-of-00006.parquet",repo_type="dataset",token=tok)
t=pq.read_table(p).to_pydict()
print("cols:", list(t.keys()))
def pil(d): return Image.open(io.BytesIO(d["bytes"])).convert("RGB")
def lv(img):
    g=cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY); return float(cv2.Laplacian(g,cv2.CV_64F).var())
import hashlib
for i in range(0,10):
    n=t["file_name_base"][i]
    f=pil(t["image_focus"][i]); b=pil(t["image_blur"][i]); pd_=t["image_pre_deblur"][i]
    hf=hashlib.md5(t["image_focus"][i]["bytes"]).hexdigest()[:10]
    hb=hashlib.md5(t["image_blur"][i]["bytes"]).hexdigest()[:10]
    hp=hashlib.md5(pd_["bytes"]).hexdigest()[:10] if pd_ and pd_.get("bytes") else "NULO"
    print(f"{n:60s} size={f.size} LVfocus={lv(f):8.1f} LVblur={lv(b):8.1f}  md5f={hf} md5b={hb} md5pre={hp}")
