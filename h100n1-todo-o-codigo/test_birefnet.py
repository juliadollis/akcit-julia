import os, io, numpy as np, torch
from PIL import Image
tok=os.environ.get("HF_TOKEN")
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
try:
    from transformers import AutoModelForImageSegmentation
    m=AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True)
    print("OK carregou BiRefNet:", type(m).__name__)
    m.eval()
    from torchvision import transforms
    tf=transforms.Compose([transforms.Resize((1024,1024)), transforms.ToTensor(),
         transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    img=Image.new("RGB",(800,600),(120,120,120))
    with torch.no_grad():
        p=m(tf(img).unsqueeze(0))[-1].sigmoid()
    print("saida:", p.shape, float(p.min()), float(p.max()))
except Exception as e:
    import traceback; traceback.print_exc(); print("FALHOU:", type(e).__name__, e)
