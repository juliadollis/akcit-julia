import os
import requests
import huggingface_hub as H
from huggingface_hub import HfApi

tok = os.environ["HF_TOKEN"]
print("huggingface_hub", H.__version__)
print("metodos com 'org':", [x for x in dir(HfApi) if "org" in x.lower()])

hdr = {"authorization": "Bearer " + tok}
r = requests.post("https://huggingface.co/api/organizations",
                  headers=hdr, json={"name": "akcit-pixel4"}, timeout=30)
print("POST /api/organizations ->", r.status_code, r.text[:300])

# quanto cada org ja usa, para achar uma com folga
api = HfApi(token=tok)
w = api.whoami()
for o in w.get("orgs", []):
    nome = o["name"]
    try:
        rr = requests.get("https://huggingface.co/api/organizations/%s/overview" % nome,
                          headers=hdr, timeout=30)
        d = rr.json() if rr.ok else {}
        print("  %-22s papel=%-8s storage=%s" % (nome, o.get("roleInOrg"),
              d.get("storageUsed", d.get("usedStorage", "?"))))
    except Exception as e:
        print("  %-22s erro %s" % (nome, e))
