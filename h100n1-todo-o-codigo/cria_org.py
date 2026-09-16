import os
import requests

tok = os.environ["HF_TOKEN"]
hdr = {"authorization": "Bearer " + tok}

payload = {
    "name": "akcit-pixel4",
    "fullname": "AKCIT Pixel 4",
    "type": "company",
}
r = requests.post("https://huggingface.co/api/organizations",
                  headers=hdr, json=payload, timeout=60)
print("POST /api/organizations ->", r.status_code)
print(r.text[:600])

if r.ok:
    rr = requests.get("https://huggingface.co/api/whoami-v2", headers=hdr, timeout=30)
    orgs = [o["name"] for o in rr.json().get("orgs", [])]
    print("akcit-pixel4 aparece nas orgs?", "akcit-pixel4" in orgs)
