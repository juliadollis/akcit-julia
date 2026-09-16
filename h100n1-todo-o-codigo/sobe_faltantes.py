#!/usr/bin/env python3
"""Sobe para o Hub os 3 checkpoints que existem no /raid e nao estavam publicados.

A conferencia foi feita por sha256: cada arquivo local foi comparado com o
sha256 de todos os arquivos LFS de todos os 1254 repos de juliadollis.
Os outros 18 arquivos da lista ja tinham copia byte a byte identica no Hub.
"""
import os, sys
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
H = "/host"

README_DEBLUR = """# deblurnet-ft

Checkpoint de treino completo (estado do otimizador incluso) do DeblurNet,
adaptador LoRA sobre `black-forest-labs/FLUX.1-dev`.

## De onde veio

- Maquina: `dgx-H100-01`
- Caminho no cluster: `/raid/user_juliadollis/julia_docker/retreinar-deblur/outputs/deblur_docker_4gpu/deblur/checkpoints/best.pt`
- Run: `deblur-docker-4gpu-h100n1` (wandb project `genrefocus-deblurnet-docker`)
- `config_hash`: `855f21ac569a0f70`, `seed`: 42, `git_commit`: `unknown` (o treino nao registrou o commit)

## Arquivos

| arquivo | bytes | o que e |
|---|---|---|
| `checkpoints/best.pt` | 5564143647 | checkpoint de treino, `torch.save`, com `lora_state`, `optimizer`, `scheduler`, `train_state`, `metadata`, `metrics_snapshot` |
| `checkpoints/latest.json` | - | ponteiro que o treino deixou para o ultimo checkpoint gravado (`step_60000.pt`, que nao esta aqui) |
| `deblur.json` | - | sidecar do export `deblur.safetensors`, copiado do disco sem alteracao |
| `deblur_best_step15500.json` | - | sidecar do export `deblur_best_step15500.safetensors`, copiado do disco sem alteracao |
| `effective_config.yaml` | - | config efetiva da run, copiada do disco sem alteracao |
| `run_metadata.json` | - | metadados da run, copiados do disco sem alteracao |

O sidecar `deblur_best_step15500.json` registra `global_step: 15500` e
`checkpoint: .../checkpoints/best.pt`, ou seja, o export `deblur_best_step15500.safetensors`
foi gerado a partir deste `best.pt`.

## Os pesos exportados desta mesma run ja estavam no Hub

Nao foram reenviados porque sao byte a byte identicos ao que ja existe:

- `deblur.safetensors` (1854533784 bytes) = `juliadollis/genrefocus-deblurnet` :: `deblurNet.safetensors`
- `deblur_best_step15500.safetensors` (1854533784 bytes) = `juliadollis/genrefocus-deblurnet-docker-4gpu` :: `deblur_best.safetensors`
  e `juliadollis/genrefocus-deblurnet-15k` :: `deblurNet.safetensors`

## Como carregar

```python
import torch
from huggingface_hub import hf_hub_download

p = hf_hub_download("juliadollis/deblurnet-ft", "checkpoints/best.pt")
ck = torch.load(p, map_location="cpu", weights_only=False)
ck.keys()                 # lora_state, optimizer, scheduler, train_state, metadata, metrics_snapshot
ck["train_state"]         # global_step, best_loss
ck["metadata"]            # mesma coisa que deblur.json
lora = ck["lora_state"]   # chaves no formato <modulo>.lora_{A,B}.default.weight
```

Para inferencia, prefira os `.safetensors` ja exportados nos repos acima: eles estao
no formato que o pipeline de inferencia consome (`transformer.<modulo>.lora_{A,B}.weight`).

## Protocolo de treino

Os hiperparametros estao em `effective_config.yaml` e `deblur.json`, que foram copiados
do disco sem edicao. Nao ha metrica nem resultado neste card: o `metrics_snapshot` esta
dentro do `best.pt` e nao foi aberto. Nada aqui foi estimado ou inferido.
"""

README_GEOB = """# bokehnet-geo-B

LoRA de bokeh sobre `black-forest-labs/FLUX.1-dev`, run `bokehnet-geo-B`.

Os arquivos `bokeh.safetensors` e `bokeh_step*.safetensors` foram enviados pelo
proprio treino. O arquivo `smoke.safetensors` foi adicionado depois, num inventario
de checkpoints que existiam no cluster e nao estavam no Hub.

## smoke.safetensors

- Caminho no cluster: `/raid/user_juliadollis/julia_docker/genrefocus_deblurnet_paper/outputs/bokehnet-geo-B/smoke.safetensors`
- Tamanho: 928115688 bytes, 688 tensores, header safetensors integro
- mtime no disco: 2026-09-05 16:55, anterior ao `bokeh_step1000.safetensors` desta run
- Conteudo distinto de todos os outros arquivos deste repo (sha256 diferente)

O nome sugere um smoke test do caminho de export, mas o protocolo que o gerou
nao esta documentado em lugar nenhum que tenha sido encontrado. Nao trate como
checkpoint de uma etapa de treino conhecida.

## Como carregar

```python
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

sd = load_file(hf_hub_download("juliadollis/bokehnet-geo-B", "smoke.safetensors"))
```

A config efetiva da run ficou no cluster e nao foi copiada para ca, porque este repo
e publico. Ela esta em
`/raid/user_juliadollis/julia_docker/genrefocus_deblurnet_paper/outputs/bokehnet-geo-B/effective_config.yaml`.
Sem metricas neste card: nenhuma foi medida nem consultada.
"""

README_NOFILTER = """# genrefocus-bokehnet-fase2-nofilter

LoRA de bokeh sobre `black-forest-labs/FLUX.1-dev`, run `bokehnet-fase2-nofilter`.

Os arquivos `bokeh_step5000/15000/30000/34000.safetensors` foram enviados pelo proprio
treino. O `bokeh_step35000.safetensors` foi adicionado depois, num inventario de
checkpoints que existiam no cluster e nao estavam no Hub.

## bokeh_step35000.safetensors

- Caminho no cluster: `/raid/user_juliadollis/julia_docker/genrefocus_deblurnet_paper/outputs/bokehnet_fase2_nofilter/bokeh/.upload_bokeh_step35000.safetensors`
- Tamanho: 928115688 bytes, 688 tensores, header safetensors integro
- mtime no disco: 2026-08-31 10:30

No disco o arquivo tem o prefixo `.upload_`, que e o nome de staging que o treino usa
antes de enviar. Como o repo tinha `bokeh_step34000` mas nao `bokeh_step35000`, o mais
provavel e que o envio deste passo tenha sido interrompido. O conteudo foi conferido
(header safetensors fecha exatamente com o tamanho do arquivo) e o sha256 nao bate com
nenhum outro arquivo dos repos de juliadollis, entao nao e duplicata.

## Como carregar

```python
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

sd = load_file(hf_hub_download("juliadollis/genrefocus-bokehnet-fase2-nofilter",
                               "bokeh_step35000.safetensors"))
```

A config efetiva da run ficou no cluster e nao foi copiada para ca, porque este repo
e publico. Ela esta em
`/raid/user_juliadollis/julia_docker/genrefocus_deblurnet_paper/outputs/bokehnet_fase2_nofilter/effective_config.yaml`.
Sem metricas neste card: nenhuma foi medida nem consultada.
"""

def envia(local, repo, dest, repo_type="model"):
    n = os.path.getsize(local)
    print(f"[envio] {local} ({n} bytes) -> {repo} :: {dest}", flush=True)
    api.upload_file(path_or_fileobj=local, path_in_repo=dest,
                    repo_id=repo, repo_type=repo_type)
    print(f"[ok   ] {repo} :: {dest}", flush=True)

def texto(conteudo, repo, dest):
    print(f"[envio] (texto) -> {repo} :: {dest}", flush=True)
    api.upload_file(path_or_fileobj=conteudo.encode("utf-8"), path_in_repo=dest,
                    repo_id=repo, repo_type="model")
    print(f"[ok   ] {repo} :: {dest}", flush=True)

# ---------- 1. deblur: repo novo, privado ----------
R1 = "juliadollis/deblurnet-ft"
api.create_repo(R1, repo_type="model", private=True, exist_ok=True)
print(f"[repo ] {R1} criado/confirmado (privado)", flush=True)
D = f"{H}/retreinar-deblur/outputs/deblur_docker_4gpu"
texto(README_DEBLUR, R1, "README.md")
for src, dest in [
    (f"{D}/deblur/checkpoints/latest.json", "checkpoints/latest.json"),
    (f"{D}/deblur/deblur.json",             "deblur.json"),
    (f"{D}/deblur/deblur_best_step15500.json", "deblur_best_step15500.json"),
    (f"{D}/effective_config.yaml",          "effective_config.yaml"),
    (f"{D}/run_metadata.json",              "run_metadata.json"),
]:
    envia(src, R1, dest)
envia(f"{D}/deblur/checkpoints/best.pt", R1, "checkpoints/best.pt")

# ---------- 2. smoke da geo-B, no repo que a familia ja usa ----------
R2 = "juliadollis/bokehnet-geo-B"
G = f"{H}/genrefocus_deblurnet_paper/outputs/bokehnet-geo-B"
envia(f"{G}/smoke.safetensors", R2, "smoke.safetensors")
texto(README_GEOB, R2, "README.md")

# ---------- 3. step35000 do nofilter, no repo que a familia ja usa ----------
R3 = "juliadollis/genrefocus-bokehnet-fase2-nofilter"
N = f"{H}/genrefocus_deblurnet_paper/outputs/bokehnet_fase2_nofilter"
envia(f"{N}/bokeh/.upload_bokeh_step35000.safetensors", R3, "bokeh_step35000.safetensors")
texto(README_NOFILTER, R3, "README.md")

print("FIM_UPLOAD", flush=True)
