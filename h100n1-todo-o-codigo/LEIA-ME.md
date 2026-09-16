# Todo o código da dgx-H100-01

Espelho de **todo o código** que existia em `/raid/user_juliadollis/julia_docker/`
na máquina, coletado para que nada se perca. Inclui projetos além do
depth-riemannian.

## O que entrou

Todo arquivo `.py`, `.sh`, `.md`, `.slurm`, `.yaml`, `.json` de configuração,
`Dockerfile` e `requirements` de cada projeto.

| pasta | projeto |
|---|---|
| `depth-riemannian/` | o reteste da curvatura (a cópia que roda nos containers) |
| `ablacao-wallisson/` | a ablação adaptada pelo Wallisson |
| `genrefocus_deblurnet_paper/` | o GenRefocus: bokeh e deblur |
| `genrefocus_deblurnet/` | a versão anterior do mesmo projeto |
| `bokehnet-regen/` | a regeneração do BokehNet (só o código; os 27.354 JSON de saída ficaram de fora) |
| `retreinar-deblur/` | o retreino do DeblurNet |
| `runs_*`, `avaliacoes/`, `filas_eq4/` | as métricas e configurações de cada campanha |
| raiz | os scripts de orquestração soltos: filas, guardas de memória, subidas ao Hub, diagnósticos |

## O que NÃO entrou, e onde está

| o que | por quê | onde está |
|---|---|---|
| `hf-cache*` (357 GB) | são downloads do próprio Hub, reproduzíveis | Hugging Face |
| `data/` (86 GB) | datasets públicos | DaRUS (Spring) e as fontes originais |
| `pylibs_*` | bibliotecas de terceiros copiadas | PyPI |
| pesos (`.pt`, `.safetensors`) | 68 GB, e o GitHub limita 100 MB por arquivo | Hugging Face |
| saídas geradas (27.354 JSON do `bokehnet-regen`) | dado, não código | Hugging Face |
| `.env` | segredo | só na máquina |

## Aviso de segurança

Em **14 arquivos** desta coleta havia um **token do Hugging Face hardcoded**,
em `genrefocus_deblurnet/scripts/*.slurm` e `*.sh`. Aqui eles aparecem como
`TOKEN_REMOVIDO_USE_ENV`.

**Os arquivos originais na máquina continuam com o token em texto puro.** O token
deve ser rotacionado e os scripts passados a ler do `.env`, como o resto do
projeto já faz:

```bash
set -a; . .env; set +a
```
