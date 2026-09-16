# Infraestrutura: a máquina, o docker e as regras

## 1. A máquina

**dgx-H100-01**, 8 GPUs H100 de 80 GB. **Docker, sem SLURM**, o que muda tudo em
relação aos nós `h100n2`/`h100n3`, que usam SLURM e Singularity.

Pasta de trabalho: `/raid/user_juliadollis/julia_docker/`.

## 2. As regras, aprendidas quebrando

| regra | por quê |
|---|---|
| **`--user $(id -u):$(id -g)` em todo `docker run`** | sem isso os arquivos nascem do root. Há 54 GB de cache `root:root` de uma rodada antiga que a usuária não consegue apagar |
| **container detached** (`setsid nohup`) | sobrevive a queda de ssh e de VPN, que já aconteceu várias vezes |
| **nunca `docker system prune`, `rmi`, nem parar container alheio** | há containers de outras 4 pessoas no mesmo host |
| **não ocupar as 8 GPUs** | não existe fila protegendo ninguém; ocupar tudo é tomar a máquina dos outros |
| **`--shm-size 32g --ipc host`** | sem isso o DataLoader trava |
| **`/raid` não é durável** | ver a regra do Hub, no README |

## 3. A armadilha da memória

Um treino a 512 px com batch 8 ocupa **~53 GB**, quase a placa inteira. Checar
**utilização** não serve: um serving fica em 0% entre requisições com dezenas de
GB reservados.

E a lição mais cara: **guarda de memória protege o seu job, não o do vizinho.**
Checar memória livre impede que o seu treino tome OOM, mas não impede que ele
cause OOM em quem está ciclando itens na mesma placa. Foi assim que dois itens de
uma fila de avaliação morreram.

Por isso as filas aqui exigem folga além do que usam e só entram em placa
realmente ociosa.

## 4. O padrão de fila

`fila_retreino.sh` e `fila_confirma.sh` usam **reivindicação atômica por
`mkdir`**: o diretório ou é criado ou falha, sem corrida. Várias GPUs consomem a
mesma fila e nenhum item é feito duas vezes, mesmo se dois processos lerem a fila
ao mesmo tempo.

Cada fila também:

- pula item que já tem `test_metrics.json`, então é retomável
- devolve o item à fila se a rodada falhar
- espera a GPU vagar em vez de morrer
- **sobe para o Hub a cada seed que fecha**, não no fim da faixa

## 5. Do zero

```bash
# dados: 23 GB do DaRUS, e o split determinístico
bash orquestracao-cluster/baixa_spring.sh
bash orquestracao-cluster/prep_passo4.sh

# a imagem: o Dockerfile está em depth-riemannian/
docker build -t riemann-depthpro:latest depth-riemannian/

# validar antes de gastar GPU (CPU, 2 min)
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD/depth-riemannian/scripts":/workspace/scripts:ro \
  -v "$PWD/depth-riemannian/riemann":/workspace/riemann:ro \
  -w /workspace riemann-depthpro:latest \
  python scripts/test_geometry_metrica.py
```

O `Dockerfile` roda um teste de geometria no build e **falha de propósito** se a
matemática da curvatura estiver quebrada, antes de gastar GPU.

## 6. O split, e por que ele é reprodutível

`prepare_spring.py` com `--particao`, frações 0,50/0,15/0,35, `--particao-seed
42`, `--camera left --passo 4 --max-depth 100`. As sequências são **disjuntas**
entre as partições.

Depois da perda de 2026-09-10 o split foi refeito do zero e saiu **idêntico**:
593/184/485 quadros, 18/6/13 sequências, e as 13 do teste são exatamente as
mesmas. É a prova de que o preparo é determinístico.

## 7. Verificação de integridade

O código científico foi conferido por **sha256** entre o cluster, a máquina local
e o `origin/main` do repositório original: **14 de 14 arquivos idênticos**. E os
`mtime` do `riemann/` são anteriores à primeira rodada, o que prova que o código
ficou congelado durante a campanha inteira.

A única alteração em arquivo versionado é o `train_single.py`, e é orquestração
de seeds (`--seed-inicio`, pular seed concluída, resumo lido do disco), sem
efeito em número nenhum.
