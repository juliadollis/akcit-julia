#!/bin/bash
#SBATCH --job-name=teste-h100n1
#SBATCH --partition=h100n1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:02:00
#SBATCH --output=/raid/user_juliadollis/julia_docker/teste-h100n1-%j.out
#SBATCH --error=/raid/user_juliadollis/julia_docker/teste-h100n1-%j.err

# =============================================================================
# JOB DE TESTE — SO para descobrir se a dgx-H100-01 aceita jobs pelo SLURM.
# Nao treina nada, nao escreve em lugar nenhum alem do proprio log.
#
# O --deadline na submissao faz o SLURM CANCELAR sozinho se o job nao iniciar
# no prazo. Isso e proposital: o no esta DOWN+NOT_RESPONDING, entao o job
# ficaria PENDING para sempre, e a regra do projeto e nunca cancelar job na mao.
# =============================================================================
echo "=========================================="
echo "SE VOCE ESTA LENDO ISTO, O SLURM RODOU NA H100-01."
echo "host:      $(hostname)"
echo "usuario:   $(whoami)"
echo "job:       ${SLURM_JOB_ID}  particao=${SLURM_JOB_PARTITION}"
echo "GPUs SLURM: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<nao setado>}"
echo "=========================================="
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader || echo "(nvidia-smi falhou)"
echo "FIM DO TESTE"
