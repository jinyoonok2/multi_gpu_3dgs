#!/bin/bash
# Multi-GPU CLM (M1 camera parallelism): auto-detects allocated GPUs.
# Usage: sbatch --gres=gpu:TYPE:COUNT run_mgclm.sh
#   Examples:
#     sbatch --gres=gpu:a40:2 run_mgclm.sh
#     sbatch --gres=gpu:a100:2 run_mgclm.sh
#     sbatch --gres=gpu:nvidia_h100_nvl:2 run_mgclm.sh
#
#SBATCH --output=./slurm/slurm%j.out
#SBATCH --error=./slurm/slurm%j.err
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --time=1-00:00:00

set -e

module purge
module load miniforge
module load cuda/12.8.1

cd /u/xna8aw/6501_GPU_architectures/multi_gpu_3dgs
conda activate clm_gs

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_TIMEOUT=1800
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Auto-detect number of GPUs allocated by SLURM
NUM_GPUS=$(nvidia-smi -L | wc -l)
GPU_IDS=$(seq -s, 0 $((NUM_GPUS - 1)))
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

# Derive a tag from GPU name for output directory
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_' | tr '[:upper:]' '[:lower:]')

echo "=== Multi-GPU CLM (M1) ==="
echo "GPUs detected: ${NUM_GPUS}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=========================="

torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --master_addr=localhost \
    --master_port=29500 \
    train_multi_gpu_clm.py \
    -s 'data/rubble-colmap' \
    --bsz 8 \
    --eval \
    -m "output/rubble_${GPU_NAME}_x${NUM_GPUS}_mgclm"
