#!/bin/bash
# Multi-GPU CLM (master branch, teammate's threading-based implementation).
# Usage:
#   sbatch multi_gpu.sh a40      # 2x A40
#   sbatch multi_gpu.sh a100     # 2x A100
#SBATCH --output=./slurm/slurm%j.out
#SBATCH --error=./slurm/slurm%j.err
#SBATCH --nodes=1
#SBATCH --mem=119G
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --time=1-00:00:00

set -e

# ---- Parse arguments ----
GPU_CHOICE="${1:-a40}"
case "${GPU_CHOICE}" in
    a40)         GRES="gpu:a40:2" ;;          # jaguar06
    nvidia_a40)  GRES="gpu:nvidia_a40:2" ;;   # jaguar01
    a100)        GRES="gpu:a100:2" ;;         # cheetah04
    nvidia_a100) GRES="gpu:nvidia_a100-pcie-40gb:2" ;;  # cheetah01
    *)    echo "Usage: sbatch multi_gpu.sh {a40|nvidia_a40|a100|nvidia_a100}"; exit 1 ;;
esac
GPU_TAG=$(echo "${GPU_CHOICE}" | sed 's/nvidia_//')

# Re-submit with correct --gres if run directly
if [[ -z "$SLURM_JOB_ID" ]]; then
    echo "Submitting: sbatch --gres=${GRES} $0 $@"
    sbatch --gres="${GRES}" "$0" "$@"
    exit 0
fi

module purge
module load miniforge
module load cuda/12.8.1

cd /u/xna8aw/6501_GPU_architectures/multi_gpu_3dgs_master
conda activate clm_gs

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Auto-detect number of GPUs allocated by SLURM
NUM_GPUS=$(nvidia-smi -L | wc -l)
GPU_IDS=$(seq -s, 0 $((NUM_GPUS - 1)))
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

export NCCL_TIMEOUT=1800
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

echo "=== Multi-GPU CLM (master, teammate impl) ==="
echo "GPU: ${GPU_CHOICE} x${NUM_GPUS}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "============================================="

torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --master_addr=localhost \
    --master_port=29500 \
    train_multi.py \
    -s 'data/rubble-colmap' \
    --clm_offload \
    --enable_distributed \
    --bsz 8 \
    --eval \
    -m "output/rubble_${GPU_TAG}_multi_clm"
