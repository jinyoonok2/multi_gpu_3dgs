#!/bin/bash
# Multi-GPU CLM training on 2 GPUs.
# Usage:
#   sbatch run_mgclm.sh a40             # M1 on 2x A40  (jaguar06: gpu:a40, or jaguar01: gpu:nvidia_a40)
#   sbatch run_mgclm.sh a100            # M1 on 2x A100 (cheetah04: gpu:a100, or cheetah01: gpu:nvidia_a100-pcie-40gb)
#   sbatch run_mgclm.sh a40  --p2p      # M1+M3 on 2x A40
#   sbatch run_mgclm.sh a100 --p2p      # M1+M3 on 2x A100
#
#SBATCH --output=./slurm/slurm%j.out
#SBATCH --error=./slurm/slurm%j.err
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --time=1-00:00:00

set -e

# ---- Parse arguments ----
GPU_CHOICE="${1:-a40}"
case "${GPU_CHOICE}" in
    a40)  GRES="gpu:a40:2" ;;          # jaguar06
    nvidia_a40)  GRES="gpu:nvidia_a40:2" ;;   # jaguar01
    a100) GRES="gpu:a100:2" ;;         # cheetah04
    nvidia_a100) GRES="gpu:nvidia_a100-pcie-40gb:2" ;;  # cheetah01
    *)    echo "Usage: sbatch run_mgclm.sh {a40|nvidia_a40|a100|nvidia_a100} [--p2p]"; exit 1 ;;
esac
# Normalize GPU_CHOICE to a40/a100 for output directory naming
GPU_TAG=$(echo "${GPU_CHOICE}" | sed 's/nvidia_//')

P2P_FLAG=""
MODE_TAG="m1"
if [[ "$2" == "--p2p" ]]; then
    P2P_FLAG="--enable_p2p_caching"
    MODE_TAG="m1m3"
fi

# Re-submit with correct --gres if run directly
if [[ -z "$SLURM_JOB_ID" ]]; then
    echo "Submitting: sbatch --gres=${GRES} $0 $@"
    sbatch --gres="${GRES}" "$0" "$@"
    exit 0
fi

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

echo "=== Multi-GPU CLM (${MODE_TAG}) ==="
echo "GPU: ${GPU_CHOICE} x${NUM_GPUS}"
echo "P2P caching: ${P2P_FLAG:-disabled}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=============================="

torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --master_addr=localhost \
    --master_port=29500 \
    train_multi_gpu_clm.py \
    -s 'data/rubble-colmap' \
    --bsz 8 \
    --eval \
    ${P2P_FLAG} \
    -m "output/rubble_${GPU_TAG}_x${NUM_GPUS}_${MODE_TAG}"
