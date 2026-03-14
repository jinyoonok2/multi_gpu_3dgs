#!/bin/bash
# CLM baseline: single-GPU CLM offload training.
# Usage:
#   sbatch run_clm_baseline.sh a40           # 1x A40  (jaguar06: gpu:a40, or jaguar01: gpu:nvidia_a40)
#   sbatch run_clm_baseline.sh a100          # 1x A100 (cheetah04: gpu:a100, or cheetah01: gpu:nvidia_a100-pcie-40gb)
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
    a40)  GRES="gpu:a40:1" ;;          # jaguar06
    nvidia_a40)  GRES="gpu:nvidia_a40:1" ;;   # jaguar01
    a100) GRES="gpu:a100:1" ;;         # cheetah04
    nvidia_a100) GRES="gpu:nvidia_a100-pcie-40gb:1" ;;  # cheetah01
    *)    echo "Usage: sbatch run_clm_baseline.sh {a40|nvidia_a40|a100|nvidia_a100}"; exit 1 ;;
esac
# Normalize GPU_CHOICE to a40/a100 for output directory naming
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

cd /u/xna8aw/6501_GPU_architectures/multi_gpu_3dgs
conda activate clm_gs

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

echo "=== CLM Baseline ==="
echo "GPU: ${GPU_CHOICE}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "===================="

python train.py \
    -s 'data/rubble-colmap' \
    --clm_offload \
    --bsz 8 \
    --eval \
    -m "output/rubble_${GPU_TAG}_clm"
