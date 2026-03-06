#!/bin/bash
# CLM baseline: single-GPU CLM offload training.
# Usage: sbatch run_clm_baseline.sh [GPU_TYPE]
#   GPU_TYPE examples: a40, a100, nvidia_a100-pcie-40gb, quadro_rtx_6000
#   Default: a40
#
#SBATCH --output=./slurm/slurm%j.out
#SBATCH --error=./slurm/slurm%j.err
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --time=1-00:00:00

set -e

# ---- GPU type from argument or default ----
GPU_TYPE="${1:-a40}"

# Request 1 GPU of the specified type via srun override
export SLURM_GRES="gpu:${GPU_TYPE}:1"
# Re-submit with correct gres if not already set by SBATCH
if [[ -z "$SLURM_JOB_ID" ]]; then
    echo "Submit with: sbatch --gres=gpu:${GPU_TYPE}:1 $0 $@"
    exit 1
fi

module purge
module load miniforge
module load cuda/12.8.1

cd /u/xna8aw/6501_GPU_architectures/multi_gpu_3dgs
conda activate clm_gs

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

echo "=== CLM Baseline ==="
echo "GPU type: ${GPU_TYPE}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "===================="

python train.py \
    -s 'data/rubble-colmap' \
    --clm_offload \
    --bsz 8 \
    --eval \
    -m "output/rubble_${GPU_TYPE}_clm"
