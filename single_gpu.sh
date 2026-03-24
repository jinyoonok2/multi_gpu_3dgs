#!/bin/bash
# Single-GPU CLM baseline.
# Usage:
#   bash single_gpu.sh a16        # 1x A16  (jaguar02)
#   bash single_gpu.sh a40        # 1x A40  (jaguar06)
#   bash single_gpu.sh a100       # 1x A100 SXM (cheetah04)
#   bash single_gpu.sh nvidia_a100 # 1x A100 PCIe (cheetah01)
#SBATCH --output=./slurm/slurm%j.out
#SBATCH --error=./slurm/slurm%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --time=3-00:00:00

set -e

# ---- Parse arguments ----
GPU_CHOICE="${1:-a100}"
case "${GPU_CHOICE}" in
    a16)         GRES="gpu:nvidia_a16:1";  MEM="90G"  ;;  # jaguar02
    a40)         GRES="gpu:a40:1";         MEM="100G" ;;  # jaguar06
    nvidia_a40)  GRES="gpu:nvidia_a40:1";  MEM="100G" ;;  # jaguar01
    a100)        GRES="gpu:a100:1";        MEM="200G" ;;  # cheetah04 (SXM)
    nvidia_a100) GRES="gpu:nvidia_a100-pcie-40gb:1";  MEM="100G" ;;  # cheetah01
    *)    echo "Usage: bash single_gpu.sh {a16|a40|nvidia_a40|a100|nvidia_a100}"; exit 1 ;;
esac
GPU_TAG=$(echo "${GPU_CHOICE}" | sed 's/nvidia_//')

# Re-submit with correct --gres if run directly
if [[ -z "$SLURM_JOB_ID" ]]; then
    echo "Submitting: sbatch --gres=${GRES} --mem=${MEM} $0 ${GPU_CHOICE}"
    sbatch --gres="${GRES}" --mem="${MEM}" --export="ALL,GPU_CHOICE=${GPU_CHOICE}" \
        "$0" "${GPU_CHOICE}"
    exit 0
fi

module purge
module load miniforge
module load cuda/12.8.1

cd /u/xna8aw/6501_GPU_architectures/multi_gpu_3dgs
conda activate clm_gs

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

# Compute safe prealloc_capacity from the requested SLURM memory budget.
# Formula: (MEM_bytes * 0.7) / (48 floats * 4 bytes * 4 tensors) rounded to 16.
# Pass it explicitly so the auto-calculation never falls back to psutil-reported
# node RAM, which would over-allocate and OOM on machines with large system RAM.
MEM_GB=${MEM%G}
PREALLOC=$(python3 -c "print(int(${MEM_GB}*1024**3*0.7/(48*4*4))//16*16)")

OUTPUT_DIR="output/rubble_${GPU_TAG}_single"

echo "================================================="
echo "  Single-GPU CLM Baseline"
echo "  GPU:    ${GPU_CHOICE}"
echo "  Output: ${OUTPUT_DIR}"
echo "================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "================================================="

python train.py \
    -s data/rubble-colmap \
    --clm_offload \
    --prealloc_capacity ${PREALLOC} \
    --bsz 8 \
    --eval \
    -m "${OUTPUT_DIR}"
