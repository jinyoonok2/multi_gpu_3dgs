#!/bin/bash
# ===========================================================================
# Unified Multi-GPU CLM Training Script
# ===========================================================================
# One script for all modes: baseline, P2P, and overlap.
#
# Usage:
#   bash multi_gpu.sh a100                     # baseline (default)
#   bash multi_gpu.sh a100 p2p                 # P2P GPU-to-GPU SH sharing
#   bash multi_gpu.sh a100 overlap             # dual-stream overlapped schedule
#   bash multi_gpu.sh a100 async               # async all-reduce (overlap grad sync)
#   bash multi_gpu.sh a100 p2p_async           # P2P + async all-reduce
#   bash multi_gpu.sh a100 overlap_async       # dual-stream + async all-reduce
#   bash multi_gpu.sh a100 p2p_overlap         # P2P + dual-stream
#   bash multi_gpu.sh a100 all                 # P2P + dual-stream + async all-reduce
#
# Self-submits to SLURM when run from the login node.
# ===========================================================================
#SBATCH --output=./slurm/slurm%j.out
#SBATCH --error=./slurm/slurm%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --time=1-00:00:00

set -e

# ---- Parse arguments ----
GPU_CHOICE="${1:-a100}"
MODE="${2:-baseline}"

case "${GPU_CHOICE}" in
    a16)         GRES="gpu:nvidia_a16:2" ;;
    a40)         GRES="gpu:a40:2" ;;
    nvidia_a40)  GRES="gpu:nvidia_a40:2" ;;
    a100)        GRES="gpu:a100:2" ;;
    nvidia_a100) GRES="gpu:nvidia_a100-pcie-40gb:2" ;;
    *)  echo "Usage: bash multi_gpu.sh {a16|a40|nvidia_a40|a100|nvidia_a100} [baseline|p2p|overlap]"
        exit 1 ;;
esac

case "${MODE}" in
    baseline|p2p|overlap|async|p2p_async|overlap_async|p2p_overlap|all) ;;   # valid
    *)  echo "Invalid mode '${MODE}'. Choose: baseline, p2p, overlap, async, p2p_async, overlap_async, p2p_overlap, or all."
        exit 1 ;;
esac

GPU_TAG=$(echo "${GPU_CHOICE}" | sed 's/nvidia_//')

# jaguar02 (A16) has only ~100 GB system RAM
if [[ "${GPU_CHOICE}" == "a16" ]]; then
    SLURM_MEM="90G"
else
    SLURM_MEM="200G"
fi

# Self-submit to SLURM if not already running inside a job
if [[ -z "$SLURM_JOB_ID" ]]; then
    echo "Submitting: sbatch --gres=${GRES} --mem=${SLURM_MEM} $0 ${GPU_CHOICE} ${MODE}"
    sbatch --gres="${GRES}" --mem="${SLURM_MEM}" --export="ALL,GPU_CHOICE=${GPU_CHOICE},MODE=${MODE}" \
           "$0" "${GPU_CHOICE}" "${MODE}"
    exit 0
fi

# ---- Environment setup ----
module purge
module load miniforge
module load cuda/12.8.1

cd /u/xna8aw/6501_GPU_architectures/multi_gpu_3dgs
conda activate clm_gs

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Auto-detect GPUs
NUM_GPUS=$(nvidia-smi -L | wc -l)
GPU_IDS=$(seq -s, 0 $((NUM_GPUS - 1)))
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

export NCCL_TIMEOUT=1800
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Dynamic port to avoid conflicts with other jobs
MASTER_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('', 0)); port=s.getsockname()[1]; s.close(); print(port)")

# ---- Build mode-specific flags ----
EXTRA_FLAGS=""
OUTPUT_SUFFIX="multi_clm"

case "${MODE}" in
    p2p)
        EXTRA_FLAGS="--p2p_fetch"
        OUTPUT_SUFFIX="multi_p2p"
        ;;
    overlap)
        EXTRA_FLAGS="--overlap_schedule"
        OUTPUT_SUFFIX="multi_overlap"
        ;;
    async)
        EXTRA_FLAGS="--async_allreduce"
        OUTPUT_SUFFIX="multi_async"
        ;;
    p2p_async)
        EXTRA_FLAGS="--p2p_fetch --async_allreduce"
        OUTPUT_SUFFIX="multi_p2p_async"
        ;;
    overlap_async)
        EXTRA_FLAGS="--overlap_schedule --async_allreduce"
        OUTPUT_SUFFIX="multi_overlap_async"
        ;;
    p2p_overlap)
        EXTRA_FLAGS="--p2p_fetch --overlap_schedule"
        OUTPUT_SUFFIX="multi_p2p_overlap"
        ;;
    all)
        EXTRA_FLAGS="--p2p_fetch --overlap_schedule --async_allreduce"
        OUTPUT_SUFFIX="multi_all"
        ;;
esac

# ---- Print run info ----
echo "================================================="
echo "  Multi-GPU CLM Training"
echo "  Mode:   ${MODE}"
echo "  GPU:    ${GPU_CHOICE} x${NUM_GPUS}"
echo "  Port:   ${MASTER_PORT}"
echo "  Output: output/rubble_${GPU_TAG}_${OUTPUT_SUFFIX}"
echo "================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "================================================="

# ---- Launch ----
torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --master_addr=localhost \
    --master_port="${MASTER_PORT}" \
    train_multi.py \
    -s 'data/rubble-colmap' \
    --clm_offload \
    --enable_distributed \
    ${EXTRA_FLAGS} \
    --bsz 8 \
    --eval \
    -m "output/rubble_${GPU_TAG}_${OUTPUT_SUFFIX}"
