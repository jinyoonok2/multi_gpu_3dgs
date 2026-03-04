#!/bin/bash
# M3 only: Spatial partitioning + AllGather (no P2P, no overlap)
#SBATCH --output=./slurm/slurm%j.out
#SBATCH --error=./slurm/slurm%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:quadro_rtx_6000:2
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --time=1-00:00:00

set -e

module purge
module load miniforge
module load cuda/12.8.1

cd /u/xna8aw/6501_GPU_architectures/multi_gpu_3dgs
conda activate clm_gs

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_TIMEOUT=1800
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

torchrun \
    --nproc_per_node=2 \
    --master_addr=localhost \
    --master_port=29500 \
    train.py \
    -s 'data/rubble-colmap' \
    --multi_gpu \
    --bsz 8 \
    --eval \
    -m output/rubble_p2p_m3
