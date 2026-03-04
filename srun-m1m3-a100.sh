#!/bin/bash
# M1+M3: Spatial partitioning + P2P collaborative caching on A100
#SBATCH --output=./slurm/slurm%j.out
#SBATCH --error=./slurm/slurm%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_a100-pcie-40gb:2
#SBATCH --mem=64G
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
    --enable_p2p_caching \
    --bsz 8 \
    --eval \
    -m output/rubble_a100_m1m3
