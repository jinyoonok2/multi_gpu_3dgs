#!/bin/bash
# One-time build: recompile all CUDA submodules for A100 (SM 8.0)
# Run this ONCE before submitting any A100 training jobs:
#   sbatch srun-build-a100.sh
#SBATCH --output=./slurm/slurm%j.out
#SBATCH --error=./slurm/slurm%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_a100-pcie-40gb:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --time=0-01:00:00

set -e

module purge
module load miniforge
module load cuda/12.8.1

cd /u/xna8aw/6501_GPU_architectures/multi_gpu_3dgs
conda activate clm_gs

# Target both RTX 6000 (7.5) and A100 (8.0) so the same build works on both nodes
export TORCH_CUDA_ARCH_LIST="7.5;8.0"
export FORCE_CUDA=1

echo "=== Building clm_kernels ==="
cd submodules/clm_kernels
rm -rf build/ clm_kernels.egg-info/
pip install -e . --no-build-isolation
cd ../..

echo "=== Building simple-knn ==="
cd submodules/simple-knn
rm -rf build/ *.egg-info/ dist/
pip install -e . --no-build-isolation
cd ../..

echo "=== Building cpu-adam ==="
cd submodules/cpu-adam
rm -rf build/ *.egg-info/
pip install -e . --no-build-isolation
cd ../..

echo "=== Building gsplat ==="
cd submodules/gsplat
rm -rf build/ *.egg-info/
pip install -e . --no-build-isolation
cd ../..

echo "=== All submodules rebuilt for SM 7.5 + 8.0 ==="
python -c "import clm_kernels; from simple_knn._C import distCUDA2; print('Import check passed.')"
