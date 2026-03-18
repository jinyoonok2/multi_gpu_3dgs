# CLM-GS Setup Guide

## 1. Clone and Initialize

```bash
git clone https://github.com/nyu-systems/CLM-GS.git
cd CLM-GS
git config --global url."https://github.com/".insteadOf git@github.com:
git submodule update --init --recursive
```

## 2. Create Conda Environment

```bash
conda create -n clm_gs python=3.10
conda activate clm_gs
```

## 3. Install PyTorch (CUDA 12.4)

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

## 4. Install Python Dependencies

```bash
pip install tqdm plyfile psutil numba opencv-python scipy matplotlib pandas imageio imageio-ffmpeg requests tabulate
```

## 5. Build & Install CUDA/C++ Submodules

These submodules contain custom CUDA kernels that must be compiled. Make sure CUDA is available (`nvcc --version`) before proceeding.

**Why `--no-build-isolation`?** Four of the five submodules import `torch` in their `setup.py` to use `torch.utils.cpp_extension`. By default, pip builds packages in an isolated virtual environment that does **not** have torch installed, causing the build to fail. The `--no-build-isolation` flag tells pip to use the current environment (where torch is already installed).

```bash
# Load CUDA module (if on SLURM cluster)
module load cuda/12.8.1   # or whichever version is available

# simple-knn: spatial KNN for Gaussian initialization (CUDA, needs torch)
pip install --no-build-isolation submodules/simple-knn

# clm_kernels: custom CUDA kernels for CLM offloading (CUDA, needs torch)
pip install --no-build-isolation submodules/clm_kernels

# cpu-adam: CPU Adam optimizer for offloaded parameters (C++, needs torch)
pip install --no-build-isolation submodules/cpu-adam

# gsplat: Gaussian splatting rasterizer (CUDA, needs torch)
pip install --no-build-isolation submodules/gsplat

# fast-tsp: TSP solver for camera ordering (C++ with pybind11, does NOT need torch)
# Regular install is fine here — it uses scikit-build, not torch extensions
pip install submodules/fast-tsp
```

## 6. Dataset Preparation (Rubble)

Download the Rubble dataset (COLMAP format):

```bash
mkdir -p data
# Download rubble-colmap from the project's data source
# Place it at: data/rubble-colmap/
```

The dataset directory should look like:
```
data/rubble-colmap/
├── images/          # Original images
├── sparse/0/        # COLMAP sparse reconstruction
│   ├── cameras.bin
│   ├── images.bin
│   └── points3D.bin
└── decoded_images/  # Auto-generated on first run
```

## 7. Running Training

### Single GPU

```bash
python train.py \
    -s data/rubble-colmap \
    --clm_offload \
    --eval \
    -m output/rubble_single_gpu
```

### Multi-GPU (2 GPUs)

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_TIMEOUT=1800
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

torchrun \
    --nproc_per_node=2 \
    --master_addr=localhost \
    --master_port=29500 \
    train_multi.py \
    -s data/rubble-colmap \
    --clm_offload \
    --enable_distributed \
    --bsz 8 \
    --eval \
    -m output/rubble_multi_gpu
```

## 8. SLURM Job Scripts

### Install submodules (one-time, needs GPU node)

```bash
#!/bin/bash
#SBATCH --output=./slurm/slurm%j.out
#SBATCH --error=./slurm/slurm%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --time=1:00:00

set -e
source ~/.bashrc
conda activate clm_gs
module load cuda/12.8.1

cd /path/to/CLM-GS

pip install --no-build-isolation submodules/simple-knn
pip install --no-build-isolation submodules/clm_kernels
pip install --no-build-isolation submodules/cpu-adam
pip install --no-build-isolation submodules/gsplat
pip install submodules/fast-tsp
```

### Multi-GPU training

```bash
#!/bin/bash
#SBATCH --output=./slurm/slurm%j.out
#SBATCH --error=./slurm/slurm%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --time=1-00:00:00

set -e
source ~/.bashrc
conda activate clm_gs
module load cuda/12.8.1

cd /path/to/CLM-GS

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_TIMEOUT=1800
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

torchrun \
    --nproc_per_node=2 \
    --master_addr=localhost \
    --master_port=29500 \
    train_multi.py \
    -s data/rubble-colmap \
    --clm_offload \
    --enable_distributed \
    --bsz 8 \
    --eval \
    -m output/rubble_multi_gpu
```

## Troubleshooting

- **`ModuleNotFoundError: No module named 'torch'` during submodule install**: You forgot `--no-build-isolation`. The setup.py imports torch at build time.
- **NCCL timeout**: Increase `NCCL_TIMEOUT` or check that GPUs can communicate (NVLink for P2P, PCIe otherwise).
- **OOM (system RAM)**: The `prealloc_capacity` auto-scales based on available memory. If `--mem` is too high on SLURM, it over-allocates. Use `--mem=200G` for 2 GPUs or set `--prealloc_capacity` manually (e.g., `40000000`).
