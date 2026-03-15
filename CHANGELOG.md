# Changelog — Our Changes to Master Multi-GPU Branch

## New Files
- **`multi_gpu.sh`** — SLURM script for 2-GPU job: `torchrun --nproc_per_node=2 train_multi.py`.
- **`single_gpu.sh`** — SLURM script for 1-GPU baseline: `python train.py`.

## Bug Fix: `train_multi.py`
**Problem:** Ranks drift out of iteration sync — no barriers exist between densification events. When one rank enters a densification block (calling `dist.barrier` + 3× `dist.all_reduce`) while the other skips it (different iteration count), NCCL deadlocks after 600s timeout.

**Fix:** Added `dist.barrier()` at the start of each training iteration so both ranks are always at the same iteration number before any densification logic runs. Original file preserved as `train_multi.py.bak`.
