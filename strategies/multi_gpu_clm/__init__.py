"""
Multi-GPU CLM — Correct Multi-GPU Scaling of CLM Offload
=========================================================

Preserves the CPU↔GPU streaming architecture of CLM (SH features on CPU,
fetched on demand) while adding multi-GPU parallelism:
  - Spatial metadata partitioned across GPUs (like multi_gpu)
  - SH features on CPU pinned memory (like CLM)
  - P2P cache sharing: GPUs borrow fetched SH from peers instead of
    redundant CPU fetches
  - Each GPU processes a subset of cameras per batch
"""

from .gaussian_model import GaussianModelMultiGPUCLM
from .engine import multi_gpu_clm_train_one_batch, multi_gpu_clm_eval_one_cam

__all__ = [
    "GaussianModelMultiGPUCLM",
    "multi_gpu_clm_train_one_batch",
    "multi_gpu_clm_eval_one_cam",
]
