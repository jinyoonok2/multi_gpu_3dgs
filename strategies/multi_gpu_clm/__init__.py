"""
Multi-GPU CLM — Camera-Parallel Multi-GPU CLM
==============================================

Camera parallelism: every GPU holds ALL Gaussians (replicated spatial
params on GPU, shared SH on CPU pinned memory). Each GPU renders a
subset of cameras per batch, then AllReduces gradients for correctness.

M1: Basic camera parallelism with AllReduced gradients.
M3: (planned) P2P collaborative SH caching.
M2: (planned) Overlapped dual-stream pipeline.
"""

from .gaussian_model import GaussianModelMultiGPUCLM
from .engine import multi_gpu_clm_train_one_batch, multi_gpu_clm_eval_one_cam

__all__ = [
    "GaussianModelMultiGPUCLM",
    "multi_gpu_clm_train_one_batch",
    "multi_gpu_clm_eval_one_cam",
]
