"""
Multi-GPU Training Engine — dispatch shim
==========================================

Public entry points used by train.py.  Picks the right strategy class
based on command-line flags and delegates all work to it.

Strategy selection:
  --enable_p2p_caching  →  M1M3Strategy (P2P point-to-point)
  (default)             →  M3Strategy   (AllGather)

To add a new method later, create a new Strategy class in its own file,
import it here, and add a branch in _get_strategy().
"""

import utils.general_utils as utils
from strategies.multi_gpu.m3_strategy import M3Strategy
from strategies.multi_gpu.m1m3_strategy import M1M3Strategy


def _get_strategy():
    """Return the appropriate strategy instance based on current args."""
    args = utils.get_args()
    if args.enable_p2p_caching:
        return M1M3Strategy()
    return M3Strategy()


def multi_gpu_train_one_batch(gaussians, scene, batched_cameras, background, pipe_args):
    """Train one batch — delegates to the active strategy."""
    return _get_strategy().train_one_batch(
        gaussians, scene, batched_cameras, background, pipe_args
    )


def multi_gpu_eval_one_cam(camera, gaussians, background, scene):
    """Eval one camera — always uses AllGather (eval is infrequent)."""
    return M3Strategy().eval_one_cam(camera, gaussians, background, scene)
