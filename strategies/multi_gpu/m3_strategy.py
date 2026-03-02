"""
M3Strategy — AllGather-based remote feature acquisition
========================================================

Method 3 (Spatial Partitioning) is always active (baked into
GaussianModelMultiGPU).  This strategy fetches remote Gaussians by
broadcasting ALL SH parameters across GPUs via dist.all_gather, then
indexing only the needed rows.

Trade-off:
  Simple and correct, but transfers N * 48 floats every forward pass
  regardless of how many remote Gaussians are actually visible.
"""

import torch
import torch.distributed as dist

from strategies.multi_gpu.base_strategy import (
    BaseMultiGPUStrategy,
    assemble_local_features,
)


# =========================================================================
# AllGather helpers
# =========================================================================

def _allgather_remote_features(gaussians, remote_global_idx, global_scaling, global_rotation):
    """
    AllGather ALL SH features across GPUs, then index only the needed entries.
    """
    if remote_global_idx.shape[0] == 0:
        return (
            torch.empty(0, 48, device="cuda"),
            torch.empty(0, 3,  device="cuda"),
            torch.empty(0, 1,  device="cuda"),
            torch.empty(0, 3,  device="cuda"),
            torch.empty(0, 4,  device="cuda"),
        )

    if dist.is_initialized() and gaussians.world_size > 1:
        all_params = [torch.zeros(s, 48, device="cuda") for s in gaussians.partition_sizes]
        dist.all_gather(all_params, gaussians._parameters.detach().contiguous())
        global_params = torch.cat(all_params, dim=0)
    else:
        global_params = gaussians._parameters.detach()

    remote_shs      = global_params[remote_global_idx]
    remote_xyz      = gaussians.global_xyz[remote_global_idx]
    remote_opacity  = gaussians.global_opacity[remote_global_idx]
    remote_scaling  = global_scaling[remote_global_idx]
    remote_rotation = global_rotation[remote_global_idx]

    return remote_shs, remote_xyz, remote_opacity, remote_scaling, remote_rotation


# =========================================================================
# M3Strategy
# =========================================================================

class M3Strategy(BaseMultiGPUStrategy):
    """
    M3 — spatial partitioning with AllGather for remote SH features.

    Each GPU owns N/world_size Gaussians.  Remote Gaussians' SH features
    are fetched by broadcasting all parameters and indexing locally.
    """

    def assemble_features(
        self,
        gaussians,
        local_idx,
        remote_global_idx,
        remote_rank,        # unused by AllGather — kept for interface compatibility
        remote_local_idx,   # unused by AllGather — kept for interface compatibility
        global_scaling,
        global_rotation,
    ):
        lx, lo, ls, lr, lsh = assemble_local_features(gaussians, local_idx)
        n_local = local_idx.shape[0]

        if remote_global_idx.shape[0] > 0:
            r_shs, r_xyz, r_opa_raw, r_scl, r_rot = _allgather_remote_features(
                gaussians, remote_global_idx, global_scaling, global_rotation
            )
            r_opa = gaussians.opacity_activation(r_opa_raw)
            return (
                torch.cat([lx, r_xyz]),  torch.cat([lo, r_opa]),
                torch.cat([ls, r_scl]),  torch.cat([lr, r_rot]),
                torch.cat([lsh, r_shs]), n_local,
            )

        return lx, lo, ls, lr, lsh, n_local
