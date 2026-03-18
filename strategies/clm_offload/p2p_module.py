"""
P2P Training Strategy Module
==============================

Hooks into the unified CLM engine to add GPU-to-GPU (peer-to-peer) SH
coefficient sharing via NCCL / NVLink.

Overridden hooks
----------------
- ``post_filters``   : Compute P2P filter partition (overlap vs local-only).
- ``load_first_shs`` : Cooperative GPU-GPU SH loading for first micro-batch.
- ``sync_gradients`` : Use ``P2PCommManager`` for gradient synchronisation.
"""

import torch
from strategies.clm_offload.strategy_base import BaseStrategy
from strategies.clm_offload.p2p_comm import P2PCommManager


class P2PStrategy(BaseStrategy):
    """P2P-enhanced training — cooperative GPU-GPU SH loading via NVLink."""

    def __init__(self, rank, world_size):
        self._p2p_mgr = P2PCommManager(rank, world_size)

    # ---------- Stage 1: compute P2P filter partition ----------
    def post_filters(self, args, filters, n_gaussians, iteration, log_file):
        if not (args.enable_distributed and args.world_size > 1):
            return {}

        torch.cuda.nvtx.range_push("p2p_filter_partition")
        all_local_indices = torch.cat(filters, dim=0).unique()
        partition = self._p2p_mgr.compute_filter_partition(
            all_local_indices, n_gaussians,
        )

        if args.rank == 0 and iteration == 1:
            n_overlap = partition["overlap_indices"].shape[0]
            n_local   = partition["local_only_indices"].shape[0]
            ratio     = partition["overlap_ratio"]
            msg = (f"[P2P] Iter {iteration}: overlap={n_overlap}, "
                   f"local_only={n_local}, ratio={ratio:.3f}")
            print(msg)
            log_file.write(msg + "\n")

        torch.cuda.nvtx.range_pop()
        return {"p2p_partition_first": partition}

    # ---------- Stage 4.1: cooperative first-SH load ----------
    def load_first_shs(self, shs, gaussians_params, local_filter,
                        n_gaussians, grid_size, block_size,
                        extra_state, args):
        from clm_kernels import send_shs2gpu_stream

        partition = extra_state.get("p2p_partition_first")
        if (args.enable_distributed and args.world_size > 1
                and partition is not None):
            torch.cuda.nvtx.range_push("p2p_share_shs")
            self._p2p_mgr.share_shs_p2p(
                shs,
                partition["overlap_indices"],
                local_filter,
                partition,
                gaussians_params,
                n_gaussians,
                grid_size,
                block_size,
            )
            torch.cuda.nvtx.range_pop()
        else:
            send_shs2gpu_stream(
                shs, gaussians_params, local_filter, grid_size, block_size,
            )

    # ---------- Stage 5.0: P2P gradient sync ----------
    def sync_gradients(self, gaussians, parameters_grad_buffer, N, args):
        torch.cuda.nvtx.range_push("p2p_sync_all_gradients")
        self._p2p_mgr.sync_gradients_p2p(gaussians, parameters_grad_buffer, N)
        torch.cuda.nvtx.range_pop()
        return True  # fully handled
