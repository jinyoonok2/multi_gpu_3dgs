"""
M1M3Strategy — P2P point-to-point remote feature acquisition
=============================================================

Extends M3 (Spatial Partitioning) with Method 1 (P2P Caching).

Instead of AllGather-ing all N * 48 floats, each GPU only fetches the
SH features it actually needs via symmetric NCCL isend/irecv exchanges
with each peer rank.

CRITICAL DESIGN — symmetric exchange:
  Both GPUs enter the same function simultaneously.  Non-blocking
  isend/irecv are used to avoid deadlocks.

  Broken pattern:
    GPU 0: send(count) → send(idx) → recv(feat)   ← DEADLOCK
    GPU 1: send(count) → send(idx) → recv(feat)   ← both stall on send

  Correct pattern:
    GPU 0: isend(count) + irecv(count) → wait → ...   ← no deadlock
    GPU 1: isend(count) + irecv(count) → wait → ...

Protocol per peer (3 phases):
  1. Exchange counts   — tell peer how many indices I need
  2. Exchange indices  — send my index list, receive peer's index list
  3. Exchange features — gather local SH for peer, send; receive my SH
"""

import torch
import torch.distributed as dist

from strategies.multi_gpu.base_strategy import (
    BaseMultiGPUStrategy,
    assemble_local_features,
)


# =========================================================================
# P2P symmetric exchange helpers
# =========================================================================

def _p2p_symmetric_feature_exchange(gaussians, peer_rank, my_need_indices):
    """
    Exchange SH features with one peer rank.
    BOTH ranks must call this simultaneously with their own need_indices.

    Returns: (n_need, 48) tensor of SH features fetched from peer.
    """
    n_need = my_need_indices.shape[0]

    # Phase 1: Exchange counts
    my_count   = torch.tensor([n_need], dtype=torch.long, device="cuda")
    peer_count = torch.empty(1, dtype=torch.long, device="cuda")
    req_s = dist.isend(my_count,   dst=peer_rank)
    req_r = dist.irecv(peer_count, src=peer_rank)
    req_s.wait(); req_r.wait()
    n_peer_needs = int(peer_count.item())

    # Phase 2: Exchange index lists
    peer_need_indices = torch.empty(n_peer_needs, dtype=my_need_indices.dtype, device="cuda")
    ops = []
    if n_need > 0:
        ops.append(dist.isend(my_need_indices.contiguous(), dst=peer_rank))
    if n_peer_needs > 0:
        ops.append(dist.irecv(peer_need_indices, src=peer_rank))
    for op in ops:
        op.wait()

    # Phase 3: Gather and exchange features
    features_for_peer = (
        gaussians._parameters.detach()[peer_need_indices].contiguous()
        if n_peer_needs > 0
        else torch.empty(0, 48, device="cuda")
    )
    features_from_peer = (
        torch.empty(n_need, 48, device="cuda")
        if n_need > 0
        else torch.empty(0, 48, device="cuda")
    )
    ops = []
    if n_peer_needs > 0:
        ops.append(dist.isend(features_for_peer, dst=peer_rank))
    if n_need > 0:
        ops.append(dist.irecv(features_from_peer, src=peer_rank))
    for op in ops:
        op.wait()

    return features_from_peer


def _p2p_symmetric_gradient_exchange(gaussians, peer_rank,
                                     grads_for_peer, indices_on_peer):
    """
    Exchange SH gradients with one peer rank (symmetric protocol).
    BOTH ranks must call this simultaneously.

    Sends gradients computed for Gaussians owned by peer.
    Receives gradients the peer computed for Gaussians we own.
    Accumulates received gradients into gaussians._parameters.grad.
    """
    n_send = grads_for_peer.shape[0]

    # Phase 1: Exchange counts
    my_count   = torch.tensor([n_send], dtype=torch.long, device="cuda")
    peer_count = torch.empty(1, dtype=torch.long, device="cuda")
    req_s = dist.isend(my_count,   dst=peer_rank)
    req_r = dist.irecv(peer_count, src=peer_rank)
    req_s.wait(); req_r.wait()
    n_recv = int(peer_count.item())

    # Phase 2: Exchange indices
    recv_indices = torch.empty(n_recv, dtype=torch.long, device="cuda")
    ops = []
    if n_send > 0:
        ops.append(dist.isend(indices_on_peer.contiguous(), dst=peer_rank))
    if n_recv > 0:
        ops.append(dist.irecv(recv_indices, src=peer_rank))
    for op in ops:
        op.wait()

    # Phase 3: Exchange gradient values
    recv_grads = (
        torch.empty(n_recv, 48, device="cuda")
        if n_recv > 0
        else torch.empty(0, 48, device="cuda")
    )
    ops = []
    if n_send > 0:
        ops.append(dist.isend(grads_for_peer.contiguous(), dst=peer_rank))
    if n_recv > 0:
        ops.append(dist.irecv(recv_grads, src=peer_rank))
    for op in ops:
        op.wait()

    # Accumulate received gradients into local param grads
    if n_recv > 0 and gaussians._parameters.grad is not None:
        gaussians._parameters.grad.index_add_(0, recv_indices, recv_grads)


# =========================================================================
# M1M3Strategy
# =========================================================================

class M1M3Strategy(BaseMultiGPUStrategy):
    """
    M1+M3 — spatial partitioning with P2P point-to-point SH fetch.

    Each GPU only transfers the SH features it needs for visible remote
    Gaussians, using symmetric NCCL isend/irecv with each peer rank.
    """

    def assemble_features(
        self,
        gaussians,
        local_idx,
        remote_global_idx,
        remote_rank,
        remote_local_idx,
        global_scaling,
        global_rotation,
    ):
        lx, lo, ls, lr, lsh = assemble_local_features(gaussians, local_idx)
        n_local = local_idx.shape[0]

        n_remote = remote_global_idx.shape[0]
        has_remote = n_remote > 0

        if has_remote:
            r_xyz = gaussians.global_xyz[remote_global_idx]
            r_opa = gaussians.opacity_activation(
                gaussians.global_opacity[remote_global_idx]
            )
            r_scl = global_scaling[remote_global_idx]
            r_rot = global_rotation[remote_global_idx]
            r_shs = torch.empty(n_remote, 48, device="cuda")

        # CRITICAL: Loop over ALL peer ranks, not just torch.unique(remote_rank).
        # Both GPUs must enter the exchange for every peer pair, even when one
        # side needs 0 Gaussians.  Skipping a peer causes a deadlock because
        # the other side is waiting in irecv with no matching isend.
        for peer in range(gaussians.world_size):
            if peer == gaussians.rank:
                continue
            if has_remote:
                mask = remote_rank == peer
                my_need_indices = remote_local_idx[mask]
            else:
                mask = None
                my_need_indices = torch.empty(0, dtype=torch.long, device="cuda")

            fetched = _p2p_symmetric_feature_exchange(
                gaussians, peer, my_need_indices
            )

            if has_remote and fetched.shape[0] > 0:
                r_shs[mask] = fetched

        if has_remote:
            return (
                torch.cat([lx, r_xyz]),  torch.cat([lo, r_opa]),
                torch.cat([ls, r_scl]),  torch.cat([lr, r_rot]),
                torch.cat([lsh, r_shs]), n_local,
            )

        return lx, lo, ls, lr, lsh, n_local
