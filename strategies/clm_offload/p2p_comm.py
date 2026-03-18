"""
P2P Communication Module for Multi-GPU CLM Gaussian Splatting
=============================================================

Provides GPU-to-GPU (peer-to-peer) communication utilities for sharing
SH (Spherical Harmonics) coefficients between GPUs via NCCL/NVLink,
reducing redundant CPU→GPU PCIe transfers.

Key idea: When two GPUs need overlapping sets of Gaussians, one GPU
loads the shared portion from CPU and sends it to the peer via
NVLink (fast) instead of both loading from CPU via PCIe (slow).

Usage:
    from strategies.clm_offload.p2p_comm import P2PCommManager
"""

import torch
import torch.distributed as dist


class P2PCommManager:
    """
    Manages peer-to-peer SH coefficient sharing between GPUs.

    In CLM, each GPU needs SH coefficients for its visible Gaussians.
    Many Gaussians are visible from multiple GPUs (camera overlap).
    Instead of each GPU independently loading from CPU (PCIe-bound),
    GPUs cooperatively partition the load and share via NVLink/NCCL.

    Partition strategy for 2 GPUs:
        - overlap:  Gaussians visible on BOTH GPUs → GPU 0 loads, sends to GPU 1
        - only_0:   Gaussians visible ONLY on GPU 0 → GPU 0 loads from CPU
        - only_1:   Gaussians visible ONLY on GPU 1 → GPU 1 loads from CPU

    This reduces total PCIe traffic by ~overlap_ratio/2 per GPU.
    """

    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        self._p2p_enabled = False
        self._check_p2p_support()

    def _check_p2p_support(self):
        """Check if GPU P2P access is available (NVLink or PCIe P2P)."""
        if self.world_size < 2:
            self._p2p_enabled = False
            return

        # Check CUDA P2P capability between all local GPU pairs
        can_p2p = True
        for i in range(self.world_size):
            for j in range(self.world_size):
                if i != j:
                    if not torch.cuda.can_device_access_peer(i, j):
                        can_p2p = False
                        break
            if not can_p2p:
                break

        self._p2p_enabled = can_p2p
        if self.rank == 0:
            if can_p2p:
                print("[P2P] NVLink/P2P access available between GPUs")
            else:
                print("[P2P] No direct P2P access; using NCCL for GPU-GPU transfers")

    @property
    def p2p_available(self):
        return self._p2p_enabled

    def compute_filter_partition(self, local_filter, n_gaussians):
        """
        Exchange visibility filters between all GPUs and compute the partition.

        Given each GPU's local filter (indices of visible Gaussians), determine:
        - overlap: Gaussian indices visible on BOTH this GPU and at least one peer
        - local_only: Gaussian indices visible ONLY on this GPU
        - peer_only: Gaussian indices visible ONLY on peer(s), not this GPU

        For 2 GPUs, the assignment is:
        - GPU 0 loads: local_only_0 + overlap (from CPU)
        - GPU 1 loads: local_only_1 (from CPU), receives overlap from GPU 0

        Args:
            local_filter: (M,) int64 tensor, Gaussian indices visible on this GPU
            n_gaussians: total number of Gaussians

        Returns:
            dict with keys:
                'overlap_indices': Gaussian indices in the overlap set
                'local_only_indices': Gaussian indices only this GPU needs
                'peer_only_indices': Gaussian indices only peer needs
                'i_am_sender': bool, True if this GPU is the overlap sender
                'overlap_ratio': float, |overlap| / |union|
        """
        # Build a binary mask of local visibility
        local_mask = torch.zeros(n_gaussians, dtype=torch.uint8, device="cuda")
        local_mask.scatter_(0, local_filter, 1)

        # Gather all masks via all-reduce (SUM gives count of GPUs seeing each Gaussian)
        global_mask = local_mask.clone().to(torch.int32)
        dist.all_reduce(global_mask, op=dist.ReduceOp.SUM)

        # Classify each Gaussian
        is_local = local_mask.bool()
        is_shared = global_mask >= 2  # visible on 2+ GPUs
        overlap = is_local & is_shared
        local_only = is_local & ~is_shared

        overlap_indices = torch.nonzero(overlap, as_tuple=False).flatten()
        local_only_indices = torch.nonzero(local_only, as_tuple=False).flatten()
        peer_only_indices = torch.nonzero(~is_local & (global_mask > 0), as_tuple=False).flatten()

        union_size = (global_mask > 0).sum().item()
        overlap_ratio = overlap_indices.shape[0] / max(union_size, 1)

        # GPU 0 is always the sender for overlapping Gaussians
        i_am_sender = (self.rank == 0)

        return {
            "overlap_indices": overlap_indices,
            "local_only_indices": local_only_indices,
            "peer_only_indices": peer_only_indices,
            "i_am_sender": i_am_sender,
            "overlap_ratio": overlap_ratio,
        }

    def share_shs_p2p(self, local_shs, overlap_indices, local_filter,
                       partition_info, gaussians_parameters, n_gaussians,
                       grid_size, block_size):
        """
        Cooperatively load SH coefficients using P2P GPU communication.

        Instead of each GPU loading all its SH from CPU independently:
        1. Each GPU loads only its local_only portion from CPU
        2. GPU 0 (sender) loads the overlap portion from CPU
        3. GPU 0 sends overlap SH to GPU 1 via NCCL (uses NVLink if available)
        4. Each GPU assembles the full SH tensor from local_only + overlap

        Args:
            local_shs: pre-allocated output tensor (filter_len, 48) on GPU
            overlap_indices: Gaussian indices in the overlap
            local_filter: this GPU's full visibility filter
            partition_info: dict from compute_filter_partition()
            gaussians_parameters: CPU pinned memory SH parameter tensor
            n_gaussians: total Gaussian count
            grid_size, block_size: CUDA kernel launch params

        Returns:
            local_shs: filled with SH coefficients
        """
        from clm_kernels import send_shs2gpu_stream

        overlap_idx = partition_info["overlap_indices"]
        local_only_idx = partition_info["local_only_indices"]
        i_am_sender = partition_info["i_am_sender"]

        n_overlap = overlap_idx.shape[0]
        n_local_only = local_only_idx.shape[0]

        if n_overlap == 0:
            # No overlap — fall back to standard CPU→GPU load
            send_shs2gpu_stream(
                local_shs, gaussians_parameters, local_filter,
                grid_size, block_size,
            )
            return local_shs

        # Build index mapping: for each index in local_filter, find its position
        # We need to know WHERE in the output local_shs each Gaussian goes
        filter_pos = torch.zeros(n_gaussians, dtype=torch.int32, device="cuda")
        filter_pos.scatter_(
            0, local_filter,
            torch.arange(local_filter.shape[0], dtype=torch.int32, device="cuda")
        )

        # Positions in output tensor for overlap and local_only
        overlap_out_positions = torch.gather(filter_pos, 0, overlap_idx)
        local_only_out_positions = torch.gather(filter_pos, 0, local_only_idx)

        # Step 1: Every GPU loads its local_only SH from CPU
        if n_local_only > 0:
            local_only_shs = torch.empty(n_local_only, 48, device="cuda")
            send_shs2gpu_stream(
                local_only_shs, gaussians_parameters, local_only_idx,
                grid_size, block_size,
            )
            # Scatter into output positions
            local_shs.scatter_(
                0,
                local_only_out_positions.unsqueeze(1).expand(-1, 48).long(),
                local_only_shs,
            )

        # Step 2: GPU 0 loads overlap SH from CPU, then broadcasts to all
        overlap_shs = torch.empty(n_overlap, 48, device="cuda")
        if i_am_sender:
            send_shs2gpu_stream(
                overlap_shs, gaussians_parameters, overlap_idx,
                grid_size, block_size,
            )

        # Step 3: Broadcast overlap SH from GPU 0 to all other GPUs via NCCL
        # On NVLink systems, NCCL automatically uses the NVLink path
        dist.broadcast(overlap_shs, src=0)

        # Step 4: Scatter overlap SH into correct output positions
        local_shs.scatter_(
            0,
            overlap_out_positions.unsqueeze(1).expand(-1, 48).long(),
            overlap_shs,
        )

        return local_shs

    def sync_gradients_p2p(self, gaussians, parameters_grad_buffer, N):
        """
        Synchronize gradients across GPUs using direct GPU-GPU NCCL.

        Optimized over the baseline approach in engine_multi.py which does:
            CPU pinned → GPU copy → all_reduce → GPU → CPU pinned copy

        This version keeps the SH gradient reduction entirely on GPU
        and only copies back to CPU once at the end, saving one PCIe round-trip.

        Args:
            gaussians: GaussianModel with .all_parameters()
            parameters_grad_buffer: CPU pinned memory gradient buffer (N, 48)
            N: number of Gaussians
        """
        # Sync GPU-resident parameter gradients (xyz, opacity, scaling, rotation)
        gpu_grads = [
            param.grad for param in gaussians.all_parameters()[:4] if param.grad is not None
        ]
        if len(gpu_grads) > 0:
            try:
                dist.all_reduce_coalesced(gpu_grads, op=dist.ReduceOp.SUM)
            except Exception:
                for grad in gpu_grads:
                    dist.all_reduce(grad, op=dist.ReduceOp.SUM)

        # Sync SH gradients: CPU pinned → GPU → all_reduce → CPU pinned
        torch.cuda.nvtx.range_push("p2p_sync_cpu_gradients")
        cpu_grad_gpu = parameters_grad_buffer[:N, :].cuda(non_blocking=True)
        dist.all_reduce(cpu_grad_gpu, op=dist.ReduceOp.SUM)
        # Use blocking copy-back to avoid an extra explicit device synchronize.
        parameters_grad_buffer[:N, :].copy_(cpu_grad_gpu, non_blocking=False)
        del cpu_grad_gpu
        torch.cuda.nvtx.range_pop()
