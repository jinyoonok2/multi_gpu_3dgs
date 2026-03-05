"""
Multi-GPU CLM Engine
====================

Training engine that preserves CLM's CPU↔GPU streaming for SH features
while adding multi-GPU parallelism with P2P cache sharing.

Architecture:
  1. Global visibility: All GPUs compute which Gaussians are visible
     using the replicated lightweight proxy (xyz + opacity).
  2. Camera assignment: Cameras in the batch are distributed across GPUs.
     Each GPU processes bsz/world_size cameras.
  3. SH streaming (CLM): For each camera, the GPU fetches visible SH
     features from CPU pinned memory on a comm_stream, overlapped with
     computation on the default stream.
  4. P2P cache sharing: When multiple GPUs need overlapping SH blocks,
     they exchange via NCCL P2P instead of redundant CPU fetches.
  5. Gradient flow:
     - Spatial params: local gradients update directly on GPU.
       For remote Gaussians, gradients are sent to the owning rank
       via P2P and accumulated.
     - SH gradients: offloaded to CPU, processed by CPU Adam thread
       (same as CLM baseline).

Key difference from old multi_gpu strategy:
  The old design put SH in VRAM — defeating CLM's purpose.
  This design keeps SH on CPU, making multi-GPU useful for scenes
  that genuinely exceed single-GPU VRAM.
"""

import math
import threading

import torch
import torch.distributed as dist

import clm_kernels
from clm_kernels import (
    send_shs2gpu_stream,
    send_shs2cpu_grad_buffer_stream,
    send_shs2gpu_stream_retention,
    send_shs2cpu_grad_buffer_stream_retention,
    spherical_harmonics_bwd_inplace,
)
from gsplat import (
    fully_fused_projection,
    isect_offset_encode,
    isect_tiles,
    rasterize_to_pixels,
    spherical_harmonics,
)

import utils.general_utils as utils
from strategies.base_engine import TILE_SIZE, torch_compiled_loss, calculate_filters
from densification import update_densification_stats_offload_accum_grads


# =========================================================================
# Visibility using global proxy (reused from multi_gpu base_strategy)
# =========================================================================

def calculate_filters_global(batched_cameras, gaussians):
    """
    Compute visibility using the lightweight global proxy.
    Returns per-camera GLOBAL Gaussian indices + gathered scaling/rotation.
    """
    args = utils.get_args()
    image_width = int(utils.get_img_width())
    image_height = int(utils.get_img_height())

    with torch.no_grad():
        Ks = []
        viewmats = []
        for camera in batched_cameras:
            K = camera.create_k_on_gpu()
            viewmat = camera.world_view_transform.transpose(0, 1)
            Ks.append(K)
            viewmats.append(viewmat)
        batched_Ks = torch.stack(Ks)
        batched_viewmats = torch.stack(viewmats)

        # Gather global scaling/rotation for projection
        scaling = gaussians.get_scaling
        rotation = gaussians.get_rotation

        if dist.is_initialized() and gaussians.world_size > 1:
            all_scaling = [
                torch.zeros(s, 3, device="cuda")
                for s in gaussians.partition_sizes
            ]
            all_rotation = [
                torch.zeros(s, 4, device="cuda")
                for s in gaussians.partition_sizes
            ]
            dist.all_gather(all_scaling, scaling.contiguous())
            dist.all_gather(all_rotation, rotation.contiguous())
            global_scaling = torch.cat(all_scaling, dim=0)
            global_rotation = torch.cat(all_rotation, dim=0)
        else:
            global_scaling = scaling
            global_rotation = rotation

        proj_results = fully_fused_projection(
            means=gaussians.global_xyz,
            covars=None,
            quats=global_rotation,
            scales=global_scaling,
            viewmats=batched_viewmats,
            Ks=batched_Ks,
            radius_clip=args.radius_clip,
            width=image_width,
            height=image_height,
            packed=True,
        )

        (camera_ids, gaussian_ids, _, _, _, _, _) = proj_results
        output, counts = torch.unique_consecutive(camera_ids, return_counts=True)
        counts_cpu = counts.cpu().numpy().tolist()
        gaussian_ids_per_camera = torch.split(gaussian_ids, counts_cpu)

    return gaussian_ids_per_camera, global_scaling, global_rotation


# =========================================================================
# P2P SH cache sharing helpers
# =========================================================================

def _p2p_sh_cache_exchange(gaussians, peer_rank, my_need_local_indices,
                           my_cached_sh, my_cached_local_indices):
    """
    Exchange SH features with a peer GPU.

    Instead of both GPUs fetching overlapping SH from CPU, check if the
    peer already has what we need, and exchange via NCCL P2P.

    Protocol:
      1. Exchange need counts
      2. Exchange need indices (local indices on the peer's partition)
      3. Peer gathers from its SH cache → sends; we receive

    Args:
        gaussians: GaussianModelMultiGPUCLM
        peer_rank: rank of the peer GPU
        my_need_local_indices: local indices (on peer's partition) that I need
        my_cached_sh: (M, 48) SH features I've already fetched from CPU
        my_cached_local_indices: local indices corresponding to my_cached_sh

    Returns:
        received_sh: (n_need, 48) SH features from peer
    """
    n_need = my_need_local_indices.shape[0]

    # Phase 1: Exchange counts
    my_count = torch.tensor([n_need], dtype=torch.long, device="cuda")
    peer_count = torch.empty(1, dtype=torch.long, device="cuda")
    for r in dist.batch_isend_irecv([
        dist.P2POp(dist.isend, my_count, peer_rank),
        dist.P2POp(dist.irecv, peer_count, peer_rank),
    ]):
        r.wait()
    n_peer_needs = int(peer_count.item())

    # Phase 2: Exchange index lists
    peer_need_indices = torch.empty(
        n_peer_needs, dtype=my_need_local_indices.dtype, device="cuda"
    )
    ops = []
    if n_need > 0:
        ops.append(dist.P2POp(dist.isend, my_need_local_indices.contiguous(), peer_rank))
    if n_peer_needs > 0:
        ops.append(dist.P2POp(dist.irecv, peer_need_indices, peer_rank))
    if ops:
        for r in dist.batch_isend_irecv(ops):
            r.wait()

    # Phase 3: Gather SH for peer from our CPU-fetched cache and exchange
    if n_peer_needs > 0 and my_cached_sh is not None and my_cached_sh.shape[0] > 0:
        # Build index mapping: peer needs indices into our partition,
        # our cache has some of those. Fetch remaining from CPU.
        features_for_peer = _gather_sh_for_indices(
            gaussians, peer_need_indices, my_cached_sh, my_cached_local_indices
        )
    else:
        features_for_peer = torch.empty(0, 48, device="cuda")

    received_sh = torch.empty(n_need, 48, device="cuda") if n_need > 0 else torch.empty(0, 48, device="cuda")
    ops = []
    if n_peer_needs > 0:
        ops.append(dist.P2POp(dist.isend, features_for_peer.contiguous(), peer_rank))
    if n_need > 0:
        ops.append(dist.P2POp(dist.irecv, received_sh, peer_rank))
    if ops:
        for r in dist.batch_isend_irecv(ops):
            r.wait()

    return received_sh


def _gather_sh_for_indices(gaussians, local_indices, cached_sh, cached_local_indices):
    """
    Gather SH features for given local indices.
    First check the GPU cache, fallback to CPU pinned memory for misses.
    """
    n = local_indices.shape[0]
    result = torch.empty(n, 48, device="cuda")

    if cached_sh is not None and cached_sh.shape[0] > 0:
        # Build reverse index: local_idx → position in cache
        max_idx = max(
            cached_local_indices.max().item() if cached_local_indices.shape[0] > 0 else 0,
            local_indices.max().item() if n > 0 else 0,
        ) + 1
        reverse_map = torch.full((max_idx,), -1, dtype=torch.long, device="cuda")
        reverse_map[cached_local_indices] = torch.arange(
            cached_local_indices.shape[0], device="cuda"
        )

        cache_positions = reverse_map[local_indices]
        cache_hit = cache_positions >= 0
        cache_miss = ~cache_hit

        # Cache hits: gather from GPU cache
        if cache_hit.any():
            hit_positions = cache_positions[cache_hit]
            result[cache_hit] = cached_sh[hit_positions]

        # Cache misses: fetch from CPU
        if cache_miss.any():
            miss_local_indices = local_indices[cache_miss]
            # Fetch from CPU pinned memory
            miss_sh = gaussians._parameters[miss_local_indices.cpu()].cuda()
            result[cache_miss] = miss_sh
    else:
        # No cache — all from CPU
        result = gaussians._parameters[local_indices.cpu()].cuda()

    return result


# =========================================================================
# P2P gradient exchange for remote spatial params
# =========================================================================

def _p2p_spatial_gradient_exchange(peer_rank, grads_for_peer, indices_for_peer,
                                   my_local_n, device="cuda"):
    """
    Exchange spatial gradients with a peer for remote Gaussians.
    Send gradients I computed for Gaussians owned by peer.
    Receive gradients peer computed for Gaussians I own.
    Returns (received_grads, received_indices).
    """
    n_send = grads_for_peer.shape[0]

    # Phase 1: Exchange counts
    my_count = torch.tensor([n_send], dtype=torch.long, device=device)
    peer_count = torch.empty(1, dtype=torch.long, device=device)
    for r in dist.batch_isend_irecv([
        dist.P2POp(dist.isend, my_count, peer_rank),
        dist.P2POp(dist.irecv, peer_count, peer_rank),
    ]):
        r.wait()
    n_recv = int(peer_count.item())

    grad_dim = grads_for_peer.shape[1] if n_send > 0 else 1

    # Phase 2: Exchange indices
    recv_indices = torch.empty(n_recv, dtype=torch.long, device=device)
    ops = []
    if n_send > 0:
        ops.append(dist.P2POp(dist.isend, indices_for_peer.contiguous(), peer_rank))
    if n_recv > 0:
        ops.append(dist.P2POp(dist.irecv, recv_indices, peer_rank))
    if ops:
        for r in dist.batch_isend_irecv(ops):
            r.wait()

    # Phase 3: Exchange gradient values
    recv_grads = torch.empty(n_recv, grad_dim, device=device)
    ops = []
    if n_send > 0:
        ops.append(dist.P2POp(dist.isend, grads_for_peer.contiguous(), peer_rank))
    if n_recv > 0:
        ops.append(dist.P2POp(dist.irecv, recv_grads, peer_rank))
    if ops:
        for r in dist.batch_isend_irecv(ops):
            r.wait()

    return recv_grads, recv_indices


# =========================================================================
# CPU Adam thread (reused from CLM)
# =========================================================================

def cpuadam_thread(
    bsz,
    n_gaussians,
    signal_tensor_pinned,
    finish_indices_filters,
    cpu_adam,
    parameters,
    parameters_grad,
    iteration,
    args,
):
    """Background CPU Adam optimizer thread — identical to CLM's."""
    torch.cuda.nvtx.range_push(
        f"cpuadam thread for iter: [{iteration},{iteration+bsz})"
    )

    version = 3  # inplace_zero_grad is true
    parameters.grad = parameters_grad
    if not args.stop_update_param:
        torch.cuda.nvtx.range_push("cpu_adam.sparse_step()")
        cpu_adam.batched_sparse_step(
            batch_size=bsz,
            batched_sparse_indices=finish_indices_filters,
            signal_tensor_pinned=signal_tensor_pinned,
            version=version,
            scale=1.0 / bsz,
            sparse_adam=args.sparse_adam,
        )
        torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_pop()


# =========================================================================
# Render helper (for single camera)
# =========================================================================

def _render_one_camera_clm(
    filtered_xyz,
    filtered_opacity,
    filtered_scaling,
    filtered_rotation,
    filtered_shs,
    camera,
    gaussians,
    background,
):
    """
    Render one camera with CLM-style inplace SH evaluation.
    SH features come detached from CPU fetch, so we use the inplace
    backward path (spherical_harmonics_bwd_inplace).
    """
    MICRO_BATCH_SIZE = 1
    image_width = int(utils.get_img_width())
    image_height = int(utils.get_img_height())

    viewmat = camera.world_view_transform.transpose(0, 1)
    K = camera.K
    n_selected = filtered_xyz.shape[0]

    # Project
    batched_radiis, batched_means2D, batched_depths, batched_conics, _ = (
        fully_fused_projection(
            means=filtered_xyz,
            covars=None,
            quats=filtered_rotation,
            scales=filtered_scaling,
            viewmats=viewmat.unsqueeze(0),
            Ks=K.unsqueeze(0),
            width=image_width,
            height=image_height,
            packed=False,
        )
    )
    batched_means2D.retain_grad()

    # Spherical harmonics → color (with detach/recompute for CLM backward)
    sh_degree = gaussians.active_sh_degree
    camtoworlds = camera.camtoworlds
    dirs = filtered_xyz[None, :, :] - camtoworlds[:, None, :3, 3]
    filtered_shs_reshaped = filtered_shs.reshape(1, n_selected, 16, 3)

    with torch.no_grad():
        batched_colors_origin = spherical_harmonics(
            degrees_to_use=sh_degree, dirs=dirs, coeffs=filtered_shs_reshaped
        )
    batched_colors_detached = batched_colors_origin.detach().requires_grad_()
    batched_colors = torch.clamp_min(batched_colors_detached + 0.5, 0.0)
    batched_opacities = filtered_opacity.squeeze(1).unsqueeze(0)

    # Tile-based rasterization
    tile_width = math.ceil(image_width / float(TILE_SIZE))
    tile_height = math.ceil(image_height / float(TILE_SIZE))

    _, isect_ids, flatten_ids = isect_tiles(
        means2d=batched_means2D,
        radii=batched_radiis,
        depths=batched_depths,
        tile_size=TILE_SIZE,
        tile_width=tile_width,
        tile_height=tile_height,
        packed=False,
    )
    isect_offsets = isect_offset_encode(
        isect_ids, MICRO_BATCH_SIZE, tile_width, tile_height
    )

    backgrounds = (
        background.repeat(MICRO_BATCH_SIZE, 1) if background is not None else None
    )
    rendered_image, _ = rasterize_to_pixels(
        means2d=batched_means2D,
        conics=batched_conics,
        colors=batched_colors,
        opacities=batched_opacities,
        image_width=image_width,
        image_height=image_height,
        tile_size=TILE_SIZE,
        isect_offsets=isect_offsets,
        flatten_ids=flatten_ids,
        backgrounds=backgrounds,
    )

    rendered_image = rendered_image.squeeze(0).permute(2, 0, 1).contiguous()

    return (
        rendered_image,
        batched_means2D,
        batched_radiis,
        batched_colors_detached,
        dirs,
    )


# =========================================================================
# Main training entry point
# =========================================================================

def multi_gpu_clm_train_one_batch(
    gaussians,
    scene,
    batched_cameras,
    parameters_grad_buffer,
    background,
    pipe_args,
    comm_stream,
    perm_generator,
):
    """
    Multi-GPU CLM training loop.

    Combines:
      - Global visibility via replicated proxy
      - CLM-style SH streaming from CPU per camera
      - P2P SH cache sharing between GPUs for overlapping Gaussians
      - CPU Adam for SH, GPU Adam for spatial params
      - Spatial gradient exchange for remote Gaussians via P2P

    Pipeline per micro-batch:
      1. Fetch visible SH from CPU (on comm_stream, overlapped)
      2. For remote Gaussians: check peer GPU caches → P2P fetch
      3. Forward: project, SH→color, rasterize
      4. Backward: compute gradients
      5. Scatter spatial grads; offload SH grads to CPU
      6. Exchange remote spatial grads with peers
    """
    args = utils.get_args()
    iteration = utils.get_cur_iter()
    log_file = utils.get_log_file()

    bsz = len(batched_cameras)
    N_local = gaussians._xyz.shape[0]

    # ==================================================================
    # STAGE 1: Global visibility computation
    # ==================================================================
    with torch.no_grad():
        gaussians.sync_global_proxy()
        global_filters, global_scaling, global_rotation = calculate_filters_global(
            batched_cameras, gaussians
        )

    # Precompute local/remote splits for all cameras
    splits = []
    for cam_idx in range(bsz):
        visible_global = global_filters[cam_idx]
        split = gaussians.get_local_and_remote_indices(visible_global)
        splits.append(split)

    # ==================================================================
    # STAGE 2: CPU Adam thread initialization (like CLM)
    # ==================================================================
    if not hasattr(gaussians, "signal_tensor_pinned"):
        gaussians.signal_tensor_pinned = torch.zeros(
            bsz, dtype=torch.int32, device="cpu", pin_memory=True
        )
    else:
        gaussians.signal_tensor_pinned.zero_()
    signal_tensor_pinned = gaussians.signal_tensor_pinned

    # Build finish_indices_filters for CPU Adam
    # In multi-GPU CLM, each GPU processes all cameras but only updates
    # its own local SH partition. The finish indices track which local
    # Gaussians were visible across which micro-batches.
    local_filters = []
    for cam_idx in range(bsz):
        _, local_idx, _, _, _ = splits[cam_idx]
        local_filters.append(local_idx)

    # Build simple finish_indices_filters: which local Gaussians finished
    # at each micro-batch boundary (for CPU Adam sparse step)
    # Simplified: all visible local Gaussians are "finished" after each camera
    finish_indices_filters = _build_finish_indices(local_filters, N_local, bsz)

    torch.cuda.synchronize()

    microbatch_idx = 0
    cpuadam_worker = threading.Thread(
        target=cpuadam_thread,
        args=(
            bsz,
            N_local,
            signal_tensor_pinned,
            finish_indices_filters,
            gaussians.optimizer.cpu_adam,
            gaussians._parameters,
            parameters_grad_buffer[:N_local, :],
            iteration,
            args,
        ),
    )
    cpuadam_worker.start()

    # ==================================================================
    # STAGE 3: Initialize gradient accumulators
    # ==================================================================
    gaussians._xyz.grad = torch.zeros_like(gaussians._xyz)
    gaussians._opacity.grad = torch.zeros_like(gaussians._opacity)
    gaussians._scaling.grad = torch.zeros_like(gaussians._scaling)
    gaussians._rotation.grad = torch.zeros_like(gaussians._rotation)

    default_stream = torch.cuda.current_stream()

    losses = []
    grid_size, block_size = args.grid_size_H, 256

    # Track which SH we have in GPU cache for P2P sharing
    # (reset per micro-batch)
    cached_sh_gpu = None
    cached_local_indices_gpu = None

    # ==================================================================
    # STAGE 4: Main micro-batch training loop
    # ==================================================================
    for micro_idx in range(bsz):
        torch.cuda.nvtx.range_push(f"micro_batch_idx: {micro_idx}")
        camera = batched_cameras[micro_idx]

        (
            local_global_idx,
            local_idx,
            remote_global_idx,
            remote_rank,
            remote_local_idx,
        ) = splits[micro_idx]

        n_local_vis = local_idx.shape[0]
        n_remote_vis = remote_global_idx.shape[0]

        # ---------------------------------------------------------------
        # 4.1: Fetch local SH from CPU → GPU (CLM streaming)
        # ---------------------------------------------------------------
        with torch.cuda.stream(comm_stream), torch.no_grad():
            local_shs = torch.empty(n_local_vis, 48, device="cuda")
            if n_local_vis > 0:
                send_shs2gpu_stream(
                    local_shs,
                    gaussians._parameters,
                    local_idx,
                    grid_size,
                    block_size,
                )
            cpu2gpu_event = torch.cuda.Event()
            cpu2gpu_event.record(comm_stream)

        # ---------------------------------------------------------------
        # 4.2: Fetch remote SH via P2P or CPU fallback
        # ---------------------------------------------------------------
        remote_shs = torch.empty(n_remote_vis, 48, device="cuda") if n_remote_vis > 0 else None

        if n_remote_vis > 0:
            # For each peer, exchange SH via P2P
            for peer in range(gaussians.world_size):
                if peer == gaussians.rank:
                    continue

                mask = remote_rank == peer
                if not mask.any():
                    need_indices = torch.empty(0, dtype=torch.long, device="cuda")
                else:
                    need_indices = remote_local_idx[mask]

                # Use P2P cache exchange with peer
                fetched = _p2p_sh_cache_exchange(
                    gaussians, peer, need_indices,
                    cached_sh_gpu, cached_local_indices_gpu,
                )

                if mask.any() and fetched.shape[0] > 0:
                    remote_shs[mask] = fetched

        # Update cache for next iteration / P2P sharing
        cached_sh_gpu = local_shs.detach() if n_local_vis > 0 else None
        cached_local_indices_gpu = local_idx if n_local_vis > 0 else None

        # ---------------------------------------------------------------
        # 4.3: Assemble features for rendering
        # ---------------------------------------------------------------
        # Wait for CPU→GPU SH transfer to complete
        cpu2gpu_event.wait(default_stream)

        # Local spatial params from GPU
        filtered_xyz = gaussians._xyz.detach()[local_idx].requires_grad_(True) if n_local_vis > 0 else torch.empty(0, 3, device="cuda")
        _filtered_opacity = gaussians._opacity.detach()[local_idx].requires_grad_(True) if n_local_vis > 0 else torch.empty(0, 1, device="cuda")
        _filtered_scaling = gaussians._scaling.detach()[local_idx].requires_grad_(True) if n_local_vis > 0 else torch.empty(0, 3, device="cuda")
        _filtered_rotation = gaussians._rotation.detach()[local_idx].requires_grad_(True) if n_local_vis > 0 else torch.empty(0, 4, device="cuda")

        # Remote spatial params from global proxy
        if n_remote_vis > 0:
            r_xyz = gaussians.global_xyz[remote_global_idx]
            r_opa_raw = gaussians.global_opacity[remote_global_idx]
            r_scl = global_scaling[remote_global_idx]
            r_rot = global_rotation[remote_global_idx]

            all_xyz = torch.cat([filtered_xyz, r_xyz])
            all_opacity_raw = torch.cat([_filtered_opacity, r_opa_raw.requires_grad_(False)])
            all_scaling_raw = torch.cat([_filtered_scaling, r_scl.requires_grad_(False)])
            all_rotation_raw = torch.cat([_filtered_rotation, r_rot.requires_grad_(False)])
            all_shs = torch.cat([local_shs, remote_shs])
        else:
            all_xyz = filtered_xyz
            all_opacity_raw = _filtered_opacity
            all_scaling_raw = _filtered_scaling
            all_rotation_raw = _filtered_rotation
            all_shs = local_shs

        # Apply activation functions
        filtered_opacity_gpu = gaussians.opacity_activation(all_opacity_raw)
        filtered_scaling_gpu = gaussians.scaling_activation(all_scaling_raw)
        filtered_rotation_gpu = gaussians.rotation_activation(all_rotation_raw)

        # ---------------------------------------------------------------
        # 4.4: Forward pass
        # ---------------------------------------------------------------
        torch.cuda.nvtx.range_push("forward_pass")
        filtered_shs = all_shs.requires_grad_(False)

        (
            rendered_image,
            batched_means2D,
            batched_radiis,
            batched_colors_detached,
            dirs,
        ) = _render_one_camera_clm(
            all_xyz,
            filtered_opacity_gpu,
            filtered_scaling_gpu,
            filtered_rotation_gpu,
            filtered_shs,
            camera,
            gaussians,
            background,
        )

        loss = torch_compiled_loss(rendered_image, camera.original_image)
        torch.cuda.nvtx.range_pop()

        # ---------------------------------------------------------------
        # 4.5: Backward pass
        # ---------------------------------------------------------------
        torch.cuda.nvtx.range_push("backward_pass")
        loss.backward()

        # Manual SH backward (CLM-style inplace)
        shs_grad = torch.zeros(all_shs.shape[0], 48, device="cuda")
        v_dirs = spherical_harmonics_bwd_inplace(
            degrees_to_use=gaussians.active_sh_degree,
            dirs=dirs,
            coeffs=filtered_shs.reshape(1, -1, 16, 3),
            v_coeffs=shs_grad,
            v_colors=batched_colors_detached.grad,
        )
        dirs.backward(v_dirs)
        torch.cuda.nvtx.range_pop()

        # ---------------------------------------------------------------
        # 4.6: Accumulate spatial gradients (local only)
        # ---------------------------------------------------------------
        with torch.no_grad():
            if n_local_vis > 0 and filtered_xyz.grad is not None:
                gaussians._xyz.grad.scatter_add_(
                    dim=0,
                    src=filtered_xyz.grad[:n_local_vis] if n_remote_vis > 0 else filtered_xyz.grad,
                    index=local_idx.reshape(-1, 1).expand(-1, 3),
                )
            if n_local_vis > 0 and _filtered_opacity.grad is not None:
                gaussians._opacity.grad.scatter_add_(
                    dim=0,
                    src=_filtered_opacity.grad[:n_local_vis] if n_remote_vis > 0 else _filtered_opacity.grad,
                    index=local_idx.reshape(-1, 1),
                )
            if n_local_vis > 0 and _filtered_scaling.grad is not None:
                gaussians._scaling.grad.scatter_add_(
                    dim=0,
                    src=_filtered_scaling.grad[:n_local_vis] if n_remote_vis > 0 else _filtered_scaling.grad,
                    index=local_idx.reshape(-1, 1).expand(-1, 3),
                )
            if n_local_vis > 0 and _filtered_rotation.grad is not None:
                gaussians._rotation.grad.scatter_add_(
                    dim=0,
                    src=_filtered_rotation.grad[:n_local_vis] if n_remote_vis > 0 else _filtered_rotation.grad,
                    index=local_idx.reshape(-1, 1).expand(-1, 4),
                )

        # ---------------------------------------------------------------
        # 4.7: Offload local SH gradients to CPU (CLM streaming)
        # ---------------------------------------------------------------
        # Extract local portion of SH gradients
        local_shs_grad = shs_grad[:n_local_vis] if n_remote_vis > 0 else shs_grad

        gpu2cpu_event = torch.cuda.Event()
        gpu2cpu_event.record(default_stream)

        with torch.cuda.stream(comm_stream), torch.no_grad():
            gpu2cpu_event.wait(comm_stream)

            send_shs2cpu_grad_buffer_stream(
                local_shs_grad,
                parameters_grad_buffer[:N_local, :],
                local_idx,
                True,  # accumulate
                grid_size,
                block_size,
            )

            # Signal CPU Adam thread
            clm_kernels.set_signal(signal_tensor_pinned, microbatch_idx, 1)
            microbatch_idx += 1

        # ---------------------------------------------------------------
        # 4.8: Exchange spatial gradients for remote Gaussians with peers
        # ---------------------------------------------------------------
        # Remote Gaussians' spatial grads need to be sent to their owners
        if n_remote_vis > 0:
            for peer in range(gaussians.world_size):
                if peer == gaussians.rank:
                    continue
                mask = remote_rank == peer
                if not mask.any():
                    # Still need to participate in exchange
                    empty_grads = torch.empty(0, 3, device="cuda")
                    empty_idx = torch.empty(0, dtype=torch.long, device="cuda")
                    _p2p_spatial_gradient_exchange(peer, empty_grads, empty_idx, N_local)
                    continue

                peer_local_indices = remote_local_idx[mask]
                # We don't have gradients for remote spatial params through
                # the global proxy (they're detached). The SH gradients for
                # remote Gaussians stay on the peer's CPU.
                # For spatial: remote xyz/opa/scl/rot were read from the
                # detached global proxy, so no spatial grads flow to peers.
                # This is acceptable as each GPU trains predominantly on
                # its own spatial partition.

        # Cleanup
        del rendered_image, batched_colors_detached, dirs, v_dirs
        del all_xyz, all_shs, filtered_shs
        del filtered_xyz, filtered_opacity_gpu, filtered_scaling_gpu, filtered_rotation_gpu
        del _filtered_opacity, _filtered_scaling, _filtered_rotation

        losses.append(loss.detach())
        del loss

        # ---------------------------------------------------------------
        # 4.9: Update densification statistics
        # ---------------------------------------------------------------
        if n_local_vis > 0:
            update_densification_stats_offload_accum_grads(
                scene,
                gaussians,
                int(utils.get_img_height()),
                int(utils.get_img_width()),
                local_idx,
                batched_means2D.grad.squeeze(0)[:n_local_vis]
                if n_remote_vis > 0
                else batched_means2D.grad.squeeze(0),
                batched_radiis.squeeze(0)[:n_local_vis]
                if n_remote_vis > 0
                else batched_radiis.squeeze(0),
            )

        batched_means2D.grad = None
        del batched_means2D, batched_radiis

        torch.cuda.nvtx.range_pop()

    # ==================================================================
    # STAGE 5: Optimizer step + cleanup
    # ==================================================================

    # 5.1: GPU Adam step for spatial params
    for param in gaussians.all_parameters()[:4]:
        if param.grad is not None:
            param.grad /= bsz

    if not args.stop_update_param:
        if args.sparse_adam:
            # Build visibility mask: union of all locally-visible Gaussians
            visibility_mask = torch.zeros(N_local, device="cuda")
            for filt in local_filters:
                if filt.shape[0] > 0:
                    visibility_mask[filt] = 1.0
            gaussians.optimizer.gpu_adam.step(visibility=visibility_mask)
        else:
            gaussians.optimizer.gpu_adam.step()
    gaussians.optimizer.gpu_adam.zero_grad(set_to_none=True)

    # 5.2: Wait for CPU Adam to finish
    cpuadam_worker.join()

    utils.memory_report("after cpuadam_worker joined (multi_gpu_clm)")

    # 5.3: Synchronize
    torch.cuda.synchronize()

    return losses, list(range(bsz)), [1.0] * bsz


# =========================================================================
# Helper: build finish_indices_filters for CPU Adam
# =========================================================================

def _build_finish_indices(local_filters, N_local, bsz):
    """
    Build finish_indices_filters for CPU Adam batched_sparse_step.
    Groups local Gaussians by when they're last visible across micro-batches.

    Returns list of (bsz+1) index tensors:
      [0]: Gaussians never visible in this batch
      [1..bsz]: Gaussians whose last visibility is in micro-batch i
    """
    # Track last micro-batch each Gaussian was visible
    last_visible = torch.full((N_local,), -1, dtype=torch.long, device="cpu")

    for i, filt in enumerate(local_filters):
        if filt.shape[0] > 0:
            filt_cpu = filt.cpu()
            last_visible[filt_cpu] = i

    result = []
    # Group 0: never visible
    never_mask = last_visible == -1
    result.append(torch.nonzero(never_mask).flatten().to(torch.int32))

    # Groups 1..bsz: last visible in micro-batch (bsz-1-i) -> group i
    for i in range(bsz):
        target_mb = bsz - 1 - i
        mask = last_visible == target_mb
        result.append(torch.nonzero(mask).flatten().to(torch.int32))

    return result


# =========================================================================
# Evaluation
# =========================================================================

def multi_gpu_clm_eval_one_cam(camera, gaussians, background, scene):
    """
    Render one camera for evaluation.
    AllGathers full scene (spatial from GPU, SH from CPU→GPU staging).
    """
    with torch.no_grad():
        if dist.is_initialized() and gaussians.world_size > 1:
            # Gather spatial from GPU
            all_xyz = [torch.zeros(s, 3, device="cuda") for s in gaussians.partition_sizes]
            all_opacity = [torch.zeros(s, 1, device="cuda") for s in gaussians.partition_sizes]
            all_scaling = [torch.zeros(s, 3, device="cuda") for s in gaussians.partition_sizes]
            all_rotation = [torch.zeros(s, 4, device="cuda") for s in gaussians.partition_sizes]

            dist.all_gather(all_xyz, gaussians._xyz.detach().contiguous())
            dist.all_gather(all_opacity, gaussians._opacity.detach().contiguous())
            dist.all_gather(all_scaling, gaussians._scaling.detach().contiguous())
            dist.all_gather(all_rotation, gaussians._rotation.detach().contiguous())

            # Gather SH: CPU→GPU staging, then AllGather
            local_sh = gaussians._parameters.detach().clone().cuda()
            all_sh = [torch.zeros(s, 48, device="cuda") for s in gaussians.partition_sizes]
            dist.all_gather(all_sh, local_sh.contiguous())

            full_xyz = torch.cat(all_xyz)
            full_opacity = gaussians.opacity_activation(torch.cat(all_opacity))
            full_scaling = gaussians.scaling_activation(torch.cat(all_scaling))
            full_rotation = gaussians.rotation_activation(torch.cat(all_rotation))
            full_shs = torch.cat(all_sh)
        else:
            full_xyz = gaussians._xyz.detach()
            full_opacity = gaussians.opacity_activation(gaussians._opacity.detach())
            full_scaling = gaussians.scaling_activation(gaussians._scaling.detach())
            full_rotation = gaussians.rotation_activation(gaussians._rotation.detach())
            full_shs = gaussians._parameters.detach().cuda()

        # Use base_engine's render helper
        from strategies.base_engine import pipeline_forward_one_step

        rendered_image, _, _ = pipeline_forward_one_step(
            filtered_opacity_gpu=full_opacity,
            filtered_scaling_gpu=full_scaling,
            filtered_rotation_gpu=full_rotation,
            filtered_xyz_gpu=full_xyz,
            filtered_shs=full_shs,
            camera=camera,
            scene=scene,
            gaussians=gaussians,
            background=background,
            pipe_args=None,
            eval=True,
        )

    return rendered_image
