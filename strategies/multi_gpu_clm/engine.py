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


def calculate_filters_local(batched_cameras, gaussians):
    """
    Compute local-only visibility: each GPU projects its OWN partition.
    No AllGather needed — avoids the memory cost of global projection.
    Returns per-camera LOCAL Gaussian indices.
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

        # Project LOCAL Gaussians only — no AllGather needed
        proj_results = fully_fused_projection(
            means=gaussians._xyz.detach(),
            covars=None,
            quats=gaussians.get_rotation,
            scales=gaussians.get_scaling,
            viewmats=batched_viewmats,
            Ks=batched_Ks,
            radius_clip=args.radius_clip,
            width=image_width,
            height=image_height,
            packed=True,
        )

        (camera_ids, gaussian_ids, _, _, _, _, _) = proj_results
        n_cams = len(batched_cameras)

        if camera_ids.numel() == 0:
            return [torch.empty(0, dtype=torch.long, device="cuda")] * n_cams

        output, counts = torch.unique_consecutive(camera_ids, return_counts=True)
        counts_cpu = counts.cpu().numpy().tolist()
        id_splits = torch.split(gaussian_ids, counts_cpu)

        # Map camera_id → split index
        cam_to_split = {}
        for i, cid in enumerate(output.cpu().numpy().tolist()):
            cam_to_split[int(cid)] = i

        result = []
        for cam_idx in range(n_cams):
            if cam_idx in cam_to_split:
                result.append(id_splits[cam_to_split[cam_idx]])
            else:
                result.append(torch.empty(0, dtype=torch.long, device="cuda"))

    return result


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

    # Phase 3: Gather SH for peer from our cache (or CPU fallback) and exchange
    if n_peer_needs > 0:
        # _gather_sh_for_indices handles cache=None by falling back to CPU
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
    Multi-GPU CLM training loop — local-only rendering.

    Each GPU renders only its own spatial partition's visible Gaussians.
    This avoids the VRAM overhead of gathering remote features while
    preserving CLM's CPU↔GPU SH streaming for memory efficiency.

    Pipeline per micro-batch:
      1. Project local Gaussians → determine visibility per camera
      2. Fetch visible local SH from CPU (on comm_stream, overlapped)
      3. Forward: project, SH→color, rasterize (local Gaussians only)
      4. Backward: compute gradients
      5. Scatter spatial grads; offload SH grads to CPU
    """
    args = utils.get_args()
    iteration = utils.get_cur_iter()
    log_file = utils.get_log_file()

    bsz = len(batched_cameras)
    N_local = gaussians._xyz.shape[0]
    rank = gaussians.rank if hasattr(gaussians, 'rank') else 0

    def _mem(tag):
        torch.cuda.synchronize()
        a = torch.cuda.memory_allocated() / 1024**3
        r = torch.cuda.memory_reserved() / 1024**3
        if rank == 0:
            print(f"[Rank {rank}] MEM {tag}: alloc={a:.2f}G reserved={r:.2f}G", flush=True)

    _mem("start_train")

    # ==================================================================
    # STAGE 1: Local visibility computation (no AllGather needed)
    # ==================================================================
    with torch.no_grad():
        _mem("before_calc_filters")
        local_visible_per_cam = calculate_filters_local(batched_cameras, gaussians)
        _mem("after_calc_filters")

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
    local_filters = list(local_visible_per_cam)

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

    _mem("after_grad_init")
    torch.cuda.empty_cache()
    _mem("after_empty_cache")

    default_stream = torch.cuda.current_stream()

    losses = []
    grid_size, block_size = args.grid_size_H, 256

    # ==================================================================
    # STAGE 4: Main micro-batch training loop (local-only rendering)
    # ==================================================================
    for micro_idx in range(bsz):
        torch.cuda.nvtx.range_push(f"micro_batch_idx: {micro_idx}")
        camera = batched_cameras[micro_idx]

        local_idx = local_visible_per_cam[micro_idx]
        n_local_vis = local_idx.shape[0]

        if micro_idx == 0:
            _mem(f"cam0_start: n_local={n_local_vis}")

        # Skip cameras with no visible local Gaussians
        if n_local_vis == 0:
            with torch.cuda.stream(comm_stream):
                clm_kernels.set_signal(signal_tensor_pinned, microbatch_idx, 1)
                microbatch_idx += 1
            torch.cuda.nvtx.range_pop()
            continue

        # ---------------------------------------------------------------
        # 4.1: Fetch local SH from CPU → GPU (CLM streaming)
        # ---------------------------------------------------------------
        with torch.cuda.stream(comm_stream), torch.no_grad():
            local_shs = torch.empty(n_local_vis, 48, device="cuda")
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
        # 4.2: Assemble local features for rendering
        # ---------------------------------------------------------------
        cpu2gpu_event.wait(default_stream)
        if micro_idx == 0:
            _mem("cam0_after_sh_fetch")

        filtered_xyz = gaussians._xyz.detach()[local_idx].requires_grad_(True)
        _filtered_opacity = gaussians._opacity.detach()[local_idx].requires_grad_(True)
        _filtered_scaling = gaussians._scaling.detach()[local_idx].requires_grad_(True)
        _filtered_rotation = gaussians._rotation.detach()[local_idx].requires_grad_(True)

        filtered_opacity_gpu = gaussians.opacity_activation(_filtered_opacity)
        filtered_scaling_gpu = gaussians.scaling_activation(_filtered_scaling)
        filtered_rotation_gpu = gaussians.rotation_activation(_filtered_rotation)

        # ---------------------------------------------------------------
        # 4.3: Forward pass
        # ---------------------------------------------------------------
        if micro_idx == 0:
            _mem(f"cam0_before_render: n_total={n_local_vis}")

        torch.cuda.nvtx.range_push("forward_pass")

        (
            rendered_image,
            batched_means2D,
            batched_radiis,
            batched_colors_detached,
            dirs,
        ) = _render_one_camera_clm(
            filtered_xyz,
            filtered_opacity_gpu,
            filtered_scaling_gpu,
            filtered_rotation_gpu,
            local_shs,
            camera,
            gaussians,
            background,
        )

        loss = torch_compiled_loss(rendered_image, camera.original_image)
        torch.cuda.nvtx.range_pop()

        # ---------------------------------------------------------------
        # 4.4: Backward pass
        # ---------------------------------------------------------------
        torch.cuda.nvtx.range_push("backward_pass")
        loss.backward()

        # Manual SH backward (CLM-style inplace)
        shs_grad = torch.zeros(n_local_vis, 48, device="cuda")
        v_dirs = spherical_harmonics_bwd_inplace(
            degrees_to_use=gaussians.active_sh_degree,
            dirs=dirs,
            coeffs=local_shs.reshape(1, -1, 16, 3),
            v_coeffs=shs_grad,
            v_colors=batched_colors_detached.grad,
        )
        dirs.backward(v_dirs)
        torch.cuda.nvtx.range_pop()

        # ---------------------------------------------------------------
        # 4.5: Accumulate spatial gradients
        # ---------------------------------------------------------------
        with torch.no_grad():
            if filtered_xyz.grad is not None:
                gaussians._xyz.grad.scatter_add_(
                    dim=0, src=filtered_xyz.grad,
                    index=local_idx.reshape(-1, 1).expand(-1, 3),
                )
            if _filtered_opacity.grad is not None:
                gaussians._opacity.grad.scatter_add_(
                    dim=0, src=_filtered_opacity.grad,
                    index=local_idx.reshape(-1, 1),
                )
            if _filtered_scaling.grad is not None:
                gaussians._scaling.grad.scatter_add_(
                    dim=0, src=_filtered_scaling.grad,
                    index=local_idx.reshape(-1, 1).expand(-1, 3),
                )
            if _filtered_rotation.grad is not None:
                gaussians._rotation.grad.scatter_add_(
                    dim=0, src=_filtered_rotation.grad,
                    index=local_idx.reshape(-1, 1).expand(-1, 4),
                )

        # ---------------------------------------------------------------
        # 4.6: Offload local SH gradients to CPU (CLM streaming)
        # ---------------------------------------------------------------
        gpu2cpu_event = torch.cuda.Event()
        gpu2cpu_event.record(default_stream)

        with torch.cuda.stream(comm_stream), torch.no_grad():
            gpu2cpu_event.wait(comm_stream)

            send_shs2cpu_grad_buffer_stream(
                shs_grad,
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
        # 4.7: Cleanup and bookkeeping
        # ---------------------------------------------------------------
        del rendered_image, batched_colors_detached, dirs, v_dirs
        del filtered_xyz, filtered_opacity_gpu, filtered_scaling_gpu, filtered_rotation_gpu
        del _filtered_opacity, _filtered_scaling, _filtered_rotation

        losses.append(loss.detach())
        del loss

        # Update densification statistics
        update_densification_stats_offload_accum_grads(
            scene,
            gaussians,
            int(utils.get_img_height()),
            int(utils.get_img_width()),
            local_idx,
            batched_means2D.grad.squeeze(0),
            batched_radiis.squeeze(0),
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
