"""
Multi-GPU CLM Engine — M1 Camera Parallelism + M3 Collaborative SH Fetch
=========================================================================

Camera parallelism: each GPU renders the FULL scene for a subset of cameras.
All Gaussians are replicated, so each GPU projects all of them. After all
micro-batches, gradients are AllReduced across ranks so every GPU ends with
identical parameter updates.

M1 Pipeline per batch:
  1. Split cameras: rank k gets cameras [k*C, (k+1)*C) where C = bsz/world_size
  2. Calculate visibility for this rank's cameras (all Gaussians)
  3. For each camera:
     a. Fetch visible SH from CPU pinned memory (CLM streaming)
     b. Forward: project, SH→color, rasterize
     c. Backward: compute gradients
     d. Scatter spatial grads; offload SH grads to CPU buffer
  4. AllReduce spatial gradients on GPU
  5. AllReduce SH gradients via GPU staging (CPU→GPU→AllReduce→GPU→CPU)
  6. AllReduce densification stats so all ranks make identical decisions
  7. GPU Adam step (spatial), CPU Adam step (SH)

M3 Enhancement (--enable_p2p_caching):
  In step 3a, instead of each GPU independently fetching all its visible SH
  from CPU, both GPUs coordinate:
    i.   Exchange visibility indices via AllGather
    ii.  Compute union of visible Gaussians
    iii. Split the union — each GPU fetches ~half from CPU
    iv.  AllGather fetched SH across GPUs
    v.   Each GPU extracts its needed subset from the full union
  Reduces per-GPU CPU→GPU bandwidth by ~50% at the cost of GPU↔GPU exchange.
"""

import math
import threading

import torch
import torch.distributed as dist

import clm_kernels
from clm_kernels import (
    send_shs2gpu_stream,
    send_shs2cpu_grad_buffer_stream,
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
# Render helper (single camera, CLM-style inplace SH backward)
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

    # Spherical harmonics → color
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
# M3: Collaborative SH fetch (P2P caching)
# =========================================================================

def _collaborative_sh_fetch(
    gaussians, this_filter, n_vis, comm_stream, grid_size, block_size,
    rank, world_size,
):
    """
    M3: Both GPUs exchange visibility, compute the union of needed Gaussians,
    split the CPU→GPU fetch, and AllGather the results.  Each GPU does ~50%
    of the CPU transfer; the other half arrives via fast GPU↔GPU AllGather.

    Returns:
        local_shs: (n_vis, 48) SH features for THIS GPU's visible Gaussians,
                   ordered to match ``this_filter``.
    """
    device = this_filter.device
    default_stream = torch.cuda.current_stream()

    # ------------------------------------------------------------------
    # 1. Exchange visibility filter sizes + indices via AllGather
    # ------------------------------------------------------------------
    local_n = torch.tensor([n_vis], dtype=torch.int64, device=device)
    all_sizes = [torch.zeros(1, dtype=torch.int64, device=device)
                 for _ in range(world_size)]
    dist.all_gather(all_sizes, local_n)
    sizes = [int(s.item()) for s in all_sizes]

    max_n = max(sizes)
    padded_filter = torch.zeros(max_n, dtype=this_filter.dtype, device=device)
    padded_filter[:n_vis] = this_filter
    all_padded = [torch.empty(max_n, dtype=this_filter.dtype, device=device)
                  for _ in range(world_size)]
    dist.all_gather(all_padded, padded_filter)
    all_filters = [all_padded[r][:sizes[r]] for r in range(world_size)]

    # ------------------------------------------------------------------
    # 2. Compute sorted union of visible indices across all GPUs
    # ------------------------------------------------------------------
    combined = torch.cat(all_filters)
    union_indices, inverse_map = torch.unique(
        combined, sorted=True, return_inverse=True,
    )
    n_union = union_indices.shape[0]

    # ------------------------------------------------------------------
    # 3. Split the union evenly across GPUs for cooperative CPU fetch
    # ------------------------------------------------------------------
    chunk_size = (n_union + world_size - 1) // world_size  # ceil division
    my_start = rank * chunk_size
    my_end = min(my_start + chunk_size, n_union)
    my_portion = union_indices[my_start:my_end]
    my_portion_len = my_end - my_start

    # ------------------------------------------------------------------
    # 4. Each GPU fetches its portion of SH from CPU (CLM kernel)
    # ------------------------------------------------------------------
    my_portion_shs = torch.empty(my_portion_len, 48, device=device)
    with torch.cuda.stream(comm_stream), torch.no_grad():
        send_shs2gpu_stream(
            my_portion_shs,
            gaussians._parameters,
            my_portion,
            grid_size,
            block_size,
        )
        fetch_event = torch.cuda.Event()
        fetch_event.record(comm_stream)
    fetch_event.wait(default_stream)

    # ------------------------------------------------------------------
    # 5. AllGather fetched SH across GPUs
    # ------------------------------------------------------------------
    # Pad to uniform chunk_size for AllGather
    my_padded_shs = torch.zeros(chunk_size, 48, device=device)
    my_padded_shs[:my_portion_len] = my_portion_shs
    all_shs = [torch.empty(chunk_size, 48, device=device)
               for _ in range(world_size)]
    dist.all_gather(all_shs, my_padded_shs)

    # Trim padding and concatenate to reconstruct full union SH
    trimmed = []
    for r in range(world_size):
        r_start = r * chunk_size
        r_end = min(r_start + chunk_size, n_union)
        trimmed.append(all_shs[r][:r_end - r_start])
    union_shs = torch.cat(trimmed, dim=0)  # (n_union, 48)

    # ------------------------------------------------------------------
    # 6. Extract this GPU's needed SH from the union
    # ------------------------------------------------------------------
    # inverse_map corresponds to combined = cat(all_filters)
    # This rank's entries are at offset sum(sizes[:rank]) .. +n_vis
    offset = sum(sizes[:rank])
    my_inverse = inverse_map[offset:offset + n_vis]
    local_shs = union_shs[my_inverse]  # (n_vis, 48)

    return local_shs


# =========================================================================
# Main training entry point — M1 Camera Parallelism
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
    M1 Camera Parallelism training loop.

    Each GPU renders the FULL scene for its subset of cameras.
    After all micro-batches, gradients are AllReduced for correctness.
    """
    args = utils.get_args()
    iteration = utils.get_cur_iter()
    log_file = utils.get_log_file()

    bsz = len(batched_cameras)
    N = gaussians._xyz.shape[0]
    rank = gaussians.rank if hasattr(gaussians, "rank") else 0
    world_size = gaussians.world_size if hasattr(gaussians, "world_size") else 1

    # ==================================================================
    # STAGE 1: Split cameras across GPUs
    # ==================================================================
    assert bsz % world_size == 0, (
        f"bsz={bsz} must be divisible by world_size={world_size}"
    )
    cams_per_gpu = bsz // world_size
    my_start = rank * cams_per_gpu
    my_end = my_start + cams_per_gpu
    my_cameras = batched_cameras[my_start:my_end]

    # ==================================================================
    # STAGE 2: Calculate visibility for MY cameras
    # ==================================================================
    with torch.no_grad():
        xyz_gpu = gaussians.get_xyz
        opacity_gpu = gaussians.get_opacity
        scaling_gpu = gaussians.get_scaling
        rotation_gpu = gaussians.get_rotation

        filters, _, _ = calculate_filters(
            my_cameras, xyz_gpu, opacity_gpu, scaling_gpu, rotation_gpu
        )
        del opacity_gpu, scaling_gpu, rotation_gpu

    # ==================================================================
    # STAGE 3: Initialize gradient accumulators
    # ==================================================================
    gaussians._xyz.grad = torch.zeros_like(gaussians._xyz)
    gaussians._opacity.grad = torch.zeros_like(gaussians._opacity)
    gaussians._scaling.grad = torch.zeros_like(gaussians._scaling)
    gaussians._rotation.grad = torch.zeros_like(gaussians._rotation)

    default_stream = torch.cuda.current_stream()
    grid_size, block_size = args.grid_size_H, 256
    losses = []

    # ==================================================================
    # STAGE 4: Main micro-batch loop (my cameras only)
    # ==================================================================
    for micro_idx in range(cams_per_gpu):
        torch.cuda.nvtx.range_push(f"micro_batch_idx: {micro_idx}")
        camera = my_cameras[micro_idx]
        this_filter = filters[micro_idx]
        n_vis = this_filter.shape[0]

        if n_vis == 0:
            # M3: still must participate in AllGather collectives to avoid deadlock
            if args.enable_p2p_caching and world_size > 1:
                _collaborative_sh_fetch(
                    gaussians, this_filter, n_vis, comm_stream,
                    grid_size, block_size, rank, world_size,
                )
            torch.cuda.nvtx.range_pop()
            continue

        # ---------------------------------------------------------------
        # 4.1: Fetch visible SH from CPU → GPU
        # ---------------------------------------------------------------
        if args.enable_p2p_caching and world_size > 1:
            # M3: Collaborative fetch — split CPU work, AllGather results
            local_shs = _collaborative_sh_fetch(
                gaussians, this_filter, n_vis, comm_stream,
                grid_size, block_size, rank, world_size,
            )
        else:
            # M1: Independent fetch — each GPU fetches its own visible SH
            with torch.cuda.stream(comm_stream), torch.no_grad():
                local_shs = torch.empty(n_vis, 48, device="cuda")
                send_shs2gpu_stream(
                    local_shs,
                    gaussians._parameters,
                    this_filter,
                    grid_size,
                    block_size,
                )
                cpu2gpu_event = torch.cuda.Event()
                cpu2gpu_event.record(comm_stream)
            cpu2gpu_event.wait(default_stream)

        filtered_xyz = gaussians._xyz.detach()[this_filter].requires_grad_(True)
        _filtered_opacity = gaussians._opacity.detach()[this_filter].requires_grad_(True)
        _filtered_scaling = gaussians._scaling.detach()[this_filter].requires_grad_(True)
        _filtered_rotation = gaussians._rotation.detach()[this_filter].requires_grad_(True)

        filtered_opacity_gpu = gaussians.opacity_activation(_filtered_opacity)
        filtered_scaling_gpu = gaussians.scaling_activation(_filtered_scaling)
        filtered_rotation_gpu = gaussians.rotation_activation(_filtered_rotation)

        # ---------------------------------------------------------------
        # 4.3: Forward pass
        # ---------------------------------------------------------------
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
        shs_grad = torch.zeros(n_vis, 48, device="cuda")
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
                    index=this_filter.reshape(-1, 1).expand(-1, 3),
                )
            if _filtered_opacity.grad is not None:
                gaussians._opacity.grad.scatter_add_(
                    dim=0, src=_filtered_opacity.grad,
                    index=this_filter.reshape(-1, 1),
                )
            if _filtered_scaling.grad is not None:
                gaussians._scaling.grad.scatter_add_(
                    dim=0, src=_filtered_scaling.grad,
                    index=this_filter.reshape(-1, 1).expand(-1, 3),
                )
            if _filtered_rotation.grad is not None:
                gaussians._rotation.grad.scatter_add_(
                    dim=0, src=_filtered_rotation.grad,
                    index=this_filter.reshape(-1, 1).expand(-1, 4),
                )

        # ---------------------------------------------------------------
        # 4.6: Offload SH gradients to CPU buffer (accumulate)
        # ---------------------------------------------------------------
        gpu2cpu_event = torch.cuda.Event()
        gpu2cpu_event.record(default_stream)

        with torch.cuda.stream(comm_stream), torch.no_grad():
            gpu2cpu_event.wait(comm_stream)
            send_shs2cpu_grad_buffer_stream(
                shs_grad,
                parameters_grad_buffer[:N, :],
                this_filter,
                True,  # accumulate
                grid_size,
                block_size,
            )

        # ---------------------------------------------------------------
        # 4.7: Densification stats + cleanup
        # ---------------------------------------------------------------
        update_densification_stats_offload_accum_grads(
            scene,
            gaussians,
            int(utils.get_img_height()),
            int(utils.get_img_width()),
            this_filter,
            batched_means2D.grad.squeeze(0),
            batched_radiis.squeeze(0),
        )

        losses.append(loss.detach())

        del rendered_image, batched_colors_detached, dirs, v_dirs, shs_grad
        del filtered_xyz, filtered_opacity_gpu, filtered_scaling_gpu, filtered_rotation_gpu
        del _filtered_opacity, _filtered_scaling, _filtered_rotation
        del loss
        batched_means2D.grad = None
        del batched_means2D, batched_radiis, local_shs

        torch.cuda.nvtx.range_pop()

    # ==================================================================
    # STAGE 5: Synchronize and AllReduce gradients
    # ==================================================================
    # Wait for all SH grad offloads to complete
    torch.cuda.current_stream().wait_stream(comm_stream)
    torch.cuda.synchronize()

    # 5.1: AllReduce spatial gradients on GPU
    if world_size > 1:
        dist.all_reduce(gaussians._xyz.grad)
        dist.all_reduce(gaussians._opacity.grad)
        dist.all_reduce(gaussians._scaling.grad)
        dist.all_reduce(gaussians._rotation.grad)

    # Scale spatial grads by 1/bsz
    for param in gaussians.all_parameters()[:4]:
        if param.grad is not None:
            param.grad /= bsz

    # 5.2: AllReduce SH gradients (CPU → GPU → AllReduce → GPU → CPU)
    if world_size > 1:
        sh_grad_staging = parameters_grad_buffer[:N].cuda()
        dist.all_reduce(sh_grad_staging)
        parameters_grad_buffer[:N].copy_(sh_grad_staging.cpu())
        del sh_grad_staging
        torch.cuda.empty_cache()

    # 5.3: AllReduce densification stats so all ranks make identical decisions
    if world_size > 1:
        dist.all_reduce(gaussians.xyz_gradient_accum)
        dist.all_reduce(gaussians.denom)
        # max_radii2D needs element-wise max
        dist.all_reduce(gaussians.max_radii2D, op=dist.ReduceOp.MAX)

    # ==================================================================
    # STAGE 6: Optimizer step
    # ==================================================================

    # 6.1: GPU Adam for spatial params
    if not args.stop_update_param:
        if args.sparse_adam:
            # Build global visibility mask (union across all GPUs)
            visibility_mask = torch.zeros(N, device="cuda", dtype=torch.bool)
            for filt in filters:
                if filt.shape[0] > 0:
                    visibility_mask[filt] = True
            if world_size > 1:
                vis_int = visibility_mask.int()
                dist.all_reduce(vis_int)
                visibility_mask = vis_int > 0
            gaussians.optimizer.gpu_adam.step(visibility=visibility_mask.float())
        else:
            gaussians.optimizer.gpu_adam.step()
    gaussians.optimizer.gpu_adam.zero_grad(set_to_none=True)

    # 6.2: CPU Adam for SH
    # Compute union of visible indices for sparse step
    visible_set = torch.zeros(N, dtype=torch.bool, device="cuda")
    for filt in filters:
        if filt.shape[0] > 0:
            visible_set[filt] = True
    if world_size > 1:
        vis_int = visible_set.int()
        dist.all_reduce(vis_int)
        visible_set = vis_int > 0
    visible_indices = torch.nonzero(visible_set).flatten().to(torch.int32).cpu()

    if not args.stop_update_param and visible_indices.shape[0] > 0:
        gaussians._parameters.grad = parameters_grad_buffer[:N]
        gaussians.optimizer.cpu_adam.sparse_adam_inc_step()
        gaussians.optimizer.cpu_adam.sparse_step(
            sparse_indices=visible_indices,
            version=3,  # inplace_zero_grad
            scale=1.0 / bsz,
        )

    mode_tag = "M1+M3" if args.enable_p2p_caching else "M1"
    utils.memory_report(f"after optimizer step (multi_gpu_clm {mode_tag})")
    torch.cuda.synchronize()

    # ==================================================================
    # STAGE 7: Gather losses across ranks for logging
    # ==================================================================
    if world_size > 1 and len(losses) > 0:
        local_losses = torch.stack(losses)
        all_loss_list = [
            torch.zeros_like(local_losses) for _ in range(world_size)
        ]
        dist.all_gather(all_loss_list, local_losses)
        all_losses = torch.cat(all_loss_list)
        losses = list(all_losses.unbind())

    return losses, list(range(bsz)), [1.0] * bsz


# =========================================================================
# Evaluation
# =========================================================================

def multi_gpu_clm_eval_one_cam(camera, gaussians, background, scene):
    """
    Render one camera for evaluation.
    Since all Gaussians are replicated, just render directly.
    """
    with torch.no_grad():
        full_xyz = gaussians._xyz.detach()
        full_opacity = gaussians.opacity_activation(gaussians._opacity.detach())
        full_scaling = gaussians.scaling_activation(gaussians._scaling.detach())
        full_rotation = gaussians.rotation_activation(gaussians._rotation.detach())
        full_shs = gaussians._parameters.detach().cuda()

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
