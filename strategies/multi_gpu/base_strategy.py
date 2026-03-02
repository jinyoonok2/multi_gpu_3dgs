"""
BaseMultiGPUStrategy
====================

Abstract base class shared by all multi-GPU training strategies.
Contains:
  - calculate_filters_global   (visibility using global proxy)
  - render_one_camera          (gsplat projection → rasterize)
  - _assemble_local_features   (gather own-GPU visible Gaussians)
  - train_one_batch            (shared training loop, calls assemble_features)
  - eval_one_cam               (AllGather full scene for evaluation)

Subclasses only need to implement one method:
  assemble_features(gaussians, local_idx, remote_global_idx,
                    remote_rank, remote_local_idx,
                    global_scaling, global_rotation)
  → returns (fxyz, fopa, fscl, frot, fshs, n_local)
"""

import math
from abc import ABC, abstractmethod

import torch
import torch.distributed as dist

import utils.general_utils as utils
from gsplat import (
    fully_fused_projection,
    isect_offset_encode,
    isect_tiles,
    rasterize_to_pixels,
    spherical_harmonics,
)
from strategies.base_engine import TILE_SIZE, torch_compiled_loss


# =========================================================================
# Visibility computation using global proxy
# =========================================================================

def calculate_filters_global(batched_cameras, gaussians):
    """
    Compute visibility using the lightweight global proxy (replicated xyz/opacity).
    Also gathers global scaling/rotation (needed for projection) and caches them.
    Returns per-camera lists of GLOBAL Gaussian indices that are visible,
    plus the gathered global scaling/rotation tensors.
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
            all_scaling = [torch.zeros(s, 3, device="cuda") for s in gaussians.partition_sizes]
            all_rotation = [torch.zeros(s, 4, device="cuda") for s in gaussians.partition_sizes]
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
# Single camera render (shared by all strategies, train + eval)
# =========================================================================

def render_one_camera(
    filtered_xyz,
    filtered_opacity,
    filtered_scaling,
    filtered_rotation,
    filtered_shs,
    camera,
    gaussians,
    background,
    eval_mode=False,
):
    """Render one camera view from the filtered (visible) Gaussians."""
    MICRO_BATCH_SIZE = 1
    image_width = int(utils.get_img_width())
    image_height = int(utils.get_img_height())

    viewmat = camera.world_view_transform.transpose(0, 1)
    K = camera.K if hasattr(camera, "K") else camera.create_k_on_gpu()
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

    if not eval_mode:
        batched_means2D.retain_grad()

    # Spherical harmonics → color
    sh_degree = gaussians.active_sh_degree
    camtoworlds = (
        camera.camtoworlds
        if hasattr(camera, "camtoworlds")
        else torch.inverse(viewmat.unsqueeze(0))
    )
    dirs = filtered_xyz[None, :, :] - camtoworlds[:, None, :3, 3]
    filtered_shs_reshaped = filtered_shs.reshape(1, n_selected, 16, 3)
    batched_colors = spherical_harmonics(
        degrees_to_use=sh_degree, dirs=dirs, coeffs=filtered_shs_reshaped
    )
    batched_colors = torch.clamp_min(batched_colors + 0.5, 0.0)
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
    return rendered_image, batched_means2D, batched_radiis


# =========================================================================
# Shared local feature assembly (used by both M3 and M1+M3)
# =========================================================================

def assemble_local_features(gaussians, local_idx):
    """Gather locally-owned visible features from own VRAM."""
    local_xyz = gaussians._xyz[local_idx]
    local_opacity = gaussians.opacity_activation(gaussians._opacity[local_idx])
    local_scaling = gaussians.scaling_activation(gaussians._scaling[local_idx])
    local_rotation = gaussians.rotation_activation(gaussians._rotation[local_idx])
    local_shs = gaussians._parameters[local_idx]
    return local_xyz, local_opacity, local_scaling, local_rotation, local_shs


# =========================================================================
# Abstract base strategy
# =========================================================================

class BaseMultiGPUStrategy(ABC):
    """
    Abstract base for multi-GPU training strategies.

    Subclasses implement assemble_features() to define how remote
    Gaussians are fetched (AllGather vs P2P), while the training loop,
    rendering, and evaluation are shared here.
    """

    @abstractmethod
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
        """
        Assemble the full set of visible Gaussian features for one camera.

        Args:
            gaussians:          GaussianModelMultiGPU instance
            local_idx:          indices into this GPU's own partition
            remote_global_idx:  global indices of remotely-owned visible Gaussians
            remote_rank:        per-element owner rank for remote_global_idx
            remote_local_idx:   per-element local index on the owning rank
            global_scaling:     (N_global, 3) gathered scaling tensor
            global_rotation:    (N_global, 4) gathered rotation tensor

        Returns:
            fxyz:    (n_vis, 3)   positions
            fopa:    (n_vis, 1)   opacities
            fscl:    (n_vis, 3)   scales
            frot:    (n_vis, 4)   rotations
            fshs:    (n_vis, 48)  SH features
            n_local: int          number of locally-owned Gaussians in the batch
        """
        ...

    # ------------------------------------------------------------------
    # Main training loop (shared across all strategies)
    # ------------------------------------------------------------------

    def train_one_batch(
        self,
        gaussians,
        scene,
        batched_cameras,
        background,
        pipe_args,
    ):
        """
        Train one batch.  Calls self.assemble_features() per camera so each
        strategy only needs to implement that single method.
        """
        iteration = utils.get_cur_iter()
        bsz = len(batched_cameras)

        # Stage 1: Visibility using global proxy
        with torch.no_grad():
            gaussians.sync_global_proxy()
            global_filters, global_scaling, global_rotation = calculate_filters_global(
                batched_cameras, gaussians,
            )

        # Stage 2: Precompute local/remote splits for all cameras
        splits = []
        for cam_idx in range(bsz):
            visible_global = global_filters[cam_idx]
            split = gaussians.get_local_and_remote_indices(visible_global)
            splits.append(split)

        # Stage 3: Render + loss + backward per camera
        losses = []
        for cam_idx in range(bsz):
            camera = batched_cameras[cam_idx]
            local_global_idx, local_idx, remote_global_idx, remote_rank, remote_local_idx = splits[cam_idx]

            fxyz, fopa, fscl, frot, fshs, n_local = self.assemble_features(
                gaussians,
                local_idx,
                remote_global_idx,
                remote_rank,
                remote_local_idx,
                global_scaling,
                global_rotation,
            )

            rendered_image, means2D, radiis = render_one_camera(
                fxyz, fopa, fscl, frot, fshs,
                camera, gaussians, background, eval_mode=False,
            )

            gt_image = camera.original_image.cuda()
            loss = torch_compiled_loss(rendered_image, gt_image)
            loss = loss / bsz
            losses.append(loss.detach())

            loss.backward()

            del rendered_image, means2D, radiis, loss
            del fxyz, fopa, fscl, frot, fshs

        # Stage 4: Optimizer step
        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none=False)

        torch.cuda.synchronize()
        return losses, list(range(bsz)), [1.0] * bsz

    # ------------------------------------------------------------------
    # Evaluation (always AllGather; called rarely — no perf concern)
    # ------------------------------------------------------------------

    def eval_one_cam(self, camera, gaussians, background, scene):
        """
        Render one camera for evaluation using the full global Gaussian set.
        Always uses AllGather regardless of which strategy is active.
        """
        with torch.no_grad():
            if dist.is_initialized() and gaussians.world_size > 1:
                all_xyz     = [torch.zeros(s, 3,  device="cuda") for s in gaussians.partition_sizes]
                all_opacity = [torch.zeros(s, 1,  device="cuda") for s in gaussians.partition_sizes]
                all_scaling = [torch.zeros(s, 3,  device="cuda") for s in gaussians.partition_sizes]
                all_rotation= [torch.zeros(s, 4,  device="cuda") for s in gaussians.partition_sizes]
                all_params  = [torch.zeros(s, 48, device="cuda") for s in gaussians.partition_sizes]

                dist.all_gather(all_xyz,      gaussians._xyz.detach().contiguous())
                dist.all_gather(all_opacity,  gaussians._opacity.detach().contiguous())
                dist.all_gather(all_scaling,  gaussians._scaling.detach().contiguous())
                dist.all_gather(all_rotation, gaussians._rotation.detach().contiguous())
                dist.all_gather(all_params,   gaussians._parameters.detach().contiguous())

                full_xyz      = torch.cat(all_xyz)
                full_opacity  = gaussians.opacity_activation(torch.cat(all_opacity))
                full_scaling  = gaussians.scaling_activation(torch.cat(all_scaling))
                full_rotation = gaussians.rotation_activation(torch.cat(all_rotation))
                full_shs      = torch.cat(all_params)
            else:
                full_xyz      = gaussians._xyz.detach()
                full_opacity  = gaussians.opacity_activation(gaussians._opacity.detach())
                full_scaling  = gaussians.scaling_activation(gaussians._scaling.detach())
                full_rotation = gaussians.rotation_activation(gaussians._rotation.detach())
                full_shs      = gaussians._parameters.detach()

            rendered_image, _, _ = render_one_camera(
                full_xyz, full_opacity, full_scaling, full_rotation, full_shs,
                camera, gaussians, background, eval_mode=True,
            )

        return rendered_image
