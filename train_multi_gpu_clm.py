#
# Standalone training script for multi_gpu_clm strategy.
#
# Camera parallelism: every GPU holds ALL Gaussians (replicated).
# Each GPU renders a subset of cameras per batch, then AllReduces gradients.
#
# M1: Basic camera parallelism with AllReduced gradients.
#
# Usage:
#   torchrun --nproc_per_node=2 train_multi_gpu_clm.py -s <data> -m <output> --bsz 8 --eval
#

import os
import sys
import json
import gc
import psutil

import torch
import torch.multiprocessing
import torch.distributed as dist
from torch.cuda import nvtx
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision

from argparse import ArgumentParser
from arguments import (
    AuxiliaryParams,
    ModelParams,
    PipelineParams,
    OptimizationParams,
    BenchmarkParams,
    DebugParams,
    print_all_args,
    init_args,
)

from scene import Scene, OffloadSceneDataset
from strategies.multi_gpu_clm import (
    GaussianModelMultiGPUCLM,
    multi_gpu_clm_train_one_batch,
    multi_gpu_clm_eval_one_cam,
)

from utils.general_utils import safe_state, prepare_output_and_logger
import utils.general_utils as utils
from utils.timer import Timer, End2endTimer
from utils.image_utils import psnr
from utils.loss_utils import l1_loss

from densification import gsplat_densification


# ============================================================================
# Training
# ============================================================================

def training(dataset_args, opt_args, pipe_args, args, log_file):
    """
    Multi-GPU CLM training loop (M1: Camera Parallelism).
    All Gaussians replicated on every GPU. Each GPU renders a subset of cameras.
    """

    # ----------------------------------------------------------------
    # 1. Initialize distributed process group
    # ----------------------------------------------------------------
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    args.gpu = local_rank
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    gc.disable()
    torch.cuda.set_device(args.gpu)

    timers = Timer(args)
    utils.set_timers(timers)
    prepare_output_and_logger(dataset_args)
    utils.log_cpu_memory_usage("at the beginning of training")
    start_from_this_iteration = 1

    if args.sharing_strategy != "default":
        torch.multiprocessing.set_sharing_strategy(args.sharing_strategy)

    # ----------------------------------------------------------------
    # 2. Create Gaussian model
    # ----------------------------------------------------------------
    # Auto-calculate prealloc_capacity for CPU pinned SH buffers.
    # Each rank preallocates 4 pinned arrays of (capacity, 48) float32 = 768 bytes/Gaussian.
    # Divide available memory by world_size so all ranks fit in physical RAM.
    if args.prealloc_capacity == -1:
        available_memory = psutil.virtual_memory().available
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        args.prealloc_capacity = (
            int((available_memory * 0.7) / world_size / (48 * 4 * 4)) // 16 * 16
        )
        utils.print_rank_0(
            f"Auto-calculated prealloc_capacity: {args.prealloc_capacity:,} Gaussians "
            f"({available_memory / (1024**3):.2f} GB available CPU memory, "
            f"{world_size} processes)"
        )
        log_file.write(
            f"Auto-calculated prealloc_capacity: {args.prealloc_capacity:,} Gaussians "
            f"({available_memory / (1024**3):.2f} GB available CPU memory, "
            f"{world_size} processes)\n"
        )

    gaussians = GaussianModelMultiGPUCLM(sh_degree=dataset_args.sh_degree)
    utils.print_rank_0("Using GaussianModelMultiGPUCLM (multi-GPU + CLM streaming)")
    log_file.write("Using GaussianModelMultiGPUCLM (multi-GPU + CLM streaming)\n")

    with torch.no_grad():
        scene = Scene(args, gaussians)
        gaussians.training_setup(opt_args)

        if args.start_checkpoint != "":
            model_params, start_from_this_iteration = utils.load_checkpoint(args)
            gaussians.restore(model_params, opt_args)
            utils.print_rank_0(f"Restored from checkpoint: {args.start_checkpoint}")
            log_file.write(f"Restored from checkpoint: {args.start_checkpoint}\n")

        scene.log_scene_info_to_file(log_file, "Scene Info Before Training")
    utils.check_initial_gpu_memory_usage("after init and before training loop")

    # ----------------------------------------------------------------
    # 3. Data loader
    # ----------------------------------------------------------------
    train_dataset = OffloadSceneDataset(scene.getTrainCamerasInfo())
    dataloader = DataLoader(
        train_dataset,
        batch_size=args.bsz,
        num_workers=1,
        shuffle=True,
        drop_last=True,
        persistent_workers=True,
        pin_memory=True,
        collate_fn=(lambda batch: batch),
    )
    dataloader_iter = iter(dataloader)

    # ----------------------------------------------------------------
    # 4. Background color + CUDA streams
    # ----------------------------------------------------------------
    background = None
    bg_color = [1, 1, 1] if dataset_args.white_background else None
    if bg_color is not None:
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    comm_stream = torch.cuda.Stream(device=args.gpu, priority=args.comm_stream_priority)

    # ----------------------------------------------------------------
    # 5. Training loop state
    # ----------------------------------------------------------------
    end2end_timers = End2endTimer(args)
    end2end_timers.start()
    progress_bar = tqdm(range(1, opt_args.iterations + 1), desc="Training progress")
    progress_bar.update(start_from_this_iteration - 1)
    num_trained_batches = 0

    perm_generator = torch.Generator(device="cuda")
    perm_generator.manual_seed(1)

    ema_loss_for_log = 0

    # ================================================================
    # MAIN TRAINING LOOP
    # ================================================================
    for iteration in range(
        start_from_this_iteration, opt_args.iterations + 1, args.bsz
    ):
        # ------------------------------------------------------------
        # Iteration setup
        # ------------------------------------------------------------
        if args.trace_cuda_mem:
            if (iteration % args.log_interval) == 1 or (
                iteration % args.densification_interval
            ) == 0:
                torch.cuda.memory._record_memory_history()
                log_file.write(f"[ITER {iteration}] Tracing cuda memory usage.\n")

        if iteration // args.bsz % 30 == 0:
            progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
        progress_bar.update(args.bsz)
        utils.set_cur_iter(iteration)
        gaussians.update_learning_rate(iteration)
        num_trained_batches += 1

        if args.reset_each_iter:
            torch.cuda.reset_max_memory_cached()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_max_memory_allocated()

        timers.clear()
        timers.start("[iteration end2end]")

        if args.nsys_profile:
            if iteration == args.nsys_profile_start_iter:
                torch.cuda.cudart().cudaProfilerStart()
            if iteration == args.nsys_profile_end_iter or iteration == opt_args.iterations:
                torch.cuda.cudart().cudaProfilerStop()
            if iteration >= args.nsys_profile_start_iter and iteration < args.nsys_profile_end_iter:
                nvtx.range_push(f"iteration[{iteration},{iteration+args.bsz})")

        if utils.check_update_at_this_iter(iteration, args.bsz, 1000, 0):
            gaussians.oneupSHdegree()

        # ------------------------------------------------------------
        # Load camera batch
        # ------------------------------------------------------------
        timers.start("dataloader: load the next image from disk and decode")
        try:
            batched_cameras = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            batched_cameras = next(dataloader_iter)
        timers.stop("dataloader: load the next image from disk and decode")

        for uid, c in enumerate(batched_cameras):
            c.uid = uid

        # ------------------------------------------------------------
        # Camera matrices → GPU
        # ------------------------------------------------------------
        timers.start("send cam matrices to gpu")
        for camera in batched_cameras:
            camera.world_view_transform = camera.world_view_transform.cuda()
            camera.full_proj_transform = camera.full_proj_transform.cuda()

        batched_wvt = []
        for camera in batched_cameras:
            camera.K = camera.create_k_on_gpu()
            batched_wvt.append(camera.world_view_transform.transpose(0, 1))

        batched_wvt = torch.stack(batched_wvt)
        batched_wvt_inv = torch.inverse(batched_wvt)
        batched_wvt_inv = torch.unbind(batched_wvt_inv, dim=0)

        for camera, wvt in zip(batched_cameras, batched_wvt_inv):
            camera.camtoworlds = wvt.unsqueeze(0)
        timers.stop("send cam matrices to gpu")

        # ------------------------------------------------------------
        # Ground-truth images → GPU
        # ------------------------------------------------------------
        with torch.no_grad():
            timers.start("load_cameras")
            for camera in batched_cameras:
                camera.original_image = camera.original_image_backup.cuda()
            timers.stop("load_cameras")

        # ------------------------------------------------------------
        # Forward/backward — multi_gpu_clm engine
        # ------------------------------------------------------------
        assert args.bsz > 1, "Multi-GPU CLM requires batch size > 1"
        N = gaussians._xyz.shape[0]

        losses, ordered_cams, sparsity = multi_gpu_clm_train_one_batch(
            gaussians,
            scene,
            batched_cameras,
            gaussians.parameters_grad_buffer,
            background,
            pipe_args,
            comm_stream,
            perm_generator,
        )

        batched_screenspace_pkg = {}

        # Log losses
        timers.start("sync_loss_and_log")
        batched_losses = torch.stack(losses)
        batched_loss_cpu = batched_losses.cpu().numpy()

        ema_loss_for_log = (
            batched_loss_cpu.mean()
            if ema_loss_for_log is None
            else 0.6 * ema_loss_for_log + 0.4 * batched_loss_cpu.mean()
        )

        train_dataset.update_losses(batched_loss_cpu)

        batched_loss_cpu = [round(loss, 6) for loss in batched_loss_cpu]
        log_string = "iteration[{},{}), loss: {} sparsity: {} image: {}\n".format(
            iteration,
            iteration + args.bsz,
            batched_loss_cpu,
            sparsity,
            [viewpoint_cam.image_name for viewpoint_cam in batched_cameras],
        )
        log_file.write(log_string)

        with torch.no_grad():
            # --------------------------------------------------------
            # Evaluation
            # --------------------------------------------------------
            end2end_timers.stop()
            training_report(
                iteration,
                l1_loss,
                args.test_iterations,
                scene,
                pipe_args,
                background,
            )
            end2end_timers.start()

            # --------------------------------------------------------
            # Densification
            # --------------------------------------------------------
            gsplat_densification(iteration, scene, gaussians, batched_screenspace_pkg)

            if (
                not args.disable_auto_densification
                and iteration <= args.densify_until_iter
                and iteration > args.densify_from_iter
                and utils.check_update_at_this_iter(
                    iteration, args.bsz, args.densification_interval, 0
                )
            ):
                pass  # gaussians may be added/removed — nothing to invalidate here

            batched_screenspace_pkg = None

            # --------------------------------------------------------
            # Save model
            # --------------------------------------------------------
            if any(
                iteration <= si < iteration + args.bsz
                for si in args.save_iterations
            ):
                utils.print_rank_0(f"\n[ITER {iteration}] Saving End2end")
                end2end_timers.stop()
                end2end_timers.print_time(log_file, iteration + args.bsz)

                if not args.do_not_save:
                    utils.print_rank_0(f"\n[ITER {iteration}] Saving Gaussians")
                    log_file.write(f"[ITER {iteration}] Saving Gaussians\n")
                    if args.save_tensors:
                        utils.print_rank_0("NOTE: Saving model as .pt files instead of .ply file.")
                        scene.save_tensors(iteration)
                    else:
                        scene.save(iteration)

                end2end_timers.start()

            # --------------------------------------------------------
            # Save checkpoint
            # --------------------------------------------------------
            if any(
                iteration <= ci < iteration + args.bsz
                for ci in args.checkpoint_iterations
            ):
                end2end_timers.stop()
                utils.print_rank_0(f"\n[ITER {iteration}] Saving Checkpoint")
                log_file.write(f"[ITER {iteration}] Saving Checkpoint\n")

                save_folder = scene.model_path + "/checkpoints/" + str(iteration) + "/"
                os.makedirs(save_folder, exist_ok=True)
                torch.save(
                    (gaussians.capture(), iteration + args.bsz),
                    save_folder + "/chkpnt.pth",
                )
                end2end_timers.start()

            # --------------------------------------------------------
            # Zero grad buffer (optimizer step is inside engine)
            # --------------------------------------------------------
            N = gaussians._xyz.shape[0]
            gaussians.parameters_grad_buffer[:N, :].zero_()

        # ------------------------------------------------------------
        # Iteration cleanup
        # ------------------------------------------------------------
        torch.cuda.synchronize()

        for viewpoint_cam in batched_cameras:
            viewpoint_cam.original_image = None

        if args.nsys_profile:
            if iteration >= args.nsys_profile_start_iter and iteration < args.nsys_profile_end_iter:
                nvtx.range_pop()

        if utils.check_enable_python_timer():
            timers.stop("[iteration end2end]")
            timers.printTimers(iteration, mode="sum")

        if args.trace_cuda_mem:
            if (iteration % args.log_interval) == 1 or (
                iteration % args.densification_interval
            ) == 0:
                dump_name = args.model_path + f"/trace_dump/iter={iteration}"
                torch.cuda.memory._dump_snapshot(filename=dump_name)
                torch.cuda.memory._record_memory_history(enabled=None)

        utils.memory_report("at the end of the iteration")
        log_file.flush()

    # ================================================================
    # Post-training
    # ================================================================
    del comm_stream

    if opt_args.iterations not in args.save_iterations:
        end2end_timers.print_time(log_file, opt_args.iterations)

    log_file.write(
        f"Max Memory usage: {torch.cuda.max_memory_allocated() / 1024**3:.3f} GB.\n"
    )

    progress_bar.close()
    scene.clean_up()

    if args.nsys_profile:
        torch.cuda.cudart().cudaProfilerStop()


# ============================================================================
# Evaluation report
# ============================================================================

def training_report(
    iteration, l1_loss, testing_iterations, scene: Scene, pipe_args, background
):
    args = utils.get_args()
    log_file = utils.get_log_file()

    while len(testing_iterations) > 0 and iteration > testing_iterations[0]:
        testing_iterations.pop(0)

    if len(testing_iterations) > 0 and utils.check_update_at_this_iter(
        iteration, args.bsz, testing_iterations[0], 0
    ):
        testing_iterations.pop(0)
        utils.print_rank_0(f"\n[ITER {iteration}] Start Testing")

        validation_configs = (
            {
                "name": "test",
                "cameras": scene.getTestCameras(),
                "cameras_info": scene.getTestCamerasInfo(),
                "num_cameras": len(
                    scene.getTestCameras()
                    if scene.getTestCameras() is not None
                    else scene.getTestCamerasInfo()
                ),
            },
            {
                "name": "train",
                "cameras": scene.getTrainCameras(),
                "cameras_info": scene.getTrainCamerasInfo(),
                "num_cameras": max(
                    len(
                        scene.getTrainCameras()
                        if scene.getTrainCameras() is not None
                        else scene.getTrainCamerasInfo()
                    )
                    // args.llffhold,
                    1,
                ),
            },
        )

        for config in validation_configs:
            l1_test = torch.scalar_tensor(0.0, device="cuda")
            psnr_test = torch.scalar_tensor(0.0, device="cuda")

            num_cameras = min(config["num_cameras"], args.max_num_images_to_evaluate)
            eval_dataset = OffloadSceneDataset(config["cameras_info"])
            eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=1,
                num_workers=1,
                pin_memory=True,
                collate_fn=(lambda batch: batch),
            )
            eval_iter = iter(eval_dataloader)

            for idx in range(1, num_cameras + 1):
                try:
                    batched_cameras = next(eval_iter)
                except StopIteration:
                    eval_iter = iter(eval_dataloader)
                    batched_cameras = next(eval_iter)

                for camera in batched_cameras:
                    camera.original_image = camera.original_image_backup.cuda()

                batched_image = []
                for camera in batched_cameras:
                    camera.world_view_transform = camera.world_view_transform.cuda()
                    camera.full_proj_transform = camera.full_proj_transform.cuda()
                    camera.K = camera.create_k_on_gpu()
                    camera.camtoworlds = torch.inverse(
                        camera.world_view_transform.transpose(0, 1)
                    ).unsqueeze(0)

                    rendered_image = multi_gpu_clm_eval_one_cam(
                        camera=camera,
                        gaussians=scene.gaussians,
                        background=background,
                        scene=scene,
                    )
                    batched_image.append(rendered_image)

                for camera_id, (image, gt_camera) in enumerate(
                    zip(batched_image, batched_cameras)
                ):
                    if image is None or len(image.shape) == 0:
                        image = torch.zeros(
                            gt_camera.original_image.shape,
                            device="cuda",
                            dtype=torch.float32,
                        )

                    image = torch.clamp(image, 0.0, 1.0)
                    gt_image = torch.clamp(gt_camera.original_image / 255.0, 0.0, 1.0)

                    if idx + camera_id < num_cameras + 1:
                        l1_test += l1_loss(image, gt_image).mean().double().item()
                        psnr_test += psnr(image, gt_image).mean().double().item()

                    if idx < args.num_save_images_during_eval:
                        save_dir = os.path.join(args.model_path, config["name"])
                        os.makedirs(save_dir, exist_ok=True)
                        img_name = gt_camera.image_name.replace("/", "_")
                        torchvision.utils.save_image(
                            image, os.path.join(save_dir, f"{iteration:06d}_{img_name}_render.png")
                        )
                        torchvision.utils.save_image(
                            gt_image, os.path.join(save_dir, f"{iteration:06d}_{img_name}_gt.png")
                        )

                    gt_camera.original_image = None

            psnr_test /= num_cameras
            l1_test /= num_cameras
            utils.print_rank_0(
                f"\n[ITER {iteration}] Evaluating {config['name']}: L1 {l1_test} PSNR {psnr_test}"
            )
            log_file.write(
                f"[ITER {iteration}] Evaluating {config['name']}: L1 {l1_test} PSNR {psnr_test}\n"
            )

        torch.cuda.empty_cache()


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    parser = ArgumentParser(description="Multi-GPU CLM training script")
    ap = AuxiliaryParams(parser)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    bench_p = BenchmarkParams(parser)
    debug_p = DebugParams(parser)
    args = parser.parse_args(sys.argv[1:])

    # Force multi_gpu_clm mode — no need for the --multi_gpu_clm flag
    args.multi_gpu_clm = True
    args.clm_offload = False
    args.naive_offload = False
    args.no_offload = False
    args.multi_gpu = False

    init_args(args)
    args = utils.get_args()

    os.makedirs(args.log_folder, exist_ok=True)
    os.makedirs(args.model_path, exist_ok=True)
    with open(args.log_folder + "/args.json", "w") as f:
        json.dump(vars(args), f)

    if args.trace_cuda_mem:
        os.makedirs(os.path.join(args.model_path, "trace_dump"))

    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    log_file = open(
        args.log_folder + "/python.log",
        "a" if args.auto_start_checkpoint else "w",
    )
    utils.set_log_file(log_file)
    print_all_args(args, log_file)

    p = psutil.Process()
    log_file.write(
        f"Initial pinned memory: {p.memory_info().shared / 1024**3} GB\n"
    )

    training(lp.extract(args), op.extract(args), pp.extract(args), args, log_file)

    utils.print_rank_0("\nTraining complete.")
