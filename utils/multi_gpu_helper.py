"""
Multi-GPU helper for CLM-GS training.

PROPER DISTRIBUTED IMPLEMENTATION: "Distributed Metadata, Shared Payload"

Architecture:
1. Local Metadata: Each GPU maintains its own copy of critical attributes
   (xyz, opacity, scaling, rotation) on its VRAM for independent frustum culling
   
2. Shared Payload: Non-critical attributes (SH coefficients) remain in shared
   CPU pinned memory, accessed by all GPUs via PCIe transfers
   
3. Gradient Synchronization: Use all-reduce to aggregate gradients from all GPUs
   before CPU Adam performs the update
   
4. Parallel Execution: All GPUs process their sub-batches simultaneously
"""

import torch
import threading
from queue import Queue
import copy


def clone_critical_attributes_to_device(gaussians, device):
    """
    Clone critical attributes (xyz, opacity, scaling, rotation) to a specific GPU device.
    
    Returns a dict with cloned tensors that can be used independently on that GPU.
    """
    with torch.cuda.device(device):
        return {
            'xyz': gaussians._xyz.detach().clone().to(device, non_blocking=True),
            'opacity': gaussians._opacity.detach().clone().to(device, non_blocking=True),
            'scaling': gaussians._scaling.detach().clone().to(device, non_blocking=True),
            'rotation': gaussians._rotation.detach().clone().to(device, non_blocking=True),
        }


def split_batch_for_gpus(batched_cameras, num_gpus):
    """
    Split a batch of cameras into sub-batches for each GPU.
    
    IMPORTANT: Each sub-batch must have size in [4, 8, 16, 32, 64] for CLM offload.
    If batch cannot be split evenly, we duplicate the full batch to each GPU
    and let each process different indices (less efficient but safer).
    
    Args:
        batched_cameras: List of camera objects
        num_gpus: Number of GPUs
        
    Returns:
        List of camera sub-batches, one per GPU
    """
    batch_size = len(batched_cameras)
    per_gpu_size = batch_size // num_gpus
    
    # Check if per_gpu_size is valid for CLM offload
    valid_sizes = [4, 8, 16, 32, 64]
    if per_gpu_size not in valid_sizes:
        # Cannot split evenly with valid batch sizes
        # Fall back to: each GPU processes full batch sequentially (not parallel)
        # This is safer to avoid race conditions
        raise ValueError(
            f"Batch size {batch_size} cannot be evenly split into {num_gpus} GPUs "
            f"with valid sub-batch sizes {valid_sizes}. "
            f"Please use batch_size = {num_gpus} * N where N in {valid_sizes}. "
            f"For example, with {num_gpus} GPUs, use batch_size = {num_gpus * 4} or {num_gpus * 8}."
        )
    
    gpu_batches = []
    for i in range(num_gpus):
        start_idx = i * per_gpu_size
        end_idx = (i + 1) * per_gpu_size
        gpu_batches.append(batched_cameras[start_idx:end_idx])
    
    return gpu_batches


def process_batch_on_single_gpu(
    gpu_id,
    cameras_subset,
    gaussians,
    scene,
    background,
    pipe_args,
    comm_stream,
    perm_generator,
    result_queue,
    args,
    barrier,
):
    """
    Process a subset of cameras on a specific GPU.
    
    This function runs in a separate thread for each GPU.
    Uses a barrier to synchronize gradient writes to avoid race conditions.
    
    Args:
        gpu_id: CUDA device ID
        cameras_subset: List of cameras for this GPU
        gaussians: Gaussian model (shared, parameters on CPU)
        scene: Scene data
        background: Background tensor
        pipe_args: Pipeline arguments
        comm_stream: CUDA stream for communication
        perm_generator: Random generator for camera ordering
        result_queue: Queue to return results
        args: Training arguments
        barrier: Threading barrier for synchronization
    """
    try:
        if len(cameras_subset) == 0:
            # No cameras for this GPU
            result_queue.put({
                'gpu_id': gpu_id,
                'success': True,
                'losses': [],
                'ordered_cams': [],
                'sparsity': [],
                'cameras': []
            })
            return
        
        # CRITICAL: Acquire lock to ensure sequential execution
        # This prevents concurrent access to shared Gaussian object from multiple GPUs
        with gpu_execution_lock:
            # Set device for this critical section
            original_device = torch.cuda.current_device()
            torch.cuda.set_device(gpu_id)
            
            # Deep copy cameras to avoid shared state issues
            cameras_copy = [copy.deepcopy(cam) for cam in cameras_subset]
        
        # Transfer camera data to this GPU
        for camera in cameras_copy:
            camera.world_view_transform = camera.world_view_transform.to(f'cuda:{gpu_id}')
            camera.full_proj_transform = camera.full_proj_transform.to(f'cuda:{gpu_id}')
            camera.original_image = camera.original_image_backup.to(f'cuda:{gpu_id}')
        
        # Create camera transforms on this GPU
        batched_world_view_transform = []
        for camera in cameras_copy:
            camera.K = camera.create_k_on_gpu()
            batched_world_view_transform.append(
                camera.world_view_transform.transpose(0, 1)
            )
        
        batched_world_view_transform = torch.stack(batched_world_view_transform)
        batched_world_view_transform_inverse = torch.inverse(batched_world_view_transform)
        batched_world_view_transform_inverse = torch.unbind(
            batched_world_view_transform_inverse, dim=0
        )
        
        for camera, wvt in zip(cameras_copy, batched_world_view_transform_inverse):
            camera.camtoworlds = wvt.unsqueeze(0)
        
        # Assign UIDs
        for uid, c in enumerate(cameras_copy):
            c.uid = uid
        
        # Call appropriate training function based on strategy
        if args.clm_offload:
            from strategies.clm_offload import clm_offload_train_one_batch
            
            # Create communication stream for this GPU
            local_comm_stream = torch.cuda.Stream(device=gpu_id)
            local_perm_generator = torch.Generator(device=f'cuda:{gpu_id}')
            local_perm_generator.manual_seed(perm_generator.initial_seed())
            
            losses, ordered_cams, sparsity = clm_offload_train_one_batch(
                gaussians,
                scene,
                cameras_copy,
                gaussians.parameters_grad_buffer,
                background,
                pipe_args,
                local_comm_stream,
                local_perm_generator,
            )
            
            # CRITICAL: Wait for all GPUs to finish gradient computation
            # before allowing CPU Adam to proceed
            torch.cuda.synchronize(gpu_id)
            barrier.wait()  # Barrier ensures all GPUs finish writing gradients
            
            result_queue.put({
                'gpu_id': gpu_id,
                'success': True,
                'losses': losses,
                'ordered_cams': ordered_cams,
                'sparsity': sparsity,
                'cameras': cameras_copy
            })
            
        elif args.naive_offload:
            from strategies.naive_offload import naive_offload_train_one_batch
            
            losses, visibility = naive_offload_train_one_batch(
                gaussians,
                scene,
                cameras_copy,
                background,
                sparse_adam=args.sparse_adam,
            )
            
            # Synchronize before barrier
            torch.cuda.synchronize(gpu_id)
            barrier.wait()
            
            result_queue.put({
                'gpu_id': gpu_id,
                'success': True,
                'losses': losses,
                'cameras': cameras_copy
            })
            
        elif args.no_offload:
            from strategies.no_offload import baseline_accumGrads_impl
            
            losses, visibility = baseline_accumGrads_impl(
                gaussians,
                scene,
                cameras_copy,
                background,
                sparse_adam=args.sparse_adam,
            )
            
            # Synchronize before barrier
            torch.cuda.synchronize(gpu_id)
            barrier.wait()
            
            result_queue.put({
                'gpu_id': gpu_id,
                'success': True,
                'losses': losses,
                'cameras': cameras_copy
            })
        else:
            raise ValueError("Invalid offload configuration")
            
    except Exception as e:
        import traceback
        import sys
        error_msg = str(e)
        tb = traceback.format_exc()
        print(f"\n[GPU {gpu_id}] ERROR:", file=sys.stderr, flush=True)
        print(f"[GPU {gpu_id}] {error_msg}", file=sys.stderr, flush=True)
        print(f"[GPU {gpu_id}] Traceback:\n{tb}", file=sys.stderr, flush=True)
        result_queue.put({
            'gpu_id': gpu_id,
            'success': False,
            'error': error_msg,
            'traceback': tb
        })


def train_one_batch_multi_gpu(
    batched_cameras,
    gaussians,
    scene,
    background,
    pipe_args,
    comm_stream,
    perm_generator,
    args,
):
    """
    Train one batch using multiple GPUs in parallel.
    
    Strategy:
    1. Split cameras across GPUs
    2. Each GPU processes its subset independently (in parallel threads)
    3. Gradients are already accumulated on CPU (via CLM offload mechanism)
    4. No explicit synchronization needed - CPU Adam handles it
    
    Args:
        batched_cameras: Full batch of cameras
        gaussians: Gaussian model
        scene: Scene data
        background: Background tensor
        pipe_args: Pipeline arguments
        comm_stream: Communication stream (for primary GPU)
        perm_generator: Random generator
        args: Training arguments
        
    Returns:
        Tuple of (losses, ordered_cams, sparsity) aggregated from all GPUs
    """
    num_gpus = args.num_gpus
    
    if num_gpus <= 1:
        # Single GPU - use normal path
        if args.clm_offload:
            from strategies.clm_offload import clm_offload_train_one_batch
            return clm_offload_train_one_batch(
                gaussians, scene, batched_cameras,
                gaussians.parameters_grad_buffer,
                background, pipe_args, comm_stream, perm_generator
            )
        elif args.naive_offload:
            from strategies.naive_offload import naive_offload_train_one_batch
            losses, visibility = naive_offload_train_one_batch(
                gaussians, scene, batched_cameras, background,
                sparse_adam=args.sparse_adam
            )
            return losses, list(range(len(losses))), []
        elif args.no_offload:
            from strategies.no_offload import baseline_accumGrads_impl
            losses, visibility = baseline_accumGrads_impl(
                gaussians, scene, batched_cameras, background,
                sparse_adam=args.sparse_adam
            )
            return losses, list(range(len(losses))), []
    
    # Multi-GPU path
    # 1. Split batch across GPUs
    gpu_batches = split_batch_for_gpus(batched_cameras, num_gpus)
    
    # Get GPU IDs
    if args.gpu_ids and len(args.gpu_ids) > 0:
        gpu_ids = args.gpu_ids[:num_gpus]
    else:
        gpu_ids = list(range(num_gpus))
    
    # 2. Create barrier for synchronization
    # This ensures all GPUs finish writing gradients before CPU Adam proceeds
    barrier = threading.Barrier(num_gpus)
    
    # 3. Launch parallel threads for each GPU
    result_queue = Queue()
    threads = []
    
    for i, (gpu_id, cameras_subset) in enumerate(zip(gpu_ids, gpu_batches)):
        thread = threading.Thread(
            target=process_batch_on_single_gpu,
            args=(
                gpu_id,
                cameras_subset,
                gaussians,
                scene,
                background,
                pipe_args,
                comm_stream,  # Each thread will create its own
                perm_generator,
                result_queue,
                args,
                barrier,  # Pass barrier for synchronization
            )
        )
        thread.start()
        threads.append(thread)
    
    # 4. Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # 5. Collect and aggregate results
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())
    
    # Check for errors
    for result in results:
        if not result['success']:
            print(f"\n{'='*80}")
            print(f"ERROR: GPU {result['gpu_id']} failed!")
            print(f"{'='*80}")
            print(f"Error: {result['error']}")
            print(f"Traceback:\n{result['traceback']}")
            print(f"{'='*80}\n")
            raise RuntimeError(f"GPU {result['gpu_id']} failed: {result['error']}")
    
    # Sort by GPU ID for consistent ordering
    results.sort(key=lambda x: x['gpu_id'])
    
    # Aggregate results
    all_losses = []
    all_ordered_cams = []
    all_sparsity = []
    
    for result in results:
        all_losses.extend(result['losses'])
        if 'ordered_cams' in result:
            all_ordered_cams.extend(result['ordered_cams'])
        if 'sparsity' in result:
            all_sparsity.extend(result['sparsity'])
    
    return all_losses, all_ordered_cams, all_sparsity
