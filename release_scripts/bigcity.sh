#!/bin/bash

# bigcity.sh - Training script for MatrixCity BigCity dataset
# This demonstrates the extreme-scale capability upper bound of the Grendel-XS system
# Note: This dataset is synthetic and designed to showcase maximum scalability

# Check arguments
if [ $# -ne 3 ]; then
    echo "Error: Please specify exactly three arguments."
    echo "Usage: bash bigcity.sh <dataset_folder> <offload_strategy> <scale>"
    echo ""
    echo "Arguments:"
    echo "  <dataset_folder>   : Path to the MatrixCity BigCity dataset folder"
    echo "                       (e.g., /path/to/matrixcity/big_city/aerial/pose/all_blocks)"
    echo "  <offload_strategy> : One of: no_offload, clm_offload, naive_offload"
    echo "  <scale>            : One of: 102m (100M Gaussians), 25m (25M Gaussians)"
    echo ""
    echo "Examples:"
    echo "  bash bigcity.sh /path/to/matrixcity/big_city/aerial/pose/all_blocks clm_offload 102m"
    echo "  bash bigcity.sh /path/to/matrixcity/big_city/aerial/pose/all_blocks clm_offload 25m"
    echo "  bash bigcity.sh /path/to/matrixcity/big_city/aerial/pose/all_blocks no_offload 25m"
    echo ""
    echo "Note: Only clm_offload can handle 102m scale without OOM on typical GPUs"
    exit 1
fi

dataset_folder=$1
offload_strategy=$2
scale=$3

# Validate offload strategy
if [[ ! "$offload_strategy" =~ ^(no_offload|clm_offload|naive_offload)$ ]]; then
    echo "Error: Invalid offload strategy '$offload_strategy'"
    echo "Must be one of: no_offload, clm_offload, naive_offload"
    exit 1
fi

# Validate scale
if [[ ! "$scale" =~ ^(102m|25m)$ ]]; then
    echo "Error: Invalid scale '$scale'"
    echo "Must be one of: 102m, 25m"
    exit 1
fi

echo "Dataset folder: $dataset_folder"
echo "Offload strategy: $offload_strategy"
echo "Scale: $scale"

# Generate timestamp for experiment naming
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Set pre-allocated capacity based on scale
if [ "$scale" == "102m" ]; then
    CAPACITY=102_231_360  # ~102M Gaussians
    CAPACITY_DESC="102M"
    PC_DOWNSAMPLE_RATIO=1.0
elif [ "$scale" == "25m" ]; then
    CAPACITY=25_557_840   # ~25M Gaussians
    CAPACITY_DESC="25M"
    PC_DOWNSAMPLE_RATIO=0.25
fi

# Experiment name
expe_name="bigcity_${scale}_${offload_strategy}_${CAPACITY_DESC}"

# Set output paths
log_folder="output/bigcity/${TIMESTAMP}_${expe_name}"
model_path=${log_folder}

echo "Output folder: $log_folder"

# Training configurations
BSZ=64
ITERATIONS=500000
# ITERATIONS=100
LOG_INTERVAL=250

# Test and save iterations
TEST_ITERATIONS="100000 200000 300000 400000 500000"
SAVE_ITERATIONS="200000 500000"
# SAVE_ITERATIONS="100"

# Densification parameters - DISABLED for extreme scale
# We use pre-allocated capacity instead of auto-densification
DENSIFY_OPTS="--disable_auto_densification"

# Configure offload options based on strategy
OFFLOAD_OPTS=""
if [ "$offload_strategy" == "clm_offload" ]; then
    OFFLOAD_OPTS="--clm_offload \
--prealloc_capacity ${CAPACITY} \
--sparse_adam"
elif [ "$offload_strategy" == "naive_offload" ]; then
    OFFLOAD_OPTS="--naive_offload \
--prealloc_capacity ${CAPACITY} \
--sparse_adam"
elif [ "$offload_strategy" == "no_offload" ]; then
    OFFLOAD_OPTS="--no_offload \
--prealloc_capacity ${CAPACITY}"
fi

# Monitoring settings
MONITOR_OPTS="--enable_timer \
--end2end_time \
--check_gpu_memory \
--check_cpu_memory"

# Configure CUDA caching allocator
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Optional: Enable detailed error tracking (uncomment if debugging)
# export TORCH_SHOW_CPP_STACKTRACES=1
# export CUDA_LAUNCH_BLOCKING=1

echo ""
echo "========================================"
echo "Training MatrixCity BigCity"
echo "========================================"
echo "Strategy: $offload_strategy"
echo "Scale: $CAPACITY_DESC Gaussians"
echo "Batch size: $BSZ"
echo "Iterations: $ITERATIONS"
echo "Pre-allocated capacity: $CAPACITY Gaussians"
echo "========================================"
echo ""

# Add warning for 102m scale or no_offload/naive_offload strategies
if [ "$scale" == "102m" ] || [ "$offload_strategy" == "no_offload" ] || [ "$offload_strategy" == "naive_offload" ]; then
    if [ "$scale" == "102m" ] && [ "$offload_strategy" != "clm_offload" ]; then
        echo "WARNING: $offload_strategy with $scale scale will likely result in OOM!"
        echo "This is expected behavior for demonstration purposes."
        echo ""
    fi
    if [ "$scale" == "102m" ]; then
        echo "NOTE: This is an extreme-scale experiment that will take"
        echo "5-10 hours and requires significant computational resources."
    fi
    echo "Press Ctrl+C within 5 seconds to cancel..."
    echo ""
    sleep 5
fi

# Run training
python train.py \
    -s ${dataset_folder} \
    --log_folder ${log_folder} \
    --model_path ${model_path} \
    --iterations ${ITERATIONS} \
    --log_interval ${LOG_INTERVAL} \
    --bsz ${BSZ} \
    --test_iterations ${TEST_ITERATIONS} \
    --save_iterations ${SAVE_ITERATIONS} \
    ${DENSIFY_OPTS} \
    ${OFFLOAD_OPTS} \
    ${MONITOR_OPTS} \
    --eval \
    --save_tensors \
    --max_num_images_to_evaluate 200 \
    --initial_point_cloud_downsampled_ratio ${PC_DOWNSAMPLE_RATIO}

# --num_save_images_during_eval 10 \

# save_tensors is very important for matrixcity. 

if [ $? -ne 0 ]; then
    echo "Error: Training failed"
    exit 1
fi

echo ""
echo "========================================"
echo "Training completed successfully!"
echo "Results saved in: ${log_folder}"
echo "========================================"

