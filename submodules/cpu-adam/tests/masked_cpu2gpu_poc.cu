// Masked CPU-to-GPU transfer Proof of Concept. 
// The masked version can also reach the same bandwidth as memcpy. 

// nvcc -o masked_cpu2gpu_poc masked_cpu2gpu_poc.cu
// ./masked_cpu2gpu_poc

#include <iostream>
#include <cuda_runtime.h>
#include <numeric>
#include <chrono>
#include <algorithm>
#include <vector>
#include <random>
#include <cub/cub.cuh>

using namespace std;

#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

__global__ void get_rank2id(int *mask,int *mask_presum, int *rank2id, int n_row) {
    int n_threads = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_row; i += n_threads) {
        if (mask[i] == 1) {
            rank2id[mask_presum[i]-1] = i;
        }
    }
}


__global__ void kernel_v1(float *source, uint n, float *dest, int *mask, int* rank2id, int n_col, int n_row, int total_mask) {
    int n_threads = gridDim.x * blockDim.x;
    for (uint i = blockIdx.x * blockDim.x + threadIdx.x; i < total_mask; i += n_threads) {
        uint row_id = rank2id[i];
        uint source_offset = row_id * n_col;
        uint dest_offset = i * n_col;
        #pragma unroll
        for (int j = 0; j < n_col; j++) {
            dest[dest_offset + j] = source[source_offset + j];
        }
    }
}

__global__ void kernel_v2(float *source, uint n, float *dest, int *mask, int* rank2id, int n_col, int n_row, int total_mask) {
    int n_threads = gridDim.x * blockDim.x;
    uint total_elements = total_mask * n_col;
    for (uint i = blockIdx.x * blockDim.x + threadIdx.x; i < total_elements; i += n_threads) {
        uint row_id = rank2id[i / n_col];
        uint source_offset = row_id * n_col + i % n_col;
        dest[i] = source[source_offset];
    }
}


void test(int version, uint N, int n_col, bool check_correctness) {

    const int n_row = N / n_col;

    // set seed for random number generator
    srand(0);

    // prepare a random mask
    int *h_mask;
    cudaMallocHost(&h_mask, n_row * sizeof(int));
    uint total_mask = 0;
    for (uint i = 0; i < n_row; i++)
    {
        if (rand() % 3 == 0) {
            h_mask[i] = 1;
            total_mask++;
        } else h_mask[i] = 0;
    }
    printf("total_mask: %d\n", total_mask);
    int* d_mask;
    cudaMalloc(&d_mask, n_row * sizeof(int));
    cudaMemcpy(d_mask, h_mask, n_row * sizeof(int), cudaMemcpyHostToDevice);

    // buffer on CPU
    float *source;
    cudaMallocHost(&source, N * sizeof(float));
    for (uint i = 0; i < N; i++) source[i] = 1.0f * i;

    printf("Data Size: %f GB\n", (float)(total_mask * n_col * sizeof(float)) / 1024.0 / 1024 / 1024);

    // buffer on GPU
    float *dest;
    cudaMalloc(&dest, total_mask * n_col * sizeof(float));

    // buffer for mask inclusive sum
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    int *d_mask_presum;
    cudaMalloc(&d_mask_presum, n_row * sizeof(int));
    cudaMemset(d_mask_presum, 0, n_row * sizeof(int));
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_mask, d_mask_presum, n_row);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // run prefix sum for mask
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_mask, d_mask_presum, n_row);
    int *rank2id;
    cudaMalloc(&rank2id, total_mask * sizeof(int));
    get_rank2id<<<64, 256>>>(d_mask, d_mask_presum, rank2id, n_row);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // CPU -> GPU transfer
    cudaEventRecord(start);
    if (version == 1) {
        int block_size = 256;
        // int grid_size = (n_row + block_size - 1) / block_size;
        // int grid_size = 32;
        int grid_size = 64;

        // run prefix sum for mask
        // cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_mask, d_mask_presum, n_row);

        kernel_v1<<<grid_size, block_size>>>(source, N, dest, d_mask, rank2id, n_col, n_row, total_mask);

    } else if (version == 2) {
        int block_size = 256;
        // int grid_size = (n_row + block_size - 1) / block_size;
        // int grid_size = 32;
        int grid_size = 64;
        kernel_v2<<<grid_size, block_size>>>(source, N, dest, d_mask, rank2id, n_col, n_row, total_mask);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);


    if (check_correctness) {
        // prefix sum check
        // int* h_mask_presum;
        // cudaMallocHost(&h_mask_presum, n_row * sizeof(int));
        // cudaMemcpy(h_mask_presum, d_mask_presum, n_row * sizeof(int), cudaMemcpyDeviceToHost);
    
        // sum the dest from gpu
        float *dest_host;
        cudaMallocHost(&dest_host, total_mask * n_col * sizeof(float));
        cudaMemcpy(dest_host, dest, total_mask * n_col * sizeof(float), cudaMemcpyDeviceToHost);
        int *h_rank2id;
        cudaMallocHost(&h_rank2id, total_mask * sizeof(int));
        cudaMemcpy(h_rank2id, rank2id, total_mask * sizeof(int), cudaMemcpyDeviceToHost);
        for (uint i = 0; i < total_mask * n_col; i++) {
            if (dest_host[i] != source[h_rank2id[i/n_col] * n_col + i % n_col]) {
                printf("Error at %u: %f\n", i, dest_host[i]);
                break;
            }
        }
        // printf("Data sum on GPU: %f\n", std::accumulate(dest_host, dest_host + N, 0.0f));
        cudaFreeHost(dest_host);
        cudaFreeHost(h_rank2id);
    }

    printf("Data transfer %.5f GB time from CPU to GPU: %.3f ms. Bandwidth: %.4f GB/s\n", 
        total_mask * n_col * sizeof(float) / 1e9, ms, total_mask * n_col * sizeof(float) / ms / 1e6);

    cudaFreeHost(source);
    cudaFree(dest);
    cudaFree(d_mask);
    cudaFree(d_mask_presum);
    cudaFreeHost(h_mask);
    cudaFree(d_temp_storage);
}

int main() {
    const uint N_list[] = {1 << 10, 1 << 20, 1 << 24, 1 << 28, 1 << 30};
    const int version_list[] = {1, 2};
    bool check_correctness = true;
    const int n_col = 64;

    for (int i = 0; i < 5; i++) {
        for (int version : version_list) {
            printf("N: %u, kernel_version: %d\n", N_list[i], version);
            test(version, N_list[i], n_col, check_correctness);
        }
        printf("\n");
    }

    return 0;
}

