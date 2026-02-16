// GPU-to-CPU transfer Proof of Concept. 
// In this code, we can show that in-kernel communication is as fast as cudaMemcpy. 

// nvcc -o gpu2cpu_poc gpu2cpu_poc.cu
// ./gpu2cpu_poc

#include <iostream>
#include <cuda_runtime.h>
#include <numeric>
#include <chrono>
#include <algorithm>
#include <vector>

using namespace std;

#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

__global__ void init_data(float *data, uint n) {
    int n_threads = gridDim.x * blockDim.x;
    for (uint i = blockIdx.x * blockDim.x + threadIdx.x;i < n; i += n_threads) {
        data[i] = 1.0f;
    }
}

__global__ void kernel_v1(float *dest, float *source, uint n) {
    int n_threads = gridDim.x * blockDim.x;
    for (uint i = blockIdx.x * blockDim.x + threadIdx.x;i < n; i += n_threads) {
        dest[i] = source[i];
    }
}

__global__ void kernel_v2(float *dest, float *source, uint n) {
    int interval = gridDim.x * blockDim.x * 4;
    for (uint i = (blockIdx.x * blockDim.x + threadIdx.x) * 4; i < n; i += interval) {
        FLOAT4(dest[i]) = FLOAT4(source[i]);
    }
}


void test(int version, uint N, int n_col, bool check_correctness) {

    const int n_row = N / n_col;

    // buffer on CPU
    float *dest;
    cudaMallocHost(&dest, N * sizeof(float));

    // buffer on GPU
    float *source;
    cudaMalloc(&source, N * sizeof(float));
    init_data<<<64, 256>>>(source, N);
    cudaDeviceSynchronize();
    // printf("Data sum on GPU: %f\n", float(N));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // CPU -> GPU transfer
    cudaEventRecord(start);
    if (version == 0) {
        cudaMemcpy(dest, source, N * sizeof(float), cudaMemcpyDeviceToHost);
    } else if (version == 1) {
        // define grid and block size
        int block_size = 256;
        // int grid_size = (n_row + block_size - 1) / block_size;
        // int grid_size = 32;
        int grid_size = 64;
        kernel_v1<<<grid_size, block_size>>>(dest, source, N);
    } else if (version == 2) {
        // define grid and block size
        int block_size = 256;
        // int grid_size = (n_row + block_size - 1) / block_size;
        // int grid_size = 32;
        int grid_size = 64;
        kernel_v2<<<grid_size, block_size>>>(dest, source, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);


    if (check_correctness) {
        for (uint i = 0; i < N; i++) {
            if (dest[i] != 1.0f) {
                printf("Error at %u: %f\n", i, dest[i]);
                break;
            }
        }
    }

    printf("Data transfer %.5f GB time from GPU to CPU: %.3f ms. Bandwidth: %.4f GB/s\n", 
        N * sizeof(float) / 1e9, ms, N * sizeof(float) / ms / 1e6);
    
    cudaFree(source);
    cudaFreeHost(dest);
}

int main() {

    const uint N_list[] = {1 << 10, 1 << 20, 1 << 24, 1 << 28, 1 << 30};
    const int version_list[] = {0, 1, 2};
    bool check_correctness = true;
    
    for (int i = 0; i < 5; i++) {
        for (int version : version_list) {
            printf("N: %u, kernel_version: %d\n", N_list[i], version);
            test(version, N_list[i], 64, check_correctness);
        }
        printf("\n");
    }

    return 0;
}

