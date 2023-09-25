#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <time.h>
#include "nvmlPower.hpp"

#define M 1024
#define N 4096
#define K 1024

__global__ void warm_up_gpu(long seed, float* __restrict__ temp){
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(seed, idx, 0, &state);
    float it = 0.0f;
    for(int i = 0; i < 5000000; i++)
        it += idx * (curand_uniform_double(&state) - 0.5);
    temp[0] = it;
}

int main() {
    float randMax = 1000.0;
    srand((unsigned)time(NULL));
    
    float *a, *b, *out;
    float *d_a, *d_b, *d_out; 
    float *d_temp;

    // Allocate host memory
    a   = (float*)malloc(sizeof(float) * M * K);
    b   = (float*)malloc(sizeof(float) * N * K);
    out = (float*)malloc(sizeof(float) * M * N);

    // Initialize host arrays
    for(int i = 0; i < M * K; i++) {
        a[i] = (float)(rand() / (float)RAND_MAX - 0.5) * 2.0 * randMax;
    }
    for(int i = 0; i < N * K; i++) {
        b[i] = (float)(rand() / (float)RAND_MAX - 0.5) * 2.0 * randMax;
    }

    // Allocate device memory 
    cudaMalloc((void**)&d_a,   sizeof(float) * M * K);
    cudaMalloc((void**)&d_b,   sizeof(float) * N * K);
    cudaMalloc((void**)&d_out, sizeof(float) * M * N);
    cudaMalloc((void**)&d_temp, sizeof(float) * 1);

    // Transfer data from host to device memory
    cudaMemcpy(d_a, a, sizeof(float) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * N * K, cudaMemcpyHostToDevice);

    // Executing kernel 
    warm_up_gpu<<<dim3(256,1,1), dim3(256,1,1)>>>(rand(), d_temp);
    cudaDeviceSynchronize();
    nvmlAPIRun();
    for(int i = 0; i < REPEAT; i++) {
        mymatmul_kernel0<<<dimGrid, dimBlock>>>(d_a, d_b, d_out);
    }
    cudaDeviceSynchronize();
    nvmlAPIEnd();
    
    // Transfer data back to host memory
    cudaMemcpy(out, d_out, sizeof(float) * N * M, cudaMemcpyDeviceToHost);

    // Deallocate device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    // Deallocate host memory
    free(a); 
    free(b); 
    free(out);
}
