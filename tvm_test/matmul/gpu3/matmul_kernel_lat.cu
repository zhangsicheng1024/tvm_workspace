#define REPEAT_WARMUP 87
#define REPEAT_RUN 1
dim3 dimGrid(32, 1, 1);
dim3 dimBlock(128, 1, 1);
extern "C" __global__ void __launch_bounds__(128) mymatmul_kernel0(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ T_matmul_NN);

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <time.h>

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
    cudaEvent_t warmup, start, stop;
    cudaEventCreate(&warmup);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float randMax = 1000.0;
    srand((unsigned)time(NULL));
    
    float *a, *b, *out;
    float *d_a, *d_b, *d_out; 
    // float *d_temp;

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
    // cudaMalloc((void**)&d_temp, sizeof(float) * 1);

    // Transfer data from host to device memory
    cudaMemcpy(d_a, a, sizeof(float) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * N * K, cudaMemcpyHostToDevice);

    // Executing kernel 
    cudaEventRecord(warmup);
    // warm_up_gpu<<<dim3(256,1,1), dim3(256,1,1)>>>(rand(), d_temp);
    for(int i = 0; i < REPEAT_WARMUP; i++) mymatmul_kernel0<<<dimGrid, dimBlock>>>(d_a, d_b, d_out);
    cudaEventRecord(start);
    for(int i = 0; i < REPEAT_RUN; i++) mymatmul_kernel0<<<dimGrid, dimBlock>>>(d_a, d_b, d_out);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // Transfer data back to host memory
    // cudaMemcpy(out, d_out, sizeof(float) * N * M, cudaMemcpyDeviceToHost);

    // Deallocate device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    // Deallocate host memory
    free(a); 
    free(b); 
    free(out);

    float timeWarmup = 0;
    float timeRun = 0;
    cudaEventElapsedTime(&timeWarmup, warmup, start);
    cudaEventElapsedTime(&timeRun, start, stop);
    // printf("warmup time= %lf ms, run time = %lf ms\n", timeWarmup, timeRun);
    printf("%f\n", timeRun/REPEAT_RUN);
}
