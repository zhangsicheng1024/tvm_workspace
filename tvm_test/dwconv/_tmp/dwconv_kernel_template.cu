#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "nvmlPower.hpp"

#define N 128
#define CI 128
#define CM 1
#define H 28
#define W 28
#define KH 3
#define KW 3




int main(){
    int repeat = REPEAT;
    float warmupRatio = 0.1;
    float randMax = 1000.0;
    srand((unsigned)time(NULL));

    float *a, *b, *out;
    float *d_a, *d_b, *d_out; 

    // Allocate host memory
    a   = (float*)malloc(sizeof(float) * N * CI * H * W);
    b   = (float*)malloc(sizeof(float) * CI * CM * KH * KW);
    out = (float*)malloc(sizeof(float) * N * CI * H * W);

    // Initialize host arrays
    // memset(a,   1.0, sizeof(float) * N * CI * H * W);
    for(int i = 0; i < N * CI * H * W; i++) {
        a[i] = (float)(rand() / (float)RAND_MAX - 0.5) * 2.0 * randMax;
    }
    // memset(b,   1.0, sizeof(float) * CI * CM * KH * KW);
    for(int i = 0; i < CI * CM * KH * KW; i++) {
        b[i] = (float)(rand() / (float)RAND_MAX - 0.5) * 2.0 * randMax;
    }

    // Allocate device memory 
    cudaMalloc((void**)&d_a,   sizeof(float) * N * CI * H * W);
    cudaMalloc((void**)&d_b,   sizeof(float) * CI * CM * KH * KW);
    cudaMalloc((void**)&d_out, sizeof(float) * N * CI * H * W);

    // Transfer data from host to device memory
    cudaMemcpy(d_a, a, sizeof(float) * N * CI * H * W, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * CI * CM * KH * KW, cudaMemcpyHostToDevice);

    // Executing kernel 
    for(int i = 0; i < repeat * warmupRatio; i++) {
        mydwconv_kernel0<<<dimGrid, dimBlock>>>(d_a, d_b, d_out);
    }
    cudaDeviceSynchronize();
    nvmlAPIRun();
    for(int i = 0; i < repeat; i++) {
        mydwconv_kernel0<<<dimGrid, dimBlock>>>(d_a, d_b, d_out);
    }
    cudaDeviceSynchronize();
    nvmlAPIEnd();
    
    // Transfer data back to host memory
    cudaMemcpy(out, d_out, sizeof(float) * N * CI * H * W, cudaMemcpyDeviceToHost);

    // Deallocate device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    // Deallocate host memory
    free(a); 
    free(b); 
    free(out);
}
