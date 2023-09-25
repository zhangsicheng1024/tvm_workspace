
#ifdef _WIN32
  using uint = unsigned int;
  using uchar = unsigned char;
  using ushort = unsigned short;
  using int64_t = long long;
  using uint64_t = unsigned long long;
#else
  #define uint unsigned int
  #define uchar unsigned char
  #define ushort unsigned short
  #define int64_t long long
  #define uint64_t unsigned long long
#endif
extern "C" __global__ void __launch_bounds__(256) mymatmul_kernel0(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ T_matmul_NN) {
  float T_matmul_NN_local[1024];
  __shared__ float data_shared[512];
  __shared__ float kernel_shared[8192];
  for (int j_c_outer_inner_init = 0; j_c_outer_inner_init < 64; ++j_c_outer_inner_init) {
    for (int j_c_inner_init = 0; j_c_inner_init < 2; ++j_c_inner_init) {
      T_matmul_NN_local[((j_c_outer_inner_init * 2) + j_c_inner_init)] = 0.000000e+00f;
      T_matmul_NN_local[(((j_c_outer_inner_init * 2) + j_c_inner_init) + 128)] = 0.000000e+00f;
      T_matmul_NN_local[(((j_c_outer_inner_init * 2) + j_c_inner_init) + 256)] = 0.000000e+00f;
      T_matmul_NN_local[(((j_c_outer_inner_init * 2) + j_c_inner_init) + 384)] = 0.000000e+00f;
      T_matmul_NN_local[(((j_c_outer_inner_init * 2) + j_c_inner_init) + 512)] = 0.000000e+00f;
      T_matmul_NN_local[(((j_c_outer_inner_init * 2) + j_c_inner_init) + 640)] = 0.000000e+00f;
      T_matmul_NN_local[(((j_c_outer_inner_init * 2) + j_c_inner_init) + 768)] = 0.000000e+00f;
      T_matmul_NN_local[(((j_c_outer_inner_init * 2) + j_c_inner_init) + 896)] = 0.000000e+00f;
    }
  }
  for (int k_outer_outer = 0; k_outer_outer < 256; ++k_outer_outer) {
    __syncthreads();
    *(float2*)(data_shared + (((int)threadIdx.x) * 2)) = *(float2*)(data + (((((((int)blockIdx.x) >> 1) * 131072) + ((((int)threadIdx.x) >> 1) * 1024)) + (k_outer_outer * 4)) + ((((int)threadIdx.x) & 1) * 2)));
    for (int ax0_ax1_fused_outer_outer = 0; ax0_ax1_fused_outer_outer < 8; ++ax0_ax1_fused_outer_outer) {
      *(float4*)(kernel_shared + ((ax0_ax1_fused_outer_outer * 1024) + (((int)threadIdx.x) * 4))) = *(float4*)(kernel + (((((k_outer_outer * 16384) + ((ax0_ax1_fused_outer_outer >> 1) * 4096)) + ((((int)blockIdx.x) & 1) * 2048)) + ((ax0_ax1_fused_outer_outer & 1) * 1024)) + (((int)threadIdx.x) * 4)));
    }
    __syncthreads();
    for (int k_outer_inner = 0; k_outer_inner < 4; ++k_outer_inner) {
      for (int j_c_outer_inner = 0; j_c_outer_inner < 64; ++j_c_outer_inner) {
        for (int j_c_inner = 0; j_c_inner < 2; ++j_c_inner) {
          T_matmul_NN_local[((j_c_outer_inner * 2) + j_c_inner)] = (T_matmul_NN_local[((j_c_outer_inner * 2) + j_c_inner)] + (data_shared[(((((int)threadIdx.x) >> 4) * 4) + k_outer_inner)] * kernel_shared[((((k_outer_inner * 2048) + ((((int)threadIdx.x) & 15) * 128)) + (j_c_outer_inner * 2)) + j_c_inner)]));
          T_matmul_NN_local[(((j_c_outer_inner * 2) + j_c_inner) + 128)] = (T_matmul_NN_local[(((j_c_outer_inner * 2) + j_c_inner) + 128)] + (data_shared[((((((int)threadIdx.x) >> 4) * 4) + k_outer_inner) + 64)] * kernel_shared[((((k_outer_inner * 2048) + ((((int)threadIdx.x) & 15) * 128)) + (j_c_outer_inner * 2)) + j_c_inner)]));
          T_matmul_NN_local[(((j_c_outer_inner * 2) + j_c_inner) + 256)] = (T_matmul_NN_local[(((j_c_outer_inner * 2) + j_c_inner) + 256)] + (data_shared[((((((int)threadIdx.x) >> 4) * 4) + k_outer_inner) + 128)] * kernel_shared[((((k_outer_inner * 2048) + ((((int)threadIdx.x) & 15) * 128)) + (j_c_outer_inner * 2)) + j_c_inner)]));
          T_matmul_NN_local[(((j_c_outer_inner * 2) + j_c_inner) + 384)] = (T_matmul_NN_local[(((j_c_outer_inner * 2) + j_c_inner) + 384)] + (data_shared[((((((int)threadIdx.x) >> 4) * 4) + k_outer_inner) + 192)] * kernel_shared[((((k_outer_inner * 2048) + ((((int)threadIdx.x) & 15) * 128)) + (j_c_outer_inner * 2)) + j_c_inner)]));
          T_matmul_NN_local[(((j_c_outer_inner * 2) + j_c_inner) + 512)] = (T_matmul_NN_local[(((j_c_outer_inner * 2) + j_c_inner) + 512)] + (data_shared[((((((int)threadIdx.x) >> 4) * 4) + k_outer_inner) + 256)] * kernel_shared[((((k_outer_inner * 2048) + ((((int)threadIdx.x) & 15) * 128)) + (j_c_outer_inner * 2)) + j_c_inner)]));
          T_matmul_NN_local[(((j_c_outer_inner * 2) + j_c_inner) + 640)] = (T_matmul_NN_local[(((j_c_outer_inner * 2) + j_c_inner) + 640)] + (data_shared[((((((int)threadIdx.x) >> 4) * 4) + k_outer_inner) + 320)] * kernel_shared[((((k_outer_inner * 2048) + ((((int)threadIdx.x) & 15) * 128)) + (j_c_outer_inner * 2)) + j_c_inner)]));
          T_matmul_NN_local[(((j_c_outer_inner * 2) + j_c_inner) + 768)] = (T_matmul_NN_local[(((j_c_outer_inner * 2) + j_c_inner) + 768)] + (data_shared[((((((int)threadIdx.x) >> 4) * 4) + k_outer_inner) + 384)] * kernel_shared[((((k_outer_inner * 2048) + ((((int)threadIdx.x) & 15) * 128)) + (j_c_outer_inner * 2)) + j_c_inner)]));
          T_matmul_NN_local[(((j_c_outer_inner * 2) + j_c_inner) + 896)] = (T_matmul_NN_local[(((j_c_outer_inner * 2) + j_c_inner) + 896)] + (data_shared[((((((int)threadIdx.x) >> 4) * 4) + k_outer_inner) + 448)] * kernel_shared[((((k_outer_inner * 2048) + ((((int)threadIdx.x) & 15) * 128)) + (j_c_outer_inner * 2)) + j_c_inner)]));
        }
      }
    }
  }
  for (int j_inner = 0; j_inner < 128; ++j_inner) {
    T_matmul_NN[((((((((int)blockIdx.x) >> 1) * 524288) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.x) & 1) * 2048)) + ((((int)threadIdx.x) & 15) * 128)) + j_inner)] = T_matmul_NN_local[j_inner];
    T_matmul_NN[(((((((((int)blockIdx.x) >> 1) * 524288) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.x) & 1) * 2048)) + ((((int)threadIdx.x) & 15) * 128)) + j_inner) + 65536)] = T_matmul_NN_local[(j_inner + 128)];
    T_matmul_NN[(((((((((int)blockIdx.x) >> 1) * 524288) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.x) & 1) * 2048)) + ((((int)threadIdx.x) & 15) * 128)) + j_inner) + 131072)] = T_matmul_NN_local[(j_inner + 256)];
    T_matmul_NN[(((((((((int)blockIdx.x) >> 1) * 524288) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.x) & 1) * 2048)) + ((((int)threadIdx.x) & 15) * 128)) + j_inner) + 196608)] = T_matmul_NN_local[(j_inner + 384)];
    T_matmul_NN[(((((((((int)blockIdx.x) >> 1) * 524288) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.x) & 1) * 2048)) + ((((int)threadIdx.x) & 15) * 128)) + j_inner) + 262144)] = T_matmul_NN_local[(j_inner + 512)];
    T_matmul_NN[(((((((((int)blockIdx.x) >> 1) * 524288) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.x) & 1) * 2048)) + ((((int)threadIdx.x) & 15) * 128)) + j_inner) + 327680)] = T_matmul_NN_local[(j_inner + 640)];
    T_matmul_NN[(((((((((int)blockIdx.x) >> 1) * 524288) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.x) & 1) * 2048)) + ((((int)threadIdx.x) & 15) * 128)) + j_inner) + 393216)] = T_matmul_NN_local[(j_inner + 768)];
    T_matmul_NN[(((((((((int)blockIdx.x) >> 1) * 524288) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.x) & 1) * 2048)) + ((((int)threadIdx.x) & 15) * 128)) + j_inner) + 458752)] = T_matmul_NN_local[(j_inner + 896)];
  }
}

