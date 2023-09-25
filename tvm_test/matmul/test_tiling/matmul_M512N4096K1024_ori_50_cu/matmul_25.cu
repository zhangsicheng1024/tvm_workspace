
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
extern "C" __global__ void __launch_bounds__(128) mymatmul_kernel0(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ T_matmul_NN) {
  float T_matmul_NN_local[512];
  __shared__ float data_shared[1024];
  __shared__ float kernel_shared[4096];
  for (int i_c_outer_inner_init = 0; i_c_outer_inner_init < 4; ++i_c_outer_inner_init) {
    for (int j_c_outer_inner_init = 0; j_c_outer_inner_init < 2; ++j_c_outer_inner_init) {
      for (int i_c_inner_init = 0; i_c_inner_init < 16; ++i_c_inner_init) {
        for (int j_c_inner_init = 0; j_c_inner_init < 2; ++j_c_inner_init) {
          T_matmul_NN_local[((((i_c_outer_inner_init * 64) + (i_c_inner_init * 4)) + (j_c_outer_inner_init * 2)) + j_c_inner_init)] = 0.000000e+00f;
          T_matmul_NN_local[(((((i_c_outer_inner_init * 64) + (i_c_inner_init * 4)) + (j_c_outer_inner_init * 2)) + j_c_inner_init) + 256)] = 0.000000e+00f;
        }
      }
    }
  }
  for (int k_outer_outer = 0; k_outer_outer < 128; ++k_outer_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_outer_outer = 0; ax0_ax1_fused_outer_outer < 8; ++ax0_ax1_fused_outer_outer) {
      data_shared[((ax0_ax1_fused_outer_outer * 128) + ((int)threadIdx.x))] = data[((((((((int)blockIdx.x) >> 3) * 131072) + (ax0_ax1_fused_outer_outer * 16384)) + ((((int)threadIdx.x) >> 3) * 1024)) + (k_outer_outer * 8)) + (((int)threadIdx.x) & 7))];
    }
    for (int ax0_ax1_fused_outer_outer_1 = 0; ax0_ax1_fused_outer_outer_1 < 32; ++ax0_ax1_fused_outer_outer_1) {
      kernel_shared[((ax0_ax1_fused_outer_outer_1 * 128) + ((int)threadIdx.x))] = kernel[(((((k_outer_outer * 32768) + ((ax0_ax1_fused_outer_outer_1 >> 2) * 4096)) + ((((int)blockIdx.x) & 7) * 512)) + ((ax0_ax1_fused_outer_outer_1 & 3) * 128)) + ((int)threadIdx.x))];
    }
    __syncthreads();
    for (int i_c_outer_inner = 0; i_c_outer_inner < 4; ++i_c_outer_inner) {
      for (int j_c_outer_inner = 0; j_c_outer_inner < 2; ++j_c_outer_inner) {
        for (int k_inner = 0; k_inner < 8; ++k_inner) {
          for (int i_c_inner = 0; i_c_inner < 16; ++i_c_inner) {
            for (int j_c_inner = 0; j_c_inner < 2; ++j_c_inner) {
              T_matmul_NN_local[((((i_c_outer_inner * 64) + (i_c_inner * 4)) + (j_c_outer_inner * 2)) + j_c_inner)] = (T_matmul_NN_local[((((i_c_outer_inner * 64) + (i_c_inner * 4)) + (j_c_outer_inner * 2)) + j_c_inner)] + (data_shared[(((i_c_outer_inner * 128) + (i_c_inner * 8)) + k_inner)] * kernel_shared[((((k_inner * 512) + (((int)threadIdx.x) * 4)) + (j_c_outer_inner * 2)) + j_c_inner)]));
              T_matmul_NN_local[(((((i_c_outer_inner * 64) + (i_c_inner * 4)) + (j_c_outer_inner * 2)) + j_c_inner) + 256)] = (T_matmul_NN_local[(((((i_c_outer_inner * 64) + (i_c_inner * 4)) + (j_c_outer_inner * 2)) + j_c_inner) + 256)] + (data_shared[((((i_c_outer_inner * 128) + (i_c_inner * 8)) + k_inner) + 512)] * kernel_shared[((((k_inner * 512) + (((int)threadIdx.x) * 4)) + (j_c_outer_inner * 2)) + j_c_inner)]));
            }
          }
        }
      }
    }
  }
  for (int i_inner = 0; i_inner < 64; ++i_inner) {
    for (int j_inner = 0; j_inner < 4; ++j_inner) {
      T_matmul_NN[((((((((int)blockIdx.x) >> 3) * 524288) + (i_inner * 4096)) + ((((int)blockIdx.x) & 7) * 512)) + (((int)threadIdx.x) * 4)) + j_inner)] = T_matmul_NN_local[((i_inner * 4) + j_inner)];
      T_matmul_NN[(((((((((int)blockIdx.x) >> 3) * 524288) + (i_inner * 4096)) + ((((int)blockIdx.x) & 7) * 512)) + (((int)threadIdx.x) * 4)) + j_inner) + 262144)] = T_matmul_NN_local[(((i_inner * 4) + j_inner) + 256)];
    }
  }
}

