
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
extern "C" __global__ void __launch_bounds__(64) mymatmul_kernel0(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ T_matmul_NN) {
  float T_matmul_NN_local[32];
  __shared__ float data_shared[4096];
  __shared__ float kernel_shared[8192];
  for (int i_c_outer_inner_init = 0; i_c_outer_inner_init < 8; ++i_c_outer_inner_init) {
    for (int j_c_inner_init = 0; j_c_inner_init < 2; ++j_c_inner_init) {
      T_matmul_NN_local[((i_c_outer_inner_init * 2) + j_c_inner_init)] = 0.000000e+00f;
      T_matmul_NN_local[(((i_c_outer_inner_init * 2) + j_c_inner_init) + 16)] = 0.000000e+00f;
    }
  }
  for (int k_outer_outer = 0; k_outer_outer < 8; ++k_outer_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_outer_outer = 0; ax0_ax1_fused_outer_outer < 64; ++ax0_ax1_fused_outer_outer) {
      data_shared[((ax0_ax1_fused_outer_outer * 64) + ((int)threadIdx.x))] = data[((((((((int)blockIdx.x) >> 6) * 32768) + ((ax0_ax1_fused_outer_outer >> 1) * 1024)) + (k_outer_outer * 128)) + ((ax0_ax1_fused_outer_outer & 1) * 64)) + ((int)threadIdx.x))];
    }
    for (int ax0_ax1_fused_outer_outer_1 = 0; ax0_ax1_fused_outer_outer_1 < 128; ++ax0_ax1_fused_outer_outer_1) {
      kernel_shared[((ax0_ax1_fused_outer_outer_1 * 64) + ((int)threadIdx.x))] = kernel[((((k_outer_outer * 524288) + (ax0_ax1_fused_outer_outer_1 * 4096)) + ((((int)blockIdx.x) & 63) * 64)) + ((int)threadIdx.x))];
    }
    __syncthreads();
    for (int k_outer_inner = 0; k_outer_inner < 8; ++k_outer_inner) {
      for (int i_c_outer_inner = 0; i_c_outer_inner < 8; ++i_c_outer_inner) {
        for (int k_inner = 0; k_inner < 16; ++k_inner) {
          for (int j_c_inner = 0; j_c_inner < 2; ++j_c_inner) {
            T_matmul_NN_local[((i_c_outer_inner * 2) + j_c_inner)] = (T_matmul_NN_local[((i_c_outer_inner * 2) + j_c_inner)] + (data_shared[(((((((int)threadIdx.x) >> 5) * 1024) + (i_c_outer_inner * 128)) + (k_outer_inner * 16)) + k_inner)] * kernel_shared[((((k_outer_inner * 1024) + (k_inner * 64)) + ((((int)threadIdx.x) & 31) * 2)) + j_c_inner)]));
            T_matmul_NN_local[(((i_c_outer_inner * 2) + j_c_inner) + 16)] = (T_matmul_NN_local[(((i_c_outer_inner * 2) + j_c_inner) + 16)] + (data_shared[((((((((int)threadIdx.x) >> 5) * 1024) + (i_c_outer_inner * 128)) + (k_outer_inner * 16)) + k_inner) + 2048)] * kernel_shared[((((k_outer_inner * 1024) + (k_inner * 64)) + ((((int)threadIdx.x) & 31) * 2)) + j_c_inner)]));
          }
        }
      }
    }
  }
  for (int i_inner = 0; i_inner < 8; ++i_inner) {
    for (int j_inner = 0; j_inner < 2; ++j_inner) {
      T_matmul_NN[(((((((((int)blockIdx.x) >> 6) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 63) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + j_inner)] = T_matmul_NN_local[((i_inner * 2) + j_inner)];
      T_matmul_NN[((((((((((int)blockIdx.x) >> 6) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 63) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + j_inner) + 65536)] = T_matmul_NN_local[(((i_inner * 2) + j_inner) + 16)];
    }
  }
}

