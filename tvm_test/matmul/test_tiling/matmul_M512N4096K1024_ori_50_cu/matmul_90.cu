
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
extern "C" __global__ void __launch_bounds__(1024) mymatmul_kernel0(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ T_matmul_NN) {
  float T_matmul_NN_local[128];
  __shared__ float data_shared[32];
  __shared__ float kernel_shared[4096];
  for (int i_c_outer_inner_init = 0; i_c_outer_inner_init < 4; ++i_c_outer_inner_init) {
    for (int j_c_outer_inner_init = 0; j_c_outer_inner_init < 4; ++j_c_outer_inner_init) {
      for (int i_c_inner_init = 0; i_c_inner_init < 4; ++i_c_inner_init) {
        T_matmul_NN_local[(((i_c_outer_inner_init * 16) + (i_c_inner_init * 4)) + j_c_outer_inner_init)] = 0.000000e+00f;
        T_matmul_NN_local[((((i_c_outer_inner_init * 16) + (i_c_inner_init * 4)) + j_c_outer_inner_init) + 64)] = 0.000000e+00f;
      }
    }
  }
  for (int k_outer_outer = 0; k_outer_outer < 1024; ++k_outer_outer) {
    __syncthreads();
    if (((int)threadIdx.x) < 32) {
      data_shared[((int)threadIdx.x)] = data[(((((int)blockIdx.x) * 32768) + (((int)threadIdx.x) * 1024)) + k_outer_outer)];
    }
    for (int ax0_ax1_fused_outer_outer = 0; ax0_ax1_fused_outer_outer < 4; ++ax0_ax1_fused_outer_outer) {
      kernel_shared[((ax0_ax1_fused_outer_outer * 1024) + ((int)threadIdx.x))] = kernel[(((k_outer_outer * 4096) + (ax0_ax1_fused_outer_outer * 1024)) + ((int)threadIdx.x))];
    }
    __syncthreads();
    for (int i_c_outer_inner = 0; i_c_outer_inner < 4; ++i_c_outer_inner) {
      for (int j_c_outer_inner = 0; j_c_outer_inner < 4; ++j_c_outer_inner) {
        for (int i_c_inner = 0; i_c_inner < 4; ++i_c_inner) {
          T_matmul_NN_local[(((i_c_outer_inner * 16) + (i_c_inner * 4)) + j_c_outer_inner)] = (T_matmul_NN_local[(((i_c_outer_inner * 16) + (i_c_inner * 4)) + j_c_outer_inner)] + (data_shared[((((((int)threadIdx.x) >> 9) * 16) + (i_c_outer_inner * 4)) + i_c_inner)] * kernel_shared[(((((int)threadIdx.x) & 511) * 4) + j_c_outer_inner)]));
          T_matmul_NN_local[((((i_c_outer_inner * 16) + (i_c_inner * 4)) + j_c_outer_inner) + 64)] = (T_matmul_NN_local[((((i_c_outer_inner * 16) + (i_c_inner * 4)) + j_c_outer_inner) + 64)] + (data_shared[((((((int)threadIdx.x) >> 9) * 16) + (i_c_outer_inner * 4)) + i_c_inner)] * kernel_shared[((((((int)threadIdx.x) & 511) * 4) + j_c_outer_inner) + 2048)]));
        }
      }
    }
  }
  for (int i_inner = 0; i_inner < 16; ++i_inner) {
    for (int j_inner = 0; j_inner < 4; ++j_inner) {
      T_matmul_NN[(((((((int)blockIdx.x) * 131072) + ((((int)threadIdx.x) >> 9) * 65536)) + (i_inner * 4096)) + ((((int)threadIdx.x) & 511) * 4)) + j_inner)] = T_matmul_NN_local[((i_inner * 4) + j_inner)];
      T_matmul_NN[((((((((int)blockIdx.x) * 131072) + ((((int)threadIdx.x) >> 9) * 65536)) + (i_inner * 4096)) + ((((int)threadIdx.x) & 511) * 4)) + j_inner) + 2048)] = T_matmul_NN_local[(((i_inner * 4) + j_inner) + 64)];
    }
  }
}

