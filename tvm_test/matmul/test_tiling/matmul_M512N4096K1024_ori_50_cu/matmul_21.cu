
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
  float T_matmul_NN_local[8192];
  __shared__ float data_shared[4096];
  __shared__ float kernel_shared[8192];
  for (int i_c_outer_inner_init = 0; i_c_outer_inner_init < 512; ++i_c_outer_inner_init) {
    for (int j_c_outer_inner_init = 0; j_c_outer_inner_init < 2; ++j_c_outer_inner_init) {
      T_matmul_NN_local[((i_c_outer_inner_init * 2) + j_c_outer_inner_init)] = 0.000000e+00f;
      T_matmul_NN_local[(((i_c_outer_inner_init * 2) + j_c_outer_inner_init) + 1024)] = 0.000000e+00f;
      T_matmul_NN_local[(((i_c_outer_inner_init * 2) + j_c_outer_inner_init) + 2048)] = 0.000000e+00f;
      T_matmul_NN_local[(((i_c_outer_inner_init * 2) + j_c_outer_inner_init) + 3072)] = 0.000000e+00f;
      T_matmul_NN_local[(((i_c_outer_inner_init * 2) + j_c_outer_inner_init) + 4096)] = 0.000000e+00f;
      T_matmul_NN_local[(((i_c_outer_inner_init * 2) + j_c_outer_inner_init) + 5120)] = 0.000000e+00f;
      T_matmul_NN_local[(((i_c_outer_inner_init * 2) + j_c_outer_inner_init) + 6144)] = 0.000000e+00f;
      T_matmul_NN_local[(((i_c_outer_inner_init * 2) + j_c_outer_inner_init) + 7168)] = 0.000000e+00f;
    }
  }
  for (int k_outer_outer = 0; k_outer_outer < 128; ++k_outer_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_outer_outer = 0; ax0_ax1_fused_outer_outer < 64; ++ax0_ax1_fused_outer_outer) {
      data_shared[((ax0_ax1_fused_outer_outer * 64) + ((int)threadIdx.x))] = data[((((ax0_ax1_fused_outer_outer * 8192) + ((((int)threadIdx.x) >> 3) * 1024)) + (k_outer_outer * 8)) + (((int)threadIdx.x) & 7))];
    }
    for (int ax0_ax1_fused_outer_outer_1 = 0; ax0_ax1_fused_outer_outer_1 < 128; ++ax0_ax1_fused_outer_outer_1) {
      kernel_shared[((ax0_ax1_fused_outer_outer_1 * 64) + ((int)threadIdx.x))] = kernel[(((((k_outer_outer * 32768) + ((ax0_ax1_fused_outer_outer_1 >> 4) * 4096)) + (((int)blockIdx.x) * 1024)) + ((ax0_ax1_fused_outer_outer_1 & 15) * 64)) + ((int)threadIdx.x))];
    }
    __syncthreads();
    for (int i_c_outer_inner = 0; i_c_outer_inner < 512; ++i_c_outer_inner) {
      for (int j_c_outer_inner = 0; j_c_outer_inner < 2; ++j_c_outer_inner) {
        for (int k_inner = 0; k_inner < 8; ++k_inner) {
          T_matmul_NN_local[((i_c_outer_inner * 2) + j_c_outer_inner)] = (T_matmul_NN_local[((i_c_outer_inner * 2) + j_c_outer_inner)] + (data_shared[((i_c_outer_inner * 8) + k_inner)] * kernel_shared[(((k_inner * 1024) + (((int)threadIdx.x) * 2)) + j_c_outer_inner)]));
          T_matmul_NN_local[(((i_c_outer_inner * 2) + j_c_outer_inner) + 1024)] = (T_matmul_NN_local[(((i_c_outer_inner * 2) + j_c_outer_inner) + 1024)] + (data_shared[((i_c_outer_inner * 8) + k_inner)] * kernel_shared[((((k_inner * 1024) + (((int)threadIdx.x) * 2)) + j_c_outer_inner) + 128)]));
          T_matmul_NN_local[(((i_c_outer_inner * 2) + j_c_outer_inner) + 2048)] = (T_matmul_NN_local[(((i_c_outer_inner * 2) + j_c_outer_inner) + 2048)] + (data_shared[((i_c_outer_inner * 8) + k_inner)] * kernel_shared[((((k_inner * 1024) + (((int)threadIdx.x) * 2)) + j_c_outer_inner) + 256)]));
          T_matmul_NN_local[(((i_c_outer_inner * 2) + j_c_outer_inner) + 3072)] = (T_matmul_NN_local[(((i_c_outer_inner * 2) + j_c_outer_inner) + 3072)] + (data_shared[((i_c_outer_inner * 8) + k_inner)] * kernel_shared[((((k_inner * 1024) + (((int)threadIdx.x) * 2)) + j_c_outer_inner) + 384)]));
          T_matmul_NN_local[(((i_c_outer_inner * 2) + j_c_outer_inner) + 4096)] = (T_matmul_NN_local[(((i_c_outer_inner * 2) + j_c_outer_inner) + 4096)] + (data_shared[((i_c_outer_inner * 8) + k_inner)] * kernel_shared[((((k_inner * 1024) + (((int)threadIdx.x) * 2)) + j_c_outer_inner) + 512)]));
          T_matmul_NN_local[(((i_c_outer_inner * 2) + j_c_outer_inner) + 5120)] = (T_matmul_NN_local[(((i_c_outer_inner * 2) + j_c_outer_inner) + 5120)] + (data_shared[((i_c_outer_inner * 8) + k_inner)] * kernel_shared[((((k_inner * 1024) + (((int)threadIdx.x) * 2)) + j_c_outer_inner) + 640)]));
          T_matmul_NN_local[(((i_c_outer_inner * 2) + j_c_outer_inner) + 6144)] = (T_matmul_NN_local[(((i_c_outer_inner * 2) + j_c_outer_inner) + 6144)] + (data_shared[((i_c_outer_inner * 8) + k_inner)] * kernel_shared[((((k_inner * 1024) + (((int)threadIdx.x) * 2)) + j_c_outer_inner) + 768)]));
          T_matmul_NN_local[(((i_c_outer_inner * 2) + j_c_outer_inner) + 7168)] = (T_matmul_NN_local[(((i_c_outer_inner * 2) + j_c_outer_inner) + 7168)] + (data_shared[((i_c_outer_inner * 8) + k_inner)] * kernel_shared[((((k_inner * 1024) + (((int)threadIdx.x) * 2)) + j_c_outer_inner) + 896)]));
        }
      }
    }
  }
  for (int i_inner = 0; i_inner < 512; ++i_inner) {
    for (int j_inner = 0; j_inner < 2; ++j_inner) {
      T_matmul_NN[((((i_inner * 4096) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.x) * 2)) + j_inner)] = T_matmul_NN_local[((i_inner * 2) + j_inner)];
      T_matmul_NN[(((((i_inner * 4096) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.x) * 2)) + j_inner) + 128)] = T_matmul_NN_local[(((i_inner * 2) + j_inner) + 1024)];
      T_matmul_NN[(((((i_inner * 4096) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.x) * 2)) + j_inner) + 256)] = T_matmul_NN_local[(((i_inner * 2) + j_inner) + 2048)];
      T_matmul_NN[(((((i_inner * 4096) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.x) * 2)) + j_inner) + 384)] = T_matmul_NN_local[(((i_inner * 2) + j_inner) + 3072)];
      T_matmul_NN[(((((i_inner * 4096) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.x) * 2)) + j_inner) + 512)] = T_matmul_NN_local[(((i_inner * 2) + j_inner) + 4096)];
      T_matmul_NN[(((((i_inner * 4096) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.x) * 2)) + j_inner) + 640)] = T_matmul_NN_local[(((i_inner * 2) + j_inner) + 5120)];
      T_matmul_NN[(((((i_inner * 4096) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.x) * 2)) + j_inner) + 768)] = T_matmul_NN_local[(((i_inner * 2) + j_inner) + 6144)];
      T_matmul_NN[(((((i_inner * 4096) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.x) * 2)) + j_inner) + 896)] = T_matmul_NN_local[(((i_inner * 2) + j_inner) + 7168)];
    }
  }
}

