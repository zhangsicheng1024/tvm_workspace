
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
extern "C" __global__ void __launch_bounds__(4) mymatmul_kernel0(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ T_matmul_NN) {
  float T_matmul_NN_local[8192];
  __shared__ float data_shared[8192];
  __shared__ float kernel_shared[1024];
  for (int i_c_outer_inner_init = 0; i_c_outer_inner_init < 64; ++i_c_outer_inner_init) {
    for (int j_c_inner_init = 0; j_c_inner_init < 16; ++j_c_inner_init) {
      T_matmul_NN_local[((i_c_outer_inner_init * 16) + j_c_inner_init)] = 0.000000e+00f;
      T_matmul_NN_local[(((i_c_outer_inner_init * 16) + j_c_inner_init) + 1024)] = 0.000000e+00f;
      T_matmul_NN_local[(((i_c_outer_inner_init * 16) + j_c_inner_init) + 2048)] = 0.000000e+00f;
      T_matmul_NN_local[(((i_c_outer_inner_init * 16) + j_c_inner_init) + 3072)] = 0.000000e+00f;
      T_matmul_NN_local[(((i_c_outer_inner_init * 16) + j_c_inner_init) + 4096)] = 0.000000e+00f;
      T_matmul_NN_local[(((i_c_outer_inner_init * 16) + j_c_inner_init) + 5120)] = 0.000000e+00f;
      T_matmul_NN_local[(((i_c_outer_inner_init * 16) + j_c_inner_init) + 6144)] = 0.000000e+00f;
      T_matmul_NN_local[(((i_c_outer_inner_init * 16) + j_c_inner_init) + 7168)] = 0.000000e+00f;
    }
  }
  for (int k_outer_outer = 0; k_outer_outer < 64; ++k_outer_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_outer_outer = 0; ax0_ax1_fused_outer_outer < 1024; ++ax0_ax1_fused_outer_outer) {
      *(float2*)(data_shared + ((ax0_ax1_fused_outer_outer * 8) + (((int)threadIdx.x) * 2))) = *(float2*)(data + (((((ax0_ax1_fused_outer_outer >> 1) * 1024) + (k_outer_outer * 16)) + ((ax0_ax1_fused_outer_outer & 1) * 8)) + (((int)threadIdx.x) * 2)));
    }
    for (int ax0_ax1_fused_outer_outer_1 = 0; ax0_ax1_fused_outer_outer_1 < 256; ++ax0_ax1_fused_outer_outer_1) {
      kernel_shared[((ax0_ax1_fused_outer_outer_1 * 4) + ((int)threadIdx.x))] = kernel[(((((k_outer_outer * 65536) + ((ax0_ax1_fused_outer_outer_1 >> 4) * 4096)) + (((int)blockIdx.x) * 64)) + ((ax0_ax1_fused_outer_outer_1 & 15) * 4)) + ((int)threadIdx.x))];
    }
    __syncthreads();
    for (int k_outer_inner = 0; k_outer_inner < 8; ++k_outer_inner) {
      for (int i_c_outer_inner = 0; i_c_outer_inner < 64; ++i_c_outer_inner) {
        for (int k_inner = 0; k_inner < 2; ++k_inner) {
          for (int j_c_inner = 0; j_c_inner < 16; ++j_c_inner) {
            T_matmul_NN_local[((i_c_outer_inner * 16) + j_c_inner)] = (T_matmul_NN_local[((i_c_outer_inner * 16) + j_c_inner)] + (data_shared[(((i_c_outer_inner * 16) + (k_outer_inner * 2)) + k_inner)] * kernel_shared[((((k_outer_inner * 128) + (k_inner * 64)) + (((int)threadIdx.x) * 16)) + j_c_inner)]));
            T_matmul_NN_local[(((i_c_outer_inner * 16) + j_c_inner) + 1024)] = (T_matmul_NN_local[(((i_c_outer_inner * 16) + j_c_inner) + 1024)] + (data_shared[((((i_c_outer_inner * 16) + (k_outer_inner * 2)) + k_inner) + 1024)] * kernel_shared[((((k_outer_inner * 128) + (k_inner * 64)) + (((int)threadIdx.x) * 16)) + j_c_inner)]));
            T_matmul_NN_local[(((i_c_outer_inner * 16) + j_c_inner) + 2048)] = (T_matmul_NN_local[(((i_c_outer_inner * 16) + j_c_inner) + 2048)] + (data_shared[((((i_c_outer_inner * 16) + (k_outer_inner * 2)) + k_inner) + 2048)] * kernel_shared[((((k_outer_inner * 128) + (k_inner * 64)) + (((int)threadIdx.x) * 16)) + j_c_inner)]));
            T_matmul_NN_local[(((i_c_outer_inner * 16) + j_c_inner) + 3072)] = (T_matmul_NN_local[(((i_c_outer_inner * 16) + j_c_inner) + 3072)] + (data_shared[((((i_c_outer_inner * 16) + (k_outer_inner * 2)) + k_inner) + 3072)] * kernel_shared[((((k_outer_inner * 128) + (k_inner * 64)) + (((int)threadIdx.x) * 16)) + j_c_inner)]));
            T_matmul_NN_local[(((i_c_outer_inner * 16) + j_c_inner) + 4096)] = (T_matmul_NN_local[(((i_c_outer_inner * 16) + j_c_inner) + 4096)] + (data_shared[((((i_c_outer_inner * 16) + (k_outer_inner * 2)) + k_inner) + 4096)] * kernel_shared[((((k_outer_inner * 128) + (k_inner * 64)) + (((int)threadIdx.x) * 16)) + j_c_inner)]));
            T_matmul_NN_local[(((i_c_outer_inner * 16) + j_c_inner) + 5120)] = (T_matmul_NN_local[(((i_c_outer_inner * 16) + j_c_inner) + 5120)] + (data_shared[((((i_c_outer_inner * 16) + (k_outer_inner * 2)) + k_inner) + 5120)] * kernel_shared[((((k_outer_inner * 128) + (k_inner * 64)) + (((int)threadIdx.x) * 16)) + j_c_inner)]));
            T_matmul_NN_local[(((i_c_outer_inner * 16) + j_c_inner) + 6144)] = (T_matmul_NN_local[(((i_c_outer_inner * 16) + j_c_inner) + 6144)] + (data_shared[((((i_c_outer_inner * 16) + (k_outer_inner * 2)) + k_inner) + 6144)] * kernel_shared[((((k_outer_inner * 128) + (k_inner * 64)) + (((int)threadIdx.x) * 16)) + j_c_inner)]));
            T_matmul_NN_local[(((i_c_outer_inner * 16) + j_c_inner) + 7168)] = (T_matmul_NN_local[(((i_c_outer_inner * 16) + j_c_inner) + 7168)] + (data_shared[((((i_c_outer_inner * 16) + (k_outer_inner * 2)) + k_inner) + 7168)] * kernel_shared[((((k_outer_inner * 128) + (k_inner * 64)) + (((int)threadIdx.x) * 16)) + j_c_inner)]));
          }
        }
      }
    }
  }
  for (int i_inner = 0; i_inner < 64; ++i_inner) {
    for (int j_inner = 0; j_inner < 16; ++j_inner) {
      T_matmul_NN[((((i_inner * 4096) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 16)) + j_inner)] = T_matmul_NN_local[((i_inner * 16) + j_inner)];
      T_matmul_NN[(((((i_inner * 4096) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 16)) + j_inner) + 262144)] = T_matmul_NN_local[(((i_inner * 16) + j_inner) + 1024)];
      T_matmul_NN[(((((i_inner * 4096) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 16)) + j_inner) + 524288)] = T_matmul_NN_local[(((i_inner * 16) + j_inner) + 2048)];
      T_matmul_NN[(((((i_inner * 4096) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 16)) + j_inner) + 786432)] = T_matmul_NN_local[(((i_inner * 16) + j_inner) + 3072)];
      T_matmul_NN[(((((i_inner * 4096) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 16)) + j_inner) + 1048576)] = T_matmul_NN_local[(((i_inner * 16) + j_inner) + 4096)];
      T_matmul_NN[(((((i_inner * 4096) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 16)) + j_inner) + 1310720)] = T_matmul_NN_local[(((i_inner * 16) + j_inner) + 5120)];
      T_matmul_NN[(((((i_inner * 4096) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 16)) + j_inner) + 1572864)] = T_matmul_NN_local[(((i_inner * 16) + j_inner) + 6144)];
      T_matmul_NN[(((((i_inner * 4096) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 16)) + j_inner) + 1835008)] = T_matmul_NN_local[(((i_inner * 16) + j_inner) + 7168)];
    }
  }
}

