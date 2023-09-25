
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
extern "C" __global__ void __launch_bounds__(16) mymatmul_kernel0(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ T_matmul_NN) {
  float T_matmul_NN_local[1024];
  __shared__ float data_shared[2048];
  __shared__ float kernel_shared[2048];
  for (int i_c_outer_inner_init = 0; i_c_outer_inner_init < 4; ++i_c_outer_inner_init) {
    for (int j_c_outer_inner_init = 0; j_c_outer_inner_init < 128; ++j_c_outer_inner_init) {
      T_matmul_NN_local[((i_c_outer_inner_init * 128) + j_c_outer_inner_init)] = 0.000000e+00f;
      T_matmul_NN_local[(((i_c_outer_inner_init * 128) + j_c_outer_inner_init) + 512)] = 0.000000e+00f;
    }
  }
  for (int k_outer_outer = 0; k_outer_outer < 64; ++k_outer_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_outer_outer = 0; ax0_ax1_fused_outer_outer < 64; ++ax0_ax1_fused_outer_outer) {
      *(float2*)(data_shared + ((ax0_ax1_fused_outer_outer * 32) + (((int)threadIdx.x) * 2))) = *(float2*)(data + ((((((((int)blockIdx.x) >> 5) * 131072) + (ax0_ax1_fused_outer_outer * 2048)) + ((((int)threadIdx.x) >> 3) * 1024)) + (k_outer_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)));
    }
    for (int ax0_ax1_fused_outer_outer_1 = 0; ax0_ax1_fused_outer_outer_1 < 128; ++ax0_ax1_fused_outer_outer_1) {
      kernel_shared[((ax0_ax1_fused_outer_outer_1 * 16) + ((int)threadIdx.x))] = kernel[(((((k_outer_outer * 65536) + ((ax0_ax1_fused_outer_outer_1 >> 3) * 4096)) + ((((int)blockIdx.x) & 31) * 128)) + ((ax0_ax1_fused_outer_outer_1 & 7) * 16)) + ((int)threadIdx.x))];
    }
    __syncthreads();
    for (int k_outer_inner = 0; k_outer_inner < 2; ++k_outer_inner) {
      for (int i_c_outer_inner = 0; i_c_outer_inner < 4; ++i_c_outer_inner) {
        for (int j_c_outer_inner = 0; j_c_outer_inner < 128; ++j_c_outer_inner) {
          for (int k_inner = 0; k_inner < 8; ++k_inner) {
            T_matmul_NN_local[((i_c_outer_inner * 128) + j_c_outer_inner)] = (T_matmul_NN_local[((i_c_outer_inner * 128) + j_c_outer_inner)] + (data_shared[((((((int)threadIdx.x) * 64) + (i_c_outer_inner * 16)) + (k_outer_inner * 8)) + k_inner)] * kernel_shared[(((k_outer_inner * 1024) + (k_inner * 128)) + j_c_outer_inner)]));
            T_matmul_NN_local[(((i_c_outer_inner * 128) + j_c_outer_inner) + 512)] = (T_matmul_NN_local[(((i_c_outer_inner * 128) + j_c_outer_inner) + 512)] + (data_shared[(((((((int)threadIdx.x) * 64) + (i_c_outer_inner * 16)) + (k_outer_inner * 8)) + k_inner) + 1024)] * kernel_shared[(((k_outer_inner * 1024) + (k_inner * 128)) + j_c_outer_inner)]));
          }
        }
      }
    }
  }
  for (int i_inner = 0; i_inner < 4; ++i_inner) {
    for (int j_inner = 0; j_inner < 128; ++j_inner) {
      T_matmul_NN[((((((((int)blockIdx.x) >> 5) * 524288) + (((int)threadIdx.x) * 16384)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 31) * 128)) + j_inner)] = T_matmul_NN_local[((i_inner * 128) + j_inner)];
      T_matmul_NN[(((((((((int)blockIdx.x) >> 5) * 524288) + (((int)threadIdx.x) * 16384)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 31) * 128)) + j_inner) + 262144)] = T_matmul_NN_local[(((i_inner * 128) + j_inner) + 512)];
    }
  }
}

