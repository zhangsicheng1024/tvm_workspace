
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
  float T_matmul_NN_local[4];
  __shared__ float data_shared[32];
  __shared__ float kernel_shared[2048];
  for (int i_c_outer_inner_init = 0; i_c_outer_inner_init < 2; ++i_c_outer_inner_init) {
    for (int j_c_outer_inner_init = 0; j_c_outer_inner_init < 2; ++j_c_outer_inner_init) {
      T_matmul_NN_local[((i_c_outer_inner_init * 2) + j_c_outer_inner_init)] = 0.000000e+00f;
    }
  }
  for (int k_outer_outer = 0; k_outer_outer < 64; ++k_outer_outer) {
    __syncthreads();
    if (((int)threadIdx.x) < 32) {
      data_shared[((int)threadIdx.x)] = data[(((((((int)blockIdx.x) >> 5) * 2048) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer_outer * 16)) + (((int)threadIdx.x) & 15))];
    }
    for (int ax0_ax1_fused_outer_outer = 0; ax0_ax1_fused_outer_outer < 16; ++ax0_ax1_fused_outer_outer) {
      *(float2*)(kernel_shared + ((ax0_ax1_fused_outer_outer * 128) + (((int)threadIdx.x) * 2))) = *(float2*)(kernel + ((((k_outer_outer * 65536) + (ax0_ax1_fused_outer_outer * 4096)) + ((((int)blockIdx.x) & 31) * 128)) + (((int)threadIdx.x) * 2)));
    }
    __syncthreads();
    for (int k_outer_inner = 0; k_outer_inner < 4; ++k_outer_inner) {
      for (int i_c_outer_inner = 0; i_c_outer_inner < 2; ++i_c_outer_inner) {
        for (int j_c_outer_inner = 0; j_c_outer_inner < 2; ++j_c_outer_inner) {
          for (int k_inner = 0; k_inner < 4; ++k_inner) {
            T_matmul_NN_local[((i_c_outer_inner * 2) + j_c_outer_inner)] = (T_matmul_NN_local[((i_c_outer_inner * 2) + j_c_outer_inner)] + (data_shared[(((i_c_outer_inner * 16) + (k_outer_inner * 4)) + k_inner)] * kernel_shared[((((k_outer_inner * 512) + (k_inner * 128)) + (((int)threadIdx.x) * 2)) + j_c_outer_inner)]));
          }
        }
      }
    }
  }
  for (int i_inner = 0; i_inner < 2; ++i_inner) {
    for (int j_inner = 0; j_inner < 2; ++j_inner) {
      T_matmul_NN[((((((((int)blockIdx.x) >> 5) * 8192) + (i_inner * 4096)) + ((((int)blockIdx.x) & 31) * 128)) + (((int)threadIdx.x) * 2)) + j_inner)] = T_matmul_NN_local[((i_inner * 2) + j_inner)];
    }
  }
}

