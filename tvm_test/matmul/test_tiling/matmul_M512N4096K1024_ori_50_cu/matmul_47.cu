
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
  float T_matmul_NN_local[32];
  __shared__ float data_shared[64];
  __shared__ float kernel_shared[1024];
  for (int i_c_outer_inner_init = 0; i_c_outer_inner_init < 4; ++i_c_outer_inner_init) {
    for (int i_c_inner_init = 0; i_c_inner_init < 4; ++i_c_inner_init) {
      T_matmul_NN_local[((i_c_outer_inner_init * 4) + i_c_inner_init)] = 0.000000e+00f;
      T_matmul_NN_local[(((i_c_outer_inner_init * 4) + i_c_inner_init) + 16)] = 0.000000e+00f;
    }
  }
  for (int k_outer_outer = 0; k_outer_outer < 256; ++k_outer_outer) {
    __syncthreads();
    if (((int)threadIdx.x) < 64) {
      data_shared[((int)threadIdx.x)] = data[(((((((int)blockIdx.x) >> 4) * 16384) + ((((int)threadIdx.x) >> 2) * 1024)) + (k_outer_outer * 4)) + (((int)threadIdx.x) & 3))];
    }
    for (int ax0_ax1_fused_outer_outer = 0; ax0_ax1_fused_outer_outer < 8; ++ax0_ax1_fused_outer_outer) {
      kernel_shared[((ax0_ax1_fused_outer_outer * 128) + ((int)threadIdx.x))] = kernel[(((((k_outer_outer * 16384) + ((ax0_ax1_fused_outer_outer >> 1) * 4096)) + ((((int)blockIdx.x) & 15) * 256)) + ((ax0_ax1_fused_outer_outer & 1) * 128)) + ((int)threadIdx.x))];
    }
    __syncthreads();
    for (int k_outer_inner = 0; k_outer_inner < 2; ++k_outer_inner) {
      for (int i_c_outer_inner = 0; i_c_outer_inner < 4; ++i_c_outer_inner) {
        for (int k_inner = 0; k_inner < 2; ++k_inner) {
          for (int i_c_inner = 0; i_c_inner < 4; ++i_c_inner) {
            T_matmul_NN_local[((i_c_outer_inner * 4) + i_c_inner)] = (T_matmul_NN_local[((i_c_outer_inner * 4) + i_c_inner)] + (data_shared[((((i_c_outer_inner * 16) + (i_c_inner * 4)) + (k_outer_inner * 2)) + k_inner)] * kernel_shared[(((k_outer_inner * 512) + (k_inner * 256)) + ((int)threadIdx.x))]));
            T_matmul_NN_local[(((i_c_outer_inner * 4) + i_c_inner) + 16)] = (T_matmul_NN_local[(((i_c_outer_inner * 4) + i_c_inner) + 16)] + (data_shared[((((i_c_outer_inner * 16) + (i_c_inner * 4)) + (k_outer_inner * 2)) + k_inner)] * kernel_shared[((((k_outer_inner * 512) + (k_inner * 256)) + ((int)threadIdx.x)) + 128)]));
          }
        }
      }
    }
  }
  for (int i_inner = 0; i_inner < 16; ++i_inner) {
    T_matmul_NN[(((((((int)blockIdx.x) >> 4) * 65536) + (i_inner * 4096)) + ((((int)blockIdx.x) & 15) * 256)) + ((int)threadIdx.x))] = T_matmul_NN_local[i_inner];
    T_matmul_NN[((((((((int)blockIdx.x) >> 4) * 65536) + (i_inner * 4096)) + ((((int)blockIdx.x) & 15) * 256)) + ((int)threadIdx.x)) + 128)] = T_matmul_NN_local[(i_inner + 16)];
  }
}

