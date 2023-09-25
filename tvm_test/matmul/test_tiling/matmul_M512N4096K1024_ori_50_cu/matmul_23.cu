
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
  float T_matmul_NN_local[128];
  __shared__ float data_shared[2048];
  __shared__ float kernel_shared[64];
  for (int i_c_outer_inner_init = 0; i_c_outer_inner_init < 2; ++i_c_outer_inner_init) {
    for (int i_c_inner_init = 0; i_c_inner_init < 4; ++i_c_inner_init) {
      for (int j_c_inner_init = 0; j_c_inner_init < 4; ++j_c_inner_init) {
        T_matmul_NN_local[(((i_c_outer_inner_init * 16) + (i_c_inner_init * 4)) + j_c_inner_init)] = 0.000000e+00f;
        T_matmul_NN_local[((((i_c_outer_inner_init * 16) + (i_c_inner_init * 4)) + j_c_inner_init) + 32)] = 0.000000e+00f;
        T_matmul_NN_local[((((i_c_outer_inner_init * 16) + (i_c_inner_init * 4)) + j_c_inner_init) + 64)] = 0.000000e+00f;
        T_matmul_NN_local[((((i_c_outer_inner_init * 16) + (i_c_inner_init * 4)) + j_c_inner_init) + 96)] = 0.000000e+00f;
      }
    }
  }
  for (int k_outer_outer = 0; k_outer_outer < 256; ++k_outer_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_outer_outer = 0; ax0_ax1_fused_outer_outer < 32; ++ax0_ax1_fused_outer_outer) {
      data_shared[((ax0_ax1_fused_outer_outer * 64) + ((int)threadIdx.x))] = data[((((ax0_ax1_fused_outer_outer * 16384) + ((((int)threadIdx.x) >> 2) * 1024)) + (k_outer_outer * 4)) + (((int)threadIdx.x) & 3))];
    }
    kernel_shared[((int)threadIdx.x)] = kernel[((((k_outer_outer * 16384) + ((((int)threadIdx.x) >> 4) * 4096)) + (((int)blockIdx.x) * 16)) + (((int)threadIdx.x) & 15))];
    __syncthreads();
    for (int i_c_outer_inner = 0; i_c_outer_inner < 2; ++i_c_outer_inner) {
      for (int k_inner = 0; k_inner < 4; ++k_inner) {
        for (int i_c_inner = 0; i_c_inner < 4; ++i_c_inner) {
          for (int j_c_inner = 0; j_c_inner < 4; ++j_c_inner) {
            T_matmul_NN_local[(((i_c_outer_inner * 16) + (i_c_inner * 4)) + j_c_inner)] = (T_matmul_NN_local[(((i_c_outer_inner * 16) + (i_c_inner * 4)) + j_c_inner)] + (data_shared[((((((int)threadIdx.x) * 32) + (i_c_outer_inner * 16)) + (i_c_inner * 4)) + k_inner)] * kernel_shared[((k_inner * 16) + j_c_inner)]));
            T_matmul_NN_local[((((i_c_outer_inner * 16) + (i_c_inner * 4)) + j_c_inner) + 32)] = (T_matmul_NN_local[((((i_c_outer_inner * 16) + (i_c_inner * 4)) + j_c_inner) + 32)] + (data_shared[((((((int)threadIdx.x) * 32) + (i_c_outer_inner * 16)) + (i_c_inner * 4)) + k_inner)] * kernel_shared[(((k_inner * 16) + j_c_inner) + 4)]));
            T_matmul_NN_local[((((i_c_outer_inner * 16) + (i_c_inner * 4)) + j_c_inner) + 64)] = (T_matmul_NN_local[((((i_c_outer_inner * 16) + (i_c_inner * 4)) + j_c_inner) + 64)] + (data_shared[((((((int)threadIdx.x) * 32) + (i_c_outer_inner * 16)) + (i_c_inner * 4)) + k_inner)] * kernel_shared[(((k_inner * 16) + j_c_inner) + 8)]));
            T_matmul_NN_local[((((i_c_outer_inner * 16) + (i_c_inner * 4)) + j_c_inner) + 96)] = (T_matmul_NN_local[((((i_c_outer_inner * 16) + (i_c_inner * 4)) + j_c_inner) + 96)] + (data_shared[((((((int)threadIdx.x) * 32) + (i_c_outer_inner * 16)) + (i_c_inner * 4)) + k_inner)] * kernel_shared[(((k_inner * 16) + j_c_inner) + 12)]));
          }
        }
      }
    }
  }
  for (int i_inner = 0; i_inner < 8; ++i_inner) {
    for (int j_inner = 0; j_inner < 4; ++j_inner) {
      T_matmul_NN[((((((int)threadIdx.x) * 32768) + (i_inner * 4096)) + (((int)blockIdx.x) * 16)) + j_inner)] = T_matmul_NN_local[((i_inner * 4) + j_inner)];
      T_matmul_NN[(((((((int)threadIdx.x) * 32768) + (i_inner * 4096)) + (((int)blockIdx.x) * 16)) + j_inner) + 4)] = T_matmul_NN_local[(((i_inner * 4) + j_inner) + 32)];
      T_matmul_NN[(((((((int)threadIdx.x) * 32768) + (i_inner * 4096)) + (((int)blockIdx.x) * 16)) + j_inner) + 8)] = T_matmul_NN_local[(((i_inner * 4) + j_inner) + 64)];
      T_matmul_NN[(((((((int)threadIdx.x) * 32768) + (i_inner * 4096)) + (((int)blockIdx.x) * 16)) + j_inner) + 12)] = T_matmul_NN_local[(((i_inner * 4) + j_inner) + 96)];
    }
  }
}

