
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
  float T_matmul_NN_local[2048];
  __shared__ float data_shared[256];
  __shared__ float kernel_shared[512];
  for (int i_c_outer_inner_init = 0; i_c_outer_inner_init < 128; ++i_c_outer_inner_init) {
    for (int j_c_outer_inner_init = 0; j_c_outer_inner_init < 8; ++j_c_outer_inner_init) {
      T_matmul_NN_local[((i_c_outer_inner_init * 8) + j_c_outer_inner_init)] = 0.000000e+00f;
      T_matmul_NN_local[(((i_c_outer_inner_init * 8) + j_c_outer_inner_init) + 1024)] = 0.000000e+00f;
    }
  }
  for (int k_outer_outer = 0; k_outer_outer < 1024; ++k_outer_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_inner_s = 0; ax0_ax1_fused_inner_s < 16; ++ax0_ax1_fused_inner_s) {
      if (((int)threadIdx.x) < 16) {
        data_shared[((((int)threadIdx.x) * 16) + ax0_ax1_fused_inner_s)] = data[(((((((int)blockIdx.x) >> 3) * 262144) + (((int)threadIdx.x) * 16384)) + (ax0_ax1_fused_inner_s * 1024)) + k_outer_outer)];
      }
    }
    for (int ax0_ax1_fused_outer_outer = 0; ax0_ax1_fused_outer_outer < 8; ++ax0_ax1_fused_outer_outer) {
      kernel_shared[((ax0_ax1_fused_outer_outer * 64) + ((int)threadIdx.x))] = kernel[((((k_outer_outer * 4096) + ((((int)blockIdx.x) & 7) * 512)) + (ax0_ax1_fused_outer_outer * 64)) + ((int)threadIdx.x))];
    }
    __syncthreads();
    for (int i_c_outer_inner = 0; i_c_outer_inner < 128; ++i_c_outer_inner) {
      for (int j_c_outer_inner = 0; j_c_outer_inner < 8; ++j_c_outer_inner) {
        T_matmul_NN_local[((i_c_outer_inner * 8) + j_c_outer_inner)] = (T_matmul_NN_local[((i_c_outer_inner * 8) + j_c_outer_inner)] + (data_shared[(((((int)threadIdx.x) >> 5) * 128) + i_c_outer_inner)] * kernel_shared[(((((int)threadIdx.x) & 31) * 8) + j_c_outer_inner)]));
        T_matmul_NN_local[(((i_c_outer_inner * 8) + j_c_outer_inner) + 1024)] = (T_matmul_NN_local[(((i_c_outer_inner * 8) + j_c_outer_inner) + 1024)] + (data_shared[(((((int)threadIdx.x) >> 5) * 128) + i_c_outer_inner)] * kernel_shared[((((((int)threadIdx.x) & 31) * 8) + j_c_outer_inner) + 256)]));
      }
    }
  }
  for (int i_inner = 0; i_inner < 128; ++i_inner) {
    for (int j_inner = 0; j_inner < 8; ++j_inner) {
      T_matmul_NN[(((((((((int)blockIdx.x) >> 3) * 1048576) + ((((int)threadIdx.x) >> 5) * 524288)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 7) * 512)) + ((((int)threadIdx.x) & 31) * 8)) + j_inner)] = T_matmul_NN_local[((i_inner * 8) + j_inner)];
      T_matmul_NN[((((((((((int)blockIdx.x) >> 3) * 1048576) + ((((int)threadIdx.x) >> 5) * 524288)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 7) * 512)) + ((((int)threadIdx.x) & 31) * 8)) + j_inner) + 256)] = T_matmul_NN_local[(((i_inner * 8) + j_inner) + 1024)];
    }
  }
}

