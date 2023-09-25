
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
  float T_matmul_NN_local[1024];
  __shared__ float data_shared[1024];
  __shared__ float kernel_shared[4096];
  for (int i_c_inner_init = 0; i_c_inner_init < 16; ++i_c_inner_init) {
    for (int j_c_inner_init = 0; j_c_inner_init < 32; ++j_c_inner_init) {
      T_matmul_NN_local[((i_c_inner_init * 32) + j_c_inner_init)] = 0.000000e+00f;
      T_matmul_NN_local[(((i_c_inner_init * 32) + j_c_inner_init) + 512)] = 0.000000e+00f;
    }
  }
  for (int k_outer_outer = 0; k_outer_outer < 512; ++k_outer_outer) {
    __syncthreads();
    data_shared[((int)threadIdx.x)] = data[((((((int)threadIdx.x) >> 1) * 1024) + (k_outer_outer * 2)) + (((int)threadIdx.x) & 1))];
    for (int ax0_ax1_fused_outer_outer = 0; ax0_ax1_fused_outer_outer < 4; ++ax0_ax1_fused_outer_outer) {
      kernel_shared[((ax0_ax1_fused_outer_outer * 1024) + ((int)threadIdx.x))] = kernel[(((((k_outer_outer * 8192) + ((ax0_ax1_fused_outer_outer >> 1) * 4096)) + (((int)blockIdx.x) * 2048)) + ((ax0_ax1_fused_outer_outer & 1) * 1024)) + ((int)threadIdx.x))];
    }
    __syncthreads();
    for (int k_inner = 0; k_inner < 2; ++k_inner) {
      for (int i_c_inner = 0; i_c_inner < 16; ++i_c_inner) {
        for (int j_c_inner = 0; j_c_inner < 32; ++j_c_inner) {
          T_matmul_NN_local[((i_c_inner * 32) + j_c_inner)] = (T_matmul_NN_local[((i_c_inner * 32) + j_c_inner)] + (data_shared[((((((int)threadIdx.x) >> 6) * 32) + (i_c_inner * 2)) + k_inner)] * kernel_shared[(((k_inner * 2048) + ((((int)threadIdx.x) & 63) * 32)) + j_c_inner)]));
          T_matmul_NN_local[(((i_c_inner * 32) + j_c_inner) + 512)] = (T_matmul_NN_local[(((i_c_inner * 32) + j_c_inner) + 512)] + (data_shared[(((((((int)threadIdx.x) >> 6) * 32) + (i_c_inner * 2)) + k_inner) + 512)] * kernel_shared[(((k_inner * 2048) + ((((int)threadIdx.x) & 63) * 32)) + j_c_inner)]));
        }
      }
    }
  }
  for (int i_inner = 0; i_inner < 16; ++i_inner) {
    for (int j_inner = 0; j_inner < 32; ++j_inner) {
      T_matmul_NN[((((((((int)threadIdx.x) >> 6) * 65536) + (i_inner * 4096)) + (((int)blockIdx.x) * 2048)) + ((((int)threadIdx.x) & 63) * 32)) + j_inner)] = T_matmul_NN_local[((i_inner * 32) + j_inner)];
      T_matmul_NN[(((((((((int)threadIdx.x) >> 6) * 65536) + (i_inner * 4096)) + (((int)blockIdx.x) * 2048)) + ((((int)threadIdx.x) & 63) * 32)) + j_inner) + 1048576)] = T_matmul_NN_local[(((i_inner * 32) + j_inner) + 512)];
    }
  }
}

