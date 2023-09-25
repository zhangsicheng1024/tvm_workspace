
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
  float T_matmul_NN_local[128];
  __shared__ float data_shared[128];
  __shared__ float kernel_shared[512];
  for (int i_c_outer_inner_init = 0; i_c_outer_inner_init < 2; ++i_c_outer_inner_init) {
    for (int j_c_inner_init = 0; j_c_inner_init < 16; ++j_c_inner_init) {
      T_matmul_NN_local[((i_c_outer_inner_init * 16) + j_c_inner_init)] = 0.000000e+00f;
      T_matmul_NN_local[(((i_c_outer_inner_init * 16) + j_c_inner_init) + 32)] = 0.000000e+00f;
      T_matmul_NN_local[(((i_c_outer_inner_init * 16) + j_c_inner_init) + 64)] = 0.000000e+00f;
      T_matmul_NN_local[(((i_c_outer_inner_init * 16) + j_c_inner_init) + 96)] = 0.000000e+00f;
    }
  }
  for (int k_outer_outer = 0; k_outer_outer < 512; ++k_outer_outer) {
    __syncthreads();
    data_shared[((int)threadIdx.x)] = data[(((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 1) * 1024)) + (k_outer_outer * 2)) + (((int)threadIdx.x) & 1))];
    for (int ax0_ax1_fused_outer_outer = 0; ax0_ax1_fused_outer_outer < 4; ++ax0_ax1_fused_outer_outer) {
      kernel_shared[((ax0_ax1_fused_outer_outer * 128) + ((int)threadIdx.x))] = kernel[(((((k_outer_outer * 8192) + ((ax0_ax1_fused_outer_outer >> 1) * 4096)) + ((((int)blockIdx.x) & 15) * 256)) + ((ax0_ax1_fused_outer_outer & 1) * 128)) + ((int)threadIdx.x))];
    }
    __syncthreads();
    for (int k_outer_inner = 0; k_outer_inner < 2; ++k_outer_inner) {
      for (int i_c_outer_inner = 0; i_c_outer_inner < 2; ++i_c_outer_inner) {
        for (int j_c_inner = 0; j_c_inner < 16; ++j_c_inner) {
          T_matmul_NN_local[((i_c_outer_inner * 16) + j_c_inner)] = (T_matmul_NN_local[((i_c_outer_inner * 16) + j_c_inner)] + (data_shared[((((((int)threadIdx.x) >> 2) * 4) + (i_c_outer_inner * 2)) + k_outer_inner)] * kernel_shared[(((k_outer_inner * 256) + ((((int)threadIdx.x) & 3) * 16)) + j_c_inner)]));
          T_matmul_NN_local[(((i_c_outer_inner * 16) + j_c_inner) + 32)] = (T_matmul_NN_local[(((i_c_outer_inner * 16) + j_c_inner) + 32)] + (data_shared[((((((int)threadIdx.x) >> 2) * 4) + (i_c_outer_inner * 2)) + k_outer_inner)] * kernel_shared[((((k_outer_inner * 256) + ((((int)threadIdx.x) & 3) * 16)) + j_c_inner) + 64)]));
          T_matmul_NN_local[(((i_c_outer_inner * 16) + j_c_inner) + 64)] = (T_matmul_NN_local[(((i_c_outer_inner * 16) + j_c_inner) + 64)] + (data_shared[((((((int)threadIdx.x) >> 2) * 4) + (i_c_outer_inner * 2)) + k_outer_inner)] * kernel_shared[((((k_outer_inner * 256) + ((((int)threadIdx.x) & 3) * 16)) + j_c_inner) + 128)]));
          T_matmul_NN_local[(((i_c_outer_inner * 16) + j_c_inner) + 96)] = (T_matmul_NN_local[(((i_c_outer_inner * 16) + j_c_inner) + 96)] + (data_shared[((((((int)threadIdx.x) >> 2) * 4) + (i_c_outer_inner * 2)) + k_outer_inner)] * kernel_shared[((((k_outer_inner * 256) + ((((int)threadIdx.x) & 3) * 16)) + j_c_inner) + 192)]));
        }
      }
    }
  }
  for (int i_inner = 0; i_inner < 2; ++i_inner) {
    for (int j_inner = 0; j_inner < 16; ++j_inner) {
      T_matmul_NN[(((((((((int)blockIdx.x) >> 4) * 262144) + ((((int)threadIdx.x) >> 2) * 8192)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 15) * 256)) + ((((int)threadIdx.x) & 3) * 16)) + j_inner)] = T_matmul_NN_local[((i_inner * 16) + j_inner)];
      T_matmul_NN[((((((((((int)blockIdx.x) >> 4) * 262144) + ((((int)threadIdx.x) >> 2) * 8192)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 15) * 256)) + ((((int)threadIdx.x) & 3) * 16)) + j_inner) + 64)] = T_matmul_NN_local[(((i_inner * 16) + j_inner) + 32)];
      T_matmul_NN[((((((((((int)blockIdx.x) >> 4) * 262144) + ((((int)threadIdx.x) >> 2) * 8192)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 15) * 256)) + ((((int)threadIdx.x) & 3) * 16)) + j_inner) + 128)] = T_matmul_NN_local[(((i_inner * 16) + j_inner) + 64)];
      T_matmul_NN[((((((((((int)blockIdx.x) >> 4) * 262144) + ((((int)threadIdx.x) >> 2) * 8192)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 15) * 256)) + ((((int)threadIdx.x) & 3) * 16)) + j_inner) + 192)] = T_matmul_NN_local[(((i_inner * 16) + j_inner) + 96)];
    }
  }
}

