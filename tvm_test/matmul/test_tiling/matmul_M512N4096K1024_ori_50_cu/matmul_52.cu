
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
  float T_matmul_NN_local[256];
  __shared__ float data_shared[1024];
  __shared__ float kernel_shared[512];
  for (int j_c_outer_inner_init = 0; j_c_outer_inner_init < 4; ++j_c_outer_inner_init) {
    for (int i_c_inner_init = 0; i_c_inner_init < 16; ++i_c_inner_init) {
      for (int j_c_inner_init = 0; j_c_inner_init < 2; ++j_c_inner_init) {
        T_matmul_NN_local[(((i_c_inner_init * 8) + (j_c_outer_inner_init * 2)) + j_c_inner_init)] = 0.000000e+00f;
        T_matmul_NN_local[((((i_c_inner_init * 8) + (j_c_outer_inner_init * 2)) + j_c_inner_init) + 128)] = 0.000000e+00f;
      }
    }
  }
  for (int k_outer_outer = 0; k_outer_outer < 256; ++k_outer_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_outer_outer = 0; ax0_ax1_fused_outer_outer < 4; ++ax0_ax1_fused_outer_outer) {
      *(float2*)(data_shared + ((ax0_ax1_fused_outer_outer * 256) + (((int)threadIdx.x) * 2))) = *(float2*)(data + ((((((((int)blockIdx.x) >> 5) * 262144) + (ax0_ax1_fused_outer_outer * 65536)) + ((((int)threadIdx.x) >> 1) * 1024)) + (k_outer_outer * 4)) + ((((int)threadIdx.x) & 1) * 2)));
    }
    for (int ax0_ax1_fused_outer_outer_1 = 0; ax0_ax1_fused_outer_outer_1 < 4; ++ax0_ax1_fused_outer_outer_1) {
      kernel_shared[((ax0_ax1_fused_outer_outer_1 * 128) + ((int)threadIdx.x))] = kernel[((((k_outer_outer * 16384) + (ax0_ax1_fused_outer_outer_1 * 4096)) + ((((int)blockIdx.x) & 31) * 128)) + ((int)threadIdx.x))];
    }
    __syncthreads();
    for (int k_outer_inner = 0; k_outer_inner < 4; ++k_outer_inner) {
      for (int j_c_outer_inner = 0; j_c_outer_inner < 4; ++j_c_outer_inner) {
        for (int i_c_inner = 0; i_c_inner < 16; ++i_c_inner) {
          for (int j_c_inner = 0; j_c_inner < 2; ++j_c_inner) {
            T_matmul_NN_local[(((i_c_inner * 8) + (j_c_outer_inner * 2)) + j_c_inner)] = (T_matmul_NN_local[(((i_c_inner * 8) + (j_c_outer_inner * 2)) + j_c_inner)] + (data_shared[((((((int)threadIdx.x) >> 4) * 64) + (i_c_inner * 4)) + k_outer_inner)] * kernel_shared[((((k_outer_inner * 128) + ((((int)threadIdx.x) & 15) * 8)) + (j_c_outer_inner * 2)) + j_c_inner)]));
            T_matmul_NN_local[((((i_c_inner * 8) + (j_c_outer_inner * 2)) + j_c_inner) + 128)] = (T_matmul_NN_local[((((i_c_inner * 8) + (j_c_outer_inner * 2)) + j_c_inner) + 128)] + (data_shared[(((((((int)threadIdx.x) >> 4) * 64) + (i_c_inner * 4)) + k_outer_inner) + 512)] * kernel_shared[((((k_outer_inner * 128) + ((((int)threadIdx.x) & 15) * 8)) + (j_c_outer_inner * 2)) + j_c_inner)]));
          }
        }
      }
    }
  }
  for (int i_inner = 0; i_inner < 16; ++i_inner) {
    for (int j_inner = 0; j_inner < 8; ++j_inner) {
      T_matmul_NN[(((((((((int)blockIdx.x) >> 5) * 1048576) + ((((int)threadIdx.x) >> 4) * 65536)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 31) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + j_inner)] = T_matmul_NN_local[((i_inner * 8) + j_inner)];
      T_matmul_NN[((((((((((int)blockIdx.x) >> 5) * 1048576) + ((((int)threadIdx.x) >> 4) * 65536)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 31) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + j_inner) + 524288)] = T_matmul_NN_local[(((i_inner * 8) + j_inner) + 128)];
    }
  }
}

