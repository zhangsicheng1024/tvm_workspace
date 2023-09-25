
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
  float T_matmul_NN_local[256];
  __shared__ float data_shared[1024];
  __shared__ float kernel_shared[1024];
  for (int i_c_outer_inner_init = 0; i_c_outer_inner_init < 4; ++i_c_outer_inner_init) {
    for (int j_c_outer_inner_init = 0; j_c_outer_inner_init < 8; ++j_c_outer_inner_init) {
      T_matmul_NN_local[((i_c_outer_inner_init * 8) + j_c_outer_inner_init)] = 0.000000e+00f;
      T_matmul_NN_local[(((i_c_outer_inner_init * 8) + j_c_outer_inner_init) + 32)] = 0.000000e+00f;
      T_matmul_NN_local[(((i_c_outer_inner_init * 8) + j_c_outer_inner_init) + 64)] = 0.000000e+00f;
      T_matmul_NN_local[(((i_c_outer_inner_init * 8) + j_c_outer_inner_init) + 96)] = 0.000000e+00f;
      T_matmul_NN_local[(((i_c_outer_inner_init * 8) + j_c_outer_inner_init) + 128)] = 0.000000e+00f;
      T_matmul_NN_local[(((i_c_outer_inner_init * 8) + j_c_outer_inner_init) + 160)] = 0.000000e+00f;
      T_matmul_NN_local[(((i_c_outer_inner_init * 8) + j_c_outer_inner_init) + 192)] = 0.000000e+00f;
      T_matmul_NN_local[(((i_c_outer_inner_init * 8) + j_c_outer_inner_init) + 224)] = 0.000000e+00f;
    }
  }
  for (int k_outer_outer = 0; k_outer_outer < 128; ++k_outer_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_outer_outer = 0; ax0_ax1_fused_outer_outer < 16; ++ax0_ax1_fused_outer_outer) {
      data_shared[((ax0_ax1_fused_outer_outer * 64) + ((int)threadIdx.x))] = data[((((((((int)blockIdx.x) >> 5) * 131072) + (ax0_ax1_fused_outer_outer * 8192)) + ((((int)threadIdx.x) >> 3) * 1024)) + (k_outer_outer * 8)) + (((int)threadIdx.x) & 7))];
    }
    for (int ax0_ax1_fused_outer_outer_1 = 0; ax0_ax1_fused_outer_outer_1 < 16; ++ax0_ax1_fused_outer_outer_1) {
      kernel_shared[((ax0_ax1_fused_outer_outer_1 * 64) + ((int)threadIdx.x))] = kernel[(((((k_outer_outer * 32768) + ((ax0_ax1_fused_outer_outer_1 >> 1) * 4096)) + ((((int)blockIdx.x) & 31) * 128)) + ((ax0_ax1_fused_outer_outer_1 & 1) * 64)) + ((int)threadIdx.x))];
    }
    __syncthreads();
    for (int k_outer_inner = 0; k_outer_inner < 2; ++k_outer_inner) {
      for (int i_c_outer_inner = 0; i_c_outer_inner < 4; ++i_c_outer_inner) {
        for (int j_c_outer_inner = 0; j_c_outer_inner < 8; ++j_c_outer_inner) {
          for (int k_inner = 0; k_inner < 4; ++k_inner) {
            T_matmul_NN_local[((i_c_outer_inner * 8) + j_c_outer_inner)] = (T_matmul_NN_local[((i_c_outer_inner * 8) + j_c_outer_inner)] + (data_shared[(((((((int)threadIdx.x) >> 4) * 32) + (i_c_outer_inner * 8)) + (k_outer_inner * 4)) + k_inner)] * kernel_shared[((((k_outer_inner * 512) + (k_inner * 128)) + ((((int)threadIdx.x) & 15) * 8)) + j_c_outer_inner)]));
            T_matmul_NN_local[(((i_c_outer_inner * 8) + j_c_outer_inner) + 32)] = (T_matmul_NN_local[(((i_c_outer_inner * 8) + j_c_outer_inner) + 32)] + (data_shared[((((((((int)threadIdx.x) >> 4) * 32) + (i_c_outer_inner * 8)) + (k_outer_inner * 4)) + k_inner) + 128)] * kernel_shared[((((k_outer_inner * 512) + (k_inner * 128)) + ((((int)threadIdx.x) & 15) * 8)) + j_c_outer_inner)]));
            T_matmul_NN_local[(((i_c_outer_inner * 8) + j_c_outer_inner) + 64)] = (T_matmul_NN_local[(((i_c_outer_inner * 8) + j_c_outer_inner) + 64)] + (data_shared[((((((((int)threadIdx.x) >> 4) * 32) + (i_c_outer_inner * 8)) + (k_outer_inner * 4)) + k_inner) + 256)] * kernel_shared[((((k_outer_inner * 512) + (k_inner * 128)) + ((((int)threadIdx.x) & 15) * 8)) + j_c_outer_inner)]));
            T_matmul_NN_local[(((i_c_outer_inner * 8) + j_c_outer_inner) + 96)] = (T_matmul_NN_local[(((i_c_outer_inner * 8) + j_c_outer_inner) + 96)] + (data_shared[((((((((int)threadIdx.x) >> 4) * 32) + (i_c_outer_inner * 8)) + (k_outer_inner * 4)) + k_inner) + 384)] * kernel_shared[((((k_outer_inner * 512) + (k_inner * 128)) + ((((int)threadIdx.x) & 15) * 8)) + j_c_outer_inner)]));
            T_matmul_NN_local[(((i_c_outer_inner * 8) + j_c_outer_inner) + 128)] = (T_matmul_NN_local[(((i_c_outer_inner * 8) + j_c_outer_inner) + 128)] + (data_shared[((((((((int)threadIdx.x) >> 4) * 32) + (i_c_outer_inner * 8)) + (k_outer_inner * 4)) + k_inner) + 512)] * kernel_shared[((((k_outer_inner * 512) + (k_inner * 128)) + ((((int)threadIdx.x) & 15) * 8)) + j_c_outer_inner)]));
            T_matmul_NN_local[(((i_c_outer_inner * 8) + j_c_outer_inner) + 160)] = (T_matmul_NN_local[(((i_c_outer_inner * 8) + j_c_outer_inner) + 160)] + (data_shared[((((((((int)threadIdx.x) >> 4) * 32) + (i_c_outer_inner * 8)) + (k_outer_inner * 4)) + k_inner) + 640)] * kernel_shared[((((k_outer_inner * 512) + (k_inner * 128)) + ((((int)threadIdx.x) & 15) * 8)) + j_c_outer_inner)]));
            T_matmul_NN_local[(((i_c_outer_inner * 8) + j_c_outer_inner) + 192)] = (T_matmul_NN_local[(((i_c_outer_inner * 8) + j_c_outer_inner) + 192)] + (data_shared[((((((((int)threadIdx.x) >> 4) * 32) + (i_c_outer_inner * 8)) + (k_outer_inner * 4)) + k_inner) + 768)] * kernel_shared[((((k_outer_inner * 512) + (k_inner * 128)) + ((((int)threadIdx.x) & 15) * 8)) + j_c_outer_inner)]));
            T_matmul_NN_local[(((i_c_outer_inner * 8) + j_c_outer_inner) + 224)] = (T_matmul_NN_local[(((i_c_outer_inner * 8) + j_c_outer_inner) + 224)] + (data_shared[((((((((int)threadIdx.x) >> 4) * 32) + (i_c_outer_inner * 8)) + (k_outer_inner * 4)) + k_inner) + 896)] * kernel_shared[((((k_outer_inner * 512) + (k_inner * 128)) + ((((int)threadIdx.x) & 15) * 8)) + j_c_outer_inner)]));
          }
        }
      }
    }
  }
  for (int i_inner = 0; i_inner < 4; ++i_inner) {
    for (int j_inner = 0; j_inner < 8; ++j_inner) {
      T_matmul_NN[(((((((((int)blockIdx.x) >> 5) * 524288) + ((((int)threadIdx.x) >> 4) * 16384)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 31) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + j_inner)] = T_matmul_NN_local[((i_inner * 8) + j_inner)];
      T_matmul_NN[((((((((((int)blockIdx.x) >> 5) * 524288) + ((((int)threadIdx.x) >> 4) * 16384)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 31) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + j_inner) + 65536)] = T_matmul_NN_local[(((i_inner * 8) + j_inner) + 32)];
      T_matmul_NN[((((((((((int)blockIdx.x) >> 5) * 524288) + ((((int)threadIdx.x) >> 4) * 16384)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 31) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + j_inner) + 131072)] = T_matmul_NN_local[(((i_inner * 8) + j_inner) + 64)];
      T_matmul_NN[((((((((((int)blockIdx.x) >> 5) * 524288) + ((((int)threadIdx.x) >> 4) * 16384)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 31) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + j_inner) + 196608)] = T_matmul_NN_local[(((i_inner * 8) + j_inner) + 96)];
      T_matmul_NN[((((((((((int)blockIdx.x) >> 5) * 524288) + ((((int)threadIdx.x) >> 4) * 16384)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 31) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + j_inner) + 262144)] = T_matmul_NN_local[(((i_inner * 8) + j_inner) + 128)];
      T_matmul_NN[((((((((((int)blockIdx.x) >> 5) * 524288) + ((((int)threadIdx.x) >> 4) * 16384)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 31) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + j_inner) + 327680)] = T_matmul_NN_local[(((i_inner * 8) + j_inner) + 160)];
      T_matmul_NN[((((((((((int)blockIdx.x) >> 5) * 524288) + ((((int)threadIdx.x) >> 4) * 16384)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 31) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + j_inner) + 393216)] = T_matmul_NN_local[(((i_inner * 8) + j_inner) + 192)];
      T_matmul_NN[((((((((((int)blockIdx.x) >> 5) * 524288) + ((((int)threadIdx.x) >> 4) * 16384)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 31) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + j_inner) + 458752)] = T_matmul_NN_local[(((i_inner * 8) + j_inner) + 224)];
    }
  }
}

