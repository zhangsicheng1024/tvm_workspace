
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
extern "C" __global__ void __launch_bounds__(256) mymatmul_kernel0(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ T_matmul_NN) {
  float T_matmul_NN_local[8192];
  __shared__ float data_shared[512];
  __shared__ float kernel_shared[4096];
  for (int j_c_outer_inner_init = 0; j_c_outer_inner_init < 4; ++j_c_outer_inner_init) {
    for (int i_c_inner_init = 0; i_c_inner_init < 8; ++i_c_inner_init) {
      for (int j_c_inner_init = 0; j_c_inner_init < 32; ++j_c_inner_init) {
        T_matmul_NN_local[(((i_c_inner_init * 128) + (j_c_outer_inner_init * 32)) + j_c_inner_init)] = 0.000000e+00f;
        T_matmul_NN_local[((((i_c_inner_init * 128) + (j_c_outer_inner_init * 32)) + j_c_inner_init) + 1024)] = 0.000000e+00f;
        T_matmul_NN_local[((((i_c_inner_init * 128) + (j_c_outer_inner_init * 32)) + j_c_inner_init) + 2048)] = 0.000000e+00f;
        T_matmul_NN_local[((((i_c_inner_init * 128) + (j_c_outer_inner_init * 32)) + j_c_inner_init) + 3072)] = 0.000000e+00f;
        T_matmul_NN_local[((((i_c_inner_init * 128) + (j_c_outer_inner_init * 32)) + j_c_inner_init) + 4096)] = 0.000000e+00f;
        T_matmul_NN_local[((((i_c_inner_init * 128) + (j_c_outer_inner_init * 32)) + j_c_inner_init) + 5120)] = 0.000000e+00f;
        T_matmul_NN_local[((((i_c_inner_init * 128) + (j_c_outer_inner_init * 32)) + j_c_inner_init) + 6144)] = 0.000000e+00f;
        T_matmul_NN_local[((((i_c_inner_init * 128) + (j_c_outer_inner_init * 32)) + j_c_inner_init) + 7168)] = 0.000000e+00f;
      }
    }
  }
  for (int k_outer_outer = 0; k_outer_outer < 1024; ++k_outer_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_inner_s = 0; ax0_ax1_fused_inner_s < 8; ++ax0_ax1_fused_inner_s) {
      if (((int)threadIdx.x) < 64) {
        data_shared[((((int)threadIdx.x) * 8) + ax0_ax1_fused_inner_s)] = data[(((((int)threadIdx.x) * 8192) + (ax0_ax1_fused_inner_s * 1024)) + k_outer_outer)];
      }
    }
    for (int ax0_ax1_fused_outer_outer = 0; ax0_ax1_fused_outer_outer < 16; ++ax0_ax1_fused_outer_outer) {
      kernel_shared[((ax0_ax1_fused_outer_outer * 256) + ((int)threadIdx.x))] = kernel[(((k_outer_outer * 4096) + (ax0_ax1_fused_outer_outer * 256)) + ((int)threadIdx.x))];
    }
    __syncthreads();
    for (int j_c_outer_inner = 0; j_c_outer_inner < 4; ++j_c_outer_inner) {
      for (int i_c_inner = 0; i_c_inner < 8; ++i_c_inner) {
        for (int j_c_inner = 0; j_c_inner < 32; ++j_c_inner) {
          T_matmul_NN_local[(((i_c_inner * 128) + (j_c_outer_inner * 32)) + j_c_inner)] = (T_matmul_NN_local[(((i_c_inner * 128) + (j_c_outer_inner * 32)) + j_c_inner)] + (data_shared[(((((int)threadIdx.x) >> 4) * 8) + i_c_inner)] * kernel_shared[((((((int)threadIdx.x) & 15) * 128) + (j_c_outer_inner * 32)) + j_c_inner)]));
          T_matmul_NN_local[((((i_c_inner * 128) + (j_c_outer_inner * 32)) + j_c_inner) + 1024)] = (T_matmul_NN_local[((((i_c_inner * 128) + (j_c_outer_inner * 32)) + j_c_inner) + 1024)] + (data_shared[(((((int)threadIdx.x) >> 4) * 8) + i_c_inner)] * kernel_shared[(((((((int)threadIdx.x) & 15) * 128) + (j_c_outer_inner * 32)) + j_c_inner) + 2048)]));
          T_matmul_NN_local[((((i_c_inner * 128) + (j_c_outer_inner * 32)) + j_c_inner) + 2048)] = (T_matmul_NN_local[((((i_c_inner * 128) + (j_c_outer_inner * 32)) + j_c_inner) + 2048)] + (data_shared[((((((int)threadIdx.x) >> 4) * 8) + i_c_inner) + 128)] * kernel_shared[((((((int)threadIdx.x) & 15) * 128) + (j_c_outer_inner * 32)) + j_c_inner)]));
          T_matmul_NN_local[((((i_c_inner * 128) + (j_c_outer_inner * 32)) + j_c_inner) + 3072)] = (T_matmul_NN_local[((((i_c_inner * 128) + (j_c_outer_inner * 32)) + j_c_inner) + 3072)] + (data_shared[((((((int)threadIdx.x) >> 4) * 8) + i_c_inner) + 128)] * kernel_shared[(((((((int)threadIdx.x) & 15) * 128) + (j_c_outer_inner * 32)) + j_c_inner) + 2048)]));
          T_matmul_NN_local[((((i_c_inner * 128) + (j_c_outer_inner * 32)) + j_c_inner) + 4096)] = (T_matmul_NN_local[((((i_c_inner * 128) + (j_c_outer_inner * 32)) + j_c_inner) + 4096)] + (data_shared[((((((int)threadIdx.x) >> 4) * 8) + i_c_inner) + 256)] * kernel_shared[((((((int)threadIdx.x) & 15) * 128) + (j_c_outer_inner * 32)) + j_c_inner)]));
          T_matmul_NN_local[((((i_c_inner * 128) + (j_c_outer_inner * 32)) + j_c_inner) + 5120)] = (T_matmul_NN_local[((((i_c_inner * 128) + (j_c_outer_inner * 32)) + j_c_inner) + 5120)] + (data_shared[((((((int)threadIdx.x) >> 4) * 8) + i_c_inner) + 256)] * kernel_shared[(((((((int)threadIdx.x) & 15) * 128) + (j_c_outer_inner * 32)) + j_c_inner) + 2048)]));
          T_matmul_NN_local[((((i_c_inner * 128) + (j_c_outer_inner * 32)) + j_c_inner) + 6144)] = (T_matmul_NN_local[((((i_c_inner * 128) + (j_c_outer_inner * 32)) + j_c_inner) + 6144)] + (data_shared[((((((int)threadIdx.x) >> 4) * 8) + i_c_inner) + 384)] * kernel_shared[((((((int)threadIdx.x) & 15) * 128) + (j_c_outer_inner * 32)) + j_c_inner)]));
          T_matmul_NN_local[((((i_c_inner * 128) + (j_c_outer_inner * 32)) + j_c_inner) + 7168)] = (T_matmul_NN_local[((((i_c_inner * 128) + (j_c_outer_inner * 32)) + j_c_inner) + 7168)] + (data_shared[((((((int)threadIdx.x) >> 4) * 8) + i_c_inner) + 384)] * kernel_shared[(((((((int)threadIdx.x) & 15) * 128) + (j_c_outer_inner * 32)) + j_c_inner) + 2048)]));
        }
      }
    }
  }
  for (int i_inner = 0; i_inner < 8; ++i_inner) {
    for (int j_inner = 0; j_inner < 128; ++j_inner) {
      T_matmul_NN[(((((((int)threadIdx.x) >> 4) * 32768) + (i_inner * 4096)) + ((((int)threadIdx.x) & 15) * 128)) + j_inner)] = T_matmul_NN_local[((i_inner * 128) + j_inner)];
      T_matmul_NN[((((((((int)threadIdx.x) >> 4) * 32768) + (i_inner * 4096)) + ((((int)threadIdx.x) & 15) * 128)) + j_inner) + 2048)] = T_matmul_NN_local[(((i_inner * 128) + j_inner) + 1024)];
      T_matmul_NN[((((((((int)threadIdx.x) >> 4) * 32768) + (i_inner * 4096)) + ((((int)threadIdx.x) & 15) * 128)) + j_inner) + 524288)] = T_matmul_NN_local[(((i_inner * 128) + j_inner) + 2048)];
      T_matmul_NN[((((((((int)threadIdx.x) >> 4) * 32768) + (i_inner * 4096)) + ((((int)threadIdx.x) & 15) * 128)) + j_inner) + 526336)] = T_matmul_NN_local[(((i_inner * 128) + j_inner) + 3072)];
      T_matmul_NN[((((((((int)threadIdx.x) >> 4) * 32768) + (i_inner * 4096)) + ((((int)threadIdx.x) & 15) * 128)) + j_inner) + 1048576)] = T_matmul_NN_local[(((i_inner * 128) + j_inner) + 4096)];
      T_matmul_NN[((((((((int)threadIdx.x) >> 4) * 32768) + (i_inner * 4096)) + ((((int)threadIdx.x) & 15) * 128)) + j_inner) + 1050624)] = T_matmul_NN_local[(((i_inner * 128) + j_inner) + 5120)];
      T_matmul_NN[((((((((int)threadIdx.x) >> 4) * 32768) + (i_inner * 4096)) + ((((int)threadIdx.x) & 15) * 128)) + j_inner) + 1572864)] = T_matmul_NN_local[(((i_inner * 128) + j_inner) + 6144)];
      T_matmul_NN[((((((((int)threadIdx.x) >> 4) * 32768) + (i_inner * 4096)) + ((((int)threadIdx.x) & 15) * 128)) + j_inner) + 1574912)] = T_matmul_NN_local[(((i_inner * 128) + j_inner) + 7168)];
    }
  }
}

