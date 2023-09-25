
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
  float T_matmul_NN_local[1024];
  __shared__ float data_shared[256];
  __shared__ float kernel_shared[4096];
  for (int i_c_outer_inner_init = 0; i_c_outer_inner_init < 4; ++i_c_outer_inner_init) {
    for (int j_c_outer_inner_init = 0; j_c_outer_inner_init < 4; ++j_c_outer_inner_init) {
      for (int i_c_inner_init = 0; i_c_inner_init < 2; ++i_c_inner_init) {
        for (int j_c_inner_init = 0; j_c_inner_init < 4; ++j_c_inner_init) {
          T_matmul_NN_local[((((i_c_outer_inner_init * 32) + (i_c_inner_init * 16)) + (j_c_outer_inner_init * 4)) + j_c_inner_init)] = 0.000000e+00f;
          T_matmul_NN_local[(((((i_c_outer_inner_init * 32) + (i_c_inner_init * 16)) + (j_c_outer_inner_init * 4)) + j_c_inner_init) + 128)] = 0.000000e+00f;
          T_matmul_NN_local[(((((i_c_outer_inner_init * 32) + (i_c_inner_init * 16)) + (j_c_outer_inner_init * 4)) + j_c_inner_init) + 256)] = 0.000000e+00f;
          T_matmul_NN_local[(((((i_c_outer_inner_init * 32) + (i_c_inner_init * 16)) + (j_c_outer_inner_init * 4)) + j_c_inner_init) + 384)] = 0.000000e+00f;
          T_matmul_NN_local[(((((i_c_outer_inner_init * 32) + (i_c_inner_init * 16)) + (j_c_outer_inner_init * 4)) + j_c_inner_init) + 512)] = 0.000000e+00f;
          T_matmul_NN_local[(((((i_c_outer_inner_init * 32) + (i_c_inner_init * 16)) + (j_c_outer_inner_init * 4)) + j_c_inner_init) + 640)] = 0.000000e+00f;
          T_matmul_NN_local[(((((i_c_outer_inner_init * 32) + (i_c_inner_init * 16)) + (j_c_outer_inner_init * 4)) + j_c_inner_init) + 768)] = 0.000000e+00f;
          T_matmul_NN_local[(((((i_c_outer_inner_init * 32) + (i_c_inner_init * 16)) + (j_c_outer_inner_init * 4)) + j_c_inner_init) + 896)] = 0.000000e+00f;
        }
      }
    }
  }
  for (int k_outer_outer = 0; k_outer_outer < 512; ++k_outer_outer) {
    __syncthreads();
    data_shared[((int)threadIdx.x)] = data[(((((((int)blockIdx.x) >> 1) * 131072) + ((((int)threadIdx.x) >> 1) * 1024)) + (k_outer_outer * 2)) + (((int)threadIdx.x) & 1))];
    for (int ax0_ax1_fused_outer_outer = 0; ax0_ax1_fused_outer_outer < 16; ++ax0_ax1_fused_outer_outer) {
      kernel_shared[((ax0_ax1_fused_outer_outer * 256) + ((int)threadIdx.x))] = kernel[(((((k_outer_outer * 8192) + ((ax0_ax1_fused_outer_outer >> 3) * 4096)) + ((((int)blockIdx.x) & 1) * 2048)) + ((ax0_ax1_fused_outer_outer & 7) * 256)) + ((int)threadIdx.x))];
    }
    __syncthreads();
    for (int i_c_outer_inner = 0; i_c_outer_inner < 4; ++i_c_outer_inner) {
      for (int j_c_outer_inner = 0; j_c_outer_inner < 4; ++j_c_outer_inner) {
        for (int k_inner = 0; k_inner < 2; ++k_inner) {
          for (int i_c_inner = 0; i_c_inner < 2; ++i_c_inner) {
            for (int j_c_inner = 0; j_c_inner < 4; ++j_c_inner) {
              T_matmul_NN_local[((((i_c_outer_inner * 32) + (i_c_inner * 16)) + (j_c_outer_inner * 4)) + j_c_inner)] = (T_matmul_NN_local[((((i_c_outer_inner * 32) + (i_c_inner * 16)) + (j_c_outer_inner * 4)) + j_c_inner)] + (data_shared[(((((((int)threadIdx.x) >> 4) * 16) + (i_c_outer_inner * 4)) + (i_c_inner * 2)) + k_inner)] * kernel_shared[((((k_inner * 2048) + ((((int)threadIdx.x) & 15) * 16)) + (j_c_outer_inner * 4)) + j_c_inner)]));
              T_matmul_NN_local[(((((i_c_outer_inner * 32) + (i_c_inner * 16)) + (j_c_outer_inner * 4)) + j_c_inner) + 128)] = (T_matmul_NN_local[(((((i_c_outer_inner * 32) + (i_c_inner * 16)) + (j_c_outer_inner * 4)) + j_c_inner) + 128)] + (data_shared[(((((((int)threadIdx.x) >> 4) * 16) + (i_c_outer_inner * 4)) + (i_c_inner * 2)) + k_inner)] * kernel_shared[(((((k_inner * 2048) + ((((int)threadIdx.x) & 15) * 16)) + (j_c_outer_inner * 4)) + j_c_inner) + 256)]));
              T_matmul_NN_local[(((((i_c_outer_inner * 32) + (i_c_inner * 16)) + (j_c_outer_inner * 4)) + j_c_inner) + 256)] = (T_matmul_NN_local[(((((i_c_outer_inner * 32) + (i_c_inner * 16)) + (j_c_outer_inner * 4)) + j_c_inner) + 256)] + (data_shared[(((((((int)threadIdx.x) >> 4) * 16) + (i_c_outer_inner * 4)) + (i_c_inner * 2)) + k_inner)] * kernel_shared[(((((k_inner * 2048) + ((((int)threadIdx.x) & 15) * 16)) + (j_c_outer_inner * 4)) + j_c_inner) + 512)]));
              T_matmul_NN_local[(((((i_c_outer_inner * 32) + (i_c_inner * 16)) + (j_c_outer_inner * 4)) + j_c_inner) + 384)] = (T_matmul_NN_local[(((((i_c_outer_inner * 32) + (i_c_inner * 16)) + (j_c_outer_inner * 4)) + j_c_inner) + 384)] + (data_shared[(((((((int)threadIdx.x) >> 4) * 16) + (i_c_outer_inner * 4)) + (i_c_inner * 2)) + k_inner)] * kernel_shared[(((((k_inner * 2048) + ((((int)threadIdx.x) & 15) * 16)) + (j_c_outer_inner * 4)) + j_c_inner) + 768)]));
              T_matmul_NN_local[(((((i_c_outer_inner * 32) + (i_c_inner * 16)) + (j_c_outer_inner * 4)) + j_c_inner) + 512)] = (T_matmul_NN_local[(((((i_c_outer_inner * 32) + (i_c_inner * 16)) + (j_c_outer_inner * 4)) + j_c_inner) + 512)] + (data_shared[(((((((int)threadIdx.x) >> 4) * 16) + (i_c_outer_inner * 4)) + (i_c_inner * 2)) + k_inner)] * kernel_shared[(((((k_inner * 2048) + ((((int)threadIdx.x) & 15) * 16)) + (j_c_outer_inner * 4)) + j_c_inner) + 1024)]));
              T_matmul_NN_local[(((((i_c_outer_inner * 32) + (i_c_inner * 16)) + (j_c_outer_inner * 4)) + j_c_inner) + 640)] = (T_matmul_NN_local[(((((i_c_outer_inner * 32) + (i_c_inner * 16)) + (j_c_outer_inner * 4)) + j_c_inner) + 640)] + (data_shared[(((((((int)threadIdx.x) >> 4) * 16) + (i_c_outer_inner * 4)) + (i_c_inner * 2)) + k_inner)] * kernel_shared[(((((k_inner * 2048) + ((((int)threadIdx.x) & 15) * 16)) + (j_c_outer_inner * 4)) + j_c_inner) + 1280)]));
              T_matmul_NN_local[(((((i_c_outer_inner * 32) + (i_c_inner * 16)) + (j_c_outer_inner * 4)) + j_c_inner) + 768)] = (T_matmul_NN_local[(((((i_c_outer_inner * 32) + (i_c_inner * 16)) + (j_c_outer_inner * 4)) + j_c_inner) + 768)] + (data_shared[(((((((int)threadIdx.x) >> 4) * 16) + (i_c_outer_inner * 4)) + (i_c_inner * 2)) + k_inner)] * kernel_shared[(((((k_inner * 2048) + ((((int)threadIdx.x) & 15) * 16)) + (j_c_outer_inner * 4)) + j_c_inner) + 1536)]));
              T_matmul_NN_local[(((((i_c_outer_inner * 32) + (i_c_inner * 16)) + (j_c_outer_inner * 4)) + j_c_inner) + 896)] = (T_matmul_NN_local[(((((i_c_outer_inner * 32) + (i_c_inner * 16)) + (j_c_outer_inner * 4)) + j_c_inner) + 896)] + (data_shared[(((((((int)threadIdx.x) >> 4) * 16) + (i_c_outer_inner * 4)) + (i_c_inner * 2)) + k_inner)] * kernel_shared[(((((k_inner * 2048) + ((((int)threadIdx.x) & 15) * 16)) + (j_c_outer_inner * 4)) + j_c_inner) + 1792)]));
            }
          }
        }
      }
    }
  }
  for (int i_inner = 0; i_inner < 8; ++i_inner) {
    for (int j_inner = 0; j_inner < 16; ++j_inner) {
      T_matmul_NN[(((((((((int)blockIdx.x) >> 1) * 524288) + ((((int)threadIdx.x) >> 4) * 32768)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 1) * 2048)) + ((((int)threadIdx.x) & 15) * 16)) + j_inner)] = T_matmul_NN_local[((i_inner * 16) + j_inner)];
      T_matmul_NN[((((((((((int)blockIdx.x) >> 1) * 524288) + ((((int)threadIdx.x) >> 4) * 32768)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 1) * 2048)) + ((((int)threadIdx.x) & 15) * 16)) + j_inner) + 256)] = T_matmul_NN_local[(((i_inner * 16) + j_inner) + 128)];
      T_matmul_NN[((((((((((int)blockIdx.x) >> 1) * 524288) + ((((int)threadIdx.x) >> 4) * 32768)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 1) * 2048)) + ((((int)threadIdx.x) & 15) * 16)) + j_inner) + 512)] = T_matmul_NN_local[(((i_inner * 16) + j_inner) + 256)];
      T_matmul_NN[((((((((((int)blockIdx.x) >> 1) * 524288) + ((((int)threadIdx.x) >> 4) * 32768)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 1) * 2048)) + ((((int)threadIdx.x) & 15) * 16)) + j_inner) + 768)] = T_matmul_NN_local[(((i_inner * 16) + j_inner) + 384)];
      T_matmul_NN[((((((((((int)blockIdx.x) >> 1) * 524288) + ((((int)threadIdx.x) >> 4) * 32768)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 1) * 2048)) + ((((int)threadIdx.x) & 15) * 16)) + j_inner) + 1024)] = T_matmul_NN_local[(((i_inner * 16) + j_inner) + 512)];
      T_matmul_NN[((((((((((int)blockIdx.x) >> 1) * 524288) + ((((int)threadIdx.x) >> 4) * 32768)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 1) * 2048)) + ((((int)threadIdx.x) & 15) * 16)) + j_inner) + 1280)] = T_matmul_NN_local[(((i_inner * 16) + j_inner) + 640)];
      T_matmul_NN[((((((((((int)blockIdx.x) >> 1) * 524288) + ((((int)threadIdx.x) >> 4) * 32768)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 1) * 2048)) + ((((int)threadIdx.x) & 15) * 16)) + j_inner) + 1536)] = T_matmul_NN_local[(((i_inner * 16) + j_inner) + 768)];
      T_matmul_NN[((((((((((int)blockIdx.x) >> 1) * 524288) + ((((int)threadIdx.x) >> 4) * 32768)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 1) * 2048)) + ((((int)threadIdx.x) & 15) * 16)) + j_inner) + 1792)] = T_matmul_NN_local[(((i_inner * 16) + j_inner) + 896)];
    }
  }
}

