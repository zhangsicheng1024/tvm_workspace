
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
extern "C" __global__ void __launch_bounds__(32) mymatmul_kernel0(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ T_matmul_NN) {
  float T_matmul_NN_local[64];
  __shared__ float data_shared[4096];
  __shared__ float kernel_shared[2048];
  for (int i_c_outer_inner_init = 0; i_c_outer_inner_init < 2; ++i_c_outer_inner_init) {
    for (int j_c_outer_inner_init = 0; j_c_outer_inner_init < 4; ++j_c_outer_inner_init) {
      T_matmul_NN_local[((i_c_outer_inner_init * 16) + j_c_outer_inner_init)] = 0.000000e+00f;
      T_matmul_NN_local[(((i_c_outer_inner_init * 16) + j_c_outer_inner_init) + 32)] = 0.000000e+00f;
      T_matmul_NN_local[(((i_c_outer_inner_init * 16) + j_c_outer_inner_init) + 4)] = 0.000000e+00f;
      T_matmul_NN_local[(((i_c_outer_inner_init * 16) + j_c_outer_inner_init) + 36)] = 0.000000e+00f;
      T_matmul_NN_local[(((i_c_outer_inner_init * 16) + j_c_outer_inner_init) + 8)] = 0.000000e+00f;
      T_matmul_NN_local[(((i_c_outer_inner_init * 16) + j_c_outer_inner_init) + 40)] = 0.000000e+00f;
      T_matmul_NN_local[(((i_c_outer_inner_init * 16) + j_c_outer_inner_init) + 12)] = 0.000000e+00f;
      T_matmul_NN_local[(((i_c_outer_inner_init * 16) + j_c_outer_inner_init) + 44)] = 0.000000e+00f;
    }
  }
  for (int k_outer_outer = 0; k_outer_outer < 16; ++k_outer_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_outer_outer = 0; ax0_ax1_fused_outer_outer < 128; ++ax0_ax1_fused_outer_outer) {
      data_shared[((ax0_ax1_fused_outer_outer * 32) + ((int)threadIdx.x))] = data[((((((((int)blockIdx.x) >> 7) * 65536) + ((ax0_ax1_fused_outer_outer >> 1) * 1024)) + (k_outer_outer * 64)) + ((ax0_ax1_fused_outer_outer & 1) * 32)) + ((int)threadIdx.x))];
    }
    for (int ax0_ax1_fused_outer_outer_1 = 0; ax0_ax1_fused_outer_outer_1 < 64; ++ax0_ax1_fused_outer_outer_1) {
      kernel_shared[((ax0_ax1_fused_outer_outer_1 * 32) + ((int)threadIdx.x))] = kernel[((((k_outer_outer * 262144) + (ax0_ax1_fused_outer_outer_1 * 4096)) + ((((int)blockIdx.x) & 127) * 32)) + ((int)threadIdx.x))];
    }
    __syncthreads();
    for (int k_outer_inner = 0; k_outer_inner < 16; ++k_outer_inner) {
      for (int i_c_outer_inner = 0; i_c_outer_inner < 2; ++i_c_outer_inner) {
        for (int j_c_outer_inner = 0; j_c_outer_inner < 4; ++j_c_outer_inner) {
          for (int k_inner = 0; k_inner < 4; ++k_inner) {
            T_matmul_NN_local[((i_c_outer_inner * 16) + j_c_outer_inner)] = (T_matmul_NN_local[((i_c_outer_inner * 16) + j_c_outer_inner)] + (data_shared[(((((((int)threadIdx.x) >> 2) * 512) + (i_c_outer_inner * 256)) + (k_outer_inner * 4)) + k_inner)] * kernel_shared[((((k_outer_inner * 128) + (k_inner * 32)) + ((((int)threadIdx.x) & 3) * 4)) + j_c_outer_inner)]));
            T_matmul_NN_local[(((i_c_outer_inner * 16) + j_c_outer_inner) + 32)] = (T_matmul_NN_local[(((i_c_outer_inner * 16) + j_c_outer_inner) + 32)] + (data_shared[(((((((int)threadIdx.x) >> 2) * 512) + (i_c_outer_inner * 256)) + (k_outer_inner * 4)) + k_inner)] * kernel_shared[(((((k_outer_inner * 128) + (k_inner * 32)) + ((((int)threadIdx.x) & 3) * 4)) + j_c_outer_inner) + 16)]));
            T_matmul_NN_local[(((i_c_outer_inner * 16) + j_c_outer_inner) + 4)] = (T_matmul_NN_local[(((i_c_outer_inner * 16) + j_c_outer_inner) + 4)] + (data_shared[((((((((int)threadIdx.x) >> 2) * 512) + (i_c_outer_inner * 256)) + (k_outer_inner * 4)) + k_inner) + 64)] * kernel_shared[((((k_outer_inner * 128) + (k_inner * 32)) + ((((int)threadIdx.x) & 3) * 4)) + j_c_outer_inner)]));
            T_matmul_NN_local[(((i_c_outer_inner * 16) + j_c_outer_inner) + 36)] = (T_matmul_NN_local[(((i_c_outer_inner * 16) + j_c_outer_inner) + 36)] + (data_shared[((((((((int)threadIdx.x) >> 2) * 512) + (i_c_outer_inner * 256)) + (k_outer_inner * 4)) + k_inner) + 64)] * kernel_shared[(((((k_outer_inner * 128) + (k_inner * 32)) + ((((int)threadIdx.x) & 3) * 4)) + j_c_outer_inner) + 16)]));
            T_matmul_NN_local[(((i_c_outer_inner * 16) + j_c_outer_inner) + 8)] = (T_matmul_NN_local[(((i_c_outer_inner * 16) + j_c_outer_inner) + 8)] + (data_shared[((((((((int)threadIdx.x) >> 2) * 512) + (i_c_outer_inner * 256)) + (k_outer_inner * 4)) + k_inner) + 128)] * kernel_shared[((((k_outer_inner * 128) + (k_inner * 32)) + ((((int)threadIdx.x) & 3) * 4)) + j_c_outer_inner)]));
            T_matmul_NN_local[(((i_c_outer_inner * 16) + j_c_outer_inner) + 40)] = (T_matmul_NN_local[(((i_c_outer_inner * 16) + j_c_outer_inner) + 40)] + (data_shared[((((((((int)threadIdx.x) >> 2) * 512) + (i_c_outer_inner * 256)) + (k_outer_inner * 4)) + k_inner) + 128)] * kernel_shared[(((((k_outer_inner * 128) + (k_inner * 32)) + ((((int)threadIdx.x) & 3) * 4)) + j_c_outer_inner) + 16)]));
            T_matmul_NN_local[(((i_c_outer_inner * 16) + j_c_outer_inner) + 12)] = (T_matmul_NN_local[(((i_c_outer_inner * 16) + j_c_outer_inner) + 12)] + (data_shared[((((((((int)threadIdx.x) >> 2) * 512) + (i_c_outer_inner * 256)) + (k_outer_inner * 4)) + k_inner) + 192)] * kernel_shared[((((k_outer_inner * 128) + (k_inner * 32)) + ((((int)threadIdx.x) & 3) * 4)) + j_c_outer_inner)]));
            T_matmul_NN_local[(((i_c_outer_inner * 16) + j_c_outer_inner) + 44)] = (T_matmul_NN_local[(((i_c_outer_inner * 16) + j_c_outer_inner) + 44)] + (data_shared[((((((((int)threadIdx.x) >> 2) * 512) + (i_c_outer_inner * 256)) + (k_outer_inner * 4)) + k_inner) + 192)] * kernel_shared[(((((k_outer_inner * 128) + (k_inner * 32)) + ((((int)threadIdx.x) & 3) * 4)) + j_c_outer_inner) + 16)]));
          }
        }
      }
    }
  }
  for (int i_inner = 0; i_inner < 8; ++i_inner) {
    for (int j_inner = 0; j_inner < 4; ++j_inner) {
      T_matmul_NN[(((((((((int)blockIdx.x) >> 7) * 262144) + ((((int)threadIdx.x) >> 2) * 32768)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 127) * 32)) + ((((int)threadIdx.x) & 3) * 4)) + j_inner)] = T_matmul_NN_local[((i_inner * 4) + j_inner)];
      T_matmul_NN[((((((((((int)blockIdx.x) >> 7) * 262144) + ((((int)threadIdx.x) >> 2) * 32768)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 127) * 32)) + ((((int)threadIdx.x) & 3) * 4)) + j_inner) + 16)] = T_matmul_NN_local[(((i_inner * 4) + j_inner) + 32)];
    }
  }
}

