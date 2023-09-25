
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
  float T_matmul_NN_local[32];
  __shared__ float data_shared[2048];
  __shared__ float kernel_shared[1024];
  for (int i_c_inner_init = 0; i_c_inner_init < 2; ++i_c_inner_init) {
    for (int j_c_inner_init = 0; j_c_inner_init < 2; ++j_c_inner_init) {
      T_matmul_NN_local[((i_c_inner_init * 2) + j_c_inner_init)] = 0.000000e+00f;
      T_matmul_NN_local[(((i_c_inner_init * 2) + j_c_inner_init) + 4)] = 0.000000e+00f;
      T_matmul_NN_local[(((i_c_inner_init * 2) + j_c_inner_init) + 8)] = 0.000000e+00f;
      T_matmul_NN_local[(((i_c_inner_init * 2) + j_c_inner_init) + 12)] = 0.000000e+00f;
      T_matmul_NN_local[(((i_c_inner_init * 2) + j_c_inner_init) + 16)] = 0.000000e+00f;
      T_matmul_NN_local[(((i_c_inner_init * 2) + j_c_inner_init) + 20)] = 0.000000e+00f;
      T_matmul_NN_local[(((i_c_inner_init * 2) + j_c_inner_init) + 24)] = 0.000000e+00f;
      T_matmul_NN_local[(((i_c_inner_init * 2) + j_c_inner_init) + 28)] = 0.000000e+00f;
    }
  }
  for (int k_outer_outer = 0; k_outer_outer < 32; ++k_outer_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_outer_outer = 0; ax0_ax1_fused_outer_outer < 32; ++ax0_ax1_fused_outer_outer) {
      data_shared[((ax0_ax1_fused_outer_outer * 64) + ((int)threadIdx.x))] = data[((((((((int)blockIdx.x) >> 7) * 65536) + (ax0_ax1_fused_outer_outer * 2048)) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx.x) & 31))];
    }
    for (int ax0_ax1_fused_outer_outer_1 = 0; ax0_ax1_fused_outer_outer_1 < 16; ++ax0_ax1_fused_outer_outer_1) {
      kernel_shared[((ax0_ax1_fused_outer_outer_1 * 64) + ((int)threadIdx.x))] = kernel[(((((k_outer_outer * 131072) + (ax0_ax1_fused_outer_outer_1 * 8192)) + ((((int)threadIdx.x) >> 5) * 4096)) + ((((int)blockIdx.x) & 127) * 32)) + (((int)threadIdx.x) & 31))];
    }
    __syncthreads();
    for (int k_outer_inner = 0; k_outer_inner < 4; ++k_outer_inner) {
      for (int k_inner = 0; k_inner < 8; ++k_inner) {
        for (int i_c_inner = 0; i_c_inner < 2; ++i_c_inner) {
          for (int j_c_inner = 0; j_c_inner < 2; ++j_c_inner) {
            T_matmul_NN_local[((i_c_inner * 2) + j_c_inner)] = (T_matmul_NN_local[((i_c_inner * 2) + j_c_inner)] + (data_shared[(((((((int)threadIdx.x) >> 3) * 64) + (i_c_inner * 32)) + (k_outer_inner * 8)) + k_inner)] * kernel_shared[((((k_outer_inner * 256) + (k_inner * 32)) + ((((int)threadIdx.x) & 7) * 2)) + j_c_inner)]));
            T_matmul_NN_local[(((i_c_inner * 2) + j_c_inner) + 4)] = (T_matmul_NN_local[(((i_c_inner * 2) + j_c_inner) + 4)] + (data_shared[(((((((int)threadIdx.x) >> 3) * 64) + (i_c_inner * 32)) + (k_outer_inner * 8)) + k_inner)] * kernel_shared[(((((k_outer_inner * 256) + (k_inner * 32)) + ((((int)threadIdx.x) & 7) * 2)) + j_c_inner) + 16)]));
            T_matmul_NN_local[(((i_c_inner * 2) + j_c_inner) + 8)] = (T_matmul_NN_local[(((i_c_inner * 2) + j_c_inner) + 8)] + (data_shared[((((((((int)threadIdx.x) >> 3) * 64) + (i_c_inner * 32)) + (k_outer_inner * 8)) + k_inner) + 512)] * kernel_shared[((((k_outer_inner * 256) + (k_inner * 32)) + ((((int)threadIdx.x) & 7) * 2)) + j_c_inner)]));
            T_matmul_NN_local[(((i_c_inner * 2) + j_c_inner) + 12)] = (T_matmul_NN_local[(((i_c_inner * 2) + j_c_inner) + 12)] + (data_shared[((((((((int)threadIdx.x) >> 3) * 64) + (i_c_inner * 32)) + (k_outer_inner * 8)) + k_inner) + 512)] * kernel_shared[(((((k_outer_inner * 256) + (k_inner * 32)) + ((((int)threadIdx.x) & 7) * 2)) + j_c_inner) + 16)]));
            T_matmul_NN_local[(((i_c_inner * 2) + j_c_inner) + 16)] = (T_matmul_NN_local[(((i_c_inner * 2) + j_c_inner) + 16)] + (data_shared[((((((((int)threadIdx.x) >> 3) * 64) + (i_c_inner * 32)) + (k_outer_inner * 8)) + k_inner) + 1024)] * kernel_shared[((((k_outer_inner * 256) + (k_inner * 32)) + ((((int)threadIdx.x) & 7) * 2)) + j_c_inner)]));
            T_matmul_NN_local[(((i_c_inner * 2) + j_c_inner) + 20)] = (T_matmul_NN_local[(((i_c_inner * 2) + j_c_inner) + 20)] + (data_shared[((((((((int)threadIdx.x) >> 3) * 64) + (i_c_inner * 32)) + (k_outer_inner * 8)) + k_inner) + 1024)] * kernel_shared[(((((k_outer_inner * 256) + (k_inner * 32)) + ((((int)threadIdx.x) & 7) * 2)) + j_c_inner) + 16)]));
            T_matmul_NN_local[(((i_c_inner * 2) + j_c_inner) + 24)] = (T_matmul_NN_local[(((i_c_inner * 2) + j_c_inner) + 24)] + (data_shared[((((((((int)threadIdx.x) >> 3) * 64) + (i_c_inner * 32)) + (k_outer_inner * 8)) + k_inner) + 1536)] * kernel_shared[((((k_outer_inner * 256) + (k_inner * 32)) + ((((int)threadIdx.x) & 7) * 2)) + j_c_inner)]));
            T_matmul_NN_local[(((i_c_inner * 2) + j_c_inner) + 28)] = (T_matmul_NN_local[(((i_c_inner * 2) + j_c_inner) + 28)] + (data_shared[((((((((int)threadIdx.x) >> 3) * 64) + (i_c_inner * 32)) + (k_outer_inner * 8)) + k_inner) + 1536)] * kernel_shared[(((((k_outer_inner * 256) + (k_inner * 32)) + ((((int)threadIdx.x) & 7) * 2)) + j_c_inner) + 16)]));
          }
        }
      }
    }
  }
  for (int i_inner = 0; i_inner < 2; ++i_inner) {
    for (int j_inner = 0; j_inner < 2; ++j_inner) {
      T_matmul_NN[(((((((((int)blockIdx.x) >> 7) * 262144) + ((((int)threadIdx.x) >> 3) * 8192)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 127) * 32)) + ((((int)threadIdx.x) & 7) * 2)) + j_inner)] = T_matmul_NN_local[((i_inner * 2) + j_inner)];
      T_matmul_NN[((((((((((int)blockIdx.x) >> 7) * 262144) + ((((int)threadIdx.x) >> 3) * 8192)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 127) * 32)) + ((((int)threadIdx.x) & 7) * 2)) + j_inner) + 16)] = T_matmul_NN_local[(((i_inner * 2) + j_inner) + 4)];
      T_matmul_NN[((((((((((int)blockIdx.x) >> 7) * 262144) + ((((int)threadIdx.x) >> 3) * 8192)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 127) * 32)) + ((((int)threadIdx.x) & 7) * 2)) + j_inner) + 65536)] = T_matmul_NN_local[(((i_inner * 2) + j_inner) + 8)];
      T_matmul_NN[((((((((((int)blockIdx.x) >> 7) * 262144) + ((((int)threadIdx.x) >> 3) * 8192)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 127) * 32)) + ((((int)threadIdx.x) & 7) * 2)) + j_inner) + 65552)] = T_matmul_NN_local[(((i_inner * 2) + j_inner) + 12)];
      T_matmul_NN[((((((((((int)blockIdx.x) >> 7) * 262144) + ((((int)threadIdx.x) >> 3) * 8192)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 127) * 32)) + ((((int)threadIdx.x) & 7) * 2)) + j_inner) + 131072)] = T_matmul_NN_local[(((i_inner * 2) + j_inner) + 16)];
      T_matmul_NN[((((((((((int)blockIdx.x) >> 7) * 262144) + ((((int)threadIdx.x) >> 3) * 8192)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 127) * 32)) + ((((int)threadIdx.x) & 7) * 2)) + j_inner) + 131088)] = T_matmul_NN_local[(((i_inner * 2) + j_inner) + 20)];
      T_matmul_NN[((((((((((int)blockIdx.x) >> 7) * 262144) + ((((int)threadIdx.x) >> 3) * 8192)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 127) * 32)) + ((((int)threadIdx.x) & 7) * 2)) + j_inner) + 196608)] = T_matmul_NN_local[(((i_inner * 2) + j_inner) + 24)];
      T_matmul_NN[((((((((((int)blockIdx.x) >> 7) * 262144) + ((((int)threadIdx.x) >> 3) * 8192)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 127) * 32)) + ((((int)threadIdx.x) & 7) * 2)) + j_inner) + 196624)] = T_matmul_NN_local[(((i_inner * 2) + j_inner) + 28)];
    }
  }
}

