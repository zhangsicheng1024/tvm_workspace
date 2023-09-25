
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
  float T_matmul_NN_local[128];
  __shared__ float data_shared[512];
  __shared__ float kernel_shared[4096];
  for (int i_c_outer_inner_init = 0; i_c_outer_inner_init < 4; ++i_c_outer_inner_init) {
    for (int j_c_outer_inner_init = 0; j_c_outer_inner_init < 2; ++j_c_outer_inner_init) {
      for (int i_c_inner_init = 0; i_c_inner_init < 2; ++i_c_inner_init) {
        T_matmul_NN_local[(((i_c_outer_inner_init * 4) + (i_c_inner_init * 2)) + j_c_outer_inner_init)] = 0.000000e+00f;
        T_matmul_NN_local[((((i_c_outer_inner_init * 4) + (i_c_inner_init * 2)) + j_c_outer_inner_init) + 16)] = 0.000000e+00f;
        T_matmul_NN_local[((((i_c_outer_inner_init * 4) + (i_c_inner_init * 2)) + j_c_outer_inner_init) + 32)] = 0.000000e+00f;
        T_matmul_NN_local[((((i_c_outer_inner_init * 4) + (i_c_inner_init * 2)) + j_c_outer_inner_init) + 48)] = 0.000000e+00f;
        T_matmul_NN_local[((((i_c_outer_inner_init * 4) + (i_c_inner_init * 2)) + j_c_outer_inner_init) + 64)] = 0.000000e+00f;
        T_matmul_NN_local[((((i_c_outer_inner_init * 4) + (i_c_inner_init * 2)) + j_c_outer_inner_init) + 80)] = 0.000000e+00f;
        T_matmul_NN_local[((((i_c_outer_inner_init * 4) + (i_c_inner_init * 2)) + j_c_outer_inner_init) + 96)] = 0.000000e+00f;
        T_matmul_NN_local[((((i_c_outer_inner_init * 4) + (i_c_inner_init * 2)) + j_c_outer_inner_init) + 112)] = 0.000000e+00f;
      }
    }
  }
  for (int k_outer_outer = 0; k_outer_outer < 128; ++k_outer_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_inner_s = 0; ax0_ax1_fused_inner_s < 64; ++ax0_ax1_fused_inner_s) {
      if (((int)threadIdx.x) < 8) {
        data_shared[((((int)threadIdx.x) * 64) + ax0_ax1_fused_inner_s)] = data[((((((((int)blockIdx.x) >> 3) * 65536) + (((int)threadIdx.x) * 8192)) + ((ax0_ax1_fused_inner_s >> 3) * 1024)) + (k_outer_outer * 8)) + (ax0_ax1_fused_inner_s & 7))];
      }
    }
    for (int ax0_ax1_fused_outer_outer = 0; ax0_ax1_fused_outer_outer < 16; ++ax0_ax1_fused_outer_outer) {
      kernel_shared[((ax0_ax1_fused_outer_outer * 256) + ((int)threadIdx.x))] = kernel[(((((k_outer_outer * 32768) + ((ax0_ax1_fused_outer_outer >> 1) * 4096)) + ((((int)blockIdx.x) & 7) * 512)) + ((ax0_ax1_fused_outer_outer & 1) * 256)) + ((int)threadIdx.x))];
    }
    __syncthreads();
    for (int i_c_outer_inner = 0; i_c_outer_inner < 4; ++i_c_outer_inner) {
      for (int j_c_outer_inner = 0; j_c_outer_inner < 2; ++j_c_outer_inner) {
        for (int k_inner = 0; k_inner < 8; ++k_inner) {
          for (int i_c_inner = 0; i_c_inner < 2; ++i_c_inner) {
            T_matmul_NN_local[(((i_c_outer_inner * 4) + (i_c_inner * 2)) + j_c_outer_inner)] = (T_matmul_NN_local[(((i_c_outer_inner * 4) + (i_c_inner * 2)) + j_c_outer_inner)] + (data_shared[(((((((int)threadIdx.x) >> 6) * 64) + (i_c_outer_inner * 16)) + (i_c_inner * 8)) + k_inner)] * kernel_shared[(((k_inner * 512) + ((((int)threadIdx.x) & 63) * 2)) + j_c_outer_inner)]));
            T_matmul_NN_local[((((i_c_outer_inner * 4) + (i_c_inner * 2)) + j_c_outer_inner) + 16)] = (T_matmul_NN_local[((((i_c_outer_inner * 4) + (i_c_inner * 2)) + j_c_outer_inner) + 16)] + (data_shared[(((((((int)threadIdx.x) >> 6) * 64) + (i_c_outer_inner * 16)) + (i_c_inner * 8)) + k_inner)] * kernel_shared[((((k_inner * 512) + ((((int)threadIdx.x) & 63) * 2)) + j_c_outer_inner) + 128)]));
            T_matmul_NN_local[((((i_c_outer_inner * 4) + (i_c_inner * 2)) + j_c_outer_inner) + 32)] = (T_matmul_NN_local[((((i_c_outer_inner * 4) + (i_c_inner * 2)) + j_c_outer_inner) + 32)] + (data_shared[(((((((int)threadIdx.x) >> 6) * 64) + (i_c_outer_inner * 16)) + (i_c_inner * 8)) + k_inner)] * kernel_shared[((((k_inner * 512) + ((((int)threadIdx.x) & 63) * 2)) + j_c_outer_inner) + 256)]));
            T_matmul_NN_local[((((i_c_outer_inner * 4) + (i_c_inner * 2)) + j_c_outer_inner) + 48)] = (T_matmul_NN_local[((((i_c_outer_inner * 4) + (i_c_inner * 2)) + j_c_outer_inner) + 48)] + (data_shared[(((((((int)threadIdx.x) >> 6) * 64) + (i_c_outer_inner * 16)) + (i_c_inner * 8)) + k_inner)] * kernel_shared[((((k_inner * 512) + ((((int)threadIdx.x) & 63) * 2)) + j_c_outer_inner) + 384)]));
            T_matmul_NN_local[((((i_c_outer_inner * 4) + (i_c_inner * 2)) + j_c_outer_inner) + 64)] = (T_matmul_NN_local[((((i_c_outer_inner * 4) + (i_c_inner * 2)) + j_c_outer_inner) + 64)] + (data_shared[((((((((int)threadIdx.x) >> 6) * 64) + (i_c_outer_inner * 16)) + (i_c_inner * 8)) + k_inner) + 256)] * kernel_shared[(((k_inner * 512) + ((((int)threadIdx.x) & 63) * 2)) + j_c_outer_inner)]));
            T_matmul_NN_local[((((i_c_outer_inner * 4) + (i_c_inner * 2)) + j_c_outer_inner) + 80)] = (T_matmul_NN_local[((((i_c_outer_inner * 4) + (i_c_inner * 2)) + j_c_outer_inner) + 80)] + (data_shared[((((((((int)threadIdx.x) >> 6) * 64) + (i_c_outer_inner * 16)) + (i_c_inner * 8)) + k_inner) + 256)] * kernel_shared[((((k_inner * 512) + ((((int)threadIdx.x) & 63) * 2)) + j_c_outer_inner) + 128)]));
            T_matmul_NN_local[((((i_c_outer_inner * 4) + (i_c_inner * 2)) + j_c_outer_inner) + 96)] = (T_matmul_NN_local[((((i_c_outer_inner * 4) + (i_c_inner * 2)) + j_c_outer_inner) + 96)] + (data_shared[((((((((int)threadIdx.x) >> 6) * 64) + (i_c_outer_inner * 16)) + (i_c_inner * 8)) + k_inner) + 256)] * kernel_shared[((((k_inner * 512) + ((((int)threadIdx.x) & 63) * 2)) + j_c_outer_inner) + 256)]));
            T_matmul_NN_local[((((i_c_outer_inner * 4) + (i_c_inner * 2)) + j_c_outer_inner) + 112)] = (T_matmul_NN_local[((((i_c_outer_inner * 4) + (i_c_inner * 2)) + j_c_outer_inner) + 112)] + (data_shared[((((((((int)threadIdx.x) >> 6) * 64) + (i_c_outer_inner * 16)) + (i_c_inner * 8)) + k_inner) + 256)] * kernel_shared[((((k_inner * 512) + ((((int)threadIdx.x) & 63) * 2)) + j_c_outer_inner) + 384)]));
          }
        }
      }
    }
  }
  for (int i_inner = 0; i_inner < 8; ++i_inner) {
    for (int j_inner = 0; j_inner < 2; ++j_inner) {
      T_matmul_NN[(((((((((int)blockIdx.x) >> 3) * 262144) + ((((int)threadIdx.x) >> 6) * 32768)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 7) * 512)) + ((((int)threadIdx.x) & 63) * 2)) + j_inner)] = T_matmul_NN_local[((i_inner * 2) + j_inner)];
      T_matmul_NN[((((((((((int)blockIdx.x) >> 3) * 262144) + ((((int)threadIdx.x) >> 6) * 32768)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 7) * 512)) + ((((int)threadIdx.x) & 63) * 2)) + j_inner) + 128)] = T_matmul_NN_local[(((i_inner * 2) + j_inner) + 16)];
      T_matmul_NN[((((((((((int)blockIdx.x) >> 3) * 262144) + ((((int)threadIdx.x) >> 6) * 32768)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 7) * 512)) + ((((int)threadIdx.x) & 63) * 2)) + j_inner) + 256)] = T_matmul_NN_local[(((i_inner * 2) + j_inner) + 32)];
      T_matmul_NN[((((((((((int)blockIdx.x) >> 3) * 262144) + ((((int)threadIdx.x) >> 6) * 32768)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 7) * 512)) + ((((int)threadIdx.x) & 63) * 2)) + j_inner) + 384)] = T_matmul_NN_local[(((i_inner * 2) + j_inner) + 48)];
      T_matmul_NN[((((((((((int)blockIdx.x) >> 3) * 262144) + ((((int)threadIdx.x) >> 6) * 32768)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 7) * 512)) + ((((int)threadIdx.x) & 63) * 2)) + j_inner) + 131072)] = T_matmul_NN_local[(((i_inner * 2) + j_inner) + 64)];
      T_matmul_NN[((((((((((int)blockIdx.x) >> 3) * 262144) + ((((int)threadIdx.x) >> 6) * 32768)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 7) * 512)) + ((((int)threadIdx.x) & 63) * 2)) + j_inner) + 131200)] = T_matmul_NN_local[(((i_inner * 2) + j_inner) + 80)];
      T_matmul_NN[((((((((((int)blockIdx.x) >> 3) * 262144) + ((((int)threadIdx.x) >> 6) * 32768)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 7) * 512)) + ((((int)threadIdx.x) & 63) * 2)) + j_inner) + 131328)] = T_matmul_NN_local[(((i_inner * 2) + j_inner) + 96)];
      T_matmul_NN[((((((((((int)blockIdx.x) >> 3) * 262144) + ((((int)threadIdx.x) >> 6) * 32768)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 7) * 512)) + ((((int)threadIdx.x) & 63) * 2)) + j_inner) + 131456)] = T_matmul_NN_local[(((i_inner * 2) + j_inner) + 112)];
    }
  }
}

