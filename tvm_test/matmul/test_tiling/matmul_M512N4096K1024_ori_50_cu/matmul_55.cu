
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
  float T_matmul_NN_local[512];
  __shared__ float data_shared[4096];
  __shared__ float kernel_shared[8192];
  for (int i_c_outer_inner_init = 0; i_c_outer_inner_init < 2; ++i_c_outer_inner_init) {
    for (int j_c_outer_inner_init = 0; j_c_outer_inner_init < 16; ++j_c_outer_inner_init) {
      for (int j_c_inner_init = 0; j_c_inner_init < 2; ++j_c_inner_init) {
        T_matmul_NN_local[(((i_c_outer_inner_init * 32) + (j_c_outer_inner_init * 2)) + j_c_inner_init)] = 0.000000e+00f;
        T_matmul_NN_local[((((i_c_outer_inner_init * 32) + (j_c_outer_inner_init * 2)) + j_c_inner_init) + 64)] = 0.000000e+00f;
        T_matmul_NN_local[((((i_c_outer_inner_init * 32) + (j_c_outer_inner_init * 2)) + j_c_inner_init) + 128)] = 0.000000e+00f;
        T_matmul_NN_local[((((i_c_outer_inner_init * 32) + (j_c_outer_inner_init * 2)) + j_c_inner_init) + 192)] = 0.000000e+00f;
        T_matmul_NN_local[((((i_c_outer_inner_init * 32) + (j_c_outer_inner_init * 2)) + j_c_inner_init) + 256)] = 0.000000e+00f;
        T_matmul_NN_local[((((i_c_outer_inner_init * 32) + (j_c_outer_inner_init * 2)) + j_c_inner_init) + 320)] = 0.000000e+00f;
        T_matmul_NN_local[((((i_c_outer_inner_init * 32) + (j_c_outer_inner_init * 2)) + j_c_inner_init) + 384)] = 0.000000e+00f;
        T_matmul_NN_local[((((i_c_outer_inner_init * 32) + (j_c_outer_inner_init * 2)) + j_c_inner_init) + 448)] = 0.000000e+00f;
      }
    }
  }
  for (int k_outer_outer = 0; k_outer_outer < 64; ++k_outer_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_outer_outer = 0; ax0_ax1_fused_outer_outer < 16; ++ax0_ax1_fused_outer_outer) {
      data_shared[((ax0_ax1_fused_outer_outer * 256) + ((int)threadIdx.x))] = data[((((((((int)blockIdx.x) >> 3) * 262144) + (ax0_ax1_fused_outer_outer * 16384)) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer_outer * 16)) + (((int)threadIdx.x) & 15))];
    }
    for (int ax0_ax1_fused_outer_outer_1 = 0; ax0_ax1_fused_outer_outer_1 < 32; ++ax0_ax1_fused_outer_outer_1) {
      kernel_shared[((ax0_ax1_fused_outer_outer_1 * 256) + ((int)threadIdx.x))] = kernel[(((((k_outer_outer * 65536) + ((ax0_ax1_fused_outer_outer_1 >> 1) * 4096)) + ((((int)blockIdx.x) & 7) * 512)) + ((ax0_ax1_fused_outer_outer_1 & 1) * 256)) + ((int)threadIdx.x))];
    }
    __syncthreads();
    for (int i_c_outer_inner = 0; i_c_outer_inner < 2; ++i_c_outer_inner) {
      for (int j_c_outer_inner = 0; j_c_outer_inner < 16; ++j_c_outer_inner) {
        for (int k_inner = 0; k_inner < 16; ++k_inner) {
          for (int j_c_inner = 0; j_c_inner < 2; ++j_c_inner) {
            T_matmul_NN_local[(((i_c_outer_inner * 32) + (j_c_outer_inner * 2)) + j_c_inner)] = (T_matmul_NN_local[(((i_c_outer_inner * 32) + (j_c_outer_inner * 2)) + j_c_inner)] + (data_shared[((((((int)threadIdx.x) >> 2) * 32) + (i_c_outer_inner * 16)) + k_inner)] * kernel_shared[((((k_inner * 512) + ((((int)threadIdx.x) & 3) * 32)) + (j_c_outer_inner * 2)) + j_c_inner)]));
            T_matmul_NN_local[((((i_c_outer_inner * 32) + (j_c_outer_inner * 2)) + j_c_inner) + 64)] = (T_matmul_NN_local[((((i_c_outer_inner * 32) + (j_c_outer_inner * 2)) + j_c_inner) + 64)] + (data_shared[((((((int)threadIdx.x) >> 2) * 32) + (i_c_outer_inner * 16)) + k_inner)] * kernel_shared[(((((k_inner * 512) + ((((int)threadIdx.x) & 3) * 32)) + (j_c_outer_inner * 2)) + j_c_inner) + 128)]));
            T_matmul_NN_local[((((i_c_outer_inner * 32) + (j_c_outer_inner * 2)) + j_c_inner) + 128)] = (T_matmul_NN_local[((((i_c_outer_inner * 32) + (j_c_outer_inner * 2)) + j_c_inner) + 128)] + (data_shared[((((((int)threadIdx.x) >> 2) * 32) + (i_c_outer_inner * 16)) + k_inner)] * kernel_shared[(((((k_inner * 512) + ((((int)threadIdx.x) & 3) * 32)) + (j_c_outer_inner * 2)) + j_c_inner) + 256)]));
            T_matmul_NN_local[((((i_c_outer_inner * 32) + (j_c_outer_inner * 2)) + j_c_inner) + 192)] = (T_matmul_NN_local[((((i_c_outer_inner * 32) + (j_c_outer_inner * 2)) + j_c_inner) + 192)] + (data_shared[((((((int)threadIdx.x) >> 2) * 32) + (i_c_outer_inner * 16)) + k_inner)] * kernel_shared[(((((k_inner * 512) + ((((int)threadIdx.x) & 3) * 32)) + (j_c_outer_inner * 2)) + j_c_inner) + 384)]));
            T_matmul_NN_local[((((i_c_outer_inner * 32) + (j_c_outer_inner * 2)) + j_c_inner) + 256)] = (T_matmul_NN_local[((((i_c_outer_inner * 32) + (j_c_outer_inner * 2)) + j_c_inner) + 256)] + (data_shared[(((((((int)threadIdx.x) >> 2) * 32) + (i_c_outer_inner * 16)) + k_inner) + 2048)] * kernel_shared[((((k_inner * 512) + ((((int)threadIdx.x) & 3) * 32)) + (j_c_outer_inner * 2)) + j_c_inner)]));
            T_matmul_NN_local[((((i_c_outer_inner * 32) + (j_c_outer_inner * 2)) + j_c_inner) + 320)] = (T_matmul_NN_local[((((i_c_outer_inner * 32) + (j_c_outer_inner * 2)) + j_c_inner) + 320)] + (data_shared[(((((((int)threadIdx.x) >> 2) * 32) + (i_c_outer_inner * 16)) + k_inner) + 2048)] * kernel_shared[(((((k_inner * 512) + ((((int)threadIdx.x) & 3) * 32)) + (j_c_outer_inner * 2)) + j_c_inner) + 128)]));
            T_matmul_NN_local[((((i_c_outer_inner * 32) + (j_c_outer_inner * 2)) + j_c_inner) + 384)] = (T_matmul_NN_local[((((i_c_outer_inner * 32) + (j_c_outer_inner * 2)) + j_c_inner) + 384)] + (data_shared[(((((((int)threadIdx.x) >> 2) * 32) + (i_c_outer_inner * 16)) + k_inner) + 2048)] * kernel_shared[(((((k_inner * 512) + ((((int)threadIdx.x) & 3) * 32)) + (j_c_outer_inner * 2)) + j_c_inner) + 256)]));
            T_matmul_NN_local[((((i_c_outer_inner * 32) + (j_c_outer_inner * 2)) + j_c_inner) + 448)] = (T_matmul_NN_local[((((i_c_outer_inner * 32) + (j_c_outer_inner * 2)) + j_c_inner) + 448)] + (data_shared[(((((((int)threadIdx.x) >> 2) * 32) + (i_c_outer_inner * 16)) + k_inner) + 2048)] * kernel_shared[(((((k_inner * 512) + ((((int)threadIdx.x) & 3) * 32)) + (j_c_outer_inner * 2)) + j_c_inner) + 384)]));
          }
        }
      }
    }
  }
  for (int i_inner = 0; i_inner < 2; ++i_inner) {
    for (int j_inner = 0; j_inner < 32; ++j_inner) {
      T_matmul_NN[(((((((((int)blockIdx.x) >> 3) * 1048576) + ((((int)threadIdx.x) >> 2) * 8192)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 7) * 512)) + ((((int)threadIdx.x) & 3) * 32)) + j_inner)] = T_matmul_NN_local[((i_inner * 32) + j_inner)];
      T_matmul_NN[((((((((((int)blockIdx.x) >> 3) * 1048576) + ((((int)threadIdx.x) >> 2) * 8192)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 7) * 512)) + ((((int)threadIdx.x) & 3) * 32)) + j_inner) + 128)] = T_matmul_NN_local[(((i_inner * 32) + j_inner) + 64)];
      T_matmul_NN[((((((((((int)blockIdx.x) >> 3) * 1048576) + ((((int)threadIdx.x) >> 2) * 8192)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 7) * 512)) + ((((int)threadIdx.x) & 3) * 32)) + j_inner) + 256)] = T_matmul_NN_local[(((i_inner * 32) + j_inner) + 128)];
      T_matmul_NN[((((((((((int)blockIdx.x) >> 3) * 1048576) + ((((int)threadIdx.x) >> 2) * 8192)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 7) * 512)) + ((((int)threadIdx.x) & 3) * 32)) + j_inner) + 384)] = T_matmul_NN_local[(((i_inner * 32) + j_inner) + 192)];
      T_matmul_NN[((((((((((int)blockIdx.x) >> 3) * 1048576) + ((((int)threadIdx.x) >> 2) * 8192)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 7) * 512)) + ((((int)threadIdx.x) & 3) * 32)) + j_inner) + 524288)] = T_matmul_NN_local[(((i_inner * 32) + j_inner) + 256)];
      T_matmul_NN[((((((((((int)blockIdx.x) >> 3) * 1048576) + ((((int)threadIdx.x) >> 2) * 8192)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 7) * 512)) + ((((int)threadIdx.x) & 3) * 32)) + j_inner) + 524416)] = T_matmul_NN_local[(((i_inner * 32) + j_inner) + 320)];
      T_matmul_NN[((((((((((int)blockIdx.x) >> 3) * 1048576) + ((((int)threadIdx.x) >> 2) * 8192)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 7) * 512)) + ((((int)threadIdx.x) & 3) * 32)) + j_inner) + 524544)] = T_matmul_NN_local[(((i_inner * 32) + j_inner) + 384)];
      T_matmul_NN[((((((((((int)blockIdx.x) >> 3) * 1048576) + ((((int)threadIdx.x) >> 2) * 8192)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 7) * 512)) + ((((int)threadIdx.x) & 3) * 32)) + j_inner) + 524672)] = T_matmul_NN_local[(((i_inner * 32) + j_inner) + 448)];
    }
  }
}

