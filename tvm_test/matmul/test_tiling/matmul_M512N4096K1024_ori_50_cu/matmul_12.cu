
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
  float T_matmul_NN_local[128];
  __shared__ float data_shared[2048];
  __shared__ float kernel_shared[2048];
  for (int j_c_outer_inner_init = 0; j_c_outer_inner_init < 2; ++j_c_outer_inner_init) {
    for (int j_c_inner_init = 0; j_c_inner_init < 16; ++j_c_inner_init) {
      T_matmul_NN_local[((j_c_outer_inner_init * 16) + j_c_inner_init)] = 0.000000e+00f;
      T_matmul_NN_local[(((j_c_outer_inner_init * 16) + j_c_inner_init) + 32)] = 0.000000e+00f;
      T_matmul_NN_local[(((j_c_outer_inner_init * 16) + j_c_inner_init) + 64)] = 0.000000e+00f;
      T_matmul_NN_local[(((j_c_outer_inner_init * 16) + j_c_inner_init) + 96)] = 0.000000e+00f;
    }
  }
  for (int k_outer_outer = 0; k_outer_outer < 32; ++k_outer_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_outer_outer = 0; ax0_ax1_fused_outer_outer < 64; ++ax0_ax1_fused_outer_outer) {
      data_shared[((ax0_ax1_fused_outer_outer * 32) + ((int)threadIdx.x))] = data[(((((((int)blockIdx.x) >> 6) * 65536) + (ax0_ax1_fused_outer_outer * 1024)) + (k_outer_outer * 32)) + ((int)threadIdx.x))];
    }
    for (int ax0_ax1_fused_outer_outer_1 = 0; ax0_ax1_fused_outer_outer_1 < 16; ++ax0_ax1_fused_outer_outer_1) {
      *(float4*)(kernel_shared + ((ax0_ax1_fused_outer_outer_1 * 128) + (((int)threadIdx.x) * 4))) = *(float4*)(kernel + (((((k_outer_outer * 131072) + (ax0_ax1_fused_outer_outer_1 * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.x) & 63) * 64)) + ((((int)threadIdx.x) & 15) * 4)));
    }
    __syncthreads();
    for (int j_c_outer_inner = 0; j_c_outer_inner < 2; ++j_c_outer_inner) {
      for (int k_inner = 0; k_inner < 32; ++k_inner) {
        for (int j_c_inner = 0; j_c_inner < 16; ++j_c_inner) {
          T_matmul_NN_local[((j_c_outer_inner * 16) + j_c_inner)] = (T_matmul_NN_local[((j_c_outer_inner * 16) + j_c_inner)] + (data_shared[(((((int)threadIdx.x) >> 1) * 32) + k_inner)] * kernel_shared[((((k_inner * 64) + ((((int)threadIdx.x) & 1) * 32)) + (j_c_outer_inner * 16)) + j_c_inner)]));
          T_matmul_NN_local[(((j_c_outer_inner * 16) + j_c_inner) + 32)] = (T_matmul_NN_local[(((j_c_outer_inner * 16) + j_c_inner) + 32)] + (data_shared[((((((int)threadIdx.x) >> 1) * 32) + k_inner) + 512)] * kernel_shared[((((k_inner * 64) + ((((int)threadIdx.x) & 1) * 32)) + (j_c_outer_inner * 16)) + j_c_inner)]));
          T_matmul_NN_local[(((j_c_outer_inner * 16) + j_c_inner) + 64)] = (T_matmul_NN_local[(((j_c_outer_inner * 16) + j_c_inner) + 64)] + (data_shared[((((((int)threadIdx.x) >> 1) * 32) + k_inner) + 1024)] * kernel_shared[((((k_inner * 64) + ((((int)threadIdx.x) & 1) * 32)) + (j_c_outer_inner * 16)) + j_c_inner)]));
          T_matmul_NN_local[(((j_c_outer_inner * 16) + j_c_inner) + 96)] = (T_matmul_NN_local[(((j_c_outer_inner * 16) + j_c_inner) + 96)] + (data_shared[((((((int)threadIdx.x) >> 1) * 32) + k_inner) + 1536)] * kernel_shared[((((k_inner * 64) + ((((int)threadIdx.x) & 1) * 32)) + (j_c_outer_inner * 16)) + j_c_inner)]));
        }
      }
    }
  }
  for (int j_inner = 0; j_inner < 32; ++j_inner) {
    T_matmul_NN[((((((((int)blockIdx.x) >> 6) * 262144) + ((((int)threadIdx.x) >> 1) * 4096)) + ((((int)blockIdx.x) & 63) * 64)) + ((((int)threadIdx.x) & 1) * 32)) + j_inner)] = T_matmul_NN_local[j_inner];
    T_matmul_NN[(((((((((int)blockIdx.x) >> 6) * 262144) + ((((int)threadIdx.x) >> 1) * 4096)) + ((((int)blockIdx.x) & 63) * 64)) + ((((int)threadIdx.x) & 1) * 32)) + j_inner) + 65536)] = T_matmul_NN_local[(j_inner + 32)];
    T_matmul_NN[(((((((((int)blockIdx.x) >> 6) * 262144) + ((((int)threadIdx.x) >> 1) * 4096)) + ((((int)blockIdx.x) & 63) * 64)) + ((((int)threadIdx.x) & 1) * 32)) + j_inner) + 131072)] = T_matmul_NN_local[(j_inner + 64)];
    T_matmul_NN[(((((((((int)blockIdx.x) >> 6) * 262144) + ((((int)threadIdx.x) >> 1) * 4096)) + ((((int)blockIdx.x) & 63) * 64)) + ((((int)threadIdx.x) & 1) * 32)) + j_inner) + 196608)] = T_matmul_NN_local[(j_inner + 96)];
  }
}

