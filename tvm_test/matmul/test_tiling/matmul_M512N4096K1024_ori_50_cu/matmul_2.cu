
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
  float T_matmul_NN_local[4];
  __shared__ float data_shared[8192];
  __shared__ float kernel_shared[2048];
  T_matmul_NN_local[0] = 0.000000e+00f;
  T_matmul_NN_local[1] = 0.000000e+00f;
  T_matmul_NN_local[2] = 0.000000e+00f;
  T_matmul_NN_local[3] = 0.000000e+00f;
  for (int k_outer_outer = 0; k_outer_outer < 8; ++k_outer_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_outer_outer = 0; ax0_ax1_fused_outer_outer < 8; ++ax0_ax1_fused_outer_outer) {
      *(float4*)(data_shared + ((ax0_ax1_fused_outer_outer * 1024) + (((int)threadIdx.x) * 4))) = *(float4*)(data + ((((((((int)blockIdx.x) >> 8) * 65536) + (ax0_ax1_fused_outer_outer * 8192)) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer_outer * 128)) + ((((int)threadIdx.x) & 31) * 4)));
    }
    for (int ax0_ax1_fused_outer_outer_1 = 0; ax0_ax1_fused_outer_outer_1 < 8; ++ax0_ax1_fused_outer_outer_1) {
      kernel_shared[((ax0_ax1_fused_outer_outer_1 * 256) + ((int)threadIdx.x))] = kernel[(((((k_outer_outer * 524288) + (ax0_ax1_fused_outer_outer_1 * 65536)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.x) & 255) * 16)) + (((int)threadIdx.x) & 15))];
    }
    __syncthreads();
    for (int k_outer_inner = 0; k_outer_inner < 64; ++k_outer_inner) {
      for (int k_inner = 0; k_inner < 2; ++k_inner) {
        T_matmul_NN_local[0] = (T_matmul_NN_local[0] + (data_shared[((((((int)threadIdx.x) >> 3) * 128) + (k_outer_inner * 2)) + k_inner)] * kernel_shared[(((k_outer_inner * 32) + (k_inner * 16)) + (((int)threadIdx.x) & 7))]));
        T_matmul_NN_local[1] = (T_matmul_NN_local[1] + (data_shared[((((((int)threadIdx.x) >> 3) * 128) + (k_outer_inner * 2)) + k_inner)] * kernel_shared[((((k_outer_inner * 32) + (k_inner * 16)) + (((int)threadIdx.x) & 7)) + 8)]));
        T_matmul_NN_local[2] = (T_matmul_NN_local[2] + (data_shared[(((((((int)threadIdx.x) >> 3) * 128) + (k_outer_inner * 2)) + k_inner) + 4096)] * kernel_shared[(((k_outer_inner * 32) + (k_inner * 16)) + (((int)threadIdx.x) & 7))]));
        T_matmul_NN_local[3] = (T_matmul_NN_local[3] + (data_shared[(((((((int)threadIdx.x) >> 3) * 128) + (k_outer_inner * 2)) + k_inner) + 4096)] * kernel_shared[((((k_outer_inner * 32) + (k_inner * 16)) + (((int)threadIdx.x) & 7)) + 8)]));
      }
    }
  }
  T_matmul_NN[(((((((int)blockIdx.x) >> 8) * 262144) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.x) & 255) * 16)) + (((int)threadIdx.x) & 7))] = T_matmul_NN_local[0];
  T_matmul_NN[((((((((int)blockIdx.x) >> 8) * 262144) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.x) & 255) * 16)) + (((int)threadIdx.x) & 7)) + 8)] = T_matmul_NN_local[1];
  T_matmul_NN[((((((((int)blockIdx.x) >> 8) * 262144) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.x) & 255) * 16)) + (((int)threadIdx.x) & 7)) + 131072)] = T_matmul_NN_local[2];
  T_matmul_NN[((((((((int)blockIdx.x) >> 8) * 262144) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.x) & 255) * 16)) + (((int)threadIdx.x) & 7)) + 131080)] = T_matmul_NN_local[3];
}

