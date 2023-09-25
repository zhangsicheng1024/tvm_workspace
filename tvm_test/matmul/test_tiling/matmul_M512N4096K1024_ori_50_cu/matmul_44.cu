
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
  float T_matmul_NN_local[16];
  __shared__ float data_shared[32];
  __shared__ float kernel_shared[512];
  for (int i_c_inner_init = 0; i_c_inner_init < 4; ++i_c_inner_init) {
    T_matmul_NN_local[i_c_inner_init] = 0.000000e+00f;
    T_matmul_NN_local[(i_c_inner_init + 4)] = 0.000000e+00f;
    T_matmul_NN_local[(i_c_inner_init + 8)] = 0.000000e+00f;
    T_matmul_NN_local[(i_c_inner_init + 12)] = 0.000000e+00f;
  }
  for (int k_outer_outer = 0; k_outer_outer < 512; ++k_outer_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_inner_s = 0; ax0_ax1_fused_inner_s < 8; ++ax0_ax1_fused_inner_s) {
      if (((int)threadIdx.x) < 4) {
        data_shared[((((int)threadIdx.x) * 8) + ax0_ax1_fused_inner_s)] = data[((((((((int)blockIdx.x) >> 4) * 16384) + (((int)threadIdx.x) * 4096)) + ((ax0_ax1_fused_inner_s >> 1) * 1024)) + (k_outer_outer * 2)) + (ax0_ax1_fused_inner_s & 1))];
      }
    }
    for (int ax0_ax1_fused_outer_outer = 0; ax0_ax1_fused_outer_outer < 2; ++ax0_ax1_fused_outer_outer) {
      kernel_shared[((ax0_ax1_fused_outer_outer * 256) + ((int)threadIdx.x))] = kernel[((((k_outer_outer * 8192) + (ax0_ax1_fused_outer_outer * 4096)) + ((((int)blockIdx.x) & 15) * 256)) + ((int)threadIdx.x))];
    }
    __syncthreads();
    for (int k_inner = 0; k_inner < 2; ++k_inner) {
      for (int i_c_inner = 0; i_c_inner < 4; ++i_c_inner) {
        T_matmul_NN_local[i_c_inner] = (T_matmul_NN_local[i_c_inner] + (data_shared[((((((int)threadIdx.x) >> 6) * 8) + (i_c_inner * 2)) + k_inner)] * kernel_shared[((k_inner * 256) + (((int)threadIdx.x) & 63))]));
        T_matmul_NN_local[(i_c_inner + 4)] = (T_matmul_NN_local[(i_c_inner + 4)] + (data_shared[((((((int)threadIdx.x) >> 6) * 8) + (i_c_inner * 2)) + k_inner)] * kernel_shared[(((k_inner * 256) + (((int)threadIdx.x) & 63)) + 64)]));
        T_matmul_NN_local[(i_c_inner + 8)] = (T_matmul_NN_local[(i_c_inner + 8)] + (data_shared[((((((int)threadIdx.x) >> 6) * 8) + (i_c_inner * 2)) + k_inner)] * kernel_shared[(((k_inner * 256) + (((int)threadIdx.x) & 63)) + 128)]));
        T_matmul_NN_local[(i_c_inner + 12)] = (T_matmul_NN_local[(i_c_inner + 12)] + (data_shared[((((((int)threadIdx.x) >> 6) * 8) + (i_c_inner * 2)) + k_inner)] * kernel_shared[(((k_inner * 256) + (((int)threadIdx.x) & 63)) + 192)]));
      }
    }
  }
  for (int i_inner = 0; i_inner < 4; ++i_inner) {
    T_matmul_NN[((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 6) * 16384)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 15) * 256)) + (((int)threadIdx.x) & 63))] = T_matmul_NN_local[i_inner];
    T_matmul_NN[(((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 6) * 16384)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 15) * 256)) + (((int)threadIdx.x) & 63)) + 64)] = T_matmul_NN_local[(i_inner + 4)];
    T_matmul_NN[(((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 6) * 16384)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 15) * 256)) + (((int)threadIdx.x) & 63)) + 128)] = T_matmul_NN_local[(i_inner + 8)];
    T_matmul_NN[(((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 6) * 16384)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 15) * 256)) + (((int)threadIdx.x) & 63)) + 192)] = T_matmul_NN_local[(i_inner + 12)];
  }
}

