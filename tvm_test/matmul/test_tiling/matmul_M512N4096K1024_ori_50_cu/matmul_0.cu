
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
  __shared__ float data_shared[512];
  __shared__ float kernel_shared[256];
  for (int i_c_inner_init = 0; i_c_inner_init < 4; ++i_c_inner_init) {
    T_matmul_NN_local[i_c_inner_init] = 0.000000e+00f;
    T_matmul_NN_local[(i_c_inner_init + 4)] = 0.000000e+00f;
    T_matmul_NN_local[(i_c_inner_init + 8)] = 0.000000e+00f;
    T_matmul_NN_local[(i_c_inner_init + 12)] = 0.000000e+00f;
    T_matmul_NN_local[(i_c_inner_init + 16)] = 0.000000e+00f;
    T_matmul_NN_local[(i_c_inner_init + 20)] = 0.000000e+00f;
    T_matmul_NN_local[(i_c_inner_init + 24)] = 0.000000e+00f;
    T_matmul_NN_local[(i_c_inner_init + 28)] = 0.000000e+00f;
  }
  for (int k_outer_outer = 0; k_outer_outer < 128; ++k_outer_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_outer_outer = 0; ax0_ax1_fused_outer_outer < 8; ++ax0_ax1_fused_outer_outer) {
      data_shared[((ax0_ax1_fused_outer_outer * 64) + ((int)threadIdx.x))] = data[((((((((int)blockIdx.x) >> 7) * 65536) + (ax0_ax1_fused_outer_outer * 8192)) + ((((int)threadIdx.x) >> 3) * 1024)) + (k_outer_outer * 8)) + (((int)threadIdx.x) & 7))];
    }
    for (int ax0_ax1_fused_inner_s = 0; ax0_ax1_fused_inner_s < 64; ++ax0_ax1_fused_inner_s) {
      if (((int)threadIdx.x) < 4) {
        kernel_shared[((((int)threadIdx.x) * 64) + ax0_ax1_fused_inner_s)] = kernel[(((((k_outer_outer * 32768) + (((int)threadIdx.x) * 8192)) + ((ax0_ax1_fused_inner_s >> 5) * 4096)) + ((((int)blockIdx.x) & 127) * 32)) + (ax0_ax1_fused_inner_s & 31))];
      }
    }
    __syncthreads();
    for (int k_outer_inner = 0; k_outer_inner < 8; ++k_outer_inner) {
      for (int i_c_inner = 0; i_c_inner < 4; ++i_c_inner) {
        T_matmul_NN_local[i_c_inner] = (T_matmul_NN_local[i_c_inner] + (data_shared[((((((int)threadIdx.x) >> 2) * 32) + (i_c_inner * 8)) + k_outer_inner)] * kernel_shared[((k_outer_inner * 32) + (((int)threadIdx.x) & 3))]));
        T_matmul_NN_local[(i_c_inner + 4)] = (T_matmul_NN_local[(i_c_inner + 4)] + (data_shared[((((((int)threadIdx.x) >> 2) * 32) + (i_c_inner * 8)) + k_outer_inner)] * kernel_shared[(((k_outer_inner * 32) + (((int)threadIdx.x) & 3)) + 4)]));
        T_matmul_NN_local[(i_c_inner + 8)] = (T_matmul_NN_local[(i_c_inner + 8)] + (data_shared[((((((int)threadIdx.x) >> 2) * 32) + (i_c_inner * 8)) + k_outer_inner)] * kernel_shared[(((k_outer_inner * 32) + (((int)threadIdx.x) & 3)) + 8)]));
        T_matmul_NN_local[(i_c_inner + 12)] = (T_matmul_NN_local[(i_c_inner + 12)] + (data_shared[((((((int)threadIdx.x) >> 2) * 32) + (i_c_inner * 8)) + k_outer_inner)] * kernel_shared[(((k_outer_inner * 32) + (((int)threadIdx.x) & 3)) + 12)]));
        T_matmul_NN_local[(i_c_inner + 16)] = (T_matmul_NN_local[(i_c_inner + 16)] + (data_shared[((((((int)threadIdx.x) >> 2) * 32) + (i_c_inner * 8)) + k_outer_inner)] * kernel_shared[(((k_outer_inner * 32) + (((int)threadIdx.x) & 3)) + 16)]));
        T_matmul_NN_local[(i_c_inner + 20)] = (T_matmul_NN_local[(i_c_inner + 20)] + (data_shared[((((((int)threadIdx.x) >> 2) * 32) + (i_c_inner * 8)) + k_outer_inner)] * kernel_shared[(((k_outer_inner * 32) + (((int)threadIdx.x) & 3)) + 20)]));
        T_matmul_NN_local[(i_c_inner + 24)] = (T_matmul_NN_local[(i_c_inner + 24)] + (data_shared[((((((int)threadIdx.x) >> 2) * 32) + (i_c_inner * 8)) + k_outer_inner)] * kernel_shared[(((k_outer_inner * 32) + (((int)threadIdx.x) & 3)) + 24)]));
        T_matmul_NN_local[(i_c_inner + 28)] = (T_matmul_NN_local[(i_c_inner + 28)] + (data_shared[((((((int)threadIdx.x) >> 2) * 32) + (i_c_inner * 8)) + k_outer_inner)] * kernel_shared[(((k_outer_inner * 32) + (((int)threadIdx.x) & 3)) + 28)]));
      }
    }
  }
  for (int i_inner = 0; i_inner < 4; ++i_inner) {
    T_matmul_NN[((((((((int)blockIdx.x) >> 7) * 262144) + ((((int)threadIdx.x) >> 2) * 16384)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 127) * 32)) + (((int)threadIdx.x) & 3))] = T_matmul_NN_local[i_inner];
    T_matmul_NN[(((((((((int)blockIdx.x) >> 7) * 262144) + ((((int)threadIdx.x) >> 2) * 16384)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 127) * 32)) + (((int)threadIdx.x) & 3)) + 4)] = T_matmul_NN_local[(i_inner + 4)];
    T_matmul_NN[(((((((((int)blockIdx.x) >> 7) * 262144) + ((((int)threadIdx.x) >> 2) * 16384)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 127) * 32)) + (((int)threadIdx.x) & 3)) + 8)] = T_matmul_NN_local[(i_inner + 8)];
    T_matmul_NN[(((((((((int)blockIdx.x) >> 7) * 262144) + ((((int)threadIdx.x) >> 2) * 16384)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 127) * 32)) + (((int)threadIdx.x) & 3)) + 12)] = T_matmul_NN_local[(i_inner + 12)];
    T_matmul_NN[(((((((((int)blockIdx.x) >> 7) * 262144) + ((((int)threadIdx.x) >> 2) * 16384)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 127) * 32)) + (((int)threadIdx.x) & 3)) + 16)] = T_matmul_NN_local[(i_inner + 16)];
    T_matmul_NN[(((((((((int)blockIdx.x) >> 7) * 262144) + ((((int)threadIdx.x) >> 2) * 16384)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 127) * 32)) + (((int)threadIdx.x) & 3)) + 20)] = T_matmul_NN_local[(i_inner + 20)];
    T_matmul_NN[(((((((((int)blockIdx.x) >> 7) * 262144) + ((((int)threadIdx.x) >> 2) * 16384)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 127) * 32)) + (((int)threadIdx.x) & 3)) + 24)] = T_matmul_NN_local[(i_inner + 24)];
    T_matmul_NN[(((((((((int)blockIdx.x) >> 7) * 262144) + ((((int)threadIdx.x) >> 2) * 16384)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 127) * 32)) + (((int)threadIdx.x) & 3)) + 28)] = T_matmul_NN_local[(i_inner + 28)];
  }
}

