
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
extern "C" __global__ void __launch_bounds__(128) mymv_kernel0(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ T_matmul_NN) {
  float T_matmul_NN_local[24];
  __shared__ float data_shared[3];
  __shared__ float kernel_shared[9216];
  for (int j_c_outer_inner_init = 0; j_c_outer_inner_init < 3; ++j_c_outer_inner_init) {
    T_matmul_NN_local[(j_c_outer_inner_init * 8)] = 0.000000e+00f;
    T_matmul_NN_local[((j_c_outer_inner_init * 8) + 1)] = 0.000000e+00f;
    T_matmul_NN_local[((j_c_outer_inner_init * 8) + 2)] = 0.000000e+00f;
    T_matmul_NN_local[((j_c_outer_inner_init * 8) + 3)] = 0.000000e+00f;
    T_matmul_NN_local[((j_c_outer_inner_init * 8) + 4)] = 0.000000e+00f;
    T_matmul_NN_local[((j_c_outer_inner_init * 8) + 5)] = 0.000000e+00f;
    T_matmul_NN_local[((j_c_outer_inner_init * 8) + 6)] = 0.000000e+00f;
    T_matmul_NN_local[((j_c_outer_inner_init * 8) + 7)] = 0.000000e+00f;
  }
  for (int k_outer_outer = 0; k_outer_outer < 4096; ++k_outer_outer) {
    __syncthreads();
    if (((int)threadIdx.x) < 1) {
      *(float3*)(data_shared + (((int)threadIdx.x) * 3)) = *(float3*)(data + ((((int)threadIdx.x) * 12288) + (k_outer_outer * 3)));
    }
    for (int ax0_ax1_fused_outer_outer = 0; ax0_ax1_fused_outer_outer < 72; ++ax0_ax1_fused_outer_outer) {
      kernel_shared[((ax0_ax1_fused_outer_outer * 128) + ((int)threadIdx.x))] = kernel[(((((k_outer_outer * 147456) + ((ax0_ax1_fused_outer_outer / 24) * 49152)) + (((int)blockIdx.x) * 3072)) + ((ax0_ax1_fused_outer_outer % 24) * 128)) + ((int)threadIdx.x))];
    }
    __syncthreads();
    for (int k_outer_inner = 0; k_outer_inner < 3; ++k_outer_inner) {
      for (int j_c_outer_inner = 0; j_c_outer_inner < 3; ++j_c_outer_inner) {
        T_matmul_NN_local[(j_c_outer_inner * 8)] = (T_matmul_NN_local[(j_c_outer_inner * 8)] + (data_shared[k_outer_inner] * kernel_shared[(((k_outer_inner * 3072) + (((int)threadIdx.x) * 24)) + (j_c_outer_inner * 8))]));
        T_matmul_NN_local[((j_c_outer_inner * 8) + 1)] = (T_matmul_NN_local[((j_c_outer_inner * 8) + 1)] + (data_shared[k_outer_inner] * kernel_shared[((((k_outer_inner * 3072) + (((int)threadIdx.x) * 24)) + (j_c_outer_inner * 8)) + 1)]));
        T_matmul_NN_local[((j_c_outer_inner * 8) + 2)] = (T_matmul_NN_local[((j_c_outer_inner * 8) + 2)] + (data_shared[k_outer_inner] * kernel_shared[((((k_outer_inner * 3072) + (((int)threadIdx.x) * 24)) + (j_c_outer_inner * 8)) + 2)]));
        T_matmul_NN_local[((j_c_outer_inner * 8) + 3)] = (T_matmul_NN_local[((j_c_outer_inner * 8) + 3)] + (data_shared[k_outer_inner] * kernel_shared[((((k_outer_inner * 3072) + (((int)threadIdx.x) * 24)) + (j_c_outer_inner * 8)) + 3)]));
        T_matmul_NN_local[((j_c_outer_inner * 8) + 4)] = (T_matmul_NN_local[((j_c_outer_inner * 8) + 4)] + (data_shared[k_outer_inner] * kernel_shared[((((k_outer_inner * 3072) + (((int)threadIdx.x) * 24)) + (j_c_outer_inner * 8)) + 4)]));
        T_matmul_NN_local[((j_c_outer_inner * 8) + 5)] = (T_matmul_NN_local[((j_c_outer_inner * 8) + 5)] + (data_shared[k_outer_inner] * kernel_shared[((((k_outer_inner * 3072) + (((int)threadIdx.x) * 24)) + (j_c_outer_inner * 8)) + 5)]));
        T_matmul_NN_local[((j_c_outer_inner * 8) + 6)] = (T_matmul_NN_local[((j_c_outer_inner * 8) + 6)] + (data_shared[k_outer_inner] * kernel_shared[((((k_outer_inner * 3072) + (((int)threadIdx.x) * 24)) + (j_c_outer_inner * 8)) + 6)]));
        T_matmul_NN_local[((j_c_outer_inner * 8) + 7)] = (T_matmul_NN_local[((j_c_outer_inner * 8) + 7)] + (data_shared[k_outer_inner] * kernel_shared[((((k_outer_inner * 3072) + (((int)threadIdx.x) * 24)) + (j_c_outer_inner * 8)) + 7)]));
      }
    }
  }
  for (int j_inner = 0; j_inner < 24; ++j_inner) {
    T_matmul_NN[(((((int)blockIdx.x) * 3072) + (((int)threadIdx.x) * 24)) + j_inner)] = T_matmul_NN_local[j_inner];
  }
}

