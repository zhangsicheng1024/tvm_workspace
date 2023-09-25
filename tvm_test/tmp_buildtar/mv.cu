
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
extern "C" __global__ void __launch_bounds__(32) mymv_kernel0(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ T_matmul_NN) {
  float T_matmul_NN_local[4];
  __shared__ float data_shared[8];
  __shared__ float kernel_shared[1024];
  for (int j_c_inner_init = 0; j_c_inner_init < 2; ++j_c_inner_init) {
    T_matmul_NN_local[j_c_inner_init] = 0.000000e+00f;
    T_matmul_NN_local[(j_c_inner_init + 2)] = 0.000000e+00f;
  }
  for (int k_outer_outer = 0; k_outer_outer < 1024; ++k_outer_outer) {
    __syncthreads();
    if (((int)threadIdx.x) < 4) {
      *(float2*)(data_shared + (((int)threadIdx.x) * 2)) = *(float2*)(data + ((k_outer_outer * 8) + (((int)threadIdx.x) * 2)));
    }
    for (int ax0_ax1_fused_outer_outer = 0; ax0_ax1_fused_outer_outer < 8; ++ax0_ax1_fused_outer_outer) {
      *(float4*)(kernel_shared + ((ax0_ax1_fused_outer_outer * 128) + (((int)threadIdx.x) * 4))) = *(float4*)(kernel + ((((k_outer_outer * 262144) + (ax0_ax1_fused_outer_outer * 32768)) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) * 4)));
    }
    __syncthreads();
    for (int k_inner = 0; k_inner < 8; ++k_inner) {
      for (int j_c_inner = 0; j_c_inner < 2; ++j_c_inner) {
        T_matmul_NN_local[j_c_inner] = (T_matmul_NN_local[j_c_inner] + (data_shared[k_inner] * kernel_shared[(((k_inner * 128) + (((int)threadIdx.x) * 2)) + j_c_inner)]));
        T_matmul_NN_local[(j_c_inner + 2)] = (T_matmul_NN_local[(j_c_inner + 2)] + (data_shared[k_inner] * kernel_shared[((((k_inner * 128) + (((int)threadIdx.x) * 2)) + j_c_inner) + 64)]));
      }
    }
  }
  for (int j_inner = 0; j_inner < 2; ++j_inner) {
    T_matmul_NN[(((((int)blockIdx.x) * 128) + (((int)threadIdx.x) * 2)) + j_inner)] = T_matmul_NN_local[j_inner];
    T_matmul_NN[((((((int)blockIdx.x) * 128) + (((int)threadIdx.x) * 2)) + j_inner) + 64)] = T_matmul_NN_local[(j_inner + 2)];
  }
}

