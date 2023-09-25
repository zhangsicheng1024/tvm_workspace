
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
extern "C" __global__ void __launch_bounds__(1024) mymatmul_kernel0(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ T_matmul_NN) {
  float T_matmul_NN_local[64];
  __shared__ float data_shared[512];
  __shared__ float kernel_shared[512];
  for (int i_c_outer_inner_init = 0; i_c_outer_inner_init < 16; ++i_c_outer_inner_init) {
    for (int j_c_outer_inner_init = 0; j_c_outer_inner_init < 2; ++j_c_outer_inner_init) {
      T_matmul_NN_local[((i_c_outer_inner_init * 2) + j_c_outer_inner_init)] = 0.000000e+00f;
      T_matmul_NN_local[(((i_c_outer_inner_init * 2) + j_c_outer_inner_init) + 32)] = 0.000000e+00f;
    }
  }
  for (int k_outer_outer = 0; k_outer_outer < 512; ++k_outer_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_inner_s = 0; ax0_ax1_fused_inner_s < 4; ++ax0_ax1_fused_inner_s) {
      if (((int)threadIdx.x) < 128) {
        data_shared[((((int)threadIdx.x) * 4) + ax0_ax1_fused_inner_s)] = data[((((((((int)blockIdx.x) >> 4) * 262144) + (((int)threadIdx.x) * 2048)) + ((ax0_ax1_fused_inner_s >> 1) * 1024)) + (k_outer_outer * 2)) + (ax0_ax1_fused_inner_s & 1))];
      }
    }
    if (((int)threadIdx.x) < 128) {
      *(float4*)(kernel_shared + (((int)threadIdx.x) * 4)) = *(float4*)(kernel + ((((k_outer_outer * 8192) + ((((int)threadIdx.x) >> 6) * 4096)) + ((((int)blockIdx.x) & 15) * 256)) + ((((int)threadIdx.x) & 63) * 4)));
    }
    __syncthreads();
    for (int k_outer_inner = 0; k_outer_inner < 2; ++k_outer_inner) {
      for (int i_c_outer_inner = 0; i_c_outer_inner < 16; ++i_c_outer_inner) {
        for (int j_c_outer_inner = 0; j_c_outer_inner < 2; ++j_c_outer_inner) {
          T_matmul_NN_local[((i_c_outer_inner * 2) + j_c_outer_inner)] = (T_matmul_NN_local[((i_c_outer_inner * 2) + j_c_outer_inner)] + (data_shared[((((((int)threadIdx.x) >> 6) * 32) + (i_c_outer_inner * 2)) + k_outer_inner)] * kernel_shared[(((k_outer_inner * 256) + ((((int)threadIdx.x) & 63) * 2)) + j_c_outer_inner)]));
          T_matmul_NN_local[(((i_c_outer_inner * 2) + j_c_outer_inner) + 32)] = (T_matmul_NN_local[(((i_c_outer_inner * 2) + j_c_outer_inner) + 32)] + (data_shared[((((((int)threadIdx.x) >> 6) * 32) + (i_c_outer_inner * 2)) + k_outer_inner)] * kernel_shared[((((k_outer_inner * 256) + ((((int)threadIdx.x) & 63) * 2)) + j_c_outer_inner) + 128)]));
        }
      }
    }
  }
  for (int i_inner = 0; i_inner < 16; ++i_inner) {
    for (int j_inner = 0; j_inner < 2; ++j_inner) {
      T_matmul_NN[(((((((((int)blockIdx.x) >> 4) * 1048576) + ((((int)threadIdx.x) >> 6) * 65536)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 15) * 256)) + ((((int)threadIdx.x) & 63) * 2)) + j_inner)] = T_matmul_NN_local[((i_inner * 2) + j_inner)];
      T_matmul_NN[((((((((((int)blockIdx.x) >> 4) * 1048576) + ((((int)threadIdx.x) >> 6) * 65536)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 15) * 256)) + ((((int)threadIdx.x) & 63) * 2)) + j_inner) + 128)] = T_matmul_NN_local[(((i_inner * 2) + j_inner) + 32)];
    }
  }
}

