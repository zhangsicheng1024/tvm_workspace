
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
extern "C" __global__ void __launch_bounds__(512) mymatmul_kernel0(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ T_matmul_NN) {
  float T_matmul_NN_local[2048];
  __shared__ float data_shared[512];
  __shared__ float kernel_shared[8192];
  for (int j_c_outer_inner_init = 0; j_c_outer_inner_init < 128; ++j_c_outer_inner_init) {
    for (int i_c_inner_init = 0; i_c_inner_init < 4; ++i_c_inner_init) {
      T_matmul_NN_local[((i_c_inner_init * 128) + j_c_outer_inner_init)] = 0.000000e+00f;
      T_matmul_NN_local[(((i_c_inner_init * 128) + j_c_outer_inner_init) + 512)] = 0.000000e+00f;
      T_matmul_NN_local[(((i_c_inner_init * 128) + j_c_outer_inner_init) + 1024)] = 0.000000e+00f;
      T_matmul_NN_local[(((i_c_inner_init * 128) + j_c_outer_inner_init) + 1536)] = 0.000000e+00f;
    }
  }
  for (int k_outer_outer = 0; k_outer_outer < 512; ++k_outer_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_inner_s = 0; ax0_ax1_fused_inner_s < 4; ++ax0_ax1_fused_inner_s) {
      if (((int)threadIdx.x) < 128) {
        data_shared[((((int)threadIdx.x) * 4) + ax0_ax1_fused_inner_s)] = data[(((((((int)blockIdx.x) * 262144) + (((int)threadIdx.x) * 2048)) + ((ax0_ax1_fused_inner_s >> 1) * 1024)) + (k_outer_outer * 2)) + (ax0_ax1_fused_inner_s & 1))];
      }
    }
    for (int ax0_ax1_fused_outer_outer = 0; ax0_ax1_fused_outer_outer < 4; ++ax0_ax1_fused_outer_outer) {
      *(float4*)(kernel_shared + ((ax0_ax1_fused_outer_outer * 2048) + (((int)threadIdx.x) * 4))) = *(float4*)(kernel + (((k_outer_outer * 8192) + (ax0_ax1_fused_outer_outer * 2048)) + (((int)threadIdx.x) * 4)));
    }
    __syncthreads();
    for (int j_c_outer_inner = 0; j_c_outer_inner < 128; ++j_c_outer_inner) {
      for (int k_inner = 0; k_inner < 2; ++k_inner) {
        for (int i_c_inner = 0; i_c_inner < 4; ++i_c_inner) {
          T_matmul_NN_local[((i_c_inner * 128) + j_c_outer_inner)] = (T_matmul_NN_local[((i_c_inner * 128) + j_c_outer_inner)] + (data_shared[((((((int)threadIdx.x) >> 3) * 8) + (i_c_inner * 2)) + k_inner)] * kernel_shared[(((k_inner * 4096) + ((((int)threadIdx.x) & 7) * 128)) + j_c_outer_inner)]));
          T_matmul_NN_local[(((i_c_inner * 128) + j_c_outer_inner) + 512)] = (T_matmul_NN_local[(((i_c_inner * 128) + j_c_outer_inner) + 512)] + (data_shared[((((((int)threadIdx.x) >> 3) * 8) + (i_c_inner * 2)) + k_inner)] * kernel_shared[((((k_inner * 4096) + ((((int)threadIdx.x) & 7) * 128)) + j_c_outer_inner) + 1024)]));
          T_matmul_NN_local[(((i_c_inner * 128) + j_c_outer_inner) + 1024)] = (T_matmul_NN_local[(((i_c_inner * 128) + j_c_outer_inner) + 1024)] + (data_shared[((((((int)threadIdx.x) >> 3) * 8) + (i_c_inner * 2)) + k_inner)] * kernel_shared[((((k_inner * 4096) + ((((int)threadIdx.x) & 7) * 128)) + j_c_outer_inner) + 2048)]));
          T_matmul_NN_local[(((i_c_inner * 128) + j_c_outer_inner) + 1536)] = (T_matmul_NN_local[(((i_c_inner * 128) + j_c_outer_inner) + 1536)] + (data_shared[((((((int)threadIdx.x) >> 3) * 8) + (i_c_inner * 2)) + k_inner)] * kernel_shared[((((k_inner * 4096) + ((((int)threadIdx.x) & 7) * 128)) + j_c_outer_inner) + 3072)]));
        }
      }
    }
  }
  for (int i_inner = 0; i_inner < 4; ++i_inner) {
    for (int j_inner = 0; j_inner < 128; ++j_inner) {
      T_matmul_NN[(((((((int)blockIdx.x) * 1048576) + ((((int)threadIdx.x) >> 3) * 16384)) + (i_inner * 4096)) + ((((int)threadIdx.x) & 7) * 128)) + j_inner)] = T_matmul_NN_local[((i_inner * 128) + j_inner)];
      T_matmul_NN[((((((((int)blockIdx.x) * 1048576) + ((((int)threadIdx.x) >> 3) * 16384)) + (i_inner * 4096)) + ((((int)threadIdx.x) & 7) * 128)) + j_inner) + 1024)] = T_matmul_NN_local[(((i_inner * 128) + j_inner) + 512)];
      T_matmul_NN[((((((((int)blockIdx.x) * 1048576) + ((((int)threadIdx.x) >> 3) * 16384)) + (i_inner * 4096)) + ((((int)threadIdx.x) & 7) * 128)) + j_inner) + 2048)] = T_matmul_NN_local[(((i_inner * 128) + j_inner) + 1024)];
      T_matmul_NN[((((((((int)blockIdx.x) * 1048576) + ((((int)threadIdx.x) >> 3) * 16384)) + (i_inner * 4096)) + ((((int)threadIdx.x) & 7) * 128)) + j_inner) + 3072)] = T_matmul_NN_local[(((i_inner * 128) + j_inner) + 1536)];
    }
  }
}

