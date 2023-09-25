
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
extern "C" __global__ void __launch_bounds__(128) mymatmul_kernel0(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ T_matmul_NN) {
  float T_matmul_NN_local[256];
  __shared__ float data_shared[256];
  __shared__ float kernel_shared[8192];
  for (int j_c_outer_inner_init = 0; j_c_outer_inner_init < 2; ++j_c_outer_inner_init) {
    for (int i_c_inner_init = 0; i_c_inner_init < 16; ++i_c_inner_init) {
      T_matmul_NN_local[((i_c_inner_init * 2) + j_c_outer_inner_init)] = 0.000000e+00f;
      T_matmul_NN_local[(((i_c_inner_init * 2) + j_c_outer_inner_init) + 32)] = 0.000000e+00f;
      T_matmul_NN_local[(((i_c_inner_init * 2) + j_c_outer_inner_init) + 64)] = 0.000000e+00f;
      T_matmul_NN_local[(((i_c_inner_init * 2) + j_c_outer_inner_init) + 96)] = 0.000000e+00f;
      T_matmul_NN_local[(((i_c_inner_init * 2) + j_c_outer_inner_init) + 128)] = 0.000000e+00f;
      T_matmul_NN_local[(((i_c_inner_init * 2) + j_c_outer_inner_init) + 160)] = 0.000000e+00f;
      T_matmul_NN_local[(((i_c_inner_init * 2) + j_c_outer_inner_init) + 192)] = 0.000000e+00f;
      T_matmul_NN_local[(((i_c_inner_init * 2) + j_c_outer_inner_init) + 224)] = 0.000000e+00f;
    }
  }
  for (int k_outer_outer = 0; k_outer_outer < 128; ++k_outer_outer) {
    __syncthreads();
    if (((int)threadIdx.x) < 64) {
      *(float4*)(data_shared + (((int)threadIdx.x) * 4)) = *(float4*)(data + (((((((int)blockIdx.x) >> 2) * 32768) + ((((int)threadIdx.x) >> 1) * 1024)) + (k_outer_outer * 8)) + ((((int)threadIdx.x) & 1) * 4)));
    }
    for (int ax0_ax1_fused_outer_outer = 0; ax0_ax1_fused_outer_outer < 64; ++ax0_ax1_fused_outer_outer) {
      kernel_shared[((ax0_ax1_fused_outer_outer * 128) + ((int)threadIdx.x))] = kernel[(((((k_outer_outer * 32768) + ((ax0_ax1_fused_outer_outer >> 3) * 4096)) + ((((int)blockIdx.x) & 3) * 1024)) + ((ax0_ax1_fused_outer_outer & 7) * 128)) + ((int)threadIdx.x))];
    }
    __syncthreads();
    for (int k_outer_inner = 0; k_outer_inner < 8; ++k_outer_inner) {
      for (int j_c_outer_inner = 0; j_c_outer_inner < 2; ++j_c_outer_inner) {
        for (int i_c_inner = 0; i_c_inner < 16; ++i_c_inner) {
          T_matmul_NN_local[((i_c_inner * 2) + j_c_outer_inner)] = (T_matmul_NN_local[((i_c_inner * 2) + j_c_outer_inner)] + (data_shared[((((((int)threadIdx.x) >> 6) * 128) + (i_c_inner * 8)) + k_outer_inner)] * kernel_shared[(((k_outer_inner * 1024) + ((((int)threadIdx.x) & 63) * 2)) + j_c_outer_inner)]));
          T_matmul_NN_local[(((i_c_inner * 2) + j_c_outer_inner) + 32)] = (T_matmul_NN_local[(((i_c_inner * 2) + j_c_outer_inner) + 32)] + (data_shared[((((((int)threadIdx.x) >> 6) * 128) + (i_c_inner * 8)) + k_outer_inner)] * kernel_shared[((((k_outer_inner * 1024) + ((((int)threadIdx.x) & 63) * 2)) + j_c_outer_inner) + 128)]));
          T_matmul_NN_local[(((i_c_inner * 2) + j_c_outer_inner) + 64)] = (T_matmul_NN_local[(((i_c_inner * 2) + j_c_outer_inner) + 64)] + (data_shared[((((((int)threadIdx.x) >> 6) * 128) + (i_c_inner * 8)) + k_outer_inner)] * kernel_shared[((((k_outer_inner * 1024) + ((((int)threadIdx.x) & 63) * 2)) + j_c_outer_inner) + 256)]));
          T_matmul_NN_local[(((i_c_inner * 2) + j_c_outer_inner) + 96)] = (T_matmul_NN_local[(((i_c_inner * 2) + j_c_outer_inner) + 96)] + (data_shared[((((((int)threadIdx.x) >> 6) * 128) + (i_c_inner * 8)) + k_outer_inner)] * kernel_shared[((((k_outer_inner * 1024) + ((((int)threadIdx.x) & 63) * 2)) + j_c_outer_inner) + 384)]));
          T_matmul_NN_local[(((i_c_inner * 2) + j_c_outer_inner) + 128)] = (T_matmul_NN_local[(((i_c_inner * 2) + j_c_outer_inner) + 128)] + (data_shared[((((((int)threadIdx.x) >> 6) * 128) + (i_c_inner * 8)) + k_outer_inner)] * kernel_shared[((((k_outer_inner * 1024) + ((((int)threadIdx.x) & 63) * 2)) + j_c_outer_inner) + 512)]));
          T_matmul_NN_local[(((i_c_inner * 2) + j_c_outer_inner) + 160)] = (T_matmul_NN_local[(((i_c_inner * 2) + j_c_outer_inner) + 160)] + (data_shared[((((((int)threadIdx.x) >> 6) * 128) + (i_c_inner * 8)) + k_outer_inner)] * kernel_shared[((((k_outer_inner * 1024) + ((((int)threadIdx.x) & 63) * 2)) + j_c_outer_inner) + 640)]));
          T_matmul_NN_local[(((i_c_inner * 2) + j_c_outer_inner) + 192)] = (T_matmul_NN_local[(((i_c_inner * 2) + j_c_outer_inner) + 192)] + (data_shared[((((((int)threadIdx.x) >> 6) * 128) + (i_c_inner * 8)) + k_outer_inner)] * kernel_shared[((((k_outer_inner * 1024) + ((((int)threadIdx.x) & 63) * 2)) + j_c_outer_inner) + 768)]));
          T_matmul_NN_local[(((i_c_inner * 2) + j_c_outer_inner) + 224)] = (T_matmul_NN_local[(((i_c_inner * 2) + j_c_outer_inner) + 224)] + (data_shared[((((((int)threadIdx.x) >> 6) * 128) + (i_c_inner * 8)) + k_outer_inner)] * kernel_shared[((((k_outer_inner * 1024) + ((((int)threadIdx.x) & 63) * 2)) + j_c_outer_inner) + 896)]));
        }
      }
    }
  }
  for (int i_inner = 0; i_inner < 16; ++i_inner) {
    for (int j_inner = 0; j_inner < 2; ++j_inner) {
      T_matmul_NN[(((((((((int)blockIdx.x) >> 2) * 131072) + ((((int)threadIdx.x) >> 6) * 65536)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 3) * 1024)) + ((((int)threadIdx.x) & 63) * 2)) + j_inner)] = T_matmul_NN_local[((i_inner * 2) + j_inner)];
      T_matmul_NN[((((((((((int)blockIdx.x) >> 2) * 131072) + ((((int)threadIdx.x) >> 6) * 65536)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 3) * 1024)) + ((((int)threadIdx.x) & 63) * 2)) + j_inner) + 128)] = T_matmul_NN_local[(((i_inner * 2) + j_inner) + 32)];
      T_matmul_NN[((((((((((int)blockIdx.x) >> 2) * 131072) + ((((int)threadIdx.x) >> 6) * 65536)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 3) * 1024)) + ((((int)threadIdx.x) & 63) * 2)) + j_inner) + 256)] = T_matmul_NN_local[(((i_inner * 2) + j_inner) + 64)];
      T_matmul_NN[((((((((((int)blockIdx.x) >> 2) * 131072) + ((((int)threadIdx.x) >> 6) * 65536)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 3) * 1024)) + ((((int)threadIdx.x) & 63) * 2)) + j_inner) + 384)] = T_matmul_NN_local[(((i_inner * 2) + j_inner) + 96)];
      T_matmul_NN[((((((((((int)blockIdx.x) >> 2) * 131072) + ((((int)threadIdx.x) >> 6) * 65536)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 3) * 1024)) + ((((int)threadIdx.x) & 63) * 2)) + j_inner) + 512)] = T_matmul_NN_local[(((i_inner * 2) + j_inner) + 128)];
      T_matmul_NN[((((((((((int)blockIdx.x) >> 2) * 131072) + ((((int)threadIdx.x) >> 6) * 65536)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 3) * 1024)) + ((((int)threadIdx.x) & 63) * 2)) + j_inner) + 640)] = T_matmul_NN_local[(((i_inner * 2) + j_inner) + 160)];
      T_matmul_NN[((((((((((int)blockIdx.x) >> 2) * 131072) + ((((int)threadIdx.x) >> 6) * 65536)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 3) * 1024)) + ((((int)threadIdx.x) & 63) * 2)) + j_inner) + 768)] = T_matmul_NN_local[(((i_inner * 2) + j_inner) + 192)];
      T_matmul_NN[((((((((((int)blockIdx.x) >> 2) * 131072) + ((((int)threadIdx.x) >> 6) * 65536)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 3) * 1024)) + ((((int)threadIdx.x) & 63) * 2)) + j_inner) + 896)] = T_matmul_NN_local[(((i_inner * 2) + j_inner) + 224)];
    }
  }
}

