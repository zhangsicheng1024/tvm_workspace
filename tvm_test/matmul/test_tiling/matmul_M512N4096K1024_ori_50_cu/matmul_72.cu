
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
  float T_matmul_NN_local[128];
  __shared__ float data_shared[512];
  __shared__ float kernel_shared[8192];
  for (int i_c_inner_init = 0; i_c_inner_init < 4; ++i_c_inner_init) {
    for (int j_c_inner_init = 0; j_c_inner_init < 4; ++j_c_inner_init) {
      T_matmul_NN_local[((i_c_inner_init * 4) + j_c_inner_init)] = 0.000000e+00f;
      T_matmul_NN_local[(((i_c_inner_init * 4) + j_c_inner_init) + 16)] = 0.000000e+00f;
      T_matmul_NN_local[(((i_c_inner_init * 4) + j_c_inner_init) + 32)] = 0.000000e+00f;
      T_matmul_NN_local[(((i_c_inner_init * 4) + j_c_inner_init) + 48)] = 0.000000e+00f;
      T_matmul_NN_local[(((i_c_inner_init * 4) + j_c_inner_init) + 64)] = 0.000000e+00f;
      T_matmul_NN_local[(((i_c_inner_init * 4) + j_c_inner_init) + 80)] = 0.000000e+00f;
      T_matmul_NN_local[(((i_c_inner_init * 4) + j_c_inner_init) + 96)] = 0.000000e+00f;
      T_matmul_NN_local[(((i_c_inner_init * 4) + j_c_inner_init) + 112)] = 0.000000e+00f;
    }
  }
  for (int k_outer_outer = 0; k_outer_outer < 128; ++k_outer_outer) {
    __syncthreads();
    data_shared[((int)threadIdx.x)] = data[(((((((int)blockIdx.x) >> 2) * 65536) + ((((int)threadIdx.x) >> 3) * 1024)) + (k_outer_outer * 8)) + (((int)threadIdx.x) & 7))];
    for (int ax0_ax1_fused_outer_outer = 0; ax0_ax1_fused_outer_outer < 4; ++ax0_ax1_fused_outer_outer) {
      *(float4*)(kernel_shared + ((ax0_ax1_fused_outer_outer * 2048) + (((int)threadIdx.x) * 4))) = *(float4*)(kernel + (((((k_outer_outer * 32768) + (ax0_ax1_fused_outer_outer * 8192)) + ((((int)threadIdx.x) >> 8) * 4096)) + ((((int)blockIdx.x) & 3) * 1024)) + ((((int)threadIdx.x) & 255) * 4)));
    }
    __syncthreads();
    for (int k_inner = 0; k_inner < 8; ++k_inner) {
      for (int i_c_inner = 0; i_c_inner < 4; ++i_c_inner) {
        for (int j_c_inner = 0; j_c_inner < 4; ++j_c_inner) {
          T_matmul_NN_local[((i_c_inner * 4) + j_c_inner)] = (T_matmul_NN_local[((i_c_inner * 4) + j_c_inner)] + (data_shared[((((((int)threadIdx.x) >> 5) * 32) + (i_c_inner * 8)) + k_inner)] * kernel_shared[(((k_inner * 1024) + ((((int)threadIdx.x) & 31) * 4)) + j_c_inner)]));
          T_matmul_NN_local[(((i_c_inner * 4) + j_c_inner) + 16)] = (T_matmul_NN_local[(((i_c_inner * 4) + j_c_inner) + 16)] + (data_shared[((((((int)threadIdx.x) >> 5) * 32) + (i_c_inner * 8)) + k_inner)] * kernel_shared[((((k_inner * 1024) + ((((int)threadIdx.x) & 31) * 4)) + j_c_inner) + 128)]));
          T_matmul_NN_local[(((i_c_inner * 4) + j_c_inner) + 32)] = (T_matmul_NN_local[(((i_c_inner * 4) + j_c_inner) + 32)] + (data_shared[((((((int)threadIdx.x) >> 5) * 32) + (i_c_inner * 8)) + k_inner)] * kernel_shared[((((k_inner * 1024) + ((((int)threadIdx.x) & 31) * 4)) + j_c_inner) + 256)]));
          T_matmul_NN_local[(((i_c_inner * 4) + j_c_inner) + 48)] = (T_matmul_NN_local[(((i_c_inner * 4) + j_c_inner) + 48)] + (data_shared[((((((int)threadIdx.x) >> 5) * 32) + (i_c_inner * 8)) + k_inner)] * kernel_shared[((((k_inner * 1024) + ((((int)threadIdx.x) & 31) * 4)) + j_c_inner) + 384)]));
          T_matmul_NN_local[(((i_c_inner * 4) + j_c_inner) + 64)] = (T_matmul_NN_local[(((i_c_inner * 4) + j_c_inner) + 64)] + (data_shared[((((((int)threadIdx.x) >> 5) * 32) + (i_c_inner * 8)) + k_inner)] * kernel_shared[((((k_inner * 1024) + ((((int)threadIdx.x) & 31) * 4)) + j_c_inner) + 512)]));
          T_matmul_NN_local[(((i_c_inner * 4) + j_c_inner) + 80)] = (T_matmul_NN_local[(((i_c_inner * 4) + j_c_inner) + 80)] + (data_shared[((((((int)threadIdx.x) >> 5) * 32) + (i_c_inner * 8)) + k_inner)] * kernel_shared[((((k_inner * 1024) + ((((int)threadIdx.x) & 31) * 4)) + j_c_inner) + 640)]));
          T_matmul_NN_local[(((i_c_inner * 4) + j_c_inner) + 96)] = (T_matmul_NN_local[(((i_c_inner * 4) + j_c_inner) + 96)] + (data_shared[((((((int)threadIdx.x) >> 5) * 32) + (i_c_inner * 8)) + k_inner)] * kernel_shared[((((k_inner * 1024) + ((((int)threadIdx.x) & 31) * 4)) + j_c_inner) + 768)]));
          T_matmul_NN_local[(((i_c_inner * 4) + j_c_inner) + 112)] = (T_matmul_NN_local[(((i_c_inner * 4) + j_c_inner) + 112)] + (data_shared[((((((int)threadIdx.x) >> 5) * 32) + (i_c_inner * 8)) + k_inner)] * kernel_shared[((((k_inner * 1024) + ((((int)threadIdx.x) & 31) * 4)) + j_c_inner) + 896)]));
        }
      }
    }
  }
  for (int i_inner = 0; i_inner < 4; ++i_inner) {
    for (int j_inner = 0; j_inner < 4; ++j_inner) {
      T_matmul_NN[(((((((((int)blockIdx.x) >> 2) * 262144) + ((((int)threadIdx.x) >> 5) * 16384)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 3) * 1024)) + ((((int)threadIdx.x) & 31) * 4)) + j_inner)] = T_matmul_NN_local[((i_inner * 4) + j_inner)];
      T_matmul_NN[((((((((((int)blockIdx.x) >> 2) * 262144) + ((((int)threadIdx.x) >> 5) * 16384)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 3) * 1024)) + ((((int)threadIdx.x) & 31) * 4)) + j_inner) + 128)] = T_matmul_NN_local[(((i_inner * 4) + j_inner) + 16)];
      T_matmul_NN[((((((((((int)blockIdx.x) >> 2) * 262144) + ((((int)threadIdx.x) >> 5) * 16384)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 3) * 1024)) + ((((int)threadIdx.x) & 31) * 4)) + j_inner) + 256)] = T_matmul_NN_local[(((i_inner * 4) + j_inner) + 32)];
      T_matmul_NN[((((((((((int)blockIdx.x) >> 2) * 262144) + ((((int)threadIdx.x) >> 5) * 16384)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 3) * 1024)) + ((((int)threadIdx.x) & 31) * 4)) + j_inner) + 384)] = T_matmul_NN_local[(((i_inner * 4) + j_inner) + 48)];
      T_matmul_NN[((((((((((int)blockIdx.x) >> 2) * 262144) + ((((int)threadIdx.x) >> 5) * 16384)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 3) * 1024)) + ((((int)threadIdx.x) & 31) * 4)) + j_inner) + 512)] = T_matmul_NN_local[(((i_inner * 4) + j_inner) + 64)];
      T_matmul_NN[((((((((((int)blockIdx.x) >> 2) * 262144) + ((((int)threadIdx.x) >> 5) * 16384)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 3) * 1024)) + ((((int)threadIdx.x) & 31) * 4)) + j_inner) + 640)] = T_matmul_NN_local[(((i_inner * 4) + j_inner) + 80)];
      T_matmul_NN[((((((((((int)blockIdx.x) >> 2) * 262144) + ((((int)threadIdx.x) >> 5) * 16384)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 3) * 1024)) + ((((int)threadIdx.x) & 31) * 4)) + j_inner) + 768)] = T_matmul_NN_local[(((i_inner * 4) + j_inner) + 96)];
      T_matmul_NN[((((((((((int)blockIdx.x) >> 2) * 262144) + ((((int)threadIdx.x) >> 5) * 16384)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 3) * 1024)) + ((((int)threadIdx.x) & 31) * 4)) + j_inner) + 896)] = T_matmul_NN_local[(((i_inner * 4) + j_inner) + 112)];
    }
  }
}

