
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
  float T_matmul_NN_local[1024];
  __shared__ float data_shared[128];
  __shared__ float kernel_shared[2048];
  for (int i_c_outer_inner_init = 0; i_c_outer_inner_init < 8; ++i_c_outer_inner_init) {
    for (int i_c_inner_init = 0; i_c_inner_init < 2; ++i_c_inner_init) {
      for (int j_c_inner_init = 0; j_c_inner_init < 8; ++j_c_inner_init) {
        T_matmul_NN_local[(((i_c_outer_inner_init * 16) + (i_c_inner_init * 8)) + j_c_inner_init)] = 0.000000e+00f;
        T_matmul_NN_local[((((i_c_outer_inner_init * 16) + (i_c_inner_init * 8)) + j_c_inner_init) + 128)] = 0.000000e+00f;
        T_matmul_NN_local[((((i_c_outer_inner_init * 16) + (i_c_inner_init * 8)) + j_c_inner_init) + 256)] = 0.000000e+00f;
        T_matmul_NN_local[((((i_c_outer_inner_init * 16) + (i_c_inner_init * 8)) + j_c_inner_init) + 384)] = 0.000000e+00f;
        T_matmul_NN_local[((((i_c_outer_inner_init * 16) + (i_c_inner_init * 8)) + j_c_inner_init) + 512)] = 0.000000e+00f;
        T_matmul_NN_local[((((i_c_outer_inner_init * 16) + (i_c_inner_init * 8)) + j_c_inner_init) + 640)] = 0.000000e+00f;
        T_matmul_NN_local[((((i_c_outer_inner_init * 16) + (i_c_inner_init * 8)) + j_c_inner_init) + 768)] = 0.000000e+00f;
        T_matmul_NN_local[((((i_c_outer_inner_init * 16) + (i_c_inner_init * 8)) + j_c_inner_init) + 896)] = 0.000000e+00f;
      }
    }
  }
  for (int k_outer_outer = 0; k_outer_outer < 512; ++k_outer_outer) {
    __syncthreads();
    if (((int)threadIdx.x) < 16) {
      data_shared[(((int)threadIdx.x) * 8)] = data[((((((int)blockIdx.x) >> 2) * 65536) + (((int)threadIdx.x) * 4096)) + (k_outer_outer * 2))];
    }
    if (((int)threadIdx.x) < 16) {
      data_shared[((((int)threadIdx.x) * 8) + 1)] = data[(((((((int)blockIdx.x) >> 2) * 65536) + (((int)threadIdx.x) * 4096)) + (k_outer_outer * 2)) + 1)];
    }
    if (((int)threadIdx.x) < 16) {
      data_shared[((((int)threadIdx.x) * 8) + 2)] = data[(((((((int)blockIdx.x) >> 2) * 65536) + (((int)threadIdx.x) * 4096)) + (k_outer_outer * 2)) + 1024)];
    }
    if (((int)threadIdx.x) < 16) {
      data_shared[((((int)threadIdx.x) * 8) + 3)] = data[(((((((int)blockIdx.x) >> 2) * 65536) + (((int)threadIdx.x) * 4096)) + (k_outer_outer * 2)) + 1025)];
    }
    if (((int)threadIdx.x) < 16) {
      data_shared[((((int)threadIdx.x) * 8) + 4)] = data[(((((((int)blockIdx.x) >> 2) * 65536) + (((int)threadIdx.x) * 4096)) + (k_outer_outer * 2)) + 2048)];
    }
    if (((int)threadIdx.x) < 16) {
      data_shared[((((int)threadIdx.x) * 8) + 5)] = data[(((((((int)blockIdx.x) >> 2) * 65536) + (((int)threadIdx.x) * 4096)) + (k_outer_outer * 2)) + 2049)];
    }
    if (((int)threadIdx.x) < 16) {
      data_shared[((((int)threadIdx.x) * 8) + 6)] = data[(((((((int)blockIdx.x) >> 2) * 65536) + (((int)threadIdx.x) * 4096)) + (k_outer_outer * 2)) + 3072)];
    }
    if (((int)threadIdx.x) < 16) {
      data_shared[((((int)threadIdx.x) * 8) + 7)] = data[(((((((int)blockIdx.x) >> 2) * 65536) + (((int)threadIdx.x) * 4096)) + (k_outer_outer * 2)) + 3073)];
    }
    for (int ax0_ax1_fused_outer_outer = 0; ax0_ax1_fused_outer_outer < 32; ++ax0_ax1_fused_outer_outer) {
      kernel_shared[((ax0_ax1_fused_outer_outer * 64) + ((int)threadIdx.x))] = kernel[(((((k_outer_outer * 8192) + ((ax0_ax1_fused_outer_outer >> 4) * 4096)) + ((((int)blockIdx.x) & 3) * 1024)) + ((ax0_ax1_fused_outer_outer & 15) * 64)) + ((int)threadIdx.x))];
    }
    __syncthreads();
    for (int k_outer_inner = 0; k_outer_inner < 2; ++k_outer_inner) {
      for (int i_c_outer_inner = 0; i_c_outer_inner < 8; ++i_c_outer_inner) {
        for (int i_c_inner = 0; i_c_inner < 2; ++i_c_inner) {
          for (int j_c_inner = 0; j_c_inner < 8; ++j_c_inner) {
            T_matmul_NN_local[(((i_c_outer_inner * 16) + (i_c_inner * 8)) + j_c_inner)] = (T_matmul_NN_local[(((i_c_outer_inner * 16) + (i_c_inner * 8)) + j_c_inner)] + (data_shared[(((i_c_outer_inner * 4) + (i_c_inner * 2)) + k_outer_inner)] * kernel_shared[(((k_outer_inner * 1024) + (((int)threadIdx.x) * 8)) + j_c_inner)]));
            T_matmul_NN_local[((((i_c_outer_inner * 16) + (i_c_inner * 8)) + j_c_inner) + 128)] = (T_matmul_NN_local[((((i_c_outer_inner * 16) + (i_c_inner * 8)) + j_c_inner) + 128)] + (data_shared[(((i_c_outer_inner * 4) + (i_c_inner * 2)) + k_outer_inner)] * kernel_shared[((((k_outer_inner * 1024) + (((int)threadIdx.x) * 8)) + j_c_inner) + 512)]));
            T_matmul_NN_local[((((i_c_outer_inner * 16) + (i_c_inner * 8)) + j_c_inner) + 256)] = (T_matmul_NN_local[((((i_c_outer_inner * 16) + (i_c_inner * 8)) + j_c_inner) + 256)] + (data_shared[((((i_c_outer_inner * 4) + (i_c_inner * 2)) + k_outer_inner) + 32)] * kernel_shared[(((k_outer_inner * 1024) + (((int)threadIdx.x) * 8)) + j_c_inner)]));
            T_matmul_NN_local[((((i_c_outer_inner * 16) + (i_c_inner * 8)) + j_c_inner) + 384)] = (T_matmul_NN_local[((((i_c_outer_inner * 16) + (i_c_inner * 8)) + j_c_inner) + 384)] + (data_shared[((((i_c_outer_inner * 4) + (i_c_inner * 2)) + k_outer_inner) + 32)] * kernel_shared[((((k_outer_inner * 1024) + (((int)threadIdx.x) * 8)) + j_c_inner) + 512)]));
            T_matmul_NN_local[((((i_c_outer_inner * 16) + (i_c_inner * 8)) + j_c_inner) + 512)] = (T_matmul_NN_local[((((i_c_outer_inner * 16) + (i_c_inner * 8)) + j_c_inner) + 512)] + (data_shared[((((i_c_outer_inner * 4) + (i_c_inner * 2)) + k_outer_inner) + 64)] * kernel_shared[(((k_outer_inner * 1024) + (((int)threadIdx.x) * 8)) + j_c_inner)]));
            T_matmul_NN_local[((((i_c_outer_inner * 16) + (i_c_inner * 8)) + j_c_inner) + 640)] = (T_matmul_NN_local[((((i_c_outer_inner * 16) + (i_c_inner * 8)) + j_c_inner) + 640)] + (data_shared[((((i_c_outer_inner * 4) + (i_c_inner * 2)) + k_outer_inner) + 64)] * kernel_shared[((((k_outer_inner * 1024) + (((int)threadIdx.x) * 8)) + j_c_inner) + 512)]));
            T_matmul_NN_local[((((i_c_outer_inner * 16) + (i_c_inner * 8)) + j_c_inner) + 768)] = (T_matmul_NN_local[((((i_c_outer_inner * 16) + (i_c_inner * 8)) + j_c_inner) + 768)] + (data_shared[((((i_c_outer_inner * 4) + (i_c_inner * 2)) + k_outer_inner) + 96)] * kernel_shared[(((k_outer_inner * 1024) + (((int)threadIdx.x) * 8)) + j_c_inner)]));
            T_matmul_NN_local[((((i_c_outer_inner * 16) + (i_c_inner * 8)) + j_c_inner) + 896)] = (T_matmul_NN_local[((((i_c_outer_inner * 16) + (i_c_inner * 8)) + j_c_inner) + 896)] + (data_shared[((((i_c_outer_inner * 4) + (i_c_inner * 2)) + k_outer_inner) + 96)] * kernel_shared[((((k_outer_inner * 1024) + (((int)threadIdx.x) * 8)) + j_c_inner) + 512)]));
          }
        }
      }
    }
  }
  for (int i_inner = 0; i_inner < 16; ++i_inner) {
    for (int j_inner = 0; j_inner < 8; ++j_inner) {
      T_matmul_NN[((((((((int)blockIdx.x) >> 2) * 262144) + (i_inner * 4096)) + ((((int)blockIdx.x) & 3) * 1024)) + (((int)threadIdx.x) * 8)) + j_inner)] = T_matmul_NN_local[((i_inner * 8) + j_inner)];
      T_matmul_NN[(((((((((int)blockIdx.x) >> 2) * 262144) + (i_inner * 4096)) + ((((int)blockIdx.x) & 3) * 1024)) + (((int)threadIdx.x) * 8)) + j_inner) + 512)] = T_matmul_NN_local[(((i_inner * 8) + j_inner) + 128)];
      T_matmul_NN[(((((((((int)blockIdx.x) >> 2) * 262144) + (i_inner * 4096)) + ((((int)blockIdx.x) & 3) * 1024)) + (((int)threadIdx.x) * 8)) + j_inner) + 65536)] = T_matmul_NN_local[(((i_inner * 8) + j_inner) + 256)];
      T_matmul_NN[(((((((((int)blockIdx.x) >> 2) * 262144) + (i_inner * 4096)) + ((((int)blockIdx.x) & 3) * 1024)) + (((int)threadIdx.x) * 8)) + j_inner) + 66048)] = T_matmul_NN_local[(((i_inner * 8) + j_inner) + 384)];
      T_matmul_NN[(((((((((int)blockIdx.x) >> 2) * 262144) + (i_inner * 4096)) + ((((int)blockIdx.x) & 3) * 1024)) + (((int)threadIdx.x) * 8)) + j_inner) + 131072)] = T_matmul_NN_local[(((i_inner * 8) + j_inner) + 512)];
      T_matmul_NN[(((((((((int)blockIdx.x) >> 2) * 262144) + (i_inner * 4096)) + ((((int)blockIdx.x) & 3) * 1024)) + (((int)threadIdx.x) * 8)) + j_inner) + 131584)] = T_matmul_NN_local[(((i_inner * 8) + j_inner) + 640)];
      T_matmul_NN[(((((((((int)blockIdx.x) >> 2) * 262144) + (i_inner * 4096)) + ((((int)blockIdx.x) & 3) * 1024)) + (((int)threadIdx.x) * 8)) + j_inner) + 196608)] = T_matmul_NN_local[(((i_inner * 8) + j_inner) + 768)];
      T_matmul_NN[(((((((((int)blockIdx.x) >> 2) * 262144) + (i_inner * 4096)) + ((((int)blockIdx.x) & 3) * 1024)) + (((int)threadIdx.x) * 8)) + j_inner) + 197120)] = T_matmul_NN_local[(((i_inner * 8) + j_inner) + 896)];
    }
  }
}

