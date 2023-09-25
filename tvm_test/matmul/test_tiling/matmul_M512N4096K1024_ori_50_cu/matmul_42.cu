
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
  float T_matmul_NN_local[128];
  __shared__ float data_shared[8192];
  __shared__ float kernel_shared[2048];
  for (int i_c_outer_inner_init = 0; i_c_outer_inner_init < 4; ++i_c_outer_inner_init) {
    for (int i_c_inner_init = 0; i_c_inner_init < 4; ++i_c_inner_init) {
      for (int j_c_inner_init = 0; j_c_inner_init < 2; ++j_c_inner_init) {
        T_matmul_NN_local[(((i_c_outer_inner_init * 8) + (i_c_inner_init * 2)) + j_c_inner_init)] = 0.000000e+00f;
        T_matmul_NN_local[((((i_c_outer_inner_init * 8) + (i_c_inner_init * 2)) + j_c_inner_init) + 32)] = 0.000000e+00f;
        T_matmul_NN_local[((((i_c_outer_inner_init * 8) + (i_c_inner_init * 2)) + j_c_inner_init) + 64)] = 0.000000e+00f;
        T_matmul_NN_local[((((i_c_outer_inner_init * 8) + (i_c_inner_init * 2)) + j_c_inner_init) + 96)] = 0.000000e+00f;
      }
    }
  }
  for (int k_outer_outer = 0; k_outer_outer < 32; ++k_outer_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_outer_outer = 0; ax0_ax1_fused_outer_outer < 64; ++ax0_ax1_fused_outer_outer) {
      data_shared[((ax0_ax1_fused_outer_outer * 128) + ((int)threadIdx.x))] = data[((((((((int)blockIdx.x) >> 6) * 262144) + (ax0_ax1_fused_outer_outer * 4096)) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx.x) & 31))];
    }
    for (int ax0_ax1_fused_outer_outer_1 = 0; ax0_ax1_fused_outer_outer_1 < 16; ++ax0_ax1_fused_outer_outer_1) {
      kernel_shared[((ax0_ax1_fused_outer_outer_1 * 128) + ((int)threadIdx.x))] = kernel[(((((k_outer_outer * 131072) + (ax0_ax1_fused_outer_outer_1 * 8192)) + ((((int)threadIdx.x) >> 6) * 4096)) + ((((int)blockIdx.x) & 63) * 64)) + (((int)threadIdx.x) & 63))];
    }
    __syncthreads();
    for (int i_c_outer_inner = 0; i_c_outer_inner < 4; ++i_c_outer_inner) {
      for (int k_inner = 0; k_inner < 32; ++k_inner) {
        for (int i_c_inner = 0; i_c_inner < 4; ++i_c_inner) {
          for (int j_c_inner = 0; j_c_inner < 2; ++j_c_inner) {
            T_matmul_NN_local[(((i_c_outer_inner * 8) + (i_c_inner * 2)) + j_c_inner)] = (T_matmul_NN_local[(((i_c_outer_inner * 8) + (i_c_inner * 2)) + j_c_inner)] + (data_shared[(((((((int)threadIdx.x) >> 3) * 512) + (i_c_outer_inner * 128)) + (i_c_inner * 32)) + k_inner)] * kernel_shared[(((k_inner * 64) + ((((int)threadIdx.x) & 7) * 2)) + j_c_inner)]));
            T_matmul_NN_local[((((i_c_outer_inner * 8) + (i_c_inner * 2)) + j_c_inner) + 32)] = (T_matmul_NN_local[((((i_c_outer_inner * 8) + (i_c_inner * 2)) + j_c_inner) + 32)] + (data_shared[(((((((int)threadIdx.x) >> 3) * 512) + (i_c_outer_inner * 128)) + (i_c_inner * 32)) + k_inner)] * kernel_shared[((((k_inner * 64) + ((((int)threadIdx.x) & 7) * 2)) + j_c_inner) + 16)]));
            T_matmul_NN_local[((((i_c_outer_inner * 8) + (i_c_inner * 2)) + j_c_inner) + 64)] = (T_matmul_NN_local[((((i_c_outer_inner * 8) + (i_c_inner * 2)) + j_c_inner) + 64)] + (data_shared[(((((((int)threadIdx.x) >> 3) * 512) + (i_c_outer_inner * 128)) + (i_c_inner * 32)) + k_inner)] * kernel_shared[((((k_inner * 64) + ((((int)threadIdx.x) & 7) * 2)) + j_c_inner) + 32)]));
            T_matmul_NN_local[((((i_c_outer_inner * 8) + (i_c_inner * 2)) + j_c_inner) + 96)] = (T_matmul_NN_local[((((i_c_outer_inner * 8) + (i_c_inner * 2)) + j_c_inner) + 96)] + (data_shared[(((((((int)threadIdx.x) >> 3) * 512) + (i_c_outer_inner * 128)) + (i_c_inner * 32)) + k_inner)] * kernel_shared[((((k_inner * 64) + ((((int)threadIdx.x) & 7) * 2)) + j_c_inner) + 48)]));
          }
        }
      }
    }
  }
  for (int i_inner = 0; i_inner < 16; ++i_inner) {
    for (int j_inner = 0; j_inner < 2; ++j_inner) {
      T_matmul_NN[(((((((((int)blockIdx.x) >> 6) * 1048576) + ((((int)threadIdx.x) >> 3) * 65536)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 63) * 64)) + ((((int)threadIdx.x) & 7) * 2)) + j_inner)] = T_matmul_NN_local[((i_inner * 2) + j_inner)];
      T_matmul_NN[((((((((((int)blockIdx.x) >> 6) * 1048576) + ((((int)threadIdx.x) >> 3) * 65536)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 63) * 64)) + ((((int)threadIdx.x) & 7) * 2)) + j_inner) + 16)] = T_matmul_NN_local[(((i_inner * 2) + j_inner) + 32)];
      T_matmul_NN[((((((((((int)blockIdx.x) >> 6) * 1048576) + ((((int)threadIdx.x) >> 3) * 65536)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 63) * 64)) + ((((int)threadIdx.x) & 7) * 2)) + j_inner) + 32)] = T_matmul_NN_local[(((i_inner * 2) + j_inner) + 64)];
      T_matmul_NN[((((((((((int)blockIdx.x) >> 6) * 1048576) + ((((int)threadIdx.x) >> 3) * 65536)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 63) * 64)) + ((((int)threadIdx.x) & 7) * 2)) + j_inner) + 48)] = T_matmul_NN_local[(((i_inner * 2) + j_inner) + 96)];
    }
  }
}

