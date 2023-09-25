
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
extern "C" __global__ void __launch_bounds__(32) mymatmul_kernel0(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ T_matmul_NN) {
  float T_matmul_NN_local[64];
  __shared__ float data_shared[8192];
  __shared__ float kernel_shared[4096];
  for (int i_c_inner_init = 0; i_c_inner_init < 4; ++i_c_inner_init) {
    for (int j_c_inner_init = 0; j_c_inner_init < 2; ++j_c_inner_init) {
      T_matmul_NN_local[((i_c_inner_init * 2) + j_c_inner_init)] = 0.000000e+00f;
      T_matmul_NN_local[(((i_c_inner_init * 2) + j_c_inner_init) + 8)] = 0.000000e+00f;
      T_matmul_NN_local[(((i_c_inner_init * 2) + j_c_inner_init) + 16)] = 0.000000e+00f;
      T_matmul_NN_local[(((i_c_inner_init * 2) + j_c_inner_init) + 24)] = 0.000000e+00f;
      T_matmul_NN_local[(((i_c_inner_init * 2) + j_c_inner_init) + 32)] = 0.000000e+00f;
      T_matmul_NN_local[(((i_c_inner_init * 2) + j_c_inner_init) + 40)] = 0.000000e+00f;
      T_matmul_NN_local[(((i_c_inner_init * 2) + j_c_inner_init) + 48)] = 0.000000e+00f;
      T_matmul_NN_local[(((i_c_inner_init * 2) + j_c_inner_init) + 56)] = 0.000000e+00f;
    }
  }
  for (int k_outer_outer = 0; k_outer_outer < 8; ++k_outer_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_outer_outer = 0; ax0_ax1_fused_outer_outer < 64; ++ax0_ax1_fused_outer_outer) {
      *(float4*)(data_shared + ((ax0_ax1_fused_outer_outer * 128) + (((int)threadIdx.x) * 4))) = *(float4*)(data + (((((((int)blockIdx.x) >> 7) * 65536) + (ax0_ax1_fused_outer_outer * 1024)) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)));
    }
    for (int ax0_ax1_fused_outer_outer_1 = 0; ax0_ax1_fused_outer_outer_1 < 64; ++ax0_ax1_fused_outer_outer_1) {
      *(float2*)(kernel_shared + ((ax0_ax1_fused_outer_outer_1 * 64) + (((int)threadIdx.x) * 2))) = *(float2*)(kernel + (((((k_outer_outer * 524288) + (ax0_ax1_fused_outer_outer_1 * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.x) & 127) * 32)) + ((((int)threadIdx.x) & 15) * 2)));
    }
    __syncthreads();
    for (int k_outer_inner = 0; k_outer_inner < 16; ++k_outer_inner) {
      for (int k_inner = 0; k_inner < 8; ++k_inner) {
        for (int i_c_inner = 0; i_c_inner < 4; ++i_c_inner) {
          for (int j_c_inner = 0; j_c_inner < 2; ++j_c_inner) {
            T_matmul_NN_local[((i_c_inner * 2) + j_c_inner)] = (T_matmul_NN_local[((i_c_inner * 2) + j_c_inner)] + (data_shared[(((((((int)threadIdx.x) >> 3) * 512) + (i_c_inner * 128)) + (k_outer_inner * 8)) + k_inner)] * kernel_shared[((((k_outer_inner * 256) + (k_inner * 32)) + ((((int)threadIdx.x) & 7) * 2)) + j_c_inner)]));
            T_matmul_NN_local[(((i_c_inner * 2) + j_c_inner) + 8)] = (T_matmul_NN_local[(((i_c_inner * 2) + j_c_inner) + 8)] + (data_shared[(((((((int)threadIdx.x) >> 3) * 512) + (i_c_inner * 128)) + (k_outer_inner * 8)) + k_inner)] * kernel_shared[(((((k_outer_inner * 256) + (k_inner * 32)) + ((((int)threadIdx.x) & 7) * 2)) + j_c_inner) + 16)]));
            T_matmul_NN_local[(((i_c_inner * 2) + j_c_inner) + 16)] = (T_matmul_NN_local[(((i_c_inner * 2) + j_c_inner) + 16)] + (data_shared[((((((((int)threadIdx.x) >> 3) * 512) + (i_c_inner * 128)) + (k_outer_inner * 8)) + k_inner) + 2048)] * kernel_shared[((((k_outer_inner * 256) + (k_inner * 32)) + ((((int)threadIdx.x) & 7) * 2)) + j_c_inner)]));
            T_matmul_NN_local[(((i_c_inner * 2) + j_c_inner) + 24)] = (T_matmul_NN_local[(((i_c_inner * 2) + j_c_inner) + 24)] + (data_shared[((((((((int)threadIdx.x) >> 3) * 512) + (i_c_inner * 128)) + (k_outer_inner * 8)) + k_inner) + 2048)] * kernel_shared[(((((k_outer_inner * 256) + (k_inner * 32)) + ((((int)threadIdx.x) & 7) * 2)) + j_c_inner) + 16)]));
            T_matmul_NN_local[(((i_c_inner * 2) + j_c_inner) + 32)] = (T_matmul_NN_local[(((i_c_inner * 2) + j_c_inner) + 32)] + (data_shared[((((((((int)threadIdx.x) >> 3) * 512) + (i_c_inner * 128)) + (k_outer_inner * 8)) + k_inner) + 4096)] * kernel_shared[((((k_outer_inner * 256) + (k_inner * 32)) + ((((int)threadIdx.x) & 7) * 2)) + j_c_inner)]));
            T_matmul_NN_local[(((i_c_inner * 2) + j_c_inner) + 40)] = (T_matmul_NN_local[(((i_c_inner * 2) + j_c_inner) + 40)] + (data_shared[((((((((int)threadIdx.x) >> 3) * 512) + (i_c_inner * 128)) + (k_outer_inner * 8)) + k_inner) + 4096)] * kernel_shared[(((((k_outer_inner * 256) + (k_inner * 32)) + ((((int)threadIdx.x) & 7) * 2)) + j_c_inner) + 16)]));
            T_matmul_NN_local[(((i_c_inner * 2) + j_c_inner) + 48)] = (T_matmul_NN_local[(((i_c_inner * 2) + j_c_inner) + 48)] + (data_shared[((((((((int)threadIdx.x) >> 3) * 512) + (i_c_inner * 128)) + (k_outer_inner * 8)) + k_inner) + 6144)] * kernel_shared[((((k_outer_inner * 256) + (k_inner * 32)) + ((((int)threadIdx.x) & 7) * 2)) + j_c_inner)]));
            T_matmul_NN_local[(((i_c_inner * 2) + j_c_inner) + 56)] = (T_matmul_NN_local[(((i_c_inner * 2) + j_c_inner) + 56)] + (data_shared[((((((((int)threadIdx.x) >> 3) * 512) + (i_c_inner * 128)) + (k_outer_inner * 8)) + k_inner) + 6144)] * kernel_shared[(((((k_outer_inner * 256) + (k_inner * 32)) + ((((int)threadIdx.x) & 7) * 2)) + j_c_inner) + 16)]));
          }
        }
      }
    }
  }
  for (int i_inner = 0; i_inner < 4; ++i_inner) {
    for (int j_inner = 0; j_inner < 2; ++j_inner) {
      T_matmul_NN[(((((((((int)blockIdx.x) >> 7) * 262144) + ((((int)threadIdx.x) >> 3) * 16384)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 127) * 32)) + ((((int)threadIdx.x) & 7) * 2)) + j_inner)] = T_matmul_NN_local[((i_inner * 2) + j_inner)];
      T_matmul_NN[((((((((((int)blockIdx.x) >> 7) * 262144) + ((((int)threadIdx.x) >> 3) * 16384)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 127) * 32)) + ((((int)threadIdx.x) & 7) * 2)) + j_inner) + 16)] = T_matmul_NN_local[(((i_inner * 2) + j_inner) + 8)];
      T_matmul_NN[((((((((((int)blockIdx.x) >> 7) * 262144) + ((((int)threadIdx.x) >> 3) * 16384)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 127) * 32)) + ((((int)threadIdx.x) & 7) * 2)) + j_inner) + 65536)] = T_matmul_NN_local[(((i_inner * 2) + j_inner) + 16)];
      T_matmul_NN[((((((((((int)blockIdx.x) >> 7) * 262144) + ((((int)threadIdx.x) >> 3) * 16384)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 127) * 32)) + ((((int)threadIdx.x) & 7) * 2)) + j_inner) + 65552)] = T_matmul_NN_local[(((i_inner * 2) + j_inner) + 24)];
      T_matmul_NN[((((((((((int)blockIdx.x) >> 7) * 262144) + ((((int)threadIdx.x) >> 3) * 16384)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 127) * 32)) + ((((int)threadIdx.x) & 7) * 2)) + j_inner) + 131072)] = T_matmul_NN_local[(((i_inner * 2) + j_inner) + 32)];
      T_matmul_NN[((((((((((int)blockIdx.x) >> 7) * 262144) + ((((int)threadIdx.x) >> 3) * 16384)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 127) * 32)) + ((((int)threadIdx.x) & 7) * 2)) + j_inner) + 131088)] = T_matmul_NN_local[(((i_inner * 2) + j_inner) + 40)];
      T_matmul_NN[((((((((((int)blockIdx.x) >> 7) * 262144) + ((((int)threadIdx.x) >> 3) * 16384)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 127) * 32)) + ((((int)threadIdx.x) & 7) * 2)) + j_inner) + 196608)] = T_matmul_NN_local[(((i_inner * 2) + j_inner) + 48)];
      T_matmul_NN[((((((((((int)blockIdx.x) >> 7) * 262144) + ((((int)threadIdx.x) >> 3) * 16384)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 127) * 32)) + ((((int)threadIdx.x) & 7) * 2)) + j_inner) + 196624)] = T_matmul_NN_local[(((i_inner * 2) + j_inner) + 56)];
    }
  }
}

