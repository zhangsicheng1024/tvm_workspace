
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
  float T_matmul_NN_local[16];
  __shared__ float data_shared[16];
  __shared__ float kernel_shared[512];
  for (int j_c_outer_inner_init = 0; j_c_outer_inner_init < 2; ++j_c_outer_inner_init) {
    T_matmul_NN_local[j_c_outer_inner_init] = 0.000000e+00f;
    T_matmul_NN_local[(j_c_outer_inner_init + 2)] = 0.000000e+00f;
    T_matmul_NN_local[(j_c_outer_inner_init + 4)] = 0.000000e+00f;
    T_matmul_NN_local[(j_c_outer_inner_init + 6)] = 0.000000e+00f;
    T_matmul_NN_local[(j_c_outer_inner_init + 8)] = 0.000000e+00f;
    T_matmul_NN_local[(j_c_outer_inner_init + 10)] = 0.000000e+00f;
    T_matmul_NN_local[(j_c_outer_inner_init + 12)] = 0.000000e+00f;
    T_matmul_NN_local[(j_c_outer_inner_init + 14)] = 0.000000e+00f;
  }
  for (int k_outer_outer = 0; k_outer_outer < 512; ++k_outer_outer) {
    __syncthreads();
    if (((int)threadIdx.x) < 8) {
      *(float2*)(data_shared + (((int)threadIdx.x) * 2)) = *(float2*)(data + ((((((int)blockIdx.x) >> 4) * 8192) + (((int)threadIdx.x) * 1024)) + (k_outer_outer * 2)));
    }
    for (int ax0_ax1_fused_outer_outer = 0; ax0_ax1_fused_outer_outer < 4; ++ax0_ax1_fused_outer_outer) {
      kernel_shared[((ax0_ax1_fused_outer_outer * 128) + ((int)threadIdx.x))] = kernel[(((((k_outer_outer * 8192) + ((ax0_ax1_fused_outer_outer >> 1) * 4096)) + ((((int)blockIdx.x) & 15) * 256)) + ((ax0_ax1_fused_outer_outer & 1) * 128)) + ((int)threadIdx.x))];
    }
    __syncthreads();
    for (int k_outer_inner = 0; k_outer_inner < 2; ++k_outer_inner) {
      for (int j_c_outer_inner = 0; j_c_outer_inner < 2; ++j_c_outer_inner) {
        T_matmul_NN_local[j_c_outer_inner] = (T_matmul_NN_local[j_c_outer_inner] + (data_shared[k_outer_inner] * kernel_shared[(((k_outer_inner * 256) + (((int)threadIdx.x) * 2)) + j_c_outer_inner)]));
        T_matmul_NN_local[(j_c_outer_inner + 2)] = (T_matmul_NN_local[(j_c_outer_inner + 2)] + (data_shared[(k_outer_inner + 2)] * kernel_shared[(((k_outer_inner * 256) + (((int)threadIdx.x) * 2)) + j_c_outer_inner)]));
        T_matmul_NN_local[(j_c_outer_inner + 4)] = (T_matmul_NN_local[(j_c_outer_inner + 4)] + (data_shared[(k_outer_inner + 4)] * kernel_shared[(((k_outer_inner * 256) + (((int)threadIdx.x) * 2)) + j_c_outer_inner)]));
        T_matmul_NN_local[(j_c_outer_inner + 6)] = (T_matmul_NN_local[(j_c_outer_inner + 6)] + (data_shared[(k_outer_inner + 6)] * kernel_shared[(((k_outer_inner * 256) + (((int)threadIdx.x) * 2)) + j_c_outer_inner)]));
        T_matmul_NN_local[(j_c_outer_inner + 8)] = (T_matmul_NN_local[(j_c_outer_inner + 8)] + (data_shared[(k_outer_inner + 8)] * kernel_shared[(((k_outer_inner * 256) + (((int)threadIdx.x) * 2)) + j_c_outer_inner)]));
        T_matmul_NN_local[(j_c_outer_inner + 10)] = (T_matmul_NN_local[(j_c_outer_inner + 10)] + (data_shared[(k_outer_inner + 10)] * kernel_shared[(((k_outer_inner * 256) + (((int)threadIdx.x) * 2)) + j_c_outer_inner)]));
        T_matmul_NN_local[(j_c_outer_inner + 12)] = (T_matmul_NN_local[(j_c_outer_inner + 12)] + (data_shared[(k_outer_inner + 12)] * kernel_shared[(((k_outer_inner * 256) + (((int)threadIdx.x) * 2)) + j_c_outer_inner)]));
        T_matmul_NN_local[(j_c_outer_inner + 14)] = (T_matmul_NN_local[(j_c_outer_inner + 14)] + (data_shared[(k_outer_inner + 14)] * kernel_shared[(((k_outer_inner * 256) + (((int)threadIdx.x) * 2)) + j_c_outer_inner)]));
      }
    }
  }
  for (int j_inner = 0; j_inner < 2; ++j_inner) {
    T_matmul_NN[(((((((int)blockIdx.x) >> 4) * 32768) + ((((int)blockIdx.x) & 15) * 256)) + (((int)threadIdx.x) * 2)) + j_inner)] = T_matmul_NN_local[j_inner];
    T_matmul_NN[((((((((int)blockIdx.x) >> 4) * 32768) + ((((int)blockIdx.x) & 15) * 256)) + (((int)threadIdx.x) * 2)) + j_inner) + 4096)] = T_matmul_NN_local[(j_inner + 2)];
    T_matmul_NN[((((((((int)blockIdx.x) >> 4) * 32768) + ((((int)blockIdx.x) & 15) * 256)) + (((int)threadIdx.x) * 2)) + j_inner) + 8192)] = T_matmul_NN_local[(j_inner + 4)];
    T_matmul_NN[((((((((int)blockIdx.x) >> 4) * 32768) + ((((int)blockIdx.x) & 15) * 256)) + (((int)threadIdx.x) * 2)) + j_inner) + 12288)] = T_matmul_NN_local[(j_inner + 6)];
    T_matmul_NN[((((((((int)blockIdx.x) >> 4) * 32768) + ((((int)blockIdx.x) & 15) * 256)) + (((int)threadIdx.x) * 2)) + j_inner) + 16384)] = T_matmul_NN_local[(j_inner + 8)];
    T_matmul_NN[((((((((int)blockIdx.x) >> 4) * 32768) + ((((int)blockIdx.x) & 15) * 256)) + (((int)threadIdx.x) * 2)) + j_inner) + 20480)] = T_matmul_NN_local[(j_inner + 10)];
    T_matmul_NN[((((((((int)blockIdx.x) >> 4) * 32768) + ((((int)blockIdx.x) & 15) * 256)) + (((int)threadIdx.x) * 2)) + j_inner) + 24576)] = T_matmul_NN_local[(j_inner + 12)];
    T_matmul_NN[((((((((int)blockIdx.x) >> 4) * 32768) + ((((int)blockIdx.x) & 15) * 256)) + (((int)threadIdx.x) * 2)) + j_inner) + 28672)] = T_matmul_NN_local[(j_inner + 14)];
  }
}

