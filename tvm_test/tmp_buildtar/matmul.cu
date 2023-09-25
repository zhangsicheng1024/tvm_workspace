
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
  __shared__ float data_shared[256];
  __shared__ float kernel_shared[1024];
  for (int i_c_outer_inner_init = 0; i_c_outer_inner_init < 16; ++i_c_outer_inner_init) {
    T_matmul_NN_local[(i_c_outer_inner_init * 2)] = 0.000000e+00f;
    T_matmul_NN_local[((i_c_outer_inner_init * 2) + 32)] = 0.000000e+00f;
    T_matmul_NN_local[((i_c_outer_inner_init * 2) + 64)] = 0.000000e+00f;
    T_matmul_NN_local[((i_c_outer_inner_init * 2) + 96)] = 0.000000e+00f;
    T_matmul_NN_local[((i_c_outer_inner_init * 2) + 1)] = 0.000000e+00f;
    T_matmul_NN_local[((i_c_outer_inner_init * 2) + 33)] = 0.000000e+00f;
    T_matmul_NN_local[((i_c_outer_inner_init * 2) + 65)] = 0.000000e+00f;
    T_matmul_NN_local[((i_c_outer_inner_init * 2) + 97)] = 0.000000e+00f;
  }
  for (int k_outer_outer = 0; k_outer_outer < 256; ++k_outer_outer) {
    __syncthreads();
    if (((int)threadIdx.x) < 64) {
      *(float4*)(data_shared + (((int)threadIdx.x) * 4)) = *(float4*)(data + ((((((int)blockIdx.x) >> 4) * 65536) + (((int)threadIdx.x) * 1024)) + (k_outer_outer * 4)));
    }
    *(float4*)(kernel_shared + (((int)threadIdx.x) * 4)) = *(float4*)(kernel + ((((k_outer_outer * 16384) + ((((int)threadIdx.x) >> 6) * 4096)) + ((((int)blockIdx.x) & 15) * 256)) + ((((int)threadIdx.x) & 63) * 4)));
    *(float4*)(kernel_shared + ((((int)threadIdx.x) * 4) + 512)) = *(float4*)(kernel + (((((k_outer_outer * 16384) + ((((int)threadIdx.x) >> 6) * 4096)) + ((((int)blockIdx.x) & 15) * 256)) + ((((int)threadIdx.x) & 63) * 4)) + 8192));
    __syncthreads();
    for (int i_c_outer_inner = 0; i_c_outer_inner < 16; ++i_c_outer_inner) {
      T_matmul_NN_local[(i_c_outer_inner * 2)] = (T_matmul_NN_local[(i_c_outer_inner * 2)] + (data_shared[(((((int)threadIdx.x) >> 6) * 64) + (i_c_outer_inner * 4))] * kernel_shared[((((int)threadIdx.x) & 63) * 2)]));
      T_matmul_NN_local[((i_c_outer_inner * 2) + 32)] = (T_matmul_NN_local[((i_c_outer_inner * 2) + 32)] + (data_shared[(((((int)threadIdx.x) >> 6) * 64) + (i_c_outer_inner * 4))] * kernel_shared[(((((int)threadIdx.x) & 63) * 2) + 128)]));
      T_matmul_NN_local[((i_c_outer_inner * 2) + 64)] = (T_matmul_NN_local[((i_c_outer_inner * 2) + 64)] + (data_shared[((((((int)threadIdx.x) >> 6) * 64) + (i_c_outer_inner * 4)) + 128)] * kernel_shared[((((int)threadIdx.x) & 63) * 2)]));
      T_matmul_NN_local[((i_c_outer_inner * 2) + 96)] = (T_matmul_NN_local[((i_c_outer_inner * 2) + 96)] + (data_shared[((((((int)threadIdx.x) >> 6) * 64) + (i_c_outer_inner * 4)) + 128)] * kernel_shared[(((((int)threadIdx.x) & 63) * 2) + 128)]));
      T_matmul_NN_local[(i_c_outer_inner * 2)] = (T_matmul_NN_local[(i_c_outer_inner * 2)] + (data_shared[((((((int)threadIdx.x) >> 6) * 64) + (i_c_outer_inner * 4)) + 1)] * kernel_shared[(((((int)threadIdx.x) & 63) * 2) + 256)]));
      T_matmul_NN_local[((i_c_outer_inner * 2) + 32)] = (T_matmul_NN_local[((i_c_outer_inner * 2) + 32)] + (data_shared[((((((int)threadIdx.x) >> 6) * 64) + (i_c_outer_inner * 4)) + 1)] * kernel_shared[(((((int)threadIdx.x) & 63) * 2) + 384)]));
      T_matmul_NN_local[((i_c_outer_inner * 2) + 64)] = (T_matmul_NN_local[((i_c_outer_inner * 2) + 64)] + (data_shared[((((((int)threadIdx.x) >> 6) * 64) + (i_c_outer_inner * 4)) + 129)] * kernel_shared[(((((int)threadIdx.x) & 63) * 2) + 256)]));
      T_matmul_NN_local[((i_c_outer_inner * 2) + 96)] = (T_matmul_NN_local[((i_c_outer_inner * 2) + 96)] + (data_shared[((((((int)threadIdx.x) >> 6) * 64) + (i_c_outer_inner * 4)) + 129)] * kernel_shared[(((((int)threadIdx.x) & 63) * 2) + 384)]));
      T_matmul_NN_local[(i_c_outer_inner * 2)] = (T_matmul_NN_local[(i_c_outer_inner * 2)] + (data_shared[((((((int)threadIdx.x) >> 6) * 64) + (i_c_outer_inner * 4)) + 2)] * kernel_shared[(((((int)threadIdx.x) & 63) * 2) + 512)]));
      T_matmul_NN_local[((i_c_outer_inner * 2) + 32)] = (T_matmul_NN_local[((i_c_outer_inner * 2) + 32)] + (data_shared[((((((int)threadIdx.x) >> 6) * 64) + (i_c_outer_inner * 4)) + 2)] * kernel_shared[(((((int)threadIdx.x) & 63) * 2) + 640)]));
      T_matmul_NN_local[((i_c_outer_inner * 2) + 64)] = (T_matmul_NN_local[((i_c_outer_inner * 2) + 64)] + (data_shared[((((((int)threadIdx.x) >> 6) * 64) + (i_c_outer_inner * 4)) + 130)] * kernel_shared[(((((int)threadIdx.x) & 63) * 2) + 512)]));
      T_matmul_NN_local[((i_c_outer_inner * 2) + 96)] = (T_matmul_NN_local[((i_c_outer_inner * 2) + 96)] + (data_shared[((((((int)threadIdx.x) >> 6) * 64) + (i_c_outer_inner * 4)) + 130)] * kernel_shared[(((((int)threadIdx.x) & 63) * 2) + 640)]));
      T_matmul_NN_local[(i_c_outer_inner * 2)] = (T_matmul_NN_local[(i_c_outer_inner * 2)] + (data_shared[((((((int)threadIdx.x) >> 6) * 64) + (i_c_outer_inner * 4)) + 3)] * kernel_shared[(((((int)threadIdx.x) & 63) * 2) + 768)]));
      T_matmul_NN_local[((i_c_outer_inner * 2) + 32)] = (T_matmul_NN_local[((i_c_outer_inner * 2) + 32)] + (data_shared[((((((int)threadIdx.x) >> 6) * 64) + (i_c_outer_inner * 4)) + 3)] * kernel_shared[(((((int)threadIdx.x) & 63) * 2) + 896)]));
      T_matmul_NN_local[((i_c_outer_inner * 2) + 64)] = (T_matmul_NN_local[((i_c_outer_inner * 2) + 64)] + (data_shared[((((((int)threadIdx.x) >> 6) * 64) + (i_c_outer_inner * 4)) + 131)] * kernel_shared[(((((int)threadIdx.x) & 63) * 2) + 768)]));
      T_matmul_NN_local[((i_c_outer_inner * 2) + 96)] = (T_matmul_NN_local[((i_c_outer_inner * 2) + 96)] + (data_shared[((((((int)threadIdx.x) >> 6) * 64) + (i_c_outer_inner * 4)) + 131)] * kernel_shared[(((((int)threadIdx.x) & 63) * 2) + 896)]));
      T_matmul_NN_local[((i_c_outer_inner * 2) + 1)] = (T_matmul_NN_local[((i_c_outer_inner * 2) + 1)] + (data_shared[(((((int)threadIdx.x) >> 6) * 64) + (i_c_outer_inner * 4))] * kernel_shared[(((((int)threadIdx.x) & 63) * 2) + 1)]));
      T_matmul_NN_local[((i_c_outer_inner * 2) + 33)] = (T_matmul_NN_local[((i_c_outer_inner * 2) + 33)] + (data_shared[(((((int)threadIdx.x) >> 6) * 64) + (i_c_outer_inner * 4))] * kernel_shared[(((((int)threadIdx.x) & 63) * 2) + 129)]));
      T_matmul_NN_local[((i_c_outer_inner * 2) + 65)] = (T_matmul_NN_local[((i_c_outer_inner * 2) + 65)] + (data_shared[((((((int)threadIdx.x) >> 6) * 64) + (i_c_outer_inner * 4)) + 128)] * kernel_shared[(((((int)threadIdx.x) & 63) * 2) + 1)]));
      T_matmul_NN_local[((i_c_outer_inner * 2) + 97)] = (T_matmul_NN_local[((i_c_outer_inner * 2) + 97)] + (data_shared[((((((int)threadIdx.x) >> 6) * 64) + (i_c_outer_inner * 4)) + 128)] * kernel_shared[(((((int)threadIdx.x) & 63) * 2) + 129)]));
      T_matmul_NN_local[((i_c_outer_inner * 2) + 1)] = (T_matmul_NN_local[((i_c_outer_inner * 2) + 1)] + (data_shared[((((((int)threadIdx.x) >> 6) * 64) + (i_c_outer_inner * 4)) + 1)] * kernel_shared[(((((int)threadIdx.x) & 63) * 2) + 257)]));
      T_matmul_NN_local[((i_c_outer_inner * 2) + 33)] = (T_matmul_NN_local[((i_c_outer_inner * 2) + 33)] + (data_shared[((((((int)threadIdx.x) >> 6) * 64) + (i_c_outer_inner * 4)) + 1)] * kernel_shared[(((((int)threadIdx.x) & 63) * 2) + 385)]));
      T_matmul_NN_local[((i_c_outer_inner * 2) + 65)] = (T_matmul_NN_local[((i_c_outer_inner * 2) + 65)] + (data_shared[((((((int)threadIdx.x) >> 6) * 64) + (i_c_outer_inner * 4)) + 129)] * kernel_shared[(((((int)threadIdx.x) & 63) * 2) + 257)]));
      T_matmul_NN_local[((i_c_outer_inner * 2) + 97)] = (T_matmul_NN_local[((i_c_outer_inner * 2) + 97)] + (data_shared[((((((int)threadIdx.x) >> 6) * 64) + (i_c_outer_inner * 4)) + 129)] * kernel_shared[(((((int)threadIdx.x) & 63) * 2) + 385)]));
      T_matmul_NN_local[((i_c_outer_inner * 2) + 1)] = (T_matmul_NN_local[((i_c_outer_inner * 2) + 1)] + (data_shared[((((((int)threadIdx.x) >> 6) * 64) + (i_c_outer_inner * 4)) + 2)] * kernel_shared[(((((int)threadIdx.x) & 63) * 2) + 513)]));
      T_matmul_NN_local[((i_c_outer_inner * 2) + 33)] = (T_matmul_NN_local[((i_c_outer_inner * 2) + 33)] + (data_shared[((((((int)threadIdx.x) >> 6) * 64) + (i_c_outer_inner * 4)) + 2)] * kernel_shared[(((((int)threadIdx.x) & 63) * 2) + 641)]));
      T_matmul_NN_local[((i_c_outer_inner * 2) + 65)] = (T_matmul_NN_local[((i_c_outer_inner * 2) + 65)] + (data_shared[((((((int)threadIdx.x) >> 6) * 64) + (i_c_outer_inner * 4)) + 130)] * kernel_shared[(((((int)threadIdx.x) & 63) * 2) + 513)]));
      T_matmul_NN_local[((i_c_outer_inner * 2) + 97)] = (T_matmul_NN_local[((i_c_outer_inner * 2) + 97)] + (data_shared[((((((int)threadIdx.x) >> 6) * 64) + (i_c_outer_inner * 4)) + 130)] * kernel_shared[(((((int)threadIdx.x) & 63) * 2) + 641)]));
      T_matmul_NN_local[((i_c_outer_inner * 2) + 1)] = (T_matmul_NN_local[((i_c_outer_inner * 2) + 1)] + (data_shared[((((((int)threadIdx.x) >> 6) * 64) + (i_c_outer_inner * 4)) + 3)] * kernel_shared[(((((int)threadIdx.x) & 63) * 2) + 769)]));
      T_matmul_NN_local[((i_c_outer_inner * 2) + 33)] = (T_matmul_NN_local[((i_c_outer_inner * 2) + 33)] + (data_shared[((((((int)threadIdx.x) >> 6) * 64) + (i_c_outer_inner * 4)) + 3)] * kernel_shared[(((((int)threadIdx.x) & 63) * 2) + 897)]));
      T_matmul_NN_local[((i_c_outer_inner * 2) + 65)] = (T_matmul_NN_local[((i_c_outer_inner * 2) + 65)] + (data_shared[((((((int)threadIdx.x) >> 6) * 64) + (i_c_outer_inner * 4)) + 131)] * kernel_shared[(((((int)threadIdx.x) & 63) * 2) + 769)]));
      T_matmul_NN_local[((i_c_outer_inner * 2) + 97)] = (T_matmul_NN_local[((i_c_outer_inner * 2) + 97)] + (data_shared[((((((int)threadIdx.x) >> 6) * 64) + (i_c_outer_inner * 4)) + 131)] * kernel_shared[(((((int)threadIdx.x) & 63) * 2) + 897)]));
    }
  }
  for (int i_inner = 0; i_inner < 16; ++i_inner) {
    for (int j_inner = 0; j_inner < 2; ++j_inner) {
      T_matmul_NN[(((((((((int)blockIdx.x) >> 4) * 262144) + ((((int)threadIdx.x) >> 6) * 65536)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 15) * 256)) + ((((int)threadIdx.x) & 63) * 2)) + j_inner)] = T_matmul_NN_local[((i_inner * 2) + j_inner)];
      T_matmul_NN[((((((((((int)blockIdx.x) >> 4) * 262144) + ((((int)threadIdx.x) >> 6) * 65536)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 15) * 256)) + ((((int)threadIdx.x) & 63) * 2)) + j_inner) + 128)] = T_matmul_NN_local[(((i_inner * 2) + j_inner) + 32)];
      T_matmul_NN[((((((((((int)blockIdx.x) >> 4) * 262144) + ((((int)threadIdx.x) >> 6) * 65536)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 15) * 256)) + ((((int)threadIdx.x) & 63) * 2)) + j_inner) + 131072)] = T_matmul_NN_local[(((i_inner * 2) + j_inner) + 64)];
      T_matmul_NN[((((((((((int)blockIdx.x) >> 4) * 262144) + ((((int)threadIdx.x) >> 6) * 65536)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 15) * 256)) + ((((int)threadIdx.x) & 63) * 2)) + j_inner) + 131200)] = T_matmul_NN_local[(((i_inner * 2) + j_inner) + 96)];
    }
  }
}

