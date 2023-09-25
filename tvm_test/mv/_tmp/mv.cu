
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
extern "C" __global__ void __launch_bounds__(64) mymv_kernel0(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ T_matmul_NN) {
  float T_matmul_NN_local[2];
  __shared__ float data_shared[8];
  __shared__ float kernel_shared[1024];
  T_matmul_NN_local[0] = 0.000000e+00f;
  T_matmul_NN_local[1] = 0.000000e+00f;
  for (int k_outer_outer = 0; k_outer_outer < 1024; ++k_outer_outer) {
    __syncthreads();
    if (((int)threadIdx.x) < 8) {
      data_shared[((int)threadIdx.x)] = data[((k_outer_outer * 8) + ((int)threadIdx.x))];
    }
    *(float2*)(kernel_shared + (((int)threadIdx.x) * 2)) = *(float2*)(kernel + (((k_outer_outer * 262144) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) * 2)));
    *(float2*)(kernel_shared + ((((int)threadIdx.x) * 2) + 128)) = *(float2*)(kernel + ((((k_outer_outer * 262144) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) * 2)) + 32768));
    *(float2*)(kernel_shared + ((((int)threadIdx.x) * 2) + 256)) = *(float2*)(kernel + ((((k_outer_outer * 262144) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) * 2)) + 65536));
    *(float2*)(kernel_shared + ((((int)threadIdx.x) * 2) + 384)) = *(float2*)(kernel + ((((k_outer_outer * 262144) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) * 2)) + 98304));
    *(float2*)(kernel_shared + ((((int)threadIdx.x) * 2) + 512)) = *(float2*)(kernel + ((((k_outer_outer * 262144) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) * 2)) + 131072));
    *(float2*)(kernel_shared + ((((int)threadIdx.x) * 2) + 640)) = *(float2*)(kernel + ((((k_outer_outer * 262144) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) * 2)) + 163840));
    *(float2*)(kernel_shared + ((((int)threadIdx.x) * 2) + 768)) = *(float2*)(kernel + ((((k_outer_outer * 262144) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) * 2)) + 196608));
    *(float2*)(kernel_shared + ((((int)threadIdx.x) * 2) + 896)) = *(float2*)(kernel + ((((k_outer_outer * 262144) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) * 2)) + 229376));
    __syncthreads();
    T_matmul_NN_local[0] = (T_matmul_NN_local[0] + (data_shared[0] * kernel_shared[((int)threadIdx.x)]));
    T_matmul_NN_local[1] = (T_matmul_NN_local[1] + (data_shared[0] * kernel_shared[(((int)threadIdx.x) + 64)]));
    T_matmul_NN_local[0] = (T_matmul_NN_local[0] + (data_shared[1] * kernel_shared[(((int)threadIdx.x) + 128)]));
    T_matmul_NN_local[1] = (T_matmul_NN_local[1] + (data_shared[1] * kernel_shared[(((int)threadIdx.x) + 192)]));
    T_matmul_NN_local[0] = (T_matmul_NN_local[0] + (data_shared[2] * kernel_shared[(((int)threadIdx.x) + 256)]));
    T_matmul_NN_local[1] = (T_matmul_NN_local[1] + (data_shared[2] * kernel_shared[(((int)threadIdx.x) + 320)]));
    T_matmul_NN_local[0] = (T_matmul_NN_local[0] + (data_shared[3] * kernel_shared[(((int)threadIdx.x) + 384)]));
    T_matmul_NN_local[1] = (T_matmul_NN_local[1] + (data_shared[3] * kernel_shared[(((int)threadIdx.x) + 448)]));
    T_matmul_NN_local[0] = (T_matmul_NN_local[0] + (data_shared[4] * kernel_shared[(((int)threadIdx.x) + 512)]));
    T_matmul_NN_local[1] = (T_matmul_NN_local[1] + (data_shared[4] * kernel_shared[(((int)threadIdx.x) + 576)]));
    T_matmul_NN_local[0] = (T_matmul_NN_local[0] + (data_shared[5] * kernel_shared[(((int)threadIdx.x) + 640)]));
    T_matmul_NN_local[1] = (T_matmul_NN_local[1] + (data_shared[5] * kernel_shared[(((int)threadIdx.x) + 704)]));
    T_matmul_NN_local[0] = (T_matmul_NN_local[0] + (data_shared[6] * kernel_shared[(((int)threadIdx.x) + 768)]));
    T_matmul_NN_local[1] = (T_matmul_NN_local[1] + (data_shared[6] * kernel_shared[(((int)threadIdx.x) + 832)]));
    T_matmul_NN_local[0] = (T_matmul_NN_local[0] + (data_shared[7] * kernel_shared[(((int)threadIdx.x) + 896)]));
    T_matmul_NN_local[1] = (T_matmul_NN_local[1] + (data_shared[7] * kernel_shared[(((int)threadIdx.x) + 960)]));
  }
  T_matmul_NN[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))] = T_matmul_NN_local[0];
  T_matmul_NN[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 64)] = T_matmul_NN_local[1];
}

