
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
  float T_matmul_NN_local[1024];
  __shared__ float data_shared[1024];
  __shared__ float kernel_shared[2048];
  for (int i_c_outer_inner_init = 0; i_c_outer_inner_init < 64; ++i_c_outer_inner_init) {
    T_matmul_NN_local[(i_c_outer_inner_init * 16)] = 0.000000e+00f;
    T_matmul_NN_local[((i_c_outer_inner_init * 16) + 8)] = 0.000000e+00f;
    T_matmul_NN_local[((i_c_outer_inner_init * 16) + 1)] = 0.000000e+00f;
    T_matmul_NN_local[((i_c_outer_inner_init * 16) + 9)] = 0.000000e+00f;
    T_matmul_NN_local[((i_c_outer_inner_init * 16) + 2)] = 0.000000e+00f;
    T_matmul_NN_local[((i_c_outer_inner_init * 16) + 10)] = 0.000000e+00f;
    T_matmul_NN_local[((i_c_outer_inner_init * 16) + 3)] = 0.000000e+00f;
    T_matmul_NN_local[((i_c_outer_inner_init * 16) + 11)] = 0.000000e+00f;
    T_matmul_NN_local[((i_c_outer_inner_init * 16) + 4)] = 0.000000e+00f;
    T_matmul_NN_local[((i_c_outer_inner_init * 16) + 12)] = 0.000000e+00f;
    T_matmul_NN_local[((i_c_outer_inner_init * 16) + 5)] = 0.000000e+00f;
    T_matmul_NN_local[((i_c_outer_inner_init * 16) + 13)] = 0.000000e+00f;
    T_matmul_NN_local[((i_c_outer_inner_init * 16) + 6)] = 0.000000e+00f;
    T_matmul_NN_local[((i_c_outer_inner_init * 16) + 14)] = 0.000000e+00f;
    T_matmul_NN_local[((i_c_outer_inner_init * 16) + 7)] = 0.000000e+00f;
    T_matmul_NN_local[((i_c_outer_inner_init * 16) + 15)] = 0.000000e+00f;
  }
  for (int k_outer_outer = 0; k_outer_outer < 256; ++k_outer_outer) {
    __syncthreads();
    *(float2*)(data_shared + (((int)threadIdx.x) * 2)) = *(float2*)(data + (((((((int)blockIdx.x) >> 3) * 262144) + ((((int)threadIdx.x) >> 1) * 1024)) + (k_outer_outer * 4)) + ((((int)threadIdx.x) & 1) * 2)));
    *(float2*)(data_shared + ((((int)threadIdx.x) * 2) + 256)) = *(float2*)(data + ((((((((int)blockIdx.x) >> 3) * 262144) + ((((int)threadIdx.x) >> 1) * 1024)) + (k_outer_outer * 4)) + ((((int)threadIdx.x) & 1) * 2)) + 65536));
    *(float2*)(data_shared + ((((int)threadIdx.x) * 2) + 512)) = *(float2*)(data + ((((((((int)blockIdx.x) >> 3) * 262144) + ((((int)threadIdx.x) >> 1) * 1024)) + (k_outer_outer * 4)) + ((((int)threadIdx.x) & 1) * 2)) + 131072));
    *(float2*)(data_shared + ((((int)threadIdx.x) * 2) + 768)) = *(float2*)(data + ((((((((int)blockIdx.x) >> 3) * 262144) + ((((int)threadIdx.x) >> 1) * 1024)) + (k_outer_outer * 4)) + ((((int)threadIdx.x) & 1) * 2)) + 196608));
    kernel_shared[((int)threadIdx.x)] = kernel[(((k_outer_outer * 16384) + ((((int)blockIdx.x) & 7) * 512)) + ((int)threadIdx.x))];
    kernel_shared[(((int)threadIdx.x) + 128)] = kernel[((((k_outer_outer * 16384) + ((((int)blockIdx.x) & 7) * 512)) + ((int)threadIdx.x)) + 128)];
    kernel_shared[(((int)threadIdx.x) + 256)] = kernel[((((k_outer_outer * 16384) + ((((int)blockIdx.x) & 7) * 512)) + ((int)threadIdx.x)) + 256)];
    kernel_shared[(((int)threadIdx.x) + 384)] = kernel[((((k_outer_outer * 16384) + ((((int)blockIdx.x) & 7) * 512)) + ((int)threadIdx.x)) + 384)];
    kernel_shared[(((int)threadIdx.x) + 512)] = kernel[((((k_outer_outer * 16384) + ((((int)blockIdx.x) & 7) * 512)) + ((int)threadIdx.x)) + 4096)];
    kernel_shared[(((int)threadIdx.x) + 640)] = kernel[((((k_outer_outer * 16384) + ((((int)blockIdx.x) & 7) * 512)) + ((int)threadIdx.x)) + 4224)];
    kernel_shared[(((int)threadIdx.x) + 768)] = kernel[((((k_outer_outer * 16384) + ((((int)blockIdx.x) & 7) * 512)) + ((int)threadIdx.x)) + 4352)];
    kernel_shared[(((int)threadIdx.x) + 896)] = kernel[((((k_outer_outer * 16384) + ((((int)blockIdx.x) & 7) * 512)) + ((int)threadIdx.x)) + 4480)];
    kernel_shared[(((int)threadIdx.x) + 1024)] = kernel[((((k_outer_outer * 16384) + ((((int)blockIdx.x) & 7) * 512)) + ((int)threadIdx.x)) + 8192)];
    kernel_shared[(((int)threadIdx.x) + 1152)] = kernel[((((k_outer_outer * 16384) + ((((int)blockIdx.x) & 7) * 512)) + ((int)threadIdx.x)) + 8320)];
    kernel_shared[(((int)threadIdx.x) + 1280)] = kernel[((((k_outer_outer * 16384) + ((((int)blockIdx.x) & 7) * 512)) + ((int)threadIdx.x)) + 8448)];
    kernel_shared[(((int)threadIdx.x) + 1408)] = kernel[((((k_outer_outer * 16384) + ((((int)blockIdx.x) & 7) * 512)) + ((int)threadIdx.x)) + 8576)];
    kernel_shared[(((int)threadIdx.x) + 1536)] = kernel[((((k_outer_outer * 16384) + ((((int)blockIdx.x) & 7) * 512)) + ((int)threadIdx.x)) + 12288)];
    kernel_shared[(((int)threadIdx.x) + 1664)] = kernel[((((k_outer_outer * 16384) + ((((int)blockIdx.x) & 7) * 512)) + ((int)threadIdx.x)) + 12416)];
    kernel_shared[(((int)threadIdx.x) + 1792)] = kernel[((((k_outer_outer * 16384) + ((((int)blockIdx.x) & 7) * 512)) + ((int)threadIdx.x)) + 12544)];
    kernel_shared[(((int)threadIdx.x) + 1920)] = kernel[((((k_outer_outer * 16384) + ((((int)blockIdx.x) & 7) * 512)) + ((int)threadIdx.x)) + 12672)];
    __syncthreads();
    for (int i_c_outer_inner = 0; i_c_outer_inner < 64; ++i_c_outer_inner) {
      T_matmul_NN_local[(i_c_outer_inner * 16)] = (T_matmul_NN_local[(i_c_outer_inner * 16)] + (data_shared[(((((int)threadIdx.x) >> 6) * 512) + (i_c_outer_inner * 8))] * kernel_shared[((((int)threadIdx.x) & 63) * 8)]));
      T_matmul_NN_local[((i_c_outer_inner * 16) + 8)] = (T_matmul_NN_local[((i_c_outer_inner * 16) + 8)] + (data_shared[((((((int)threadIdx.x) >> 6) * 512) + (i_c_outer_inner * 8)) + 4)] * kernel_shared[((((int)threadIdx.x) & 63) * 8)]));
      T_matmul_NN_local[(i_c_outer_inner * 16)] = (T_matmul_NN_local[(i_c_outer_inner * 16)] + (data_shared[((((((int)threadIdx.x) >> 6) * 512) + (i_c_outer_inner * 8)) + 1)] * kernel_shared[(((((int)threadIdx.x) & 63) * 8) + 512)]));
      T_matmul_NN_local[((i_c_outer_inner * 16) + 8)] = (T_matmul_NN_local[((i_c_outer_inner * 16) + 8)] + (data_shared[((((((int)threadIdx.x) >> 6) * 512) + (i_c_outer_inner * 8)) + 5)] * kernel_shared[(((((int)threadIdx.x) & 63) * 8) + 512)]));
      T_matmul_NN_local[(i_c_outer_inner * 16)] = (T_matmul_NN_local[(i_c_outer_inner * 16)] + (data_shared[((((((int)threadIdx.x) >> 6) * 512) + (i_c_outer_inner * 8)) + 2)] * kernel_shared[(((((int)threadIdx.x) & 63) * 8) + 1024)]));
      T_matmul_NN_local[((i_c_outer_inner * 16) + 8)] = (T_matmul_NN_local[((i_c_outer_inner * 16) + 8)] + (data_shared[((((((int)threadIdx.x) >> 6) * 512) + (i_c_outer_inner * 8)) + 6)] * kernel_shared[(((((int)threadIdx.x) & 63) * 8) + 1024)]));
      T_matmul_NN_local[(i_c_outer_inner * 16)] = (T_matmul_NN_local[(i_c_outer_inner * 16)] + (data_shared[((((((int)threadIdx.x) >> 6) * 512) + (i_c_outer_inner * 8)) + 3)] * kernel_shared[(((((int)threadIdx.x) & 63) * 8) + 1536)]));
      T_matmul_NN_local[((i_c_outer_inner * 16) + 8)] = (T_matmul_NN_local[((i_c_outer_inner * 16) + 8)] + (data_shared[((((((int)threadIdx.x) >> 6) * 512) + (i_c_outer_inner * 8)) + 7)] * kernel_shared[(((((int)threadIdx.x) & 63) * 8) + 1536)]));
      T_matmul_NN_local[((i_c_outer_inner * 16) + 1)] = (T_matmul_NN_local[((i_c_outer_inner * 16) + 1)] + (data_shared[(((((int)threadIdx.x) >> 6) * 512) + (i_c_outer_inner * 8))] * kernel_shared[(((((int)threadIdx.x) & 63) * 8) + 1)]));
      T_matmul_NN_local[((i_c_outer_inner * 16) + 9)] = (T_matmul_NN_local[((i_c_outer_inner * 16) + 9)] + (data_shared[((((((int)threadIdx.x) >> 6) * 512) + (i_c_outer_inner * 8)) + 4)] * kernel_shared[(((((int)threadIdx.x) & 63) * 8) + 1)]));
      T_matmul_NN_local[((i_c_outer_inner * 16) + 1)] = (T_matmul_NN_local[((i_c_outer_inner * 16) + 1)] + (data_shared[((((((int)threadIdx.x) >> 6) * 512) + (i_c_outer_inner * 8)) + 1)] * kernel_shared[(((((int)threadIdx.x) & 63) * 8) + 513)]));
      T_matmul_NN_local[((i_c_outer_inner * 16) + 9)] = (T_matmul_NN_local[((i_c_outer_inner * 16) + 9)] + (data_shared[((((((int)threadIdx.x) >> 6) * 512) + (i_c_outer_inner * 8)) + 5)] * kernel_shared[(((((int)threadIdx.x) & 63) * 8) + 513)]));
      T_matmul_NN_local[((i_c_outer_inner * 16) + 1)] = (T_matmul_NN_local[((i_c_outer_inner * 16) + 1)] + (data_shared[((((((int)threadIdx.x) >> 6) * 512) + (i_c_outer_inner * 8)) + 2)] * kernel_shared[(((((int)threadIdx.x) & 63) * 8) + 1025)]));
      T_matmul_NN_local[((i_c_outer_inner * 16) + 9)] = (T_matmul_NN_local[((i_c_outer_inner * 16) + 9)] + (data_shared[((((((int)threadIdx.x) >> 6) * 512) + (i_c_outer_inner * 8)) + 6)] * kernel_shared[(((((int)threadIdx.x) & 63) * 8) + 1025)]));
      T_matmul_NN_local[((i_c_outer_inner * 16) + 1)] = (T_matmul_NN_local[((i_c_outer_inner * 16) + 1)] + (data_shared[((((((int)threadIdx.x) >> 6) * 512) + (i_c_outer_inner * 8)) + 3)] * kernel_shared[(((((int)threadIdx.x) & 63) * 8) + 1537)]));
      T_matmul_NN_local[((i_c_outer_inner * 16) + 9)] = (T_matmul_NN_local[((i_c_outer_inner * 16) + 9)] + (data_shared[((((((int)threadIdx.x) >> 6) * 512) + (i_c_outer_inner * 8)) + 7)] * kernel_shared[(((((int)threadIdx.x) & 63) * 8) + 1537)]));
      T_matmul_NN_local[((i_c_outer_inner * 16) + 2)] = (T_matmul_NN_local[((i_c_outer_inner * 16) + 2)] + (data_shared[(((((int)threadIdx.x) >> 6) * 512) + (i_c_outer_inner * 8))] * kernel_shared[(((((int)threadIdx.x) & 63) * 8) + 2)]));
      T_matmul_NN_local[((i_c_outer_inner * 16) + 10)] = (T_matmul_NN_local[((i_c_outer_inner * 16) + 10)] + (data_shared[((((((int)threadIdx.x) >> 6) * 512) + (i_c_outer_inner * 8)) + 4)] * kernel_shared[(((((int)threadIdx.x) & 63) * 8) + 2)]));
      T_matmul_NN_local[((i_c_outer_inner * 16) + 2)] = (T_matmul_NN_local[((i_c_outer_inner * 16) + 2)] + (data_shared[((((((int)threadIdx.x) >> 6) * 512) + (i_c_outer_inner * 8)) + 1)] * kernel_shared[(((((int)threadIdx.x) & 63) * 8) + 514)]));
      T_matmul_NN_local[((i_c_outer_inner * 16) + 10)] = (T_matmul_NN_local[((i_c_outer_inner * 16) + 10)] + (data_shared[((((((int)threadIdx.x) >> 6) * 512) + (i_c_outer_inner * 8)) + 5)] * kernel_shared[(((((int)threadIdx.x) & 63) * 8) + 514)]));
      T_matmul_NN_local[((i_c_outer_inner * 16) + 2)] = (T_matmul_NN_local[((i_c_outer_inner * 16) + 2)] + (data_shared[((((((int)threadIdx.x) >> 6) * 512) + (i_c_outer_inner * 8)) + 2)] * kernel_shared[(((((int)threadIdx.x) & 63) * 8) + 1026)]));
      T_matmul_NN_local[((i_c_outer_inner * 16) + 10)] = (T_matmul_NN_local[((i_c_outer_inner * 16) + 10)] + (data_shared[((((((int)threadIdx.x) >> 6) * 512) + (i_c_outer_inner * 8)) + 6)] * kernel_shared[(((((int)threadIdx.x) & 63) * 8) + 1026)]));
      T_matmul_NN_local[((i_c_outer_inner * 16) + 2)] = (T_matmul_NN_local[((i_c_outer_inner * 16) + 2)] + (data_shared[((((((int)threadIdx.x) >> 6) * 512) + (i_c_outer_inner * 8)) + 3)] * kernel_shared[(((((int)threadIdx.x) & 63) * 8) + 1538)]));
      T_matmul_NN_local[((i_c_outer_inner * 16) + 10)] = (T_matmul_NN_local[((i_c_outer_inner * 16) + 10)] + (data_shared[((((((int)threadIdx.x) >> 6) * 512) + (i_c_outer_inner * 8)) + 7)] * kernel_shared[(((((int)threadIdx.x) & 63) * 8) + 1538)]));
      T_matmul_NN_local[((i_c_outer_inner * 16) + 3)] = (T_matmul_NN_local[((i_c_outer_inner * 16) + 3)] + (data_shared[(((((int)threadIdx.x) >> 6) * 512) + (i_c_outer_inner * 8))] * kernel_shared[(((((int)threadIdx.x) & 63) * 8) + 3)]));
      T_matmul_NN_local[((i_c_outer_inner * 16) + 11)] = (T_matmul_NN_local[((i_c_outer_inner * 16) + 11)] + (data_shared[((((((int)threadIdx.x) >> 6) * 512) + (i_c_outer_inner * 8)) + 4)] * kernel_shared[(((((int)threadIdx.x) & 63) * 8) + 3)]));
      T_matmul_NN_local[((i_c_outer_inner * 16) + 3)] = (T_matmul_NN_local[((i_c_outer_inner * 16) + 3)] + (data_shared[((((((int)threadIdx.x) >> 6) * 512) + (i_c_outer_inner * 8)) + 1)] * kernel_shared[(((((int)threadIdx.x) & 63) * 8) + 515)]));
      T_matmul_NN_local[((i_c_outer_inner * 16) + 11)] = (T_matmul_NN_local[((i_c_outer_inner * 16) + 11)] + (data_shared[((((((int)threadIdx.x) >> 6) * 512) + (i_c_outer_inner * 8)) + 5)] * kernel_shared[(((((int)threadIdx.x) & 63) * 8) + 515)]));
      T_matmul_NN_local[((i_c_outer_inner * 16) + 3)] = (T_matmul_NN_local[((i_c_outer_inner * 16) + 3)] + (data_shared[((((((int)threadIdx.x) >> 6) * 512) + (i_c_outer_inner * 8)) + 2)] * kernel_shared[(((((int)threadIdx.x) & 63) * 8) + 1027)]));
      T_matmul_NN_local[((i_c_outer_inner * 16) + 11)] = (T_matmul_NN_local[((i_c_outer_inner * 16) + 11)] + (data_shared[((((((int)threadIdx.x) >> 6) * 512) + (i_c_outer_inner * 8)) + 6)] * kernel_shared[(((((int)threadIdx.x) & 63) * 8) + 1027)]));
      T_matmul_NN_local[((i_c_outer_inner * 16) + 3)] = (T_matmul_NN_local[((i_c_outer_inner * 16) + 3)] + (data_shared[((((((int)threadIdx.x) >> 6) * 512) + (i_c_outer_inner * 8)) + 3)] * kernel_shared[(((((int)threadIdx.x) & 63) * 8) + 1539)]));
      T_matmul_NN_local[((i_c_outer_inner * 16) + 11)] = (T_matmul_NN_local[((i_c_outer_inner * 16) + 11)] + (data_shared[((((((int)threadIdx.x) >> 6) * 512) + (i_c_outer_inner * 8)) + 7)] * kernel_shared[(((((int)threadIdx.x) & 63) * 8) + 1539)]));
      T_matmul_NN_local[((i_c_outer_inner * 16) + 4)] = (T_matmul_NN_local[((i_c_outer_inner * 16) + 4)] + (data_shared[(((((int)threadIdx.x) >> 6) * 512) + (i_c_outer_inner * 8))] * kernel_shared[(((((int)threadIdx.x) & 63) * 8) + 4)]));
      T_matmul_NN_local[((i_c_outer_inner * 16) + 12)] = (T_matmul_NN_local[((i_c_outer_inner * 16) + 12)] + (data_shared[((((((int)threadIdx.x) >> 6) * 512) + (i_c_outer_inner * 8)) + 4)] * kernel_shared[(((((int)threadIdx.x) & 63) * 8) + 4)]));
      T_matmul_NN_local[((i_c_outer_inner * 16) + 4)] = (T_matmul_NN_local[((i_c_outer_inner * 16) + 4)] + (data_shared[((((((int)threadIdx.x) >> 6) * 512) + (i_c_outer_inner * 8)) + 1)] * kernel_shared[(((((int)threadIdx.x) & 63) * 8) + 516)]));
      T_matmul_NN_local[((i_c_outer_inner * 16) + 12)] = (T_matmul_NN_local[((i_c_outer_inner * 16) + 12)] + (data_shared[((((((int)threadIdx.x) >> 6) * 512) + (i_c_outer_inner * 8)) + 5)] * kernel_shared[(((((int)threadIdx.x) & 63) * 8) + 516)]));
      T_matmul_NN_local[((i_c_outer_inner * 16) + 4)] = (T_matmul_NN_local[((i_c_outer_inner * 16) + 4)] + (data_shared[((((((int)threadIdx.x) >> 6) * 512) + (i_c_outer_inner * 8)) + 2)] * kernel_shared[(((((int)threadIdx.x) & 63) * 8) + 1028)]));
      T_matmul_NN_local[((i_c_outer_inner * 16) + 12)] = (T_matmul_NN_local[((i_c_outer_inner * 16) + 12)] + (data_shared[((((((int)threadIdx.x) >> 6) * 512) + (i_c_outer_inner * 8)) + 6)] * kernel_shared[(((((int)threadIdx.x) & 63) * 8) + 1028)]));
      T_matmul_NN_local[((i_c_outer_inner * 16) + 4)] = (T_matmul_NN_local[((i_c_outer_inner * 16) + 4)] + (data_shared[((((((int)threadIdx.x) >> 6) * 512) + (i_c_outer_inner * 8)) + 3)] * kernel_shared[(((((int)threadIdx.x) & 63) * 8) + 1540)]));
      T_matmul_NN_local[((i_c_outer_inner * 16) + 12)] = (T_matmul_NN_local[((i_c_outer_inner * 16) + 12)] + (data_shared[((((((int)threadIdx.x) >> 6) * 512) + (i_c_outer_inner * 8)) + 7)] * kernel_shared[(((((int)threadIdx.x) & 63) * 8) + 1540)]));
      T_matmul_NN_local[((i_c_outer_inner * 16) + 5)] = (T_matmul_NN_local[((i_c_outer_inner * 16) + 5)] + (data_shared[(((((int)threadIdx.x) >> 6) * 512) + (i_c_outer_inner * 8))] * kernel_shared[(((((int)threadIdx.x) & 63) * 8) + 5)]));
      T_matmul_NN_local[((i_c_outer_inner * 16) + 13)] = (T_matmul_NN_local[((i_c_outer_inner * 16) + 13)] + (data_shared[((((((int)threadIdx.x) >> 6) * 512) + (i_c_outer_inner * 8)) + 4)] * kernel_shared[(((((int)threadIdx.x) & 63) * 8) + 5)]));
      T_matmul_NN_local[((i_c_outer_inner * 16) + 5)] = (T_matmul_NN_local[((i_c_outer_inner * 16) + 5)] + (data_shared[((((((int)threadIdx.x) >> 6) * 512) + (i_c_outer_inner * 8)) + 1)] * kernel_shared[(((((int)threadIdx.x) & 63) * 8) + 517)]));
      T_matmul_NN_local[((i_c_outer_inner * 16) + 13)] = (T_matmul_NN_local[((i_c_outer_inner * 16) + 13)] + (data_shared[((((((int)threadIdx.x) >> 6) * 512) + (i_c_outer_inner * 8)) + 5)] * kernel_shared[(((((int)threadIdx.x) & 63) * 8) + 517)]));
      T_matmul_NN_local[((i_c_outer_inner * 16) + 5)] = (T_matmul_NN_local[((i_c_outer_inner * 16) + 5)] + (data_shared[((((((int)threadIdx.x) >> 6) * 512) + (i_c_outer_inner * 8)) + 2)] * kernel_shared[(((((int)threadIdx.x) & 63) * 8) + 1029)]));
      T_matmul_NN_local[((i_c_outer_inner * 16) + 13)] = (T_matmul_NN_local[((i_c_outer_inner * 16) + 13)] + (data_shared[((((((int)threadIdx.x) >> 6) * 512) + (i_c_outer_inner * 8)) + 6)] * kernel_shared[(((((int)threadIdx.x) & 63) * 8) + 1029)]));
      T_matmul_NN_local[((i_c_outer_inner * 16) + 5)] = (T_matmul_NN_local[((i_c_outer_inner * 16) + 5)] + (data_shared[((((((int)threadIdx.x) >> 6) * 512) + (i_c_outer_inner * 8)) + 3)] * kernel_shared[(((((int)threadIdx.x) & 63) * 8) + 1541)]));
      T_matmul_NN_local[((i_c_outer_inner * 16) + 13)] = (T_matmul_NN_local[((i_c_outer_inner * 16) + 13)] + (data_shared[((((((int)threadIdx.x) >> 6) * 512) + (i_c_outer_inner * 8)) + 7)] * kernel_shared[(((((int)threadIdx.x) & 63) * 8) + 1541)]));
      T_matmul_NN_local[((i_c_outer_inner * 16) + 6)] = (T_matmul_NN_local[((i_c_outer_inner * 16) + 6)] + (data_shared[(((((int)threadIdx.x) >> 6) * 512) + (i_c_outer_inner * 8))] * kernel_shared[(((((int)threadIdx.x) & 63) * 8) + 6)]));
      T_matmul_NN_local[((i_c_outer_inner * 16) + 14)] = (T_matmul_NN_local[((i_c_outer_inner * 16) + 14)] + (data_shared[((((((int)threadIdx.x) >> 6) * 512) + (i_c_outer_inner * 8)) + 4)] * kernel_shared[(((((int)threadIdx.x) & 63) * 8) + 6)]));
      T_matmul_NN_local[((i_c_outer_inner * 16) + 6)] = (T_matmul_NN_local[((i_c_outer_inner * 16) + 6)] + (data_shared[((((((int)threadIdx.x) >> 6) * 512) + (i_c_outer_inner * 8)) + 1)] * kernel_shared[(((((int)threadIdx.x) & 63) * 8) + 518)]));
      T_matmul_NN_local[((i_c_outer_inner * 16) + 14)] = (T_matmul_NN_local[((i_c_outer_inner * 16) + 14)] + (data_shared[((((((int)threadIdx.x) >> 6) * 512) + (i_c_outer_inner * 8)) + 5)] * kernel_shared[(((((int)threadIdx.x) & 63) * 8) + 518)]));
      T_matmul_NN_local[((i_c_outer_inner * 16) + 6)] = (T_matmul_NN_local[((i_c_outer_inner * 16) + 6)] + (data_shared[((((((int)threadIdx.x) >> 6) * 512) + (i_c_outer_inner * 8)) + 2)] * kernel_shared[(((((int)threadIdx.x) & 63) * 8) + 1030)]));
      T_matmul_NN_local[((i_c_outer_inner * 16) + 14)] = (T_matmul_NN_local[((i_c_outer_inner * 16) + 14)] + (data_shared[((((((int)threadIdx.x) >> 6) * 512) + (i_c_outer_inner * 8)) + 6)] * kernel_shared[(((((int)threadIdx.x) & 63) * 8) + 1030)]));
      T_matmul_NN_local[((i_c_outer_inner * 16) + 6)] = (T_matmul_NN_local[((i_c_outer_inner * 16) + 6)] + (data_shared[((((((int)threadIdx.x) >> 6) * 512) + (i_c_outer_inner * 8)) + 3)] * kernel_shared[(((((int)threadIdx.x) & 63) * 8) + 1542)]));
      T_matmul_NN_local[((i_c_outer_inner * 16) + 14)] = (T_matmul_NN_local[((i_c_outer_inner * 16) + 14)] + (data_shared[((((((int)threadIdx.x) >> 6) * 512) + (i_c_outer_inner * 8)) + 7)] * kernel_shared[(((((int)threadIdx.x) & 63) * 8) + 1542)]));
      T_matmul_NN_local[((i_c_outer_inner * 16) + 7)] = (T_matmul_NN_local[((i_c_outer_inner * 16) + 7)] + (data_shared[(((((int)threadIdx.x) >> 6) * 512) + (i_c_outer_inner * 8))] * kernel_shared[(((((int)threadIdx.x) & 63) * 8) + 7)]));
      T_matmul_NN_local[((i_c_outer_inner * 16) + 15)] = (T_matmul_NN_local[((i_c_outer_inner * 16) + 15)] + (data_shared[((((((int)threadIdx.x) >> 6) * 512) + (i_c_outer_inner * 8)) + 4)] * kernel_shared[(((((int)threadIdx.x) & 63) * 8) + 7)]));
      T_matmul_NN_local[((i_c_outer_inner * 16) + 7)] = (T_matmul_NN_local[((i_c_outer_inner * 16) + 7)] + (data_shared[((((((int)threadIdx.x) >> 6) * 512) + (i_c_outer_inner * 8)) + 1)] * kernel_shared[(((((int)threadIdx.x) & 63) * 8) + 519)]));
      T_matmul_NN_local[((i_c_outer_inner * 16) + 15)] = (T_matmul_NN_local[((i_c_outer_inner * 16) + 15)] + (data_shared[((((((int)threadIdx.x) >> 6) * 512) + (i_c_outer_inner * 8)) + 5)] * kernel_shared[(((((int)threadIdx.x) & 63) * 8) + 519)]));
      T_matmul_NN_local[((i_c_outer_inner * 16) + 7)] = (T_matmul_NN_local[((i_c_outer_inner * 16) + 7)] + (data_shared[((((((int)threadIdx.x) >> 6) * 512) + (i_c_outer_inner * 8)) + 2)] * kernel_shared[(((((int)threadIdx.x) & 63) * 8) + 1031)]));
      T_matmul_NN_local[((i_c_outer_inner * 16) + 15)] = (T_matmul_NN_local[((i_c_outer_inner * 16) + 15)] + (data_shared[((((((int)threadIdx.x) >> 6) * 512) + (i_c_outer_inner * 8)) + 6)] * kernel_shared[(((((int)threadIdx.x) & 63) * 8) + 1031)]));
      T_matmul_NN_local[((i_c_outer_inner * 16) + 7)] = (T_matmul_NN_local[((i_c_outer_inner * 16) + 7)] + (data_shared[((((((int)threadIdx.x) >> 6) * 512) + (i_c_outer_inner * 8)) + 3)] * kernel_shared[(((((int)threadIdx.x) & 63) * 8) + 1543)]));
      T_matmul_NN_local[((i_c_outer_inner * 16) + 15)] = (T_matmul_NN_local[((i_c_outer_inner * 16) + 15)] + (data_shared[((((((int)threadIdx.x) >> 6) * 512) + (i_c_outer_inner * 8)) + 7)] * kernel_shared[(((((int)threadIdx.x) & 63) * 8) + 1543)]));
    }
  }
  for (int i_inner = 0; i_inner < 128; ++i_inner) {
    for (int j_inner = 0; j_inner < 8; ++j_inner) {
      T_matmul_NN[(((((((((int)blockIdx.x) >> 3) * 1048576) + ((((int)threadIdx.x) >> 6) * 524288)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 7) * 512)) + ((((int)threadIdx.x) & 63) * 8)) + j_inner)] = T_matmul_NN_local[((i_inner * 8) + j_inner)];
    }
  }
}

