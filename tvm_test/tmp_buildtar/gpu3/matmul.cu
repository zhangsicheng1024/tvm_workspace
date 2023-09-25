
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
  __shared__ float data_shared[2048];
  __shared__ float kernel_shared[8192];
  for (int i_c_outer_inner_init = 0; i_c_outer_inner_init < 4; ++i_c_outer_inner_init) {
    T_matmul_NN_local[(i_c_outer_inner_init * 4)] = 0.000000e+00f;
    T_matmul_NN_local[((i_c_outer_inner_init * 4) + 16)] = 0.000000e+00f;
    T_matmul_NN_local[((i_c_outer_inner_init * 4) + 32)] = 0.000000e+00f;
    T_matmul_NN_local[((i_c_outer_inner_init * 4) + 48)] = 0.000000e+00f;
    T_matmul_NN_local[((i_c_outer_inner_init * 4) + 64)] = 0.000000e+00f;
    T_matmul_NN_local[((i_c_outer_inner_init * 4) + 80)] = 0.000000e+00f;
    T_matmul_NN_local[((i_c_outer_inner_init * 4) + 96)] = 0.000000e+00f;
    T_matmul_NN_local[((i_c_outer_inner_init * 4) + 112)] = 0.000000e+00f;
    T_matmul_NN_local[((i_c_outer_inner_init * 4) + 1)] = 0.000000e+00f;
    T_matmul_NN_local[((i_c_outer_inner_init * 4) + 17)] = 0.000000e+00f;
    T_matmul_NN_local[((i_c_outer_inner_init * 4) + 33)] = 0.000000e+00f;
    T_matmul_NN_local[((i_c_outer_inner_init * 4) + 49)] = 0.000000e+00f;
    T_matmul_NN_local[((i_c_outer_inner_init * 4) + 65)] = 0.000000e+00f;
    T_matmul_NN_local[((i_c_outer_inner_init * 4) + 81)] = 0.000000e+00f;
    T_matmul_NN_local[((i_c_outer_inner_init * 4) + 97)] = 0.000000e+00f;
    T_matmul_NN_local[((i_c_outer_inner_init * 4) + 113)] = 0.000000e+00f;
    T_matmul_NN_local[((i_c_outer_inner_init * 4) + 2)] = 0.000000e+00f;
    T_matmul_NN_local[((i_c_outer_inner_init * 4) + 18)] = 0.000000e+00f;
    T_matmul_NN_local[((i_c_outer_inner_init * 4) + 34)] = 0.000000e+00f;
    T_matmul_NN_local[((i_c_outer_inner_init * 4) + 50)] = 0.000000e+00f;
    T_matmul_NN_local[((i_c_outer_inner_init * 4) + 66)] = 0.000000e+00f;
    T_matmul_NN_local[((i_c_outer_inner_init * 4) + 82)] = 0.000000e+00f;
    T_matmul_NN_local[((i_c_outer_inner_init * 4) + 98)] = 0.000000e+00f;
    T_matmul_NN_local[((i_c_outer_inner_init * 4) + 114)] = 0.000000e+00f;
    T_matmul_NN_local[((i_c_outer_inner_init * 4) + 3)] = 0.000000e+00f;
    T_matmul_NN_local[((i_c_outer_inner_init * 4) + 19)] = 0.000000e+00f;
    T_matmul_NN_local[((i_c_outer_inner_init * 4) + 35)] = 0.000000e+00f;
    T_matmul_NN_local[((i_c_outer_inner_init * 4) + 51)] = 0.000000e+00f;
    T_matmul_NN_local[((i_c_outer_inner_init * 4) + 67)] = 0.000000e+00f;
    T_matmul_NN_local[((i_c_outer_inner_init * 4) + 83)] = 0.000000e+00f;
    T_matmul_NN_local[((i_c_outer_inner_init * 4) + 99)] = 0.000000e+00f;
    T_matmul_NN_local[((i_c_outer_inner_init * 4) + 115)] = 0.000000e+00f;
  }
  for (int k_outer_outer = 0; k_outer_outer < 32; ++k_outer_outer) {
    __syncthreads();
    *(float4*)(data_shared + (((int)threadIdx.x) * 4)) = *(float4*)(data + (((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 3) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx.x) & 7) * 4)));
    *(float4*)(data_shared + ((((int)threadIdx.x) * 4) + 512)) = *(float4*)(data + ((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 3) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 16384));
    *(float4*)(data_shared + ((((int)threadIdx.x) * 4) + 1024)) = *(float4*)(data + ((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 3) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 32768));
    *(float4*)(data_shared + ((((int)threadIdx.x) * 4) + 1536)) = *(float4*)(data + ((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 3) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 49152));
    kernel_shared[((int)threadIdx.x)] = kernel[(((k_outer_outer * 131072) + ((((int)blockIdx.x) & 15) * 256)) + ((int)threadIdx.x))];
    kernel_shared[(((int)threadIdx.x) + 128)] = kernel[((((k_outer_outer * 131072) + ((((int)blockIdx.x) & 15) * 256)) + ((int)threadIdx.x)) + 128)];
    kernel_shared[(((int)threadIdx.x) + 256)] = kernel[((((k_outer_outer * 131072) + ((((int)blockIdx.x) & 15) * 256)) + ((int)threadIdx.x)) + 4096)];
    kernel_shared[(((int)threadIdx.x) + 384)] = kernel[((((k_outer_outer * 131072) + ((((int)blockIdx.x) & 15) * 256)) + ((int)threadIdx.x)) + 4224)];
    kernel_shared[(((int)threadIdx.x) + 512)] = kernel[((((k_outer_outer * 131072) + ((((int)blockIdx.x) & 15) * 256)) + ((int)threadIdx.x)) + 8192)];
    kernel_shared[(((int)threadIdx.x) + 640)] = kernel[((((k_outer_outer * 131072) + ((((int)blockIdx.x) & 15) * 256)) + ((int)threadIdx.x)) + 8320)];
    kernel_shared[(((int)threadIdx.x) + 768)] = kernel[((((k_outer_outer * 131072) + ((((int)blockIdx.x) & 15) * 256)) + ((int)threadIdx.x)) + 12288)];
    kernel_shared[(((int)threadIdx.x) + 896)] = kernel[((((k_outer_outer * 131072) + ((((int)blockIdx.x) & 15) * 256)) + ((int)threadIdx.x)) + 12416)];
    kernel_shared[(((int)threadIdx.x) + 1024)] = kernel[((((k_outer_outer * 131072) + ((((int)blockIdx.x) & 15) * 256)) + ((int)threadIdx.x)) + 16384)];
    kernel_shared[(((int)threadIdx.x) + 1152)] = kernel[((((k_outer_outer * 131072) + ((((int)blockIdx.x) & 15) * 256)) + ((int)threadIdx.x)) + 16512)];
    kernel_shared[(((int)threadIdx.x) + 1280)] = kernel[((((k_outer_outer * 131072) + ((((int)blockIdx.x) & 15) * 256)) + ((int)threadIdx.x)) + 20480)];
    kernel_shared[(((int)threadIdx.x) + 1408)] = kernel[((((k_outer_outer * 131072) + ((((int)blockIdx.x) & 15) * 256)) + ((int)threadIdx.x)) + 20608)];
    kernel_shared[(((int)threadIdx.x) + 1536)] = kernel[((((k_outer_outer * 131072) + ((((int)blockIdx.x) & 15) * 256)) + ((int)threadIdx.x)) + 24576)];
    kernel_shared[(((int)threadIdx.x) + 1664)] = kernel[((((k_outer_outer * 131072) + ((((int)blockIdx.x) & 15) * 256)) + ((int)threadIdx.x)) + 24704)];
    kernel_shared[(((int)threadIdx.x) + 1792)] = kernel[((((k_outer_outer * 131072) + ((((int)blockIdx.x) & 15) * 256)) + ((int)threadIdx.x)) + 28672)];
    kernel_shared[(((int)threadIdx.x) + 1920)] = kernel[((((k_outer_outer * 131072) + ((((int)blockIdx.x) & 15) * 256)) + ((int)threadIdx.x)) + 28800)];
    kernel_shared[(((int)threadIdx.x) + 2048)] = kernel[((((k_outer_outer * 131072) + ((((int)blockIdx.x) & 15) * 256)) + ((int)threadIdx.x)) + 32768)];
    kernel_shared[(((int)threadIdx.x) + 2176)] = kernel[((((k_outer_outer * 131072) + ((((int)blockIdx.x) & 15) * 256)) + ((int)threadIdx.x)) + 32896)];
    kernel_shared[(((int)threadIdx.x) + 2304)] = kernel[((((k_outer_outer * 131072) + ((((int)blockIdx.x) & 15) * 256)) + ((int)threadIdx.x)) + 36864)];
    kernel_shared[(((int)threadIdx.x) + 2432)] = kernel[((((k_outer_outer * 131072) + ((((int)blockIdx.x) & 15) * 256)) + ((int)threadIdx.x)) + 36992)];
    kernel_shared[(((int)threadIdx.x) + 2560)] = kernel[((((k_outer_outer * 131072) + ((((int)blockIdx.x) & 15) * 256)) + ((int)threadIdx.x)) + 40960)];
    kernel_shared[(((int)threadIdx.x) + 2688)] = kernel[((((k_outer_outer * 131072) + ((((int)blockIdx.x) & 15) * 256)) + ((int)threadIdx.x)) + 41088)];
    kernel_shared[(((int)threadIdx.x) + 2816)] = kernel[((((k_outer_outer * 131072) + ((((int)blockIdx.x) & 15) * 256)) + ((int)threadIdx.x)) + 45056)];
    kernel_shared[(((int)threadIdx.x) + 2944)] = kernel[((((k_outer_outer * 131072) + ((((int)blockIdx.x) & 15) * 256)) + ((int)threadIdx.x)) + 45184)];
    kernel_shared[(((int)threadIdx.x) + 3072)] = kernel[((((k_outer_outer * 131072) + ((((int)blockIdx.x) & 15) * 256)) + ((int)threadIdx.x)) + 49152)];
    kernel_shared[(((int)threadIdx.x) + 3200)] = kernel[((((k_outer_outer * 131072) + ((((int)blockIdx.x) & 15) * 256)) + ((int)threadIdx.x)) + 49280)];
    kernel_shared[(((int)threadIdx.x) + 3328)] = kernel[((((k_outer_outer * 131072) + ((((int)blockIdx.x) & 15) * 256)) + ((int)threadIdx.x)) + 53248)];
    kernel_shared[(((int)threadIdx.x) + 3456)] = kernel[((((k_outer_outer * 131072) + ((((int)blockIdx.x) & 15) * 256)) + ((int)threadIdx.x)) + 53376)];
    kernel_shared[(((int)threadIdx.x) + 3584)] = kernel[((((k_outer_outer * 131072) + ((((int)blockIdx.x) & 15) * 256)) + ((int)threadIdx.x)) + 57344)];
    kernel_shared[(((int)threadIdx.x) + 3712)] = kernel[((((k_outer_outer * 131072) + ((((int)blockIdx.x) & 15) * 256)) + ((int)threadIdx.x)) + 57472)];
    kernel_shared[(((int)threadIdx.x) + 3840)] = kernel[((((k_outer_outer * 131072) + ((((int)blockIdx.x) & 15) * 256)) + ((int)threadIdx.x)) + 61440)];
    kernel_shared[(((int)threadIdx.x) + 3968)] = kernel[((((k_outer_outer * 131072) + ((((int)blockIdx.x) & 15) * 256)) + ((int)threadIdx.x)) + 61568)];
    kernel_shared[(((int)threadIdx.x) + 4096)] = kernel[((((k_outer_outer * 131072) + ((((int)blockIdx.x) & 15) * 256)) + ((int)threadIdx.x)) + 65536)];
    kernel_shared[(((int)threadIdx.x) + 4224)] = kernel[((((k_outer_outer * 131072) + ((((int)blockIdx.x) & 15) * 256)) + ((int)threadIdx.x)) + 65664)];
    kernel_shared[(((int)threadIdx.x) + 4352)] = kernel[((((k_outer_outer * 131072) + ((((int)blockIdx.x) & 15) * 256)) + ((int)threadIdx.x)) + 69632)];
    kernel_shared[(((int)threadIdx.x) + 4480)] = kernel[((((k_outer_outer * 131072) + ((((int)blockIdx.x) & 15) * 256)) + ((int)threadIdx.x)) + 69760)];
    kernel_shared[(((int)threadIdx.x) + 4608)] = kernel[((((k_outer_outer * 131072) + ((((int)blockIdx.x) & 15) * 256)) + ((int)threadIdx.x)) + 73728)];
    kernel_shared[(((int)threadIdx.x) + 4736)] = kernel[((((k_outer_outer * 131072) + ((((int)blockIdx.x) & 15) * 256)) + ((int)threadIdx.x)) + 73856)];
    kernel_shared[(((int)threadIdx.x) + 4864)] = kernel[((((k_outer_outer * 131072) + ((((int)blockIdx.x) & 15) * 256)) + ((int)threadIdx.x)) + 77824)];
    kernel_shared[(((int)threadIdx.x) + 4992)] = kernel[((((k_outer_outer * 131072) + ((((int)blockIdx.x) & 15) * 256)) + ((int)threadIdx.x)) + 77952)];
    kernel_shared[(((int)threadIdx.x) + 5120)] = kernel[((((k_outer_outer * 131072) + ((((int)blockIdx.x) & 15) * 256)) + ((int)threadIdx.x)) + 81920)];
    kernel_shared[(((int)threadIdx.x) + 5248)] = kernel[((((k_outer_outer * 131072) + ((((int)blockIdx.x) & 15) * 256)) + ((int)threadIdx.x)) + 82048)];
    kernel_shared[(((int)threadIdx.x) + 5376)] = kernel[((((k_outer_outer * 131072) + ((((int)blockIdx.x) & 15) * 256)) + ((int)threadIdx.x)) + 86016)];
    kernel_shared[(((int)threadIdx.x) + 5504)] = kernel[((((k_outer_outer * 131072) + ((((int)blockIdx.x) & 15) * 256)) + ((int)threadIdx.x)) + 86144)];
    kernel_shared[(((int)threadIdx.x) + 5632)] = kernel[((((k_outer_outer * 131072) + ((((int)blockIdx.x) & 15) * 256)) + ((int)threadIdx.x)) + 90112)];
    kernel_shared[(((int)threadIdx.x) + 5760)] = kernel[((((k_outer_outer * 131072) + ((((int)blockIdx.x) & 15) * 256)) + ((int)threadIdx.x)) + 90240)];
    kernel_shared[(((int)threadIdx.x) + 5888)] = kernel[((((k_outer_outer * 131072) + ((((int)blockIdx.x) & 15) * 256)) + ((int)threadIdx.x)) + 94208)];
    kernel_shared[(((int)threadIdx.x) + 6016)] = kernel[((((k_outer_outer * 131072) + ((((int)blockIdx.x) & 15) * 256)) + ((int)threadIdx.x)) + 94336)];
    kernel_shared[(((int)threadIdx.x) + 6144)] = kernel[((((k_outer_outer * 131072) + ((((int)blockIdx.x) & 15) * 256)) + ((int)threadIdx.x)) + 98304)];
    kernel_shared[(((int)threadIdx.x) + 6272)] = kernel[((((k_outer_outer * 131072) + ((((int)blockIdx.x) & 15) * 256)) + ((int)threadIdx.x)) + 98432)];
    kernel_shared[(((int)threadIdx.x) + 6400)] = kernel[((((k_outer_outer * 131072) + ((((int)blockIdx.x) & 15) * 256)) + ((int)threadIdx.x)) + 102400)];
    kernel_shared[(((int)threadIdx.x) + 6528)] = kernel[((((k_outer_outer * 131072) + ((((int)blockIdx.x) & 15) * 256)) + ((int)threadIdx.x)) + 102528)];
    kernel_shared[(((int)threadIdx.x) + 6656)] = kernel[((((k_outer_outer * 131072) + ((((int)blockIdx.x) & 15) * 256)) + ((int)threadIdx.x)) + 106496)];
    kernel_shared[(((int)threadIdx.x) + 6784)] = kernel[((((k_outer_outer * 131072) + ((((int)blockIdx.x) & 15) * 256)) + ((int)threadIdx.x)) + 106624)];
    kernel_shared[(((int)threadIdx.x) + 6912)] = kernel[((((k_outer_outer * 131072) + ((((int)blockIdx.x) & 15) * 256)) + ((int)threadIdx.x)) + 110592)];
    kernel_shared[(((int)threadIdx.x) + 7040)] = kernel[((((k_outer_outer * 131072) + ((((int)blockIdx.x) & 15) * 256)) + ((int)threadIdx.x)) + 110720)];
    kernel_shared[(((int)threadIdx.x) + 7168)] = kernel[((((k_outer_outer * 131072) + ((((int)blockIdx.x) & 15) * 256)) + ((int)threadIdx.x)) + 114688)];
    kernel_shared[(((int)threadIdx.x) + 7296)] = kernel[((((k_outer_outer * 131072) + ((((int)blockIdx.x) & 15) * 256)) + ((int)threadIdx.x)) + 114816)];
    kernel_shared[(((int)threadIdx.x) + 7424)] = kernel[((((k_outer_outer * 131072) + ((((int)blockIdx.x) & 15) * 256)) + ((int)threadIdx.x)) + 118784)];
    kernel_shared[(((int)threadIdx.x) + 7552)] = kernel[((((k_outer_outer * 131072) + ((((int)blockIdx.x) & 15) * 256)) + ((int)threadIdx.x)) + 118912)];
    kernel_shared[(((int)threadIdx.x) + 7680)] = kernel[((((k_outer_outer * 131072) + ((((int)blockIdx.x) & 15) * 256)) + ((int)threadIdx.x)) + 122880)];
    kernel_shared[(((int)threadIdx.x) + 7808)] = kernel[((((k_outer_outer * 131072) + ((((int)blockIdx.x) & 15) * 256)) + ((int)threadIdx.x)) + 123008)];
    kernel_shared[(((int)threadIdx.x) + 7936)] = kernel[((((k_outer_outer * 131072) + ((((int)blockIdx.x) & 15) * 256)) + ((int)threadIdx.x)) + 126976)];
    kernel_shared[(((int)threadIdx.x) + 8064)] = kernel[((((k_outer_outer * 131072) + ((((int)blockIdx.x) & 15) * 256)) + ((int)threadIdx.x)) + 127104)];
    __syncthreads();
    for (int k_outer_inner = 0; k_outer_inner < 4; ++k_outer_inner) {
      for (int i_c_outer_inner = 0; i_c_outer_inner < 4; ++i_c_outer_inner) {
        for (int k_inner = 0; k_inner < 8; ++k_inner) {
          T_matmul_NN_local[(i_c_outer_inner * 4)] = (T_matmul_NN_local[(i_c_outer_inner * 4)] + (data_shared[(((((((int)threadIdx.x) >> 5) * 512) + (i_c_outer_inner * 128)) + (k_outer_inner * 8)) + k_inner)] * kernel_shared[(((k_outer_inner * 2048) + (k_inner * 256)) + (((int)threadIdx.x) & 31))]));
          T_matmul_NN_local[((i_c_outer_inner * 4) + 16)] = (T_matmul_NN_local[((i_c_outer_inner * 4) + 16)] + (data_shared[(((((((int)threadIdx.x) >> 5) * 512) + (i_c_outer_inner * 128)) + (k_outer_inner * 8)) + k_inner)] * kernel_shared[((((k_outer_inner * 2048) + (k_inner * 256)) + (((int)threadIdx.x) & 31)) + 32)]));
          T_matmul_NN_local[((i_c_outer_inner * 4) + 32)] = (T_matmul_NN_local[((i_c_outer_inner * 4) + 32)] + (data_shared[(((((((int)threadIdx.x) >> 5) * 512) + (i_c_outer_inner * 128)) + (k_outer_inner * 8)) + k_inner)] * kernel_shared[((((k_outer_inner * 2048) + (k_inner * 256)) + (((int)threadIdx.x) & 31)) + 64)]));
          T_matmul_NN_local[((i_c_outer_inner * 4) + 48)] = (T_matmul_NN_local[((i_c_outer_inner * 4) + 48)] + (data_shared[(((((((int)threadIdx.x) >> 5) * 512) + (i_c_outer_inner * 128)) + (k_outer_inner * 8)) + k_inner)] * kernel_shared[((((k_outer_inner * 2048) + (k_inner * 256)) + (((int)threadIdx.x) & 31)) + 96)]));
          T_matmul_NN_local[((i_c_outer_inner * 4) + 64)] = (T_matmul_NN_local[((i_c_outer_inner * 4) + 64)] + (data_shared[(((((((int)threadIdx.x) >> 5) * 512) + (i_c_outer_inner * 128)) + (k_outer_inner * 8)) + k_inner)] * kernel_shared[((((k_outer_inner * 2048) + (k_inner * 256)) + (((int)threadIdx.x) & 31)) + 128)]));
          T_matmul_NN_local[((i_c_outer_inner * 4) + 80)] = (T_matmul_NN_local[((i_c_outer_inner * 4) + 80)] + (data_shared[(((((((int)threadIdx.x) >> 5) * 512) + (i_c_outer_inner * 128)) + (k_outer_inner * 8)) + k_inner)] * kernel_shared[((((k_outer_inner * 2048) + (k_inner * 256)) + (((int)threadIdx.x) & 31)) + 160)]));
          T_matmul_NN_local[((i_c_outer_inner * 4) + 96)] = (T_matmul_NN_local[((i_c_outer_inner * 4) + 96)] + (data_shared[(((((((int)threadIdx.x) >> 5) * 512) + (i_c_outer_inner * 128)) + (k_outer_inner * 8)) + k_inner)] * kernel_shared[((((k_outer_inner * 2048) + (k_inner * 256)) + (((int)threadIdx.x) & 31)) + 192)]));
          T_matmul_NN_local[((i_c_outer_inner * 4) + 112)] = (T_matmul_NN_local[((i_c_outer_inner * 4) + 112)] + (data_shared[(((((((int)threadIdx.x) >> 5) * 512) + (i_c_outer_inner * 128)) + (k_outer_inner * 8)) + k_inner)] * kernel_shared[((((k_outer_inner * 2048) + (k_inner * 256)) + (((int)threadIdx.x) & 31)) + 224)]));
          T_matmul_NN_local[((i_c_outer_inner * 4) + 1)] = (T_matmul_NN_local[((i_c_outer_inner * 4) + 1)] + (data_shared[((((((((int)threadIdx.x) >> 5) * 512) + (i_c_outer_inner * 128)) + (k_outer_inner * 8)) + k_inner) + 32)] * kernel_shared[(((k_outer_inner * 2048) + (k_inner * 256)) + (((int)threadIdx.x) & 31))]));
          T_matmul_NN_local[((i_c_outer_inner * 4) + 17)] = (T_matmul_NN_local[((i_c_outer_inner * 4) + 17)] + (data_shared[((((((((int)threadIdx.x) >> 5) * 512) + (i_c_outer_inner * 128)) + (k_outer_inner * 8)) + k_inner) + 32)] * kernel_shared[((((k_outer_inner * 2048) + (k_inner * 256)) + (((int)threadIdx.x) & 31)) + 32)]));
          T_matmul_NN_local[((i_c_outer_inner * 4) + 33)] = (T_matmul_NN_local[((i_c_outer_inner * 4) + 33)] + (data_shared[((((((((int)threadIdx.x) >> 5) * 512) + (i_c_outer_inner * 128)) + (k_outer_inner * 8)) + k_inner) + 32)] * kernel_shared[((((k_outer_inner * 2048) + (k_inner * 256)) + (((int)threadIdx.x) & 31)) + 64)]));
          T_matmul_NN_local[((i_c_outer_inner * 4) + 49)] = (T_matmul_NN_local[((i_c_outer_inner * 4) + 49)] + (data_shared[((((((((int)threadIdx.x) >> 5) * 512) + (i_c_outer_inner * 128)) + (k_outer_inner * 8)) + k_inner) + 32)] * kernel_shared[((((k_outer_inner * 2048) + (k_inner * 256)) + (((int)threadIdx.x) & 31)) + 96)]));
          T_matmul_NN_local[((i_c_outer_inner * 4) + 65)] = (T_matmul_NN_local[((i_c_outer_inner * 4) + 65)] + (data_shared[((((((((int)threadIdx.x) >> 5) * 512) + (i_c_outer_inner * 128)) + (k_outer_inner * 8)) + k_inner) + 32)] * kernel_shared[((((k_outer_inner * 2048) + (k_inner * 256)) + (((int)threadIdx.x) & 31)) + 128)]));
          T_matmul_NN_local[((i_c_outer_inner * 4) + 81)] = (T_matmul_NN_local[((i_c_outer_inner * 4) + 81)] + (data_shared[((((((((int)threadIdx.x) >> 5) * 512) + (i_c_outer_inner * 128)) + (k_outer_inner * 8)) + k_inner) + 32)] * kernel_shared[((((k_outer_inner * 2048) + (k_inner * 256)) + (((int)threadIdx.x) & 31)) + 160)]));
          T_matmul_NN_local[((i_c_outer_inner * 4) + 97)] = (T_matmul_NN_local[((i_c_outer_inner * 4) + 97)] + (data_shared[((((((((int)threadIdx.x) >> 5) * 512) + (i_c_outer_inner * 128)) + (k_outer_inner * 8)) + k_inner) + 32)] * kernel_shared[((((k_outer_inner * 2048) + (k_inner * 256)) + (((int)threadIdx.x) & 31)) + 192)]));
          T_matmul_NN_local[((i_c_outer_inner * 4) + 113)] = (T_matmul_NN_local[((i_c_outer_inner * 4) + 113)] + (data_shared[((((((((int)threadIdx.x) >> 5) * 512) + (i_c_outer_inner * 128)) + (k_outer_inner * 8)) + k_inner) + 32)] * kernel_shared[((((k_outer_inner * 2048) + (k_inner * 256)) + (((int)threadIdx.x) & 31)) + 224)]));
          T_matmul_NN_local[((i_c_outer_inner * 4) + 2)] = (T_matmul_NN_local[((i_c_outer_inner * 4) + 2)] + (data_shared[((((((((int)threadIdx.x) >> 5) * 512) + (i_c_outer_inner * 128)) + (k_outer_inner * 8)) + k_inner) + 64)] * kernel_shared[(((k_outer_inner * 2048) + (k_inner * 256)) + (((int)threadIdx.x) & 31))]));
          T_matmul_NN_local[((i_c_outer_inner * 4) + 18)] = (T_matmul_NN_local[((i_c_outer_inner * 4) + 18)] + (data_shared[((((((((int)threadIdx.x) >> 5) * 512) + (i_c_outer_inner * 128)) + (k_outer_inner * 8)) + k_inner) + 64)] * kernel_shared[((((k_outer_inner * 2048) + (k_inner * 256)) + (((int)threadIdx.x) & 31)) + 32)]));
          T_matmul_NN_local[((i_c_outer_inner * 4) + 34)] = (T_matmul_NN_local[((i_c_outer_inner * 4) + 34)] + (data_shared[((((((((int)threadIdx.x) >> 5) * 512) + (i_c_outer_inner * 128)) + (k_outer_inner * 8)) + k_inner) + 64)] * kernel_shared[((((k_outer_inner * 2048) + (k_inner * 256)) + (((int)threadIdx.x) & 31)) + 64)]));
          T_matmul_NN_local[((i_c_outer_inner * 4) + 50)] = (T_matmul_NN_local[((i_c_outer_inner * 4) + 50)] + (data_shared[((((((((int)threadIdx.x) >> 5) * 512) + (i_c_outer_inner * 128)) + (k_outer_inner * 8)) + k_inner) + 64)] * kernel_shared[((((k_outer_inner * 2048) + (k_inner * 256)) + (((int)threadIdx.x) & 31)) + 96)]));
          T_matmul_NN_local[((i_c_outer_inner * 4) + 66)] = (T_matmul_NN_local[((i_c_outer_inner * 4) + 66)] + (data_shared[((((((((int)threadIdx.x) >> 5) * 512) + (i_c_outer_inner * 128)) + (k_outer_inner * 8)) + k_inner) + 64)] * kernel_shared[((((k_outer_inner * 2048) + (k_inner * 256)) + (((int)threadIdx.x) & 31)) + 128)]));
          T_matmul_NN_local[((i_c_outer_inner * 4) + 82)] = (T_matmul_NN_local[((i_c_outer_inner * 4) + 82)] + (data_shared[((((((((int)threadIdx.x) >> 5) * 512) + (i_c_outer_inner * 128)) + (k_outer_inner * 8)) + k_inner) + 64)] * kernel_shared[((((k_outer_inner * 2048) + (k_inner * 256)) + (((int)threadIdx.x) & 31)) + 160)]));
          T_matmul_NN_local[((i_c_outer_inner * 4) + 98)] = (T_matmul_NN_local[((i_c_outer_inner * 4) + 98)] + (data_shared[((((((((int)threadIdx.x) >> 5) * 512) + (i_c_outer_inner * 128)) + (k_outer_inner * 8)) + k_inner) + 64)] * kernel_shared[((((k_outer_inner * 2048) + (k_inner * 256)) + (((int)threadIdx.x) & 31)) + 192)]));
          T_matmul_NN_local[((i_c_outer_inner * 4) + 114)] = (T_matmul_NN_local[((i_c_outer_inner * 4) + 114)] + (data_shared[((((((((int)threadIdx.x) >> 5) * 512) + (i_c_outer_inner * 128)) + (k_outer_inner * 8)) + k_inner) + 64)] * kernel_shared[((((k_outer_inner * 2048) + (k_inner * 256)) + (((int)threadIdx.x) & 31)) + 224)]));
          T_matmul_NN_local[((i_c_outer_inner * 4) + 3)] = (T_matmul_NN_local[((i_c_outer_inner * 4) + 3)] + (data_shared[((((((((int)threadIdx.x) >> 5) * 512) + (i_c_outer_inner * 128)) + (k_outer_inner * 8)) + k_inner) + 96)] * kernel_shared[(((k_outer_inner * 2048) + (k_inner * 256)) + (((int)threadIdx.x) & 31))]));
          T_matmul_NN_local[((i_c_outer_inner * 4) + 19)] = (T_matmul_NN_local[((i_c_outer_inner * 4) + 19)] + (data_shared[((((((((int)threadIdx.x) >> 5) * 512) + (i_c_outer_inner * 128)) + (k_outer_inner * 8)) + k_inner) + 96)] * kernel_shared[((((k_outer_inner * 2048) + (k_inner * 256)) + (((int)threadIdx.x) & 31)) + 32)]));
          T_matmul_NN_local[((i_c_outer_inner * 4) + 35)] = (T_matmul_NN_local[((i_c_outer_inner * 4) + 35)] + (data_shared[((((((((int)threadIdx.x) >> 5) * 512) + (i_c_outer_inner * 128)) + (k_outer_inner * 8)) + k_inner) + 96)] * kernel_shared[((((k_outer_inner * 2048) + (k_inner * 256)) + (((int)threadIdx.x) & 31)) + 64)]));
          T_matmul_NN_local[((i_c_outer_inner * 4) + 51)] = (T_matmul_NN_local[((i_c_outer_inner * 4) + 51)] + (data_shared[((((((((int)threadIdx.x) >> 5) * 512) + (i_c_outer_inner * 128)) + (k_outer_inner * 8)) + k_inner) + 96)] * kernel_shared[((((k_outer_inner * 2048) + (k_inner * 256)) + (((int)threadIdx.x) & 31)) + 96)]));
          T_matmul_NN_local[((i_c_outer_inner * 4) + 67)] = (T_matmul_NN_local[((i_c_outer_inner * 4) + 67)] + (data_shared[((((((((int)threadIdx.x) >> 5) * 512) + (i_c_outer_inner * 128)) + (k_outer_inner * 8)) + k_inner) + 96)] * kernel_shared[((((k_outer_inner * 2048) + (k_inner * 256)) + (((int)threadIdx.x) & 31)) + 128)]));
          T_matmul_NN_local[((i_c_outer_inner * 4) + 83)] = (T_matmul_NN_local[((i_c_outer_inner * 4) + 83)] + (data_shared[((((((((int)threadIdx.x) >> 5) * 512) + (i_c_outer_inner * 128)) + (k_outer_inner * 8)) + k_inner) + 96)] * kernel_shared[((((k_outer_inner * 2048) + (k_inner * 256)) + (((int)threadIdx.x) & 31)) + 160)]));
          T_matmul_NN_local[((i_c_outer_inner * 4) + 99)] = (T_matmul_NN_local[((i_c_outer_inner * 4) + 99)] + (data_shared[((((((((int)threadIdx.x) >> 5) * 512) + (i_c_outer_inner * 128)) + (k_outer_inner * 8)) + k_inner) + 96)] * kernel_shared[((((k_outer_inner * 2048) + (k_inner * 256)) + (((int)threadIdx.x) & 31)) + 192)]));
          T_matmul_NN_local[((i_c_outer_inner * 4) + 115)] = (T_matmul_NN_local[((i_c_outer_inner * 4) + 115)] + (data_shared[((((((((int)threadIdx.x) >> 5) * 512) + (i_c_outer_inner * 128)) + (k_outer_inner * 8)) + k_inner) + 96)] * kernel_shared[((((k_outer_inner * 2048) + (k_inner * 256)) + (((int)threadIdx.x) & 31)) + 224)]));
        }
      }
    }
  }
  for (int i_inner = 0; i_inner < 16; ++i_inner) {
    T_matmul_NN[((((((((int)blockIdx.x) >> 4) * 262144) + ((((int)threadIdx.x) >> 5) * 65536)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 15) * 256)) + (((int)threadIdx.x) & 31))] = T_matmul_NN_local[i_inner];
    T_matmul_NN[(((((((((int)blockIdx.x) >> 4) * 262144) + ((((int)threadIdx.x) >> 5) * 65536)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 15) * 256)) + (((int)threadIdx.x) & 31)) + 32)] = T_matmul_NN_local[(i_inner + 16)];
    T_matmul_NN[(((((((((int)blockIdx.x) >> 4) * 262144) + ((((int)threadIdx.x) >> 5) * 65536)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 15) * 256)) + (((int)threadIdx.x) & 31)) + 64)] = T_matmul_NN_local[(i_inner + 32)];
    T_matmul_NN[(((((((((int)blockIdx.x) >> 4) * 262144) + ((((int)threadIdx.x) >> 5) * 65536)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 15) * 256)) + (((int)threadIdx.x) & 31)) + 96)] = T_matmul_NN_local[(i_inner + 48)];
    T_matmul_NN[(((((((((int)blockIdx.x) >> 4) * 262144) + ((((int)threadIdx.x) >> 5) * 65536)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 15) * 256)) + (((int)threadIdx.x) & 31)) + 128)] = T_matmul_NN_local[(i_inner + 64)];
    T_matmul_NN[(((((((((int)blockIdx.x) >> 4) * 262144) + ((((int)threadIdx.x) >> 5) * 65536)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 15) * 256)) + (((int)threadIdx.x) & 31)) + 160)] = T_matmul_NN_local[(i_inner + 80)];
    T_matmul_NN[(((((((((int)blockIdx.x) >> 4) * 262144) + ((((int)threadIdx.x) >> 5) * 65536)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 15) * 256)) + (((int)threadIdx.x) & 31)) + 192)] = T_matmul_NN_local[(i_inner + 96)];
    T_matmul_NN[(((((((((int)blockIdx.x) >> 4) * 262144) + ((((int)threadIdx.x) >> 5) * 65536)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 15) * 256)) + (((int)threadIdx.x) & 31)) + 224)] = T_matmul_NN_local[(i_inner + 112)];
  }
}

