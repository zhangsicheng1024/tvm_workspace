
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
  float T_matmul_NN_local[512];
  __shared__ float data_shared[8192];
  __shared__ float kernel_shared[512];
  for (int j_c_outer_inner_init = 0; j_c_outer_inner_init < 32; ++j_c_outer_inner_init) {
    T_matmul_NN_local[j_c_outer_inner_init] = 0.000000e+00f;
    T_matmul_NN_local[(j_c_outer_inner_init + 128)] = 0.000000e+00f;
    T_matmul_NN_local[(j_c_outer_inner_init + 256)] = 0.000000e+00f;
    T_matmul_NN_local[(j_c_outer_inner_init + 384)] = 0.000000e+00f;
    T_matmul_NN_local[(j_c_outer_inner_init + 32)] = 0.000000e+00f;
    T_matmul_NN_local[(j_c_outer_inner_init + 160)] = 0.000000e+00f;
    T_matmul_NN_local[(j_c_outer_inner_init + 288)] = 0.000000e+00f;
    T_matmul_NN_local[(j_c_outer_inner_init + 416)] = 0.000000e+00f;
    T_matmul_NN_local[(j_c_outer_inner_init + 64)] = 0.000000e+00f;
    T_matmul_NN_local[(j_c_outer_inner_init + 192)] = 0.000000e+00f;
    T_matmul_NN_local[(j_c_outer_inner_init + 320)] = 0.000000e+00f;
    T_matmul_NN_local[(j_c_outer_inner_init + 448)] = 0.000000e+00f;
    T_matmul_NN_local[(j_c_outer_inner_init + 96)] = 0.000000e+00f;
    T_matmul_NN_local[(j_c_outer_inner_init + 224)] = 0.000000e+00f;
    T_matmul_NN_local[(j_c_outer_inner_init + 352)] = 0.000000e+00f;
    T_matmul_NN_local[(j_c_outer_inner_init + 480)] = 0.000000e+00f;
  }
  for (int k_outer_outer = 0; k_outer_outer < 64; ++k_outer_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_outer_outer = 0; ax0_ax1_fused_outer_outer < 256; ++ax0_ax1_fused_outer_outer) {
      data_shared[((ax0_ax1_fused_outer_outer * 32) + ((int)threadIdx.x))] = data[((((ax0_ax1_fused_outer_outer * 2048) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer_outer * 16)) + (((int)threadIdx.x) & 15))];
    }
    kernel_shared[((int)threadIdx.x)] = kernel[(((k_outer_outer * 65536) + (((int)blockIdx.x) * 32)) + ((int)threadIdx.x))];
    kernel_shared[(((int)threadIdx.x) + 32)] = kernel[((((k_outer_outer * 65536) + (((int)blockIdx.x) * 32)) + ((int)threadIdx.x)) + 4096)];
    kernel_shared[(((int)threadIdx.x) + 64)] = kernel[((((k_outer_outer * 65536) + (((int)blockIdx.x) * 32)) + ((int)threadIdx.x)) + 8192)];
    kernel_shared[(((int)threadIdx.x) + 96)] = kernel[((((k_outer_outer * 65536) + (((int)blockIdx.x) * 32)) + ((int)threadIdx.x)) + 12288)];
    kernel_shared[(((int)threadIdx.x) + 128)] = kernel[((((k_outer_outer * 65536) + (((int)blockIdx.x) * 32)) + ((int)threadIdx.x)) + 16384)];
    kernel_shared[(((int)threadIdx.x) + 160)] = kernel[((((k_outer_outer * 65536) + (((int)blockIdx.x) * 32)) + ((int)threadIdx.x)) + 20480)];
    kernel_shared[(((int)threadIdx.x) + 192)] = kernel[((((k_outer_outer * 65536) + (((int)blockIdx.x) * 32)) + ((int)threadIdx.x)) + 24576)];
    kernel_shared[(((int)threadIdx.x) + 224)] = kernel[((((k_outer_outer * 65536) + (((int)blockIdx.x) * 32)) + ((int)threadIdx.x)) + 28672)];
    kernel_shared[(((int)threadIdx.x) + 256)] = kernel[((((k_outer_outer * 65536) + (((int)blockIdx.x) * 32)) + ((int)threadIdx.x)) + 32768)];
    kernel_shared[(((int)threadIdx.x) + 288)] = kernel[((((k_outer_outer * 65536) + (((int)blockIdx.x) * 32)) + ((int)threadIdx.x)) + 36864)];
    kernel_shared[(((int)threadIdx.x) + 320)] = kernel[((((k_outer_outer * 65536) + (((int)blockIdx.x) * 32)) + ((int)threadIdx.x)) + 40960)];
    kernel_shared[(((int)threadIdx.x) + 352)] = kernel[((((k_outer_outer * 65536) + (((int)blockIdx.x) * 32)) + ((int)threadIdx.x)) + 45056)];
    kernel_shared[(((int)threadIdx.x) + 384)] = kernel[((((k_outer_outer * 65536) + (((int)blockIdx.x) * 32)) + ((int)threadIdx.x)) + 49152)];
    kernel_shared[(((int)threadIdx.x) + 416)] = kernel[((((k_outer_outer * 65536) + (((int)blockIdx.x) * 32)) + ((int)threadIdx.x)) + 53248)];
    kernel_shared[(((int)threadIdx.x) + 448)] = kernel[((((k_outer_outer * 65536) + (((int)blockIdx.x) * 32)) + ((int)threadIdx.x)) + 57344)];
    kernel_shared[(((int)threadIdx.x) + 480)] = kernel[((((k_outer_outer * 65536) + (((int)blockIdx.x) * 32)) + ((int)threadIdx.x)) + 61440)];
    __syncthreads();
    for (int k_outer_inner = 0; k_outer_inner < 16; ++k_outer_inner) {
      for (int j_c_outer_inner = 0; j_c_outer_inner < 32; ++j_c_outer_inner) {
        T_matmul_NN_local[j_c_outer_inner] = (T_matmul_NN_local[j_c_outer_inner] + (data_shared[((((int)threadIdx.x) * 64) + k_outer_inner)] * kernel_shared[((k_outer_inner * 32) + j_c_outer_inner)]));
        T_matmul_NN_local[(j_c_outer_inner + 128)] = (T_matmul_NN_local[(j_c_outer_inner + 128)] + (data_shared[(((((int)threadIdx.x) * 64) + k_outer_inner) + 2048)] * kernel_shared[((k_outer_inner * 32) + j_c_outer_inner)]));
        T_matmul_NN_local[(j_c_outer_inner + 256)] = (T_matmul_NN_local[(j_c_outer_inner + 256)] + (data_shared[(((((int)threadIdx.x) * 64) + k_outer_inner) + 4096)] * kernel_shared[((k_outer_inner * 32) + j_c_outer_inner)]));
        T_matmul_NN_local[(j_c_outer_inner + 384)] = (T_matmul_NN_local[(j_c_outer_inner + 384)] + (data_shared[(((((int)threadIdx.x) * 64) + k_outer_inner) + 6144)] * kernel_shared[((k_outer_inner * 32) + j_c_outer_inner)]));
        T_matmul_NN_local[(j_c_outer_inner + 32)] = (T_matmul_NN_local[(j_c_outer_inner + 32)] + (data_shared[(((((int)threadIdx.x) * 64) + k_outer_inner) + 16)] * kernel_shared[((k_outer_inner * 32) + j_c_outer_inner)]));
        T_matmul_NN_local[(j_c_outer_inner + 160)] = (T_matmul_NN_local[(j_c_outer_inner + 160)] + (data_shared[(((((int)threadIdx.x) * 64) + k_outer_inner) + 2064)] * kernel_shared[((k_outer_inner * 32) + j_c_outer_inner)]));
        T_matmul_NN_local[(j_c_outer_inner + 288)] = (T_matmul_NN_local[(j_c_outer_inner + 288)] + (data_shared[(((((int)threadIdx.x) * 64) + k_outer_inner) + 4112)] * kernel_shared[((k_outer_inner * 32) + j_c_outer_inner)]));
        T_matmul_NN_local[(j_c_outer_inner + 416)] = (T_matmul_NN_local[(j_c_outer_inner + 416)] + (data_shared[(((((int)threadIdx.x) * 64) + k_outer_inner) + 6160)] * kernel_shared[((k_outer_inner * 32) + j_c_outer_inner)]));
        T_matmul_NN_local[(j_c_outer_inner + 64)] = (T_matmul_NN_local[(j_c_outer_inner + 64)] + (data_shared[(((((int)threadIdx.x) * 64) + k_outer_inner) + 32)] * kernel_shared[((k_outer_inner * 32) + j_c_outer_inner)]));
        T_matmul_NN_local[(j_c_outer_inner + 192)] = (T_matmul_NN_local[(j_c_outer_inner + 192)] + (data_shared[(((((int)threadIdx.x) * 64) + k_outer_inner) + 2080)] * kernel_shared[((k_outer_inner * 32) + j_c_outer_inner)]));
        T_matmul_NN_local[(j_c_outer_inner + 320)] = (T_matmul_NN_local[(j_c_outer_inner + 320)] + (data_shared[(((((int)threadIdx.x) * 64) + k_outer_inner) + 4128)] * kernel_shared[((k_outer_inner * 32) + j_c_outer_inner)]));
        T_matmul_NN_local[(j_c_outer_inner + 448)] = (T_matmul_NN_local[(j_c_outer_inner + 448)] + (data_shared[(((((int)threadIdx.x) * 64) + k_outer_inner) + 6176)] * kernel_shared[((k_outer_inner * 32) + j_c_outer_inner)]));
        T_matmul_NN_local[(j_c_outer_inner + 96)] = (T_matmul_NN_local[(j_c_outer_inner + 96)] + (data_shared[(((((int)threadIdx.x) * 64) + k_outer_inner) + 48)] * kernel_shared[((k_outer_inner * 32) + j_c_outer_inner)]));
        T_matmul_NN_local[(j_c_outer_inner + 224)] = (T_matmul_NN_local[(j_c_outer_inner + 224)] + (data_shared[(((((int)threadIdx.x) * 64) + k_outer_inner) + 2096)] * kernel_shared[((k_outer_inner * 32) + j_c_outer_inner)]));
        T_matmul_NN_local[(j_c_outer_inner + 352)] = (T_matmul_NN_local[(j_c_outer_inner + 352)] + (data_shared[(((((int)threadIdx.x) * 64) + k_outer_inner) + 4144)] * kernel_shared[((k_outer_inner * 32) + j_c_outer_inner)]));
        T_matmul_NN_local[(j_c_outer_inner + 480)] = (T_matmul_NN_local[(j_c_outer_inner + 480)] + (data_shared[(((((int)threadIdx.x) * 64) + k_outer_inner) + 6192)] * kernel_shared[((k_outer_inner * 32) + j_c_outer_inner)]));
      }
    }
  }
  for (int i_inner = 0; i_inner < 4; ++i_inner) {
    for (int j_inner = 0; j_inner < 32; ++j_inner) {
      T_matmul_NN[((((((int)threadIdx.x) * 16384) + (i_inner * 4096)) + (((int)blockIdx.x) * 32)) + j_inner)] = T_matmul_NN_local[((i_inner * 32) + j_inner)];
      T_matmul_NN[(((((((int)threadIdx.x) * 16384) + (i_inner * 4096)) + (((int)blockIdx.x) * 32)) + j_inner) + 524288)] = T_matmul_NN_local[(((i_inner * 32) + j_inner) + 128)];
      T_matmul_NN[(((((((int)threadIdx.x) * 16384) + (i_inner * 4096)) + (((int)blockIdx.x) * 32)) + j_inner) + 1048576)] = T_matmul_NN_local[(((i_inner * 32) + j_inner) + 256)];
      T_matmul_NN[(((((((int)threadIdx.x) * 16384) + (i_inner * 4096)) + (((int)blockIdx.x) * 32)) + j_inner) + 1572864)] = T_matmul_NN_local[(((i_inner * 32) + j_inner) + 384)];
    }
  }
}

