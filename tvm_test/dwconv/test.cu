
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
extern "C" __global__ void __launch_bounds__(56) mydwconv_kernel0(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ DepthwiseConv2d) {
  float DepthwiseConv2d_local[16];
  __shared__ float PaddedInput_shared[896];
  __shared__ float kernel_shared[8];
  for (int b_c_inner_init = 0; b_c_inner_init < 2; ++b_c_inner_init) {
    for (int c_c_inner_init = 0; c_c_inner_init < 4; ++c_c_inner_init) {
      DepthwiseConv2d_local[((b_c_inner_init * 4) + c_c_inner_init)] = 0.000000e+00f;
      DepthwiseConv2d_local[(((b_c_inner_init * 4) + c_c_inner_init) + 8)] = 0.000000e+00f;
    }
  }
  for (int di_outer_outer = 0; di_outer_outer < 3; ++di_outer_outer) {
    for (int dj_outer_outer = 0; dj_outer_outer < 3; ++dj_outer_outer) {
      __syncthreads();
      for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer < 16; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer) {
        PaddedInput_shared[((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 56) + ((int)threadIdx.x))] = (((((1 <= (di_outer_outer + (((int)blockIdx.x) % 28))) && ((di_outer_outer + (((int)blockIdx.x) % 28)) < 29)) && (1 <= (dj_outer_outer + (((int)threadIdx.x) % 28)))) && ((dj_outer_outer + (((int)threadIdx.x) % 28)) < 29)) ? data[(((((((((((((int)blockIdx.x) / 448) * 401408) + ((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer >> 2) * 100352)) + (((((int)blockIdx.x) % 448) / 28) * 6272)) + ((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer & 3) * 1568)) + ((((int)threadIdx.x) / 28) * 784)) + (di_outer_outer * 28)) + ((((int)blockIdx.x) % 28) * 28)) + dj_outer_outer) + (((int)threadIdx.x) % 28)) - 29)] : 0.000000e+00f);
      }
      if (((int)threadIdx.x) < 8) {
        kernel_shared[((int)threadIdx.x)] = kernel[((((((((int)blockIdx.x) % 448) / 28) * 72) + (((int)threadIdx.x) * 9)) + (di_outer_outer * 3)) + dj_outer_outer)];
      }
      __syncthreads();
      for (int b_c_inner = 0; b_c_inner < 2; ++b_c_inner) {
        for (int c_c_inner = 0; c_c_inner < 4; ++c_c_inner) {
          DepthwiseConv2d_local[((b_c_inner * 4) + c_c_inner)] = (DepthwiseConv2d_local[((b_c_inner * 4) + c_c_inner)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 28) * 448) + (b_c_inner * 224)) + (c_c_inner * 28)) + (((int)threadIdx.x) % 28))] * kernel_shared[c_c_inner]));
          DepthwiseConv2d_local[(((b_c_inner * 4) + c_c_inner) + 8)] = (DepthwiseConv2d_local[(((b_c_inner * 4) + c_c_inner) + 8)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 28) * 448) + (b_c_inner * 224)) + (c_c_inner * 28)) + (((int)threadIdx.x) % 28)) + 112)] * kernel_shared[(c_c_inner + 4)]));
        }
      }
    }
  }
  for (int b_inner = 0; b_inner < 2; ++b_inner) {
    for (int c_inner = 0; c_inner < 4; ++c_inner) {
      DepthwiseConv2d[((((((((((int)blockIdx.x) / 448) * 401408) + ((((int)threadIdx.x) / 28) * 200704)) + (b_inner * 100352)) + (((((int)blockIdx.x) % 448) / 28) * 6272)) + (c_inner * 784)) + ((((int)blockIdx.x) % 28) * 28)) + (((int)threadIdx.x) % 28))] = DepthwiseConv2d_local[((b_inner * 4) + c_inner)];
      DepthwiseConv2d[(((((((((((int)blockIdx.x) / 448) * 401408) + ((((int)threadIdx.x) / 28) * 200704)) + (b_inner * 100352)) + (((((int)blockIdx.x) % 448) / 28) * 6272)) + (c_inner * 784)) + ((((int)blockIdx.x) % 28) * 28)) + (((int)threadIdx.x) % 28)) + 3136)] = DepthwiseConv2d_local[(((b_inner * 4) + c_inner) + 8)];
    }
  }
}

