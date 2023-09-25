
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
extern "C" __global__ void __launch_bounds__(4) mydwconv_kernel0(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ DepthwiseConv2d) {
  float DepthwiseConv2d_local[896];
  __shared__ float PaddedInput_shared[5760];
  __shared__ float kernel_shared[18];
  for (int b_c_outer_inner_init = 0; b_c_outer_inner_init < 16; ++b_c_outer_inner_init) {
    for (int i_c_outer_inner_init = 0; i_c_outer_inner_init < 4; ++i_c_outer_inner_init) {
      for (int c_c_inner_init = 0; c_c_inner_init < 2; ++c_c_inner_init) {
        DepthwiseConv2d_local[(((b_c_outer_inner_init * 8) + (c_c_inner_init * 4)) + i_c_outer_inner_init)] = 0.000000e+00f;
        DepthwiseConv2d_local[((((b_c_outer_inner_init * 8) + (c_c_inner_init * 4)) + i_c_outer_inner_init) + 128)] = 0.000000e+00f;
        DepthwiseConv2d_local[((((b_c_outer_inner_init * 8) + (c_c_inner_init * 4)) + i_c_outer_inner_init) + 256)] = 0.000000e+00f;
        DepthwiseConv2d_local[((((b_c_outer_inner_init * 8) + (c_c_inner_init * 4)) + i_c_outer_inner_init) + 384)] = 0.000000e+00f;
        DepthwiseConv2d_local[((((b_c_outer_inner_init * 8) + (c_c_inner_init * 4)) + i_c_outer_inner_init) + 512)] = 0.000000e+00f;
        DepthwiseConv2d_local[((((b_c_outer_inner_init * 8) + (c_c_inner_init * 4)) + i_c_outer_inner_init) + 640)] = 0.000000e+00f;
        DepthwiseConv2d_local[((((b_c_outer_inner_init * 8) + (c_c_inner_init * 4)) + i_c_outer_inner_init) + 768)] = 0.000000e+00f;
      }
    }
  }
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer < 1440; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer) {
    PaddedInput_shared[((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 4) + ((int)threadIdx.x))] = (((((1 <= (((((int)blockIdx.x) % 7) * 4) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer % 45) * 2) + (((int)threadIdx.x) >> 1)) / 15))) && ((((((int)blockIdx.x) % 7) * 4) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer % 45) * 2) + (((int)threadIdx.x) >> 1)) / 15)) < 29)) && (1 <= (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 4) + ((int)threadIdx.x)) % 30))) && ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 4) + ((int)threadIdx.x)) % 30) < 29)) ? data[(((((((((((int)blockIdx.x) / 448) * 1605632) + ((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer / 90) * 100352)) + (((((int)blockIdx.x) % 448) / 7) * 1568)) + (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer % 90) / 45) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer % 45) * 2) + (((int)threadIdx.x) >> 1)) / 15) * 28)) + (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 4) + ((int)threadIdx.x)) % 30)) - 29)] : 0.000000e+00f);
  }
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_1 < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_1) {
    if (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_1 * 2) + (((int)threadIdx.x) >> 1)) < 9) {
      kernel_shared[((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_1 * 4) + ((int)threadIdx.x))] = kernel[(((((((int)blockIdx.x) % 448) / 7) * 18) + (ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_1 * 4)) + ((int)threadIdx.x))];
    }
  }
  __syncthreads();
  for (int b_c_outer_inner = 0; b_c_outer_inner < 16; ++b_c_outer_inner) {
    for (int i_c_outer_inner = 0; i_c_outer_inner < 4; ++i_c_outer_inner) {
      for (int di_inner = 0; di_inner < 3; ++di_inner) {
        for (int dj_inner = 0; dj_inner < 3; ++dj_inner) {
          for (int c_c_inner = 0; c_c_inner < 2; ++c_c_inner) {
            DepthwiseConv2d_local[(((b_c_outer_inner * 8) + (c_c_inner * 4)) + i_c_outer_inner)] = (DepthwiseConv2d_local[(((b_c_outer_inner * 8) + (c_c_inner * 4)) + i_c_outer_inner)] + (PaddedInput_shared[((((((b_c_outer_inner * 360) + (c_c_inner * 180)) + (i_c_outer_inner * 30)) + (di_inner * 30)) + ((int)threadIdx.x)) + dj_inner)] * kernel_shared[(((c_c_inner * 9) + (di_inner * 3)) + dj_inner)]));
            DepthwiseConv2d_local[((((b_c_outer_inner * 8) + (c_c_inner * 4)) + i_c_outer_inner) + 128)] = (DepthwiseConv2d_local[((((b_c_outer_inner * 8) + (c_c_inner * 4)) + i_c_outer_inner) + 128)] + (PaddedInput_shared[(((((((b_c_outer_inner * 360) + (c_c_inner * 180)) + (i_c_outer_inner * 30)) + (di_inner * 30)) + ((int)threadIdx.x)) + dj_inner) + 4)] * kernel_shared[(((c_c_inner * 9) + (di_inner * 3)) + dj_inner)]));
            DepthwiseConv2d_local[((((b_c_outer_inner * 8) + (c_c_inner * 4)) + i_c_outer_inner) + 256)] = (DepthwiseConv2d_local[((((b_c_outer_inner * 8) + (c_c_inner * 4)) + i_c_outer_inner) + 256)] + (PaddedInput_shared[(((((((b_c_outer_inner * 360) + (c_c_inner * 180)) + (i_c_outer_inner * 30)) + (di_inner * 30)) + ((int)threadIdx.x)) + dj_inner) + 8)] * kernel_shared[(((c_c_inner * 9) + (di_inner * 3)) + dj_inner)]));
            DepthwiseConv2d_local[((((b_c_outer_inner * 8) + (c_c_inner * 4)) + i_c_outer_inner) + 384)] = (DepthwiseConv2d_local[((((b_c_outer_inner * 8) + (c_c_inner * 4)) + i_c_outer_inner) + 384)] + (PaddedInput_shared[(((((((b_c_outer_inner * 360) + (c_c_inner * 180)) + (i_c_outer_inner * 30)) + (di_inner * 30)) + ((int)threadIdx.x)) + dj_inner) + 12)] * kernel_shared[(((c_c_inner * 9) + (di_inner * 3)) + dj_inner)]));
            DepthwiseConv2d_local[((((b_c_outer_inner * 8) + (c_c_inner * 4)) + i_c_outer_inner) + 512)] = (DepthwiseConv2d_local[((((b_c_outer_inner * 8) + (c_c_inner * 4)) + i_c_outer_inner) + 512)] + (PaddedInput_shared[(((((((b_c_outer_inner * 360) + (c_c_inner * 180)) + (i_c_outer_inner * 30)) + (di_inner * 30)) + ((int)threadIdx.x)) + dj_inner) + 16)] * kernel_shared[(((c_c_inner * 9) + (di_inner * 3)) + dj_inner)]));
            DepthwiseConv2d_local[((((b_c_outer_inner * 8) + (c_c_inner * 4)) + i_c_outer_inner) + 640)] = (DepthwiseConv2d_local[((((b_c_outer_inner * 8) + (c_c_inner * 4)) + i_c_outer_inner) + 640)] + (PaddedInput_shared[(((((((b_c_outer_inner * 360) + (c_c_inner * 180)) + (i_c_outer_inner * 30)) + (di_inner * 30)) + ((int)threadIdx.x)) + dj_inner) + 20)] * kernel_shared[(((c_c_inner * 9) + (di_inner * 3)) + dj_inner)]));
            DepthwiseConv2d_local[((((b_c_outer_inner * 8) + (c_c_inner * 4)) + i_c_outer_inner) + 768)] = (DepthwiseConv2d_local[((((b_c_outer_inner * 8) + (c_c_inner * 4)) + i_c_outer_inner) + 768)] + (PaddedInput_shared[(((((((b_c_outer_inner * 360) + (c_c_inner * 180)) + (i_c_outer_inner * 30)) + (di_inner * 30)) + ((int)threadIdx.x)) + dj_inner) + 24)] * kernel_shared[(((c_c_inner * 9) + (di_inner * 3)) + dj_inner)]));
          }
        }
      }
    }
  }
  for (int b_inner = 0; b_inner < 16; ++b_inner) {
    for (int c_inner = 0; c_inner < 2; ++c_inner) {
      for (int i_inner = 0; i_inner < 4; ++i_inner) {
        DepthwiseConv2d[((((((((((int)blockIdx.x) / 448) * 1605632) + (b_inner * 100352)) + (((((int)blockIdx.x) % 448) / 7) * 1568)) + (c_inner * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (i_inner * 28)) + ((int)threadIdx.x))] = DepthwiseConv2d_local[(((b_inner * 8) + (c_inner * 4)) + i_inner)];
        DepthwiseConv2d[(((((((((((int)blockIdx.x) / 448) * 1605632) + (b_inner * 100352)) + (((((int)blockIdx.x) % 448) / 7) * 1568)) + (c_inner * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (i_inner * 28)) + ((int)threadIdx.x)) + 4)] = DepthwiseConv2d_local[((((b_inner * 8) + (c_inner * 4)) + i_inner) + 128)];
        DepthwiseConv2d[(((((((((((int)blockIdx.x) / 448) * 1605632) + (b_inner * 100352)) + (((((int)blockIdx.x) % 448) / 7) * 1568)) + (c_inner * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (i_inner * 28)) + ((int)threadIdx.x)) + 8)] = DepthwiseConv2d_local[((((b_inner * 8) + (c_inner * 4)) + i_inner) + 256)];
        DepthwiseConv2d[(((((((((((int)blockIdx.x) / 448) * 1605632) + (b_inner * 100352)) + (((((int)blockIdx.x) % 448) / 7) * 1568)) + (c_inner * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (i_inner * 28)) + ((int)threadIdx.x)) + 12)] = DepthwiseConv2d_local[((((b_inner * 8) + (c_inner * 4)) + i_inner) + 384)];
        DepthwiseConv2d[(((((((((((int)blockIdx.x) / 448) * 1605632) + (b_inner * 100352)) + (((((int)blockIdx.x) % 448) / 7) * 1568)) + (c_inner * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (i_inner * 28)) + ((int)threadIdx.x)) + 16)] = DepthwiseConv2d_local[((((b_inner * 8) + (c_inner * 4)) + i_inner) + 512)];
        DepthwiseConv2d[(((((((((((int)blockIdx.x) / 448) * 1605632) + (b_inner * 100352)) + (((((int)blockIdx.x) % 448) / 7) * 1568)) + (c_inner * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (i_inner * 28)) + ((int)threadIdx.x)) + 20)] = DepthwiseConv2d_local[((((b_inner * 8) + (c_inner * 4)) + i_inner) + 640)];
        DepthwiseConv2d[(((((((((((int)blockIdx.x) / 448) * 1605632) + (b_inner * 100352)) + (((((int)blockIdx.x) % 448) / 7) * 1568)) + (c_inner * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (i_inner * 28)) + ((int)threadIdx.x)) + 24)] = DepthwiseConv2d_local[((((b_inner * 8) + (c_inner * 4)) + i_inner) + 768)];
      }
    }
  }
}

