// 1024 1 1
// 256 1 1

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
extern "C" __global__ void __launch_bounds__(256) mymatmul_kernel0(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ T_matmul_NN) {
  // block:

  // load [128*1024] from data (128 rows) (512/128=4)
  // load [1024*16] from kernel (16 cols) (4096/16=256)
  // compute [128*1024] x [1024*16] = [128*16]

  // K for 64: 
  // load [128*16] (data_shared[2048]) (* 64)
  // load [16*16] (kernel_shared[512]) (64 *)
  // compute [128*16] x [16*16] = [128*16] (256 threads * T_matmul_NN_local[8])
  float T_matmul_NN_local[8];
  __shared__ float data_shared[2048];
  __shared__ float kernel_shared[256];

  for (int j_c_outer_inner_init = 0; j_c_outer_inner_init < 4; ++j_c_outer_inner_init) {
    T_matmul_NN_local[j_c_outer_inner_init] = 0.000000e+00f;
    T_matmul_NN_local[(j_c_outer_inner_init + 4)] = 0.000000e+00f;
  }

  for (int k_outer_outer = 0; k_outer_outer < 64; ++k_outer_outer) {
    __syncthreads();
    // 1 thread: 1 float 
    // 128 threads: tid[:4]*1024(K), tid[3:0]*1, 256(16*16) threads -> [16*16] from data
    // 8 line: 16384 = 16*1024 -> 8 * [16*16] -> [128*16] from data (data_shared[2048])
    // block-for: k_outer_outer[0-64)*16 -> [128*16] * 64 -> [128*1024] from data -> 128 rows from data
    // M_size: line_count * 2^(tid_high) = 8 * 2^4 = 8*16 = 128
    // M = 512 / 128 = 4
    for (int ax0_ax1_fused_outer_outer = 0; ax0_ax1_fused_outer_outer < 8; ++ax0_ax1_fused_outer_outer) {
      data_shared[((ax0_ax1_fused_outer_outer * 256) + ((int)threadIdx.x))] = data[((((((((int)blockIdx.x) >> 8) * 131072) + (ax0_ax1_fused_outer_outer * 16384)) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer_outer * 16)) + (((int)threadIdx.x) & 15))];
    }

    // kernel -> kernel shared
    // 1 thread: 1 float
    // 128 threads: tid[:4]*4096(N), tid[3:0]*1, 256(16*16) threads -> [16*16] from kernel (kernel_shared[256])
    // block-for: k_outer_outer[0-64)*65536(16*4096(N)) -> 64 * [16*16] -> [1024*16] from kernel -> 16 cols from kernel
    // N_size: g * line_count * 2^(tid_low) = 1 * 1 * 2^4 = 16
    // N = 4096 / 16 = 256
    kernel_shared[((int)threadIdx.x)] = kernel[((((k_outer_outer * 65536) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.x) & 255) * 16)) + (((int)threadIdx.x) & 15))];
    __syncthreads();

    // M: line_count * 2^(tid_high) = 1 * 2^6 = 64
    // N: g * line_count * 2^(tid_low) = 1 * 1 * 2^2 = 4
    for (int k_outer_inner = 0; k_outer_inner < 4; ++k_outer_inner) {
      for (int j_c_outer_inner = 0; j_c_outer_inner < 4; ++j_c_outer_inner) {
        for (int k_inner = 0; k_inner < 4; ++k_inner) {
          T_matmul_NN_local[(j_c_outer_inner + 0)] = (T_matmul_NN_local[(j_c_outer_inner + 0)] + (data_shared[(((((((int)threadIdx.x) >> 2) * 16) + (k_outer_inner * 4)) + k_inner) + 0   )] * kernel_shared[((((k_outer_inner * 64) + (k_inner * 16)) + ((((int)threadIdx.x) & 3) * 4)) + j_c_outer_inner)]));
          T_matmul_NN_local[(j_c_outer_inner + 4)] = (T_matmul_NN_local[(j_c_outer_inner + 4)] + (data_shared[(((((((int)threadIdx.x) >> 2) * 16) + (k_outer_inner * 4)) + k_inner) + 1024)] * kernel_shared[((((k_outer_inner * 64) + (k_inner * 16)) + ((((int)threadIdx.x) & 3) * 4)) + j_c_outer_inner)]));
        }
      }
    }
  }

  for (int j_inner = 0; j_inner < 4; ++j_inner) {
    T_matmul_NN[((((((((int)blockIdx.x) >> 8) * 524288) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)blockIdx.x) & 255) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + j_inner)] = T_matmul_NN_local[j_inner];
    T_matmul_NN[(((((((((int)blockIdx.x) >> 8) * 524288) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)blockIdx.x) & 255) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + j_inner) + 262144)] = T_matmul_NN_local[(j_inner + 4)];
  }
}

