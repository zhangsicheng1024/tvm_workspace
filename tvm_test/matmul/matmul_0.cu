// 16384 1 1
// 128 1 1

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
  // block:

  // load [256*1024] from data (256 rows) (65536/256=256)
  // load [1024*64] from kernel (64 cols) (4096/64=64)
  // compute [256*1024] x [1024*64] = [256*64]

  // K for 128: 
  // load [256*8] (data_shared[2048]) (* 128)
  // load [8*64] (kernel_shared[512]) (128 *)
  // compute [256*8] x [8*64] = [256*64] (128 threads * T_matmul_NN_local[128])

  // write back [256*64] to T_matmul_NN   (65536/256) * (4096/64) = 256*64 = 16384 = gridDim

  float T_matmul_NN_local[128];         // result
  __shared__ float data_shared[2048];   // data shared memory
  __shared__ float kernel_shared[512];  // kernel shared memory

  // Init
  for (int i_c_inner_init = 0; i_c_inner_init < 4; ++i_c_inner_init) {
    for (int j_c_inner_init = 0; j_c_inner_init < 8; ++j_c_inner_init) {
      T_matmul_NN_local[(((i_c_inner_init * 8) + j_c_inner_init) + 0 )] = 0.000000e+00f;
      T_matmul_NN_local[(((i_c_inner_init * 8) + j_c_inner_init) + 32)] = 0.000000e+00f;
      T_matmul_NN_local[(((i_c_inner_init * 8) + j_c_inner_init) + 64)] = 0.000000e+00f;
      T_matmul_NN_local[(((i_c_inner_init * 8) + j_c_inner_init) + 96)] = 0.000000e+00f;
    }
  }

  // [4, 1024] x [1024, 8]
  // K 1024 / 128 = 8

  for (int k_outer_outer = 0; k_outer_outer < 128; ++k_outer_outer) {
    __syncthreads();

    // data -> data shared
    // 1 thread: 2 float 
    // 128 threads: tid[:2]*1024(K), tid[1:0]*2, 128(32*4) threads -> [32*8] from data
    // 8 line: 32768 = 32*1024 -> 8 * [32*8] -> [256*8] from data (data_shared[2048])
    // block-for: k_outer_outer[0-128)*8 -> [256*8] * 128 -> [256*1024] from data -> 256 rows from data
    // M: line_count * 2^(tid_high) = 8 * 2^4 = 8*16 = 128
    *(float2*)(data_shared + ((((int)threadIdx.x) * 2) + 0   )) = *(float2*)(data + ((((((((int)blockIdx.x) >> 6) * 262144) + ((((int)threadIdx.x) >> 2) * 1024)) + (k_outer_outer * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 0     ));
    *(float2*)(data_shared + ((((int)threadIdx.x) * 2) + 256 )) = *(float2*)(data + ((((((((int)blockIdx.x) >> 6) * 262144) + ((((int)threadIdx.x) >> 2) * 1024)) + (k_outer_outer * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 32768 ));
    *(float2*)(data_shared + ((((int)threadIdx.x) * 2) + 512 )) = *(float2*)(data + ((((((((int)blockIdx.x) >> 6) * 262144) + ((((int)threadIdx.x) >> 2) * 1024)) + (k_outer_outer * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 65536 ));
    *(float2*)(data_shared + ((((int)threadIdx.x) * 2) + 768 )) = *(float2*)(data + ((((((((int)blockIdx.x) >> 6) * 262144) + ((((int)threadIdx.x) >> 2) * 1024)) + (k_outer_outer * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 98304 ));
    *(float2*)(data_shared + ((((int)threadIdx.x) * 2) + 1024)) = *(float2*)(data + ((((((((int)blockIdx.x) >> 6) * 262144) + ((((int)threadIdx.x) >> 2) * 1024)) + (k_outer_outer * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 131072));
    *(float2*)(data_shared + ((((int)threadIdx.x) * 2) + 1280)) = *(float2*)(data + ((((((((int)blockIdx.x) >> 6) * 262144) + ((((int)threadIdx.x) >> 2) * 1024)) + (k_outer_outer * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 163840));
    *(float2*)(data_shared + ((((int)threadIdx.x) * 2) + 1536)) = *(float2*)(data + ((((((((int)blockIdx.x) >> 6) * 262144) + ((((int)threadIdx.x) >> 2) * 1024)) + (k_outer_outer * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 196608));
    *(float2*)(data_shared + ((((int)threadIdx.x) * 2) + 1792)) = *(float2*)(data + ((((((((int)blockIdx.x) >> 6) * 262144) + ((((int)threadIdx.x) >> 2) * 1024)) + (k_outer_outer * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 229376));
    
    // kernel -> kernel shared  4*128 = 512
    // 1 thread: 4 float
    // 128 threads: tid[:4]*4096(N), tid[3:0]*4, 128(8*16) threads -> [8*64] from kernel (kernel_shared[512])
    // block-for: k_outer_outer[0-128)*32768(8*4096(N)) -> 128 * [8*64] -> [1024*64] from kernel -> 64 cols from kernel
    // N: g * line_count * 2^(tid_low) = 1 * 1 * 2^4 = 16
    *(float4*)(kernel_shared + (((int)threadIdx.x) * 4)) = *(float4*)(kernel + ((((k_outer_outer * 32768) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.x) & 63) * 64)) + ((((int)threadIdx.x) & 15) * 4)));
    __syncthreads();
    
    // compute
    // 1 line: [4*8]
    // 4 line: 4 * [4*8]
    // 128 threads: 128 * 4 * [4*8]

    // 
    for (int k_inner = 0; k_inner < 8; ++k_inner) {
      for (int i_c_inner = 0; i_c_inner < 4; ++i_c_inner) {
        for (int j_c_inner = 0; j_c_inner < 8; ++j_c_inner) {
          T_matmul_NN_local[(((i_c_inner * 8) + j_c_inner) + 0 )] += (/*T_matmul_NN_local[(((i_c_inner * 8) + j_c_inner) + 0 )]*/ + (data_shared[(((((((int)threadIdx.x) >> 3) * 32) + (i_c_inner * 8)) + k_inner) + 0   )]   * kernel_shared[(((k_inner * 64) + ((((int)threadIdx.x) & 7) * 8)) + j_c_inner)]));
          T_matmul_NN_local[(((i_c_inner * 8) + j_c_inner) + 32)] += (/*T_matmul_NN_local[(((i_c_inner * 8) + j_c_inner) + 32)]*/ + (data_shared[(((((((int)threadIdx.x) >> 3) * 32) + (i_c_inner * 8)) + k_inner) + 512 )]   * kernel_shared[(((k_inner * 64) + ((((int)threadIdx.x) & 7) * 8)) + j_c_inner)]));
          T_matmul_NN_local[(((i_c_inner * 8) + j_c_inner) + 64)] += (/*T_matmul_NN_local[(((i_c_inner * 8) + j_c_inner) + 64)]*/ + (data_shared[(((((((int)threadIdx.x) >> 3) * 32) + (i_c_inner * 8)) + k_inner) + 1024)]   * kernel_shared[(((k_inner * 64) + ((((int)threadIdx.x) & 7) * 8)) + j_c_inner)]));
          T_matmul_NN_local[(((i_c_inner * 8) + j_c_inner) + 96)] += (/*T_matmul_NN_local[(((i_c_inner * 8) + j_c_inner) + 96)]*/ + (data_shared[(((((((int)threadIdx.x) >> 3) * 32) + (i_c_inner * 8)) + k_inner) + 1536)]   * kernel_shared[(((k_inner * 64) + ((((int)threadIdx.x) & 7) * 8)) + j_c_inner)]));
        }
      }
    }
  }

  // Write back
  // j_inner [0,8) -> [1*8] 
  // i_inner [0,4) -> [4*8]
  // 1 thread: [4*8] float
  // 128 threads: tid[:3]*16384(4*4096), tid[2:0]*8, 128(16*8) threads -> [16*8] x [4*8] -> [64*64]
  // 4 line: 262144=64*4096 -> 4 * [64*64] -> [256*64] to T_matmul_NN
  for (int i_inner = 0; i_inner < 4; ++i_inner) {
    for (int j_inner = 0; j_inner < 8; ++j_inner) {
      T_matmul_NN[((((((((((int)blockIdx.x) >> 6) * 1048576) + ((((int)threadIdx.x) >> 3) * 16384)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 63) * 64)) + ((((int)threadIdx.x) & 7) * 8)) + j_inner) + 0     )] = T_matmul_NN_local[(((i_inner * 8) + j_inner) + 0 )];
      T_matmul_NN[((((((((((int)blockIdx.x) >> 6) * 1048576) + ((((int)threadIdx.x) >> 3) * 16384)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 63) * 64)) + ((((int)threadIdx.x) & 7) * 8)) + j_inner) + 262144)] = T_matmul_NN_local[(((i_inner * 8) + j_inner) + 32)];
      T_matmul_NN[((((((((((int)blockIdx.x) >> 6) * 1048576) + ((((int)threadIdx.x) >> 3) * 16384)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 63) * 64)) + ((((int)threadIdx.x) & 7) * 8)) + j_inner) + 524288)] = T_matmul_NN_local[(((i_inner * 8) + j_inner) + 64)];
      T_matmul_NN[((((((((((int)blockIdx.x) >> 6) * 1048576) + ((((int)threadIdx.x) >> 3) * 16384)) + (i_inner * 4096)) + ((((int)blockIdx.x) & 63) * 64)) + ((((int)threadIdx.x) & 7) * 8)) + j_inner) + 786432)] = T_matmul_NN_local[(((i_inner * 8) + j_inner) + 96)];
    }
  }
}

