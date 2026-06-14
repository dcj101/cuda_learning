#include <stdio.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include "cuda_runtime.h"
#include <cstdlib>
#include <algorithm>
#include <cmath>

// ==================== 控制变量 ====================
constexpr bool USE_TANH_APPROX = false;   // 是否使用 tanh.approx.f32 指令
constexpr bool USE_APPLY2     = false;    // 是否使用 half2 向量化 apply2
// =================================================

template <typename T, int Size>
struct alignas(sizeof(T) * Size) AlignedVector {
  T val[Size];
  __host__ __device__ inline const T& operator[](int i) const { return val[i]; }
  __host__ __device__ inline T& operator[](int i) { return val[i]; }
};

__device__ float TanhApprox(float x) {
  if constexpr (USE_TANH_APPROX) {
    float r;
    asm("tanh.approx.f32 %0,%1; \n\t" : "=f"(r) : "f"(x));
    return r;
  } else {
    return tanhf(x);
  }
}

template<typename T>
struct GeluFunctor {
  static constexpr T alpha = static_cast<T>(0.7978845608028654);
  static constexpr T beta = static_cast<T>(0.044714998453855515);

  __host__ __device__ GeluFunctor() {}

  __host__ __device__ T operator()(T x) const {
    const T half = static_cast<T>(0.5);
    const T one = static_cast<T>(1);
    const T tanh_in = alpha * (x + beta * x * x * x);
    return half * x * (one + tanhf(tanh_in));
  }
};

template<>
struct GeluFunctor<half> {
  static constexpr float alpha = GeluFunctor<float>::alpha;
  static constexpr float beta = GeluFunctor<float>::beta;
  GeluFunctor<float> float_functor;

  __device__ GeluFunctor() {}

  __device__ half operator()(const half x) const {
    if constexpr (USE_APPLY2) {
      const float tanh_in =
         __half2float(__float2half_rn(alpha) * (x + __float2half_rn(beta) * x * x * x));
      const float tanh_out = TanhApprox(tanh_in);
      return __float2half_rn(0.5f) * x * (__float2half_rn(1.0f) + __float2half_rn(tanh_out));
    } else {
      return static_cast<half>(float_functor(static_cast<float>(x)));
    }
  }

  __device__ void apply2(half* y, const half* x) const {
    const half2 x2 = *(reinterpret_cast<const half2*>(x));
    const float2 tanh_in = __half22float2(
       __hmul2(__float2half2_rn(alpha),
               __hadd2(x2, __hmul2(__hmul2(__hmul2(__float2half2_rn(beta), x2), x2), x2))));
    float2 tanh_out;
    tanh_out.x = TanhApprox(tanh_in.x);
    tanh_out.y = TanhApprox(tanh_in.y);
    const half2 y2 = __hmul2(__hmul2(__float2half2_rn(0.5F), x2),
                                    __hadd2(__float2half2_rn(1.0F), __float22half2_rn(tanh_out)));
    *reinterpret_cast<half2*>(y) = y2;
  }
};

template <int VecSize>
__global__ void FP16GeluCUDAKernel(const __half* x, __half* y, int n) {
  int offset = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x) * VecSize;
  int stride = static_cast<int>(blockDim.x * gridDim.x) * VecSize;
  GeluFunctor<half> gelu_fwd;
  __half y_reg[VecSize];
  using ArrT = AlignedVector<__half, VecSize>;

  for (; offset < n; offset += stride) {
    const __half* in = x + offset;

    if (VecSize == 1){
        y_reg[0] = gelu_fwd(in[0]);
    } else {
      if constexpr (USE_APPLY2) {
        for (int i = 0; i < VecSize; i += 2) {
          gelu_fwd.apply2(y_reg + i, in + i);   // 已修正：写回寄存器而非全局内存
        }
      } else {
        for (int i = 0; i < VecSize; i++) {
            y_reg[i] = gelu_fwd(in[i]);
        }
      }
    }
    *reinterpret_cast<ArrT*>(y + offset) = *reinterpret_cast<ArrT*>(y_reg);
  }
}

int main() {
    int n = 1000;

    constexpr size_t kAlignment = alignof(AlignedVector<__half, 8>);
    __half *x = static_cast<__half*>(aligned_alloc(kAlignment, n * sizeof(__half)));
    __half *y = static_cast<__half*>(aligned_alloc(kAlignment, n * sizeof(__half)));
    __half *y_ref = static_cast<__half*>(aligned_alloc(kAlignment, n * sizeof(__half)));

    for (int i = 0; i < n; i++) {
      x[i] = static_cast<__half>(static_cast<float>(i));
    }

    // CPU 参考值（现在 GeluFunctor<float> 可同时用于主机和设备）
    GeluFunctor<float> cpu_gelu;
    for (int i = 0; i < n; i++) {
      float val = static_cast<float>(x[i]);
      y_ref[i] = static_cast<__half>(cpu_gelu(val));
    }

    __half *d_x, *d_y;
    cudaMalloc((void **)&d_x, n * sizeof(__half));
    cudaMalloc((void **)&d_y, n * sizeof(__half));
    cudaMemcpy(d_x, x, sizeof(__half) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, sizeof(__half) * n, cudaMemcpyHostToDevice);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    auto is_aligned = [](const void* p, int alignment) {
        return reinterpret_cast<uintptr_t>(p) % alignment == 0;
    };

    printf("kAlignment = %zu\n", kAlignment);
    printf("USE_TANH_APPROX = %d, USE_APPLY2 = %d\n", USE_TANH_APPROX, USE_APPLY2);

    if (n % 8 == 0 && is_aligned(x, kAlignment) && is_aligned(y, kAlignment)) {
      int thread = std::min<int>(512, deviceProp.maxThreadsPerBlock);
      int block = (n / 8 + thread - 1) / thread;
      block = std::min<int>(block, deviceProp.maxGridSize[0]);

      cudaEvent_t start, stop;
      cudaEventCreate(&start);
      cudaEventCreate(&stop);
      cudaEventRecord(start);
      FP16GeluCUDAKernel<8><<<block, thread>>>(d_x, d_y, n);
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      float milliseconds = 0;
      cudaEventElapsedTime(&milliseconds, start, stop);
      printf("Kernel execution time: %.4f ms\n", milliseconds);
      cudaEventDestroy(start);
      cudaEventDestroy(stop);

      cudaMemcpy(y, d_y, sizeof(__half) * n, cudaMemcpyDeviceToHost);

      float max_error = 0.0f;
      for (int i = 0; i < n; i++) {
        float gpu_val = static_cast<float>(y[i]);
        float ref_val = static_cast<float>(y_ref[i]);
        float err = fabs(gpu_val - ref_val);
        if (err > max_error) max_error = err;
      }
      printf("Max absolute error: %e\n", max_error);
      printf("Calibration %s\n", (max_error < 1e-3f) ? "PASSED" : "FAILED");
    } else {
      printf("Data not aligned or n not divisible by 8, kernel not launched.\n");
    }

    free(x);
    free(y);
    free(y_ref);
    cudaFree(d_x);
    cudaFree(d_y);
    return 0;
}