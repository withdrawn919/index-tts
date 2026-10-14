#include <musa_runtime.h>

namespace {

constexpr int kNfft = 1024;
constexpr int kFreq = 513;
constexpr int kThreads = 512;
constexpr int kFramesPerBlock = 1;

__device__ __forceinline__ int bit_reverse_10(int value) {
  int result = 0;
#pragma unroll
  for (int bit = 0; bit < 10; ++bit) {
    result = (result << 1) | ((value >> bit) & 1);
  }
  return result;
}

__global__ void real_stft_1024_kernel(const float* __restrict__ input,
                                      const float* __restrict__ window,
                                      const float* __restrict__ twiddle_cos,
                                      const float* __restrict__ twiddle_sin,
                                      float* __restrict__ output,
                                      int batch,
                                      int samples,
                                      int hop,
                                      int frames) {
  extern __shared__ float shared[];
  float* real = shared;
  float* imag = shared + kNfft;
  const int frame_id = static_cast<int>(blockIdx.x) * kFramesPerBlock;
  const int batch_id = frame_id / frames;
  if (batch_id >= batch) {
    return;
  }

  const int tid = static_cast<int>(threadIdx.x);
  for (int local_frame = 0; local_frame < kFramesPerBlock; ++local_frame) {
    const int frame = frame_id - batch_id * frames + local_frame;
    if (frame >= frames) {
      break;
    }
    const int start = frame * hop;
    const int first = tid * 2;
    if (first < kNfft) {
      real[bit_reverse_10(first)] = input[batch_id * samples + start + first] * window[first];
    }
    if (first + 1 < kNfft) {
      real[bit_reverse_10(first + 1)] =
          input[batch_id * samples + start + first + 1] * window[first + 1];
    }
    if (first < kNfft) {
      imag[bit_reverse_10(first)] = 0.0f;
    }
    if (first + 1 < kNfft) {
      imag[bit_reverse_10(first + 1)] = 0.0f;
    }
    __syncthreads();

    for (int width = 2; width <= kNfft; width <<= 1) {
      const int butterflies = kNfft / 2;
      for (int butterfly = tid; butterfly < butterflies; butterfly += kThreads) {
        const int half = width >> 1;
        const int group = butterfly / half;
        const int j = butterfly - group * half;
        const int left = group * width + j;
        const int right = left + half;
        const int table_index = j * kNfft / width;
        const float cosine = twiddle_cos[table_index];
        const float sine = twiddle_sin[table_index];
        const float right_real = real[right] * cosine - imag[right] * sine;
        const float right_imag = real[right] * sine + imag[right] * cosine;
        const float left_real = real[left];
        const float left_imag = imag[left];
        real[left] = left_real + right_real;
        imag[left] = left_imag + right_imag;
        real[right] = left_real - right_real;
        imag[right] = left_imag - right_imag;
      }
      __syncthreads();
    }

    for (int frequency = tid; frequency < kFreq; frequency += kThreads) {
      const int output_index = ((batch_id * kFreq + frequency) * frames + frame) * 2;
      output[output_index] = real[frequency];
      output[output_index + 1] = imag[frequency];
    }
    __syncthreads();
  }
}

}  // namespace

extern "C" int launch_real_stft_1024(const float* input,
                                      const float* window,
                                      const float* twiddle_cos,
                                      const float* twiddle_sin,
                                      float* output,
                                      int batch,
                                      int samples,
                                      int hop,
                                      uintptr_t stream) {
  if (batch <= 0 || samples < kNfft || hop <= 0 || hop > kNfft) {
    return 1;
  }
  const int frames = (samples - kNfft) / hop + 1;
  const dim3 grid(static_cast<unsigned>((batch * frames + kFramesPerBlock - 1) / kFramesPerBlock), 1, 1);
  const dim3 block(kThreads, 1, 1);
  real_stft_1024_kernel<<<grid, block, 2 * kNfft * sizeof(float),
                          reinterpret_cast<musaStream_t>(stream)>>>(
      input, window, twiddle_cos, twiddle_sin, output, batch, samples, hop, frames);
  return static_cast<int>(musaGetLastError());
}

namespace {

constexpr int kNfft512 = 512;
constexpr int kFreq512 = 257;
constexpr int kThreads512 = 256;

__device__ __forceinline__ int bit_reverse_9(int value) {
  int result = 0;
#pragma unroll
  for (int bit = 0; bit < 9; ++bit) {
    result = (result << 1) | ((value >> bit) & 1);
  }
  return result;
}

__global__ void real_stft_512_kernel(const float* __restrict__ input,
                                     const float* __restrict__ window,
                                     const float* __restrict__ twiddle_cos,
                                     const float* __restrict__ twiddle_sin,
                                     float* __restrict__ output,
                                     int batch,
                                     int samples,
                                     int hop,
                                     int frames) {
  extern __shared__ float shared[];
  float* real = shared;
  float* imag = shared + kNfft512;
  const int frame_id = static_cast<int>(blockIdx.x);
  const int batch_id = frame_id / frames;
  if (batch_id >= batch) {
    return;
  }

  const int tid = static_cast<int>(threadIdx.x);
  const int start = (frame_id - batch_id * frames) * hop;
  const int first = tid * 2;
  if (first < kNfft512) {
    real[bit_reverse_9(first)] = input[batch_id * samples + start + first] * window[first];
    imag[bit_reverse_9(first)] = 0.0f;
  }
  if (first + 1 < kNfft512) {
    real[bit_reverse_9(first + 1)] =
        input[batch_id * samples + start + first + 1] * window[first + 1];
    imag[bit_reverse_9(first + 1)] = 0.0f;
  }
  __syncthreads();

  for (int width = 2; width <= kNfft512; width <<= 1) {
    for (int butterfly = tid; butterfly < kNfft512 / 2; butterfly += kThreads512) {
      const int half = width >> 1;
      const int group = butterfly / half;
      const int j = butterfly - group * half;
      const int left = group * width + j;
      const int right = left + half;
      const int table_index = j * kNfft512 / width;
      const float cosine = twiddle_cos[table_index];
      const float sine = twiddle_sin[table_index];
      const float right_real = real[right] * cosine - imag[right] * sine;
      const float right_imag = real[right] * sine + imag[right] * cosine;
      const float left_real = real[left];
      const float left_imag = imag[left];
      real[left] = left_real + right_real;
      imag[left] = left_imag + right_imag;
      real[right] = left_real - right_real;
      imag[right] = left_imag - right_imag;
    }
    __syncthreads();
  }

  for (int frequency = tid; frequency < kFreq512; frequency += kThreads512) {
    const int output_index = ((batch_id * kFreq512 + frequency) * frames +
                              (frame_id - batch_id * frames)) * 2;
    output[output_index] = real[frequency];
    output[output_index + 1] = imag[frequency];
  }
}

}  // namespace

extern "C" int launch_real_stft_512(const float* input,
                                     const float* window,
                                     const float* twiddle_cos,
                                     const float* twiddle_sin,
                                     float* output,
                                     int batch,
                                     int samples,
                                     int hop,
                                     uintptr_t stream) {
  if (batch <= 0 || samples < kNfft512 || hop <= 0 || hop > kNfft512) {
    return 1;
  }
  const int frames = (samples - kNfft512) / hop + 1;
  const dim3 grid(static_cast<unsigned>(batch * frames), 1, 1);
  const dim3 block(kThreads512, 1, 1);
  real_stft_512_kernel<<<grid, block, 2 * kNfft512 * sizeof(float),
                         reinterpret_cast<musaStream_t>(stream)>>>(
      input, window, twiddle_cos, twiddle_sin, output, batch, samples, hop, frames);
  return static_cast<int>(musaGetLastError());
}
