#include <ATen/ATen.h>
#include <torch/library.h>
#include <torch_musa/csrc/core/MUSAStream.h>
#include <torch_musa/csrc/core/MUSAGuard.h>

#include <cstdint>

extern "C" int launch_real_stft_1024(const float* input,
                                      const float* window,
                                      const float* twiddle_cos,
                                      const float* twiddle_sin,
                                      float* output,
                                      int batch,
                                      int samples,
                                      int hop,
                                      uintptr_t stream);
extern "C" int launch_real_stft_512(const float* input,
                                     const float* window,
                                     const float* twiddle_cos,
                                     const float* twiddle_sin,
                                     float* output,
                                     int batch,
                                     int samples,
                                     int hop,
                                     uintptr_t stream);

namespace {

at::Tensor real_stft_musa(const at::Tensor& input,
                          const at::Tensor& window,
                          const at::Tensor& twiddle_cos,
                          const at::Tensor& twiddle_sin,
                          int64_t hop) {
  TORCH_CHECK(input.device().type() == c10::DeviceType::PrivateUse1,
              "real_stft input must be a MUSA tensor");
  TORCH_CHECK(window.device() == input.device(),
              "real_stft window must be on the input device");
  TORCH_CHECK(twiddle_cos.device() == input.device() &&
                  twiddle_sin.device() == input.device(),
              "real_stft twiddle tables must be on the input device");
  TORCH_CHECK(input.scalar_type() == at::kFloat && window.scalar_type() == at::kFloat,
              "real_stft input and window must be float32");
  TORCH_CHECK(twiddle_cos.scalar_type() == at::kFloat &&
                  twiddle_sin.scalar_type() == at::kFloat,
              "real_stft twiddle tables must be float32");
  TORCH_CHECK(input.dim() == 2 && window.dim() == 1 && window.numel() == 1024,
              "real_stft expects input [B,T] and a 1024 element window");
  TORCH_CHECK(twiddle_cos.dim() == 1 && twiddle_sin.dim() == 1 &&
                  twiddle_cos.numel() == 512 && twiddle_sin.numel() == 512,
              "real_stft twiddle tables must contain 512 elements");
  TORCH_CHECK(input.is_contiguous() && window.is_contiguous() &&
                  twiddle_cos.is_contiguous() && twiddle_sin.is_contiguous(),
              "real_stft inputs must be contiguous");
  TORCH_CHECK(hop > 0 && hop <= 1024 && input.size(1) >= 1024,
              "real_stft received an invalid hop or input length");

  const c10::musa::MUSAGuard guard(input.device());
  const int64_t frames = (input.size(1) - 1024) / hop + 1;
  auto output = at::empty({input.size(0), 513, frames, 2}, input.options());
  auto stream = c10::musa::getCurrentMUSAStream(input.device().index());
  const int rc = launch_real_stft_1024(
      input.data_ptr<float>(), window.data_ptr<float>(),
      twiddle_cos.data_ptr<float>(), twiddle_sin.data_ptr<float>(),
      output.data_ptr<float>(),
      static_cast<int>(input.size(0)), static_cast<int>(input.size(1)),
      static_cast<int>(hop), reinterpret_cast<uintptr_t>(stream.stream()));
  TORCH_CHECK(rc == 0, "real_stft_1024 launch failed, rc=", rc);
  return output;
}

at::Tensor real_stft_512_musa(const at::Tensor& input,
                              const at::Tensor& window,
                              const at::Tensor& twiddle_cos,
                              const at::Tensor& twiddle_sin,
                              int64_t hop) {
  TORCH_CHECK(input.device().type() == c10::DeviceType::PrivateUse1,
              "real_stft_512 input must be a MUSA tensor");
  TORCH_CHECK(window.device() == input.device() &&
                  twiddle_cos.device() == input.device() &&
                  twiddle_sin.device() == input.device(),
              "real_stft_512 inputs must share the input device");
  TORCH_CHECK(input.scalar_type() == at::kFloat && window.scalar_type() == at::kFloat &&
                  twiddle_cos.scalar_type() == at::kFloat && twiddle_sin.scalar_type() == at::kFloat,
              "real_stft_512 inputs must be float32");
  TORCH_CHECK(input.dim() == 2 && window.dim() == 1 && window.numel() == 512 &&
                  twiddle_cos.dim() == 1 && twiddle_cos.numel() == 256 &&
                  twiddle_sin.dim() == 1 && twiddle_sin.numel() == 256,
              "real_stft_512 expects [B,T], a 512 element window and 256 element twiddles");
  TORCH_CHECK(input.is_contiguous() && window.is_contiguous() &&
                  twiddle_cos.is_contiguous() && twiddle_sin.is_contiguous(),
              "real_stft_512 inputs must be contiguous");
  TORCH_CHECK(hop > 0 && hop <= 512 && input.size(1) >= 512,
              "real_stft_512 received an invalid hop or input length");

  const c10::musa::MUSAGuard guard(input.device());
  const int64_t frames = (input.size(1) - 512) / hop + 1;
  auto output = at::empty({input.size(0), 257, frames, 2}, input.options());
  auto stream = c10::musa::getCurrentMUSAStream(input.device().index());
  const int rc = launch_real_stft_512(
      input.data_ptr<float>(), window.data_ptr<float>(),
      twiddle_cos.data_ptr<float>(), twiddle_sin.data_ptr<float>(),
      output.data_ptr<float>(), static_cast<int>(input.size(0)),
      static_cast<int>(input.size(1)), static_cast<int>(hop),
      reinterpret_cast<uintptr_t>(stream.stream()));
  TORCH_CHECK(rc == 0, "real_stft_512 launch failed, rc=", rc);
  return output;
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(musa_audio, m) {
  m.def("real_stft(Tensor input, Tensor window, Tensor twiddle_cos, Tensor twiddle_sin, int hop) -> Tensor");
  m.def("real_stft_512(Tensor input, Tensor window, Tensor twiddle_cos, Tensor twiddle_sin, int hop) -> Tensor");
}

TORCH_LIBRARY_IMPL(musa_audio, PrivateUse1, m) {
  m.impl("real_stft", real_stft_musa);
  m.impl("real_stft_512", real_stft_512_musa);
}
