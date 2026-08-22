#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPGuard.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <torch/extension.h>

template <typename scalar_t>
__device__ __forceinline__ float as_float(scalar_t value) {
  return static_cast<float>(value);
}

__device__ __forceinline__ float stable_softplus(float value) {
  if (value > 20.0f) return value;
  if (value < -20.0f) return expf(value);
  return log1pf(expf(value));
}

template <typename scalar_t>
__global__ void plamo_ssd_forward_kernel(
    const scalar_t* x,
    const scalar_t* dt,
    const float* a,
    const scalar_t* b,
    const scalar_t* c,
    const float* d,
    const scalar_t* z,
    const float* dt_bias,
    const int32_t* seq_idx,
    scalar_t* output,
    float* states,
    int batch,
    int length,
    int heads,
    int channels,
    int state_size,
    int chunk_size,
    bool has_seq_idx) {
  const int linear = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = batch * heads * channels;
  if (linear >= total) return;

  const int channel = linear % channels;
  const int head = (linear / channels) % heads;
  const int batch_index = linear / (channels * heads);
  float state[64];
  #pragma unroll
  for (int n = 0; n < 64; ++n) state[n] = 0.0f;

  int32_t previous_sequence = 0;
  for (int token = 0; token < length; ++token) {
    if (has_seq_idx) {
      const int32_t sequence = seq_idx[batch_index * length + token];
      if (token > 0 && sequence != previous_sequence) {
        #pragma unroll
        for (int n = 0; n < 64; ++n) state[n] = 0.0f;
      }
      previous_sequence = sequence;
    }

    const int scalar_index = (batch_index * length + token) * heads + head;
    const int x_index = scalar_index * channels + channel;
    const int state_index = scalar_index * state_size;
    const float delta = stable_softplus(as_float(dt[scalar_index]) + dt_bias[head]);
    const float decay = expf(delta * a[head]);
    const float x_value = as_float(x[x_index]);
    float reduced = 0.0f;
    for (int n = 0; n < state_size; ++n) {
      state[n] = state[n] * decay + delta * x_value * as_float(b[state_index + n]);
      reduced += state[n] * as_float(c[state_index + n]);
      if ((token + 1) % chunk_size == 0 || token == length - 1) {
        const int chunk = token / chunk_size;
        const int state_history_index =
            ((((batch_index * ((length + chunk_size - 1) / chunk_size) + chunk)
                * heads + head) * channels + channel) * state_size) + n;
        states[state_history_index] = state[n];
      }
    }
    const float gate = as_float(z[x_index]);
    const float silu = gate / (1.0f + expf(-gate));
    output[x_index] = static_cast<scalar_t>((reduced + x_value * d[head]) * silu);
  }
}

std::vector<torch::Tensor> plamo_ssd_forward_hip(
    torch::Tensor x,
    torch::Tensor dt,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor c,
    torch::Tensor d,
    torch::Tensor z,
    torch::Tensor dt_bias,
    torch::Tensor seq_idx) {
  const c10::cuda::CUDAGuard guard(x.device());
  auto output = torch::empty_like(x);
  const bool has_seq_idx = seq_idx.numel() != 0;
  constexpr int chunk_size = 1;
  const int chunks = (x.size(1) + chunk_size - 1) / chunk_size;
  auto states = torch::empty(
      {x.size(0), chunks, x.size(2), x.size(3), b.size(3)},
      x.options().dtype(torch::kFloat32));
  const int batch = static_cast<int>(x.size(0));
  const int length = static_cast<int>(x.size(1));
  const int heads = static_cast<int>(x.size(2));
  const int channels = static_cast<int>(x.size(3));
  const int state_size = static_cast<int>(b.size(3));
  const int total = batch * heads * channels;
  constexpr int threads = 256;
  const int blocks = (total + threads - 1) / threads;
  const auto stream = at::cuda::getCurrentCUDAStream(x.device().index());

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      x.scalar_type(),
      "plamo_ssd_forward_hip",
      [&] {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(plamo_ssd_forward_kernel<scalar_t>),
            dim3(blocks), dim3(threads), 0, stream,
            x.const_data_ptr<scalar_t>(),
            dt.const_data_ptr<scalar_t>(),
            a.const_data_ptr<float>(),
            b.const_data_ptr<scalar_t>(),
            c.const_data_ptr<scalar_t>(),
            d.const_data_ptr<float>(),
            z.const_data_ptr<scalar_t>(),
            dt_bias.const_data_ptr<float>(),
            has_seq_idx ? seq_idx.const_data_ptr<int32_t>() : nullptr,
            output.mutable_data_ptr<scalar_t>(),
            states.mutable_data_ptr<float>(),
            batch, length, heads, channels, state_size, chunk_size, has_seq_idx);
      });
  TORCH_CHECK(hipGetLastError() == hipSuccess, "PLaMo SSD HIP kernel launch failed");
  return {output, states};
}

template <typename scalar_t>
__global__ void plamo_ssd_backward_kernel(
    const scalar_t* grad_output,
    const float* states,
    const scalar_t* x,
    const scalar_t* dt,
    const float* a,
    const scalar_t* b,
    const scalar_t* c,
    const float* d,
    const scalar_t* z,
    const float* dt_bias,
    const int32_t* seq_idx,
    scalar_t* grad_x,
    float* grad_dt,
    float* grad_a,
    float* grad_b,
    float* grad_c,
    float* grad_d,
    scalar_t* grad_z,
    float* grad_dt_bias,
    int batch, int length, int heads, int channels, int state_size,
    int chunk_size,
    bool has_seq_idx) {
  const int linear = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = batch * heads * channels;
  if (linear >= total) return;
  const int channel = linear % channels;
  const int head = (linear / channels) % heads;
  const int batch_index = linear / (channels * heads);
  float adjoint[64];
  #pragma unroll
  for (int n = 0; n < 64; ++n) adjoint[n] = 0.0f;

  float local_a = 0.0f;
  float local_d = 0.0f;
  float local_bias = 0.0f;
  const int chunks = (length + chunk_size - 1) / chunk_size;
  float current_state[64];
  #pragma unroll
  for (int n = 0; n < 64; ++n) current_state[n] = 0.0f;
  for (int token = length - 1; token >= 0; --token) {
    const int scalar_index = (batch_index * length + token) * heads + head;
    const int x_index = scalar_index * channels + channel;
    const int state_index = scalar_index * state_size;
    const float raw_delta = as_float(dt[scalar_index]) + dt_bias[head];
    const float delta = stable_softplus(raw_delta);
    const float sigmoid_delta = 1.0f / (1.0f + expf(-raw_delta));
    const float decay = expf(delta * a[head]);
    const float x_value = as_float(x[x_index]);
    const float gate = as_float(z[x_index]);
    const float sigmoid_gate = 1.0f / (1.0f + expf(-gate));
    const float silu = gate * sigmoid_gate;
    const float grad = as_float(grad_output[x_index]);

    if (token == length - 1 || (token + 1) % chunk_size == 0) {
      const int chunk = token / chunk_size;
      for (int n = 0; n < state_size; ++n) {
        const int history_index =
            ((((batch_index * chunks + chunk) * heads + head)
                * channels + channel) * state_size) + n;
        current_state[n] = states[history_index];
      }
    }

    float reduced = 0.0f;
    for (int n = 0; n < state_size; ++n) {
      reduced += current_state[n] * as_float(c[state_index + n]);
    }
    const float raw_output = reduced + x_value * d[head];
    const float grad_raw = grad * silu;
    grad_z[x_index] = static_cast<scalar_t>(
        grad * raw_output * sigmoid_gate * (1.0f + gate * (1.0f - sigmoid_gate)));
    local_d += grad_raw * x_value;
    float grad_x_value = grad_raw * d[head];
    float grad_delta = 0.0f;
    float previous_state[64];

    const bool reset_here = has_seq_idx && token > 0 &&
        seq_idx[batch_index * length + token] != seq_idx[batch_index * length + token - 1];
    for (int n = 0; n < state_size; ++n) {
      const float state = current_state[n];
      const float c_value = as_float(c[state_index + n]);
      adjoint[n] += grad_raw * c_value;
      atomicAdd(&grad_c[state_index + n], grad_raw * state);
      if (token == 0 || reset_here || decay < 1.0e-20f) {
        previous_state[n] = 0.0f;
      } else {
        previous_state[n] =
            (state - delta * x_value * as_float(b[state_index + n])) / decay;
      }
      const float b_value = as_float(b[state_index + n]);
      atomicAdd(&grad_b[state_index + n], adjoint[n] * delta * x_value);
      grad_x_value += adjoint[n] * delta * b_value;
      grad_delta += adjoint[n] * (previous_state[n] * decay * a[head] + x_value * b_value);
      local_a += adjoint[n] * previous_state[n] * decay * delta;
      adjoint[n] = reset_here ? 0.0f : adjoint[n] * decay;
      current_state[n] = previous_state[n];
    }
    grad_x[x_index] = static_cast<scalar_t>(grad_x_value);
    const float grad_raw_delta = grad_delta * sigmoid_delta;
    atomicAdd(&grad_dt[scalar_index], grad_raw_delta);
    local_bias += grad_raw_delta;
  }
  atomicAdd(&grad_a[head], local_a);
  atomicAdd(&grad_d[head], local_d);
  atomicAdd(&grad_dt_bias[head], local_bias);
}

std::vector<torch::Tensor> plamo_ssd_backward_hip(
    torch::Tensor grad_output, torch::Tensor states, torch::Tensor x,
    torch::Tensor dt, torch::Tensor a, torch::Tensor b, torch::Tensor c,
    torch::Tensor d, torch::Tensor z, torch::Tensor dt_bias,
    torch::Tensor seq_idx) {
  const c10::cuda::CUDAGuard guard(x.device());
  auto grad_x = torch::zeros_like(x);
  auto grad_dt_float = torch::zeros_like(dt, dt.options().dtype(torch::kFloat32));
  auto grad_a = torch::zeros_like(a, torch::MemoryFormat::Contiguous);
  auto grad_b_float = torch::zeros_like(b, b.options().dtype(torch::kFloat32));
  auto grad_c_float = torch::zeros_like(c, c.options().dtype(torch::kFloat32));
  auto grad_d = torch::zeros_like(d, torch::MemoryFormat::Contiguous);
  auto grad_z = torch::zeros_like(z);
  auto grad_dt_bias = torch::zeros_like(dt_bias, torch::MemoryFormat::Contiguous);
  const int batch = x.size(0), length = x.size(1), heads = x.size(2);
  const int channels = x.size(3), state_size = b.size(3);
  const int total = batch * heads * channels;
  constexpr int threads = 256;
  const int blocks = (total + threads - 1) / threads;
  const auto stream = at::cuda::getCurrentCUDAStream(x.device().index());
  const bool has_seq_idx = seq_idx.numel() != 0;
  constexpr int chunk_size = 1;
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(),
      "plamo_ssd_backward_hip", [&] {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(plamo_ssd_backward_kernel<scalar_t>),
            dim3(blocks), dim3(threads), 0, stream,
            grad_output.const_data_ptr<scalar_t>(), states.const_data_ptr<float>(),
            x.const_data_ptr<scalar_t>(), dt.const_data_ptr<scalar_t>(),
            a.const_data_ptr<float>(), b.const_data_ptr<scalar_t>(),
            c.const_data_ptr<scalar_t>(), d.const_data_ptr<float>(),
            z.const_data_ptr<scalar_t>(), dt_bias.const_data_ptr<float>(),
            has_seq_idx ? seq_idx.const_data_ptr<int32_t>() : nullptr,
            grad_x.mutable_data_ptr<scalar_t>(), grad_dt_float.mutable_data_ptr<float>(),
            grad_a.mutable_data_ptr<float>(), grad_b_float.mutable_data_ptr<float>(),
            grad_c_float.mutable_data_ptr<float>(), grad_d.mutable_data_ptr<float>(),
            grad_z.mutable_data_ptr<scalar_t>(), grad_dt_bias.mutable_data_ptr<float>(),
            batch, length, heads, channels, state_size, chunk_size, has_seq_idx);
      });
  TORCH_CHECK(hipGetLastError() == hipSuccess, "PLaMo SSD HIP backward launch failed");
  return {grad_x, grad_dt_float.to(dt.scalar_type()), grad_a,
          grad_b_float.to(b.scalar_type()), grad_c_float.to(c.scalar_type()),
          grad_d, grad_z, grad_dt_bias};
}

template <typename scalar_t>
__global__ void plamo_conv_forward_kernel(
    const scalar_t* x, const scalar_t* weight, const int32_t* seq_idx,
    scalar_t* output, int batch, int channels, int length, int width,
    bool has_seq_idx) {
  const int linear = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = batch * channels * length;
  if (linear >= total) return;
  const int token = linear % length;
  const int channel = (linear / length) % channels;
  const int batch_index = linear / (length * channels);
  float raw = 0.0f;
  for (int kernel_index = 0; kernel_index < width; ++kernel_index) {
    const int source = token - (width - 1 - kernel_index);
    if (source < 0) continue;
    bool valid = true;
    if (has_seq_idx) {
      for (int boundary = source + 1; boundary <= token; ++boundary) {
        valid = valid && seq_idx[batch_index * length + boundary]
            == seq_idx[batch_index * length + boundary - 1];
      }
    }
    if (valid) {
      raw += as_float(x[(batch_index * channels + channel) * length + source])
          * as_float(weight[channel * width + kernel_index]);
    }
  }
  output[linear] = static_cast<scalar_t>(raw / (1.0f + expf(-raw)));
}

template <typename scalar_t>
__device__ __forceinline__ float conv_raw_at(
    const scalar_t* x, const scalar_t* weight, const int32_t* seq_idx,
    int batch_index, int channel, int token, int channels, int length,
    int width, bool has_seq_idx) {
  float raw = 0.0f;
  for (int kernel_index = 0; kernel_index < width; ++kernel_index) {
    const int source = token - (width - 1 - kernel_index);
    if (source < 0) continue;
    bool valid = true;
    if (has_seq_idx) {
      for (int boundary = source + 1; boundary <= token; ++boundary) {
        valid = valid && seq_idx[batch_index * length + boundary]
            == seq_idx[batch_index * length + boundary - 1];
      }
    }
    if (valid) {
      raw += as_float(x[(batch_index * channels + channel) * length + source])
          * as_float(weight[channel * width + kernel_index]);
    }
  }
  return raw;
}

template <typename scalar_t>
__global__ void plamo_conv_grad_x_kernel(
    const scalar_t* grad_output, const scalar_t* x, const scalar_t* weight,
    const int32_t* seq_idx, scalar_t* grad_x,
    int batch, int channels, int length, int width, bool has_seq_idx) {
  const int linear = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = batch * channels * length;
  if (linear >= total) return;
  const int source = linear % length;
  const int channel = (linear / length) % channels;
  const int batch_index = linear / (length * channels);
  float accumulated = 0.0f;
  for (int kernel_index = 0; kernel_index < width; ++kernel_index) {
    const int target = source + (width - 1 - kernel_index);
    if (target >= length) continue;
    bool valid = true;
    if (has_seq_idx) {
      for (int boundary = source + 1; boundary <= target; ++boundary) {
        valid = valid && seq_idx[batch_index * length + boundary]
            == seq_idx[batch_index * length + boundary - 1];
      }
    }
    if (!valid) continue;
    const float raw = conv_raw_at(
        x, weight, seq_idx, batch_index, channel, target, channels, length,
        width, has_seq_idx);
    const float sigmoid = 1.0f / (1.0f + expf(-raw));
    const float grad_raw = as_float(
        grad_output[(batch_index * channels + channel) * length + target])
        * sigmoid * (1.0f + raw * (1.0f - sigmoid));
    accumulated += grad_raw * as_float(weight[channel * width + kernel_index]);
  }
  grad_x[linear] = static_cast<scalar_t>(accumulated);
}

template <typename scalar_t>
__global__ void plamo_conv_grad_weight_kernel(
    const scalar_t* grad_output, const scalar_t* x, const scalar_t* weight,
    const int32_t* seq_idx, float* grad_weight,
    int batch, int channels, int length, int width, bool has_seq_idx) {
  const int linear = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = channels * width;
  if (linear >= total) return;
  const int kernel_index = linear % width;
  const int channel = linear / width;
  float accumulated = 0.0f;
  const int lag = width - 1 - kernel_index;
  for (int batch_index = 0; batch_index < batch; ++batch_index) {
    for (int target = lag; target < length; ++target) {
      const int source = target - lag;
      bool valid = true;
      if (has_seq_idx) {
        for (int boundary = source + 1; boundary <= target; ++boundary) {
          valid = valid && seq_idx[batch_index * length + boundary]
              == seq_idx[batch_index * length + boundary - 1];
        }
      }
      if (!valid) continue;
      const float raw = conv_raw_at(
          x, weight, seq_idx, batch_index, channel, target, channels, length,
          width, has_seq_idx);
      const float sigmoid = 1.0f / (1.0f + expf(-raw));
      const int target_index = (batch_index * channels + channel) * length + target;
      const int source_index = (batch_index * channels + channel) * length + source;
      const float grad_raw = as_float(grad_output[target_index])
          * sigmoid * (1.0f + raw * (1.0f - sigmoid));
      accumulated += grad_raw * as_float(x[source_index]);
    }
  }
  grad_weight[linear] = accumulated;
}

torch::Tensor plamo_conv_forward_hip(
    torch::Tensor x, torch::Tensor weight, torch::Tensor seq_idx) {
  const c10::cuda::CUDAGuard guard(x.device());
  auto output = torch::empty_like(x);
  const int batch = x.size(0), channels = x.size(1), length = x.size(2);
  const int width = weight.size(2), total = batch * channels * length;
  constexpr int threads = 256;
  const int blocks = (total + threads - 1) / threads;
  const bool has_seq_idx = seq_idx.numel() != 0;
  const auto stream = at::cuda::getCurrentCUDAStream(x.device().index());
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(),
      "plamo_conv_forward_hip", [&] {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(plamo_conv_forward_kernel<scalar_t>),
            dim3(blocks), dim3(threads), 0, stream,
            x.const_data_ptr<scalar_t>(), weight.const_data_ptr<scalar_t>(),
            has_seq_idx ? seq_idx.const_data_ptr<int32_t>() : nullptr,
            output.mutable_data_ptr<scalar_t>(), batch, channels, length, width, has_seq_idx);
      });
  TORCH_CHECK(hipGetLastError() == hipSuccess, "PLaMo conv HIP forward launch failed");
  return output;
}

std::vector<torch::Tensor> plamo_conv_backward_hip(
    torch::Tensor grad_output, torch::Tensor x, torch::Tensor weight,
    torch::Tensor seq_idx, bool need_weight_grad) {
  const c10::cuda::CUDAGuard guard(x.device());
  auto grad_x = torch::empty_like(x);
  auto grad_weight = torch::zeros_like(weight, weight.options().dtype(torch::kFloat32));
  const int batch = x.size(0), channels = x.size(1), length = x.size(2);
  const int width = weight.size(2), total = batch * channels * length;
  constexpr int threads = 256;
  const int blocks = (total + threads - 1) / threads;
  const bool has_seq_idx = seq_idx.numel() != 0;
  const auto stream = at::cuda::getCurrentCUDAStream(x.device().index());
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(),
      "plamo_conv_backward_hip", [&] {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(plamo_conv_grad_x_kernel<scalar_t>),
            dim3(blocks), dim3(threads), 0, stream,
            grad_output.const_data_ptr<scalar_t>(), x.const_data_ptr<scalar_t>(),
            weight.const_data_ptr<scalar_t>(),
            has_seq_idx ? seq_idx.const_data_ptr<int32_t>() : nullptr,
            grad_x.mutable_data_ptr<scalar_t>(), batch, channels, length, width,
            has_seq_idx);
        if (need_weight_grad) {
          const int weight_total = channels * width;
          const int weight_blocks = (weight_total + threads - 1) / threads;
          hipLaunchKernelGGL(HIP_KERNEL_NAME(plamo_conv_grad_weight_kernel<scalar_t>),
              dim3(weight_blocks), dim3(threads), 0, stream,
              grad_output.const_data_ptr<scalar_t>(), x.const_data_ptr<scalar_t>(),
              weight.const_data_ptr<scalar_t>(),
              has_seq_idx ? seq_idx.const_data_ptr<int32_t>() : nullptr,
              grad_weight.mutable_data_ptr<float>(), batch, channels, length, width,
              has_seq_idx);
        }
      });
  TORCH_CHECK(hipGetLastError() == hipSuccess, "PLaMo conv HIP backward launch failed");
  return {grad_x, grad_weight.to(weight.scalar_type())};
}
