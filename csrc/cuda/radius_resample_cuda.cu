#include "radius_resample_cuda.h"

#include <ATen/cuda/CUDAContext.h>

#include "utils.cuh"

#define THREADS 256

#include <curand_kernel.h>

template <typename scalar_t>
__global__ void
radius_kernel_resample(const scalar_t *__restrict__ x, const scalar_t *__restrict__ y,
              const int64_t *__restrict__ ptr_x,
              const int64_t *__restrict__ ptr_y, 
              int64_t *__restrict__ row,
              int64_t *__restrict__ col,
              int64_t *__restrict__ matches, 
              const scalar_t r, const int64_t n,
              const int64_t m, const int64_t dim, const int64_t num_examples,
              const int64_t max_num_neighbors,
              const bool ignore_same_index,
              unsigned long long seed)
{

  const int64_t n_y = blockIdx.x * blockDim.x + threadIdx.x;
  if (n_y >= m)
    return;

  // Each thread gets different state with same seed, different sequence number
  curandStatePhilox4_32_10_t state;
  curand_init(seed, n_y, 0, &state);

  int64_t count = 0;
  const int64_t example_idx = get_example_idx(n_y, ptr_y, num_examples);

  for (int64_t n_x = ptr_x[example_idx]; n_x < ptr_x[example_idx + 1]; n_x++)
  {
    scalar_t dist = 0;
    for (int64_t d = 0; d < dim; d++)
    {
      dist += (x[n_x * dim + d] - y[n_y * dim + d]) *
              (x[n_x * dim + d] - y[n_y * dim + d]);
    }

    if (dist < r && !(ignore_same_index && n_y == n_x))
    {
      if (count  < max_num_neighbors) {
        row[n_y * max_num_neighbors + count] = n_y;
        col[n_y * max_num_neighbors + count] = n_x;
      } else {
        // Resevoir sampling
        int64_t j = curand(&state) % (count + 1);
        if (j < max_num_neighbors) {
          row[n_y * max_num_neighbors + j] = n_y;
          col[n_y * max_num_neighbors + j] = n_x;
        }
      }
      count++; 
    }
  }

  // Give the total number of neighbors found
  matches[n_y] = count;
}

std::vector<torch::Tensor> radius_cuda_resample(const torch::Tensor x, const torch::Tensor y,
                          std::optional<torch::Tensor> ptr_x,
                          std::optional<torch::Tensor> ptr_y, const double r,
                          const int64_t max_num_neighbors,
                          const bool ignore_same_index)
{
  CHECK_CUDA(x);
  CHECK_CONTIGUOUS(x);
  CHECK_INPUT(x.dim() == 2);
  CHECK_CUDA(y);
  CHECK_CONTIGUOUS(y);
  CHECK_INPUT(y.dim() == 2);
  CHECK_INPUT(x.size(1) == y.size(1));

  c10::cuda::MaybeSetDevice(x.get_device());

  if (ptr_x.has_value())
  {
    CHECK_CUDA(ptr_x.value());
    CHECK_INPUT(ptr_x.value().dim() == 1);
  }
  else
    ptr_x = torch::arange(0, x.size(0) + 1, x.size(0),
                          x.options().dtype(torch::kLong));

  if (ptr_y.has_value())
  {
    CHECK_CUDA(ptr_y.value());
    CHECK_INPUT(ptr_y.value().dim() == 1);
  }
  else
    ptr_y = torch::arange(0, y.size(0) + 1, y.size(0),
                          y.options().dtype(torch::kLong));

  CHECK_INPUT(ptr_x.value().numel() == ptr_y.value().numel());

  auto row =
      torch::full(y.size(0) * max_num_neighbors, -1, ptr_y.value().options());
  auto col =
      torch::full(y.size(0) * max_num_neighbors, -1, ptr_y.value().options());
  auto matches =
      torch::full(y.size(0), -1, ptr_y.value().options());

  dim3 BLOCKS((y.size(0) + THREADS - 1) / THREADS);

  // Generate a random global seed, C++
  const auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count() % 1000000;
  auto stream = at::cuda::getCurrentCUDAStream();
  auto scalar_type = x.scalar_type();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, scalar_type, "_", [&]
      { radius_kernel_resample<scalar_t><<<BLOCKS, THREADS, 0, stream>>>(
            x.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(),
            ptr_x.value().data_ptr<int64_t>(),
            ptr_y.value().data_ptr<int64_t>(), 
            row.data_ptr<int64_t>(),
            col.data_ptr<int64_t>(), 
            matches.data_ptr<int64_t>(),
            r * r, x.size(0), y.size(0), x.size(1),
            ptr_x.value().numel() - 1, max_num_neighbors, ignore_same_index,
            seed); });

  auto mask = row != -1;
  return {torch::stack({row.masked_select(mask), col.masked_select(mask)}, 0), matches};
}