#ifdef WITH_PYTHON
#include <Python.h>
#endif
#include <torch/script.h>

// No implementation
// #include "cpu/radius_resample_cpu.h"

#ifdef WITH_CUDA
#include "cuda/radius_resample_cuda.h"
#endif

#ifdef _WIN32
#ifdef WITH_PYTHON
#ifdef WITH_CUDA
PyMODINIT_FUNC PyInit__radius_resample_cuda(void) { return NULL; }
#else
PyMODINIT_FUNC PyInit__radius_resample_cpu(void) { return NULL; }
#endif
#endif
#endif

CLUSTER_API std::vector<torch::Tensor> radius_resample(torch::Tensor x, torch::Tensor y,
  std::optional<torch::Tensor> ptr_x,
  std::optional<torch::Tensor> ptr_y, double r,
  int64_t max_num_neighbors, int64_t num_workers,
  bool ignore_same_index) {
if (x.device().is_cuda()) {
  #ifdef WITH_CUDA
  return radius_cuda_resample(x, y, ptr_x, ptr_y, r, max_num_neighbors, ignore_same_index);
  #else
  AT_ERROR("Not compiled with CUDA support");
  #endif
} else {
  AT_ERROR("Not compiled with CPU support");
  //return radius_cpu(x, y, ptr_x, ptr_y, r, max_num_neighbors, num_workers, ignore_same_index);
}
}

static auto registry =
    torch::RegisterOperators().op("torch_cluster::radius_resample", &radius_resample);
