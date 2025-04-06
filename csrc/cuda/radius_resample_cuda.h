#pragma once

#include "../extensions.h"

std::vector<torch::Tensor> radius_cuda_resample(torch::Tensor x, torch::Tensor y,
                          std::optional<torch::Tensor> ptr_x,
                          std::optional<torch::Tensor> ptr_y, double r,
                          int64_t max_num_neighbors,
                          bool ignore_same_index);

                          