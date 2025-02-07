#pragma once

#include <torch/torch.h>

void get_position_ids(torch::Tensor &out, torch::Tensor &input_ids,
                               torch::Tensor &image_grid_thw);