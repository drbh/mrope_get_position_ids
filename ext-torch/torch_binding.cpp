#include "torch_binding.h"
#include "registration.h"
#include <torch/library.h>

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("get_position_ids(Tensor out, Tensor input_ids, Tensor "
          "image_grid_thw) -> ()");
  ops.impl("get_position_ids", torch::kCUDA, &get_position_ids);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
