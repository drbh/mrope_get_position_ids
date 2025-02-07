#include <cuda_runtime.h>
#include <torch/torch.h>
#include <vector>
#include <stdio.h>

#define SPATIAL_MERGE_SIZE 2
#define MAX_THREADS_PER_BLOCK 256

// Kernel: each block processes one vision segment.
// For a given segment, the kernel computes image positions by "unraveling" a 1D index
// into 3D coordinates (t_idx, h_idx, w_idx) and then adds a per‑segment offset.
__global__ void create_image_positions_kernel(
    const int *image_grid_thw,                // shape: [num_segments * 3]
    const int *segment_offsets,               // shape: [num_segments]
    const int *vision_segment_lengths_cumsum, // shape: [num_segments]
    int *image_positions)                     // output: shape [total_image_positions, 3]
{
    int segment_idx = blockIdx.x;

    // Load grid dims for this segment.
    int t = image_grid_thw[segment_idx * 3];
    int h = image_grid_thw[segment_idx * 3 + 1] / SPATIAL_MERGE_SIZE;
    int w = image_grid_thw[segment_idx * 3 + 2] / SPATIAL_MERGE_SIZE;
    int total_length = t * h * w;

    // Get the starting output position for this segment.
    int pos_offset = segment_offsets[segment_idx];
    // The per‐segment offset to add to each coordinate.
    int offset_add = vision_segment_lengths_cumsum[segment_idx];

    // Process all positions in this segment using a grid–stride loop.
    for (int pos_idx = threadIdx.x; pos_idx < total_length; pos_idx += blockDim.x)
    {
        // Compute the "unraveled" coordinates.
        int t_idx = pos_idx / (h * w);
        int h_idx = (pos_idx / w) % h;
        int w_idx = pos_idx % w;
        // Write out the 3 coordinates (each image token gets 3 ints).
        int out_index = (pos_offset + pos_idx) * 3;
        image_positions[out_index] = t_idx + offset_add;
        image_positions[out_index + 1] = h_idx + offset_add;
        image_positions[out_index + 2] = w_idx + offset_add;
    }
}

// This function computes text and image position ids then interleaves them as:
//    [text segment 0, image segment 0, text segment 1, image segment 1, ...].
// If extra text tokens exist after the last vision segment, they are appended at the end.
void get_position_ids(
    torch::Tensor &out,            // Final output tensor
    torch::Tensor &input_ids,      // tensor holding token ids
    torch::Tensor &image_grid_thw) // tensor of shape [num_segments, 3]: each row is [t, h, w]
{
    TORCH_CHECK(input_ids.device().is_cuda(), "input_ids must be a CUDA tensor");
    TORCH_CHECK(image_grid_thw.device().is_cuda(), "image_grid_thw must be a CUDA tensor");
    TORCH_CHECK(out.device().is_cuda(), "out must be a CUDA tensor");

    const int input_len = input_ids.size(0);
    auto options_int = torch::TensorOptions().device(input_ids.device()).dtype(torch::kInt);
    auto options_long = torch::TensorOptions().device(input_ids.device()).dtype(torch::kLong);

    const int VISION_START_TOKEN_ID = 151652;
    const int VISION_END_TOKEN_ID = 151653;

    // Find vision segments
    auto vision_starts_mask = input_ids == VISION_START_TOKEN_ID;
    auto vision_ends_mask = input_ids == VISION_END_TOKEN_ID;

    auto starts = torch::where(vision_starts_mask)[0].to(torch::kInt);
    auto ends = torch::where(vision_ends_mask)[0].to(torch::kInt);

    int actual_segments = starts.size(0);
    auto prev_end = torch::cat({torch::zeros({1}, options_long), ends.slice(0, 0, actual_segments - 1)});

    // Compute text lengths between vision tokens.
    auto text_lengths_between_vision = starts - prev_end + 1;
    auto zeros = torch::zeros({1}, options_long);
    auto widths = image_grid_thw.slice(0, 0, actual_segments).select(1, 2);
    auto divided_widths = widths / SPATIAL_MERGE_SIZE;
    auto vision_widths_max = torch::cat({zeros, divided_widths.slice(0, 0, actual_segments - 1)});
    // The vision segment length is the sum of text tokens plus the (merged) image width.
    auto vision_segment_lengths = text_lengths_between_vision + vision_widths_max;
    auto vision_segment_lengths_cumsum = vision_segment_lengths.cumsum(0);
    auto text_segment_lengths = vision_segment_lengths_cumsum - text_lengths_between_vision;

    // Compute per‑segment starting indices for image positions.
    std::vector<int> segment_offsets_vec(actual_segments);
    int total_image_positions = 0;
    // (Using a CPU copy because the number of segments is small.)
    auto image_grid_cpu = image_grid_thw.to(torch::kCPU);
    auto image_grid_accessor = image_grid_cpu.accessor<int, 2>(); // shape: [actual_segments, 3]
    for (int i = 0; i < actual_segments; i++)
    {
        int t = image_grid_accessor[i][0];
        int h = image_grid_accessor[i][1] / SPATIAL_MERGE_SIZE;
        int w = image_grid_accessor[i][2] / SPATIAL_MERGE_SIZE;
        segment_offsets_vec[i] = total_image_positions;
        total_image_positions += t * h * w;
    }

    // IMPORTANT: Create the segment_offsets tensor directly so that its memory is on the device.
    auto segment_offsets_tensor = torch::tensor(segment_offsets_vec, options_int);

    // Make sure vision_segment_lengths_cumsum is int and on the correct device.
    auto vision_segment_lengths_cumsum_int = vision_segment_lengths_cumsum.to(torch::kInt);

    // Allocate one contiguous output tensor for all image positions.
    // Each image token produces 3 ints.
    auto image_positions_tensor = torch::empty({total_image_positions, 3}, options_int);

    // Launch one block per vision segment.
    int threads = MAX_THREADS_PER_BLOCK;
    int blocks = actual_segments;
    create_image_positions_kernel<<<blocks, threads>>>(
        image_grid_thw.data_ptr<int>(),
        segment_offsets_tensor.data_ptr<int>(),
        vision_segment_lengths_cumsum_int.data_ptr<int>(),
        image_positions_tensor.data_ptr<int>());
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    TORCH_CHECK(error == cudaSuccess, "CUDA error: ", cudaGetErrorString(error));

    // Process text segments on host
    // Each text segment is computed as a tensor of shape [3, seq_len] with all entries equal to text_segment_lengths[i].
    std::vector<torch::Tensor> text_positions_list;
    for (int i = 0; i < actual_segments; i++)
    {
        int seq_len = text_lengths_between_vision[i].item<int>();
        auto text_range = torch::zeros({3, seq_len}, options_long) + text_segment_lengths[i];
        text_positions_list.push_back(text_range);
    }

    // Interleave text and image segments
    std::vector<torch::Tensor> full_positions_list;
    // For each vision segment, first add its text positions then add its image positions.
    for (int i = 0; i < actual_segments; i++)
    {
        // Append text segment for vision segment i.
        full_positions_list.push_back(text_positions_list[i]);
        // Determine the slice boundaries for this vision segment’s image positions.
        int start = segment_offsets_vec[i];
        int seg_length = 0;
        if (i == actual_segments - 1)
            seg_length = total_image_positions - segment_offsets_vec[i];
        else
            seg_length = segment_offsets_vec[i + 1] - segment_offsets_vec[i];
        // Slice the image_positions_tensor for this segment.
        // (Kernel output is [total_image_positions, 3]; we want to obtain a tensor of shape [3, seg_length] as in the Python reference.)
        torch::Tensor image_segment = image_positions_tensor.slice(0, start, start + seg_length).t();
        full_positions_list.push_back(image_segment);
    }
    // If there are extra text tokens after the last vision segment, add them.
    int full_text_len = input_len - ends[actual_segments - 1].item<int>();
    if (full_text_len > 0)
    {
        int max_s = full_positions_list.back().max().item<int>() + 1;
        auto extra_text = torch::arange(full_text_len, options_long).view({1, -1}).expand({3, -1}) + max_s;
        full_positions_list.push_back(extra_text);
    }

    // Concatenate along dimension 1 (the "position" dimension), then transpose so that the final tensor is [total_tokens, 3].
    auto full_positions_concatenated = torch::cat(full_positions_list, 1);
    auto full_positions_concatenated_transposed = full_positions_concatenated.t();

    // Write final result to output tensor.
    out.copy_(full_positions_concatenated_transposed);
}
