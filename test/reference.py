import torch
from typing import Optional

class DummyModel:
    spatial_merge_size = 2
    vision_start_token_id = 151652
    vision_end_token_id = 151653
    
    # based on https://github.com/huggingface/transformers/blob/e284c7e954abe12c34b50461c17f8115a0afe115/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py#L1391
    # modified to first find segments then initialize position ids for each segment
    # Steps:
    #  locate all vision and text segments
    #  calculate `vision_segment_lengths` for each vision segment to be use as offset
    #  calculate `text_segment_lengths` for each text segment to be used as offset
    #  create position ids for each vision segment based on the image grid
    #  create position ids for each text segment
    #  combine all the position ids
    #  the final segment is the difference between the last vision segment and the end of the input
    #  combine all the position ids and reshape to (3, input_ids_len) then swap dimensions to (input_ids_len, 3)
    def get_position_ids(
        self,
        input_ids: torch.Tensor,
        image_grid_thw: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if image_grid_thw is None:
            return (
                torch.arange(input_ids.shape[0], device=input_ids.device)
                .unsqueeze(1)
                .repeat(1, 3)
            )

        spatial_merge_size = self.spatial_merge_size
        vision_start_token_id = self.vision_start_token_id
        vision_end_token_id = self.vision_end_token_id
        device = input_ids.device
        dtype = input_ids.dtype
        input_ids_len = input_ids.shape[0]

        vision_starts = torch.where(input_ids == vision_start_token_id)[0]
        vision_ends = torch.where(input_ids == vision_end_token_id)[0]
        vision_segments = torch.stack((vision_starts, vision_ends), dim=1)
        prev_vision_end = torch.cat(
            [torch.zeros(1, device=vision_ends.device, dtype=dtype), vision_ends[:-1]]
        )
        text_lengths_between_vision = vision_segments[:, 0] - prev_vision_end + 1
        vision_widths_max = torch.cat(
            [
                torch.zeros(1, device=image_grid_thw.device, dtype=dtype),
                image_grid_thw[:-1, 2] // spatial_merge_size,
            ]
        )
        vision_segment_lengths = vision_widths_max + text_lengths_between_vision
        vision_segment_lengths = vision_segment_lengths.cumsum(dim=0)
        text_segment_lengths = vision_segment_lengths - text_lengths_between_vision

        # create position ids for each vision segment based on the image grid
        llm_pos_ids_list = []
        for i, _ in enumerate(vision_segments):
            t, h, w = (
                image_grid_thw[i][0],
                image_grid_thw[i][1] // spatial_merge_size,
                image_grid_thw[i][2] // spatial_merge_size,
            )
            t_indices = torch.arange(t, device=device).repeat_interleave(h * w)
            h_indices = torch.arange(h, device=device).repeat_interleave(w).repeat(t)
            w_indices = torch.arange(w, device=device).repeat(t * h)
            image_position_ids = torch.stack([t_indices, h_indices, w_indices], dim=0)

            # offset by the position of the last vision segment
            im = image_position_ids + vision_segment_lengths[i]
            llm_pos_ids_list.append(im)

        # create position ids for each text segment
        text_ranges = [
            torch.zeros(3, seq_len, device=device) + text_segment_lengths[i]
            for i, seq_len in enumerate(text_lengths_between_vision)
        ]

        full_llm_pos_ids_list = [
            item for sublist in zip(text_ranges, llm_pos_ids_list) for item in sublist
        ]
        max_s = full_llm_pos_ids_list[-1].max() + 1
        final_text_len = input_ids_len - vision_ends[-1]
        if final_text_len > 0:
            m = torch.arange(final_text_len, device=device).view(1, -1).expand(3, -1)
            full_llm_pos_ids_list.append(m + max_s)

        position_ids = (
            torch.cat(full_llm_pos_ids_list, dim=1).reshape(3, -1).transpose(0, 1)
        )
        return position_ids