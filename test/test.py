import time
import torch
import pytest
import get_position_ids # noqa: E402
from reference import DummyModel

# Each configuration includes:
#   - name: A label for the test case.
#   - input_ids: A list of token IDs (with vision start (151652) and vision end (151653) tokens embedded).
#   - grid: A list of [t, h, w] values (one per vision segment).
#
# The cases below include:
#   1. one_segment: a single vision segment.
#   2. two_segments: two vision segments with extra text tokens afterward.
#   3. three_segments: three vision segments.
VISION_CONFIGS = [
    {
        "name": "one_segment",
        "input_ids": (
            [10] * 5 +            # 5 text tokens before vision segment
            [151652, 151653] +    # vision tokens for segment 1
            [20] * 5              # 5 extra text tokens after vision segment
        ),
        "grid": [[2, 4, 6]]       # one vision segment grid
    },
    {
        "name": "two_segments",
        "input_ids": (
            [100] * 5 +           # 5 text tokens for segment 1
            [151652, 151653] +    # vision tokens for segment 1
            [101] * 5 +           # 5 text tokens for segment 2
            [151652, 151653] +    # vision tokens for segment 2
            [102] * 5             # 5 extra text tokens after last vision segment
        ),
        "grid": [
            [2, 4, 6],          # vision segment 1 grid
            [3, 4, 6]           # vision segment 2 grid
        ],
    },
    {
        "name": "three_segments",
        "input_ids": (
            [11] * 5 +            # Segment 1: 5 text tokens
            [151652, 151653] +    # vision tokens for segment 1
            [12] * 6 +            # Segment 2: 6 text tokens
            [151652, 151653] +    # vision tokens for segment 2
            [13] * 7 +            # Segment 3: 7 text tokens
            [151652, 151653] +    # vision tokens for segment 3
            [14] * 8              # 8 extra text tokens after the last vision segment
        ),
        "grid": [
            [2, 4, 6],          # vision segment 1 grid
            [3, 6, 6],          # vision segment 2 grid
            [4, 4, 8]           # vision segment 3 grid
        ],
    },
]

CUDA_DEVICES = ["cuda"]     # List of CUDA devices; you can add more if needed.
SEEDS = [42]                # Seeds for reproducibility.
DTYPES = [torch.int32]      # In our test the tokens and grid are created with int32.


@pytest.mark.parametrize("vision_config", 
                         VISION_CONFIGS, 
                         ids=[cfg["name"] for cfg in VISION_CONFIGS])
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_get_position_ids(vision_config, seed, device):
    torch.manual_seed(seed)
    input_ids = torch.tensor(vision_config["input_ids"], dtype=torch.int32, device=device)
    image_grid_thw = torch.tensor(vision_config["grid"], dtype=torch.int32, device=device)

    # Create a DummyModel instance from the reference implementation.
    dummy_model = DummyModel()

    # reference implementation
    torch.cuda.synchronize()
    start_ref = time.perf_counter()
    pos_ids_ref = dummy_model.get_position_ids(input_ids, image_grid_thw)
    torch.cuda.synchronize()
    end_ref = time.perf_counter()
    ref_time = (end_ref - start_ref) * 1000  # ms
    print(f"\nVision config {vision_config['name']} - Reference time: {ref_time:.2f} ms")
    # Convert reference output to int32 for comparison (since its returned as a float tensor).
    pos_ids_ref = pos_ids_ref.to(dtype=torch.int32)

    # kernel implementation
    torch.cuda.synchronize()
    start_ext = time.perf_counter()
    out = torch.empty(pos_ids_ref.shape, dtype=torch.int32, device=device)
    get_position_ids.get_position_ids(out, input_ids, image_grid_thw)
    torch.cuda.synchronize()
    end_ext = time.perf_counter()
    ext_time = (end_ext - start_ext) * 1000  # ms
    print(f"Vision config {vision_config['name']} - Extension time: {ext_time:.2f} ms\n")
    ext_out = out.clone()

    # verify the results
    torch.testing.assert_close(ext_out.cpu(), pos_ids_ref.cpu())
