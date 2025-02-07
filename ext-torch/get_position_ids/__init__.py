import torch

try:
    from ._ops import ops
except ImportError as e:
    # Fallback for local development.
    try:
        import _get_position_ids

        ops = torch.ops._get_position_ids
    except ImportError:
        raise e
    
def get_position_ids(out: torch.Tensor, input_ids: torch.Tensor, image_grid_thw: torch.Tensor) -> torch.Tensor:
    ops.get_position_ids(out, input_ids, image_grid_thw)
    return out