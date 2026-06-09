import torch
import torch.nn.functional as F


def pad_to_multiple(
    tensor: torch.Tensor,
    multiple: int,
    *,
    mode: str,
    value: float = 0.0,
) -> torch.Tensor:
    """
    Pad a tensor's size to a multiple of a number.

    Args:
        tensor: Tensor to pad.
        multiple: Size multiple to pad to.
        mode: Padding mode; see `torch.nn.functional.pad`.
        value: Padding value; see `torch.nn.functional.pad`.

    Returns:
        Padded tensor, or the original tensor if no padding was needed.
    """
    _, _, h, w = tensor.size()
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h or pad_w:
        return F.pad(tensor, (0, pad_w, 0, pad_h), mode, value)
    return tensor
