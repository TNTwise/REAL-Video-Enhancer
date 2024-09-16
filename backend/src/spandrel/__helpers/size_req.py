from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SizeRequirements:
    """
    A set of requirements for the size of an input image.
    """

    minimum: int = 0
    """
    The minimum size of the input image in pixels.

    `minimum` is guaranteed to be a multiple of `multiple_of` and to be >= 0.

    On initialization, if `minimum` is not a multiple of `multiple_of`, it will be rounded up to the next multiple of `multiple_of`.

    Default/neutral value: `0`
    """
    multiple_of: int = 1
    """
    The width and height of the image must be a multiple of this value.

    `multiple_of` is guaranteed to be >= 1.

    Default/neutral value: `1`
    """
    square: bool = False
    """
    The image must be square.

    Default/neutral value: `False`
    """

    def __post_init__(self):
        assert self.minimum >= 0, "minimum must be >= 0"
        assert self.multiple_of >= 1, "multiple_of must be >= 1"

        if self.minimum % self.multiple_of != 0:
            self.minimum = (self.minimum // self.multiple_of + 1) * self.multiple_of

    @property
    def none(self) -> bool:
        """
        Returns True if no size requirements are specified.

        If True, then `check` is guaranteed to always return True.
        """
        return self.minimum == 0 and self.multiple_of == 1 and not self.square

    def check(self, width: int, height: int) -> bool:
        """
        Returns whether the given width and height satisfy the size requirements.
        """
        return self.get_padding(width, height) == (0, 0)

    def get_padding(self, width: int, height: int) -> tuple[int, int]:
        """
        Given an image size, this returns the minimum amount of padding necessary to satisfy the size requirements. The returned padding is in the format `(pad_width, pad_height)` and is guaranteed to be non-negative.
        """

        def ceil_modulo(x: int, mod: int) -> int:
            if x % mod == 0:
                return x
            return (x // mod + 1) * mod

        w: int = max(self.minimum, width)
        h: int = max(self.minimum, height)

        w = ceil_modulo(w, self.multiple_of)
        h = ceil_modulo(h, self.multiple_of)

        if self.square:
            w = h = max(w, h)

        return w - width, h - height


def pad_tensor(t: torch.Tensor, req: SizeRequirements):
    w = t.shape[-1]
    h = t.shape[-2]

    pad_w, pad_h = req.get_padding(w, h)

    if pad_w or pad_h:
        # reflect padding only allows a maximum padding of size - 1
        reflect_pad_w = min(pad_w, w - 1)
        reflect_pad_h = min(pad_h, h - 1)
        t = torch.nn.functional.pad(t, (0, reflect_pad_w, 0, reflect_pad_h), "reflect")

        # do the rest of the padding (if any) with replicate, which has no such restrictions
        pad_w -= reflect_pad_w
        pad_h -= reflect_pad_h
        t = torch.nn.functional.pad(t, (0, pad_w, 0, pad_h), "replicate")

        return True, t
    else:
        return False, t
