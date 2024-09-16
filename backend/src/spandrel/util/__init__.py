"""
A module containing commonly-used functionality to implement architectures.
"""

from __future__ import annotations

import functools
import inspect
import math
from typing import Any, Literal, Mapping, Protocol, TypeVar


class KeyCondition:
    """
    A condition that checks if a state dict has the given keys.
    """

    def __init__(
        self, kind: Literal["all", "any"], keys: tuple[str | KeyCondition, ...]
    ):
        self._keys = keys
        self._kind: Literal["all", "any"] = kind

    @staticmethod
    def has_all(*keys: str | KeyCondition) -> KeyCondition:
        return KeyCondition("all", keys)

    @staticmethod
    def has_any(*keys: str | KeyCondition) -> KeyCondition:
        return KeyCondition("any", keys)

    def __call__(self, state_dict: Mapping[str, object]) -> bool:
        def _detect(key: str | KeyCondition) -> bool:
            if isinstance(key, KeyCondition):
                return key(state_dict)
            return key in state_dict

        if self._kind == "all":
            return all(_detect(key) for key in self._keys)
        else:
            return any(_detect(key) for key in self._keys)


def get_first_seq_index(state_dict: Mapping[str, object], key_pattern: str) -> int:
    """
    Returns the maximum index `i` such that `key_pattern.format(str(i))` is in `state`.

    If no such key is in state, then `-1` is returned.

    Example:
        get_first_seq_index(state, "body.{}.weight") -> -1
        get_first_seq_index(state, "body.{}.weight") -> 3
    """
    for i in range(100):
        if key_pattern.format(str(i)) in state_dict:
            return i
    return -1


def get_seq_len(state_dict: Mapping[str, object], seq_key: str) -> int:
    """
    Returns the length of a sequence in the state dict.

    The length is detected by finding the maximum index `i` such that
    `{seq_key}.{i}.{suffix}` is in `state` for some suffix.

    Example:
        get_seq_len(state, "body") -> 5
    """
    prefix = seq_key + "."

    keys: set[int] = set()
    for k in state_dict.keys():
        if k.startswith(prefix):
            index = k[len(prefix) :].split(".", maxsplit=1)[0]
            keys.add(int(index))

    if len(keys) == 0:
        return 0
    return max(keys) + 1


def get_scale_and_output_channels(x: int, input_channels: int) -> tuple[int, int]:
    """
    Returns a scale and number of output channels such that `scale**2 * out_nc = x`.

    This is commonly used for pixelshuffel layers.
    """
    # Unfortunately, we do not have enough information to determine both the scale and
    # number output channels correctly *in general*. However, we can make some
    # assumptions to make it good enough.
    #
    # What we know:
    # - x = scale * scale * output_channels
    # - output_channels is likely equal to input_channels
    # - output_channels and input_channels is likely 1, 3, or 4
    # - scale is likely 1, 2, 4, or 8

    def is_square(n: int) -> bool:
        return math.sqrt(n) == int(math.sqrt(n))

    # just try out a few candidates and see which ones fulfill the requirements
    candidates = [input_channels, 3, 4, 1]
    for c in candidates:
        if x % c == 0 and is_square(x // c):
            return int(math.sqrt(x // c)), c

    raise AssertionError(
        f"Expected output channels to be either 1, 3, or 4."
        f" Could not find a pair (scale, out_nc) such that `scale**2 * out_nc = {x}`"
    )


def get_pixelshuffle_params(
    state_dict: Mapping[str, object],
    upsample_key: str = "upsample",
    default_nf: int = 64,
) -> tuple[int, int]:
    """
    This will detect the upscale factor and number of features of a pixelshuffle module in the state dict.

    A pixelshuffle module is a sequence of alternating up convolutions and pixelshuffle.
    The class of this module is commonyl called `Upsample`.
    Examples of such modules can be found in most SISR architectures, such as SwinIR, HAT, RGT, and many more.
    """
    upscale = 1
    num_feat = default_nf

    for i in range(0, 10, 2):
        key = f"{upsample_key}.{i}.weight"
        if key not in state_dict:
            break

        tensor = state_dict[key]
        # we'll assume that the state dict contains tensors
        shape: tuple[int, ...] = tensor.shape  # type: ignore
        num_feat = shape[1]
        upscale *= math.isqrt(shape[0] // num_feat)

    return upscale, num_feat


def store_hyperparameters(*, extra_parameters: Mapping[str, object] = {}):
    """
    Stores the hyperparameters of a class in a `hyperparameters` attribute.
    """

    def get_arg_defaults(spec: inspect.FullArgSpec) -> dict[str, Any]:
        defaults = {}
        if spec.kwonlydefaults is not None:
            defaults = spec.kwonlydefaults

        if spec.defaults is not None:
            defaults = {
                **defaults,
                **dict(zip(spec.args[-len(spec.defaults) :], spec.defaults)),
            }

        return defaults

    class WithHyperparameters(Protocol):
        hyperparameters: dict[str, Any]

    C = TypeVar("C", bound=WithHyperparameters)

    def inner(cls: type[C]) -> type[C]:
        old_init = cls.__init__

        spec = inspect.getfullargspec(old_init)
        defaults = get_arg_defaults(spec)

        if spec.varargs is not None:
            raise UserWarning(
                "Class has *args, which is not allowed in combination with @store_hyperparameters"
            )
        if spec.varkw is not None:
            raise UserWarning(
                "Class has **kwargs, which is not allowed in combination with @store_hyperparameters"
            )
        if spec.args != ["self"]:
            raise UserWarning(
                "@store_hyperparameters requires all arguments of `"
                + cls.__name__
                + ".__init__` after `self` to be keyword arguments. Use `def __init__(self, *, a, b, c):`."
            )

        @functools.wraps(old_init)
        def new_init(self: C, **kwargs):
            # remove extra parameters from kwargs
            for k, v in extra_parameters.items():
                if k in kwargs:
                    if kwargs[k] != v:
                        raise ValueError(
                            f"Expected hyperparameter {k} to be {v}, but got {kwargs[k]}"
                        )
                    del kwargs[k]

            self.hyperparameters = {**extra_parameters, **defaults, **kwargs}
            old_init(self, **kwargs)

        cls.__init__ = new_init
        return cls

    return inner


__all__ = [
    "get_first_seq_index",
    "get_pixelshuffle_params",
    "get_scale_and_output_channels",
    "get_seq_len",
    "KeyCondition",
    "store_hyperparameters",
]
