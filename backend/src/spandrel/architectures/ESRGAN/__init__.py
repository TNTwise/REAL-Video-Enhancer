from __future__ import annotations

import functools
import math
import re
from collections import OrderedDict

from typing_extensions import override

from spandrel.util import KeyCondition, get_seq_len

from ...__helpers.model_descriptor import (
    Architecture,
    ImageModelDescriptor,
    SizeRequirements,
    StateDict,
)
from .__arch.RRDB import RRDBNet as ESRGAN


def _new_to_old_arch(state: StateDict, state_map: dict, num_blocks: int):
    """Convert a new-arch model state dictionary to an old-arch dictionary."""
    if "params_ema" in state:
        state = state["params_ema"]

    if "conv_first.weight" not in state:
        # model is already old arch, this is a loose check, but should be sufficient
        return state

    # add nb to state keys
    for kind in ("weight", "bias"):
        state_map[f"model.1.sub.{num_blocks}.{kind}"] = state_map[
            f"model.1.sub./NB/.{kind}"
        ]
        del state_map[f"model.1.sub./NB/.{kind}"]

    old_state = OrderedDict()
    for old_key, new_keys in state_map.items():
        for new_key in new_keys:
            if r"\1" in old_key:
                for k, v in state.items():
                    sub = re.sub(new_key, old_key, k)
                    if sub != k:
                        old_state[sub] = v
            else:
                if new_key in state:
                    old_state[old_key] = state[new_key]

    # upconv layers
    max_upconv = 0
    for key in state.keys():
        match = re.match(r"(upconv|conv_up)(\d)\.(weight|bias)", key)
        if match is not None:
            _, key_num, key_type = match.groups()
            old_state[f"model.{int(key_num) * 3}.{key_type}"] = state[key]
            max_upconv = max(max_upconv, int(key_num) * 3)

    # final layers
    for key in state.keys():
        if key in ("HRconv.weight", "conv_hr.weight"):
            old_state[f"model.{max_upconv + 2}.weight"] = state[key]
        elif key in ("HRconv.bias", "conv_hr.bias"):
            old_state[f"model.{max_upconv + 2}.bias"] = state[key]
        elif key in ("conv_last.weight",):
            old_state[f"model.{max_upconv + 4}.weight"] = state[key]
        elif key in ("conv_last.bias",):
            old_state[f"model.{max_upconv + 4}.bias"] = state[key]

    # Sort by first numeric value of each layer
    def compare(item1: str, item2: str):
        parts1 = item1.split(".")
        parts2 = item2.split(".")
        int1 = int(parts1[1])
        int2 = int(parts2[1])
        return int1 - int2

    sorted_keys = sorted(old_state.keys(), key=functools.cmp_to_key(compare))

    # Rebuild the output dict in the right order
    out_dict = OrderedDict((k, old_state[k]) for k in sorted_keys)

    return out_dict


def _get_scale(state: StateDict) -> int:
    # model is composed of a few blocks that look like this flattened:
    #
    #   Conv2d
    #   B.ShortcutBlock
    #   [nn.Upsample, Conv2d, nn.LeakyReLU] for i in range(log2(scale))
    #   Conv2d
    #   nn.LeakyReLU (activation)
    #   Conv2d
    seq_len = get_seq_len(state, "model")
    log2_scale = (seq_len - 5) // 3
    return 2**log2_scale


def _get_num_blocks(state: StateDict, state_map: dict) -> int:
    nbs = []
    state_keys = state_map[r"model.1.sub.\1.RDB\2.conv\3.0.\4"] + (
        r"model\.\d+\.sub\.(\d+)\.RDB(\d+)\.conv(\d+)\.0\.(weight|bias)",
    )
    for state_key in state_keys:
        for k in state:
            m = re.search(state_key, k)
            if m:
                nbs.append(int(m.group(1)))
        if nbs:
            break
    return max(*nbs) + 1


def _to_old_arch(state: StateDict) -> StateDict:
    state_map = {
        # currently supports old, new, and newer RRDBNet arch models
        # ESRGAN, BSRGAN/RealSR, Real-ESRGAN
        "model.0.weight": ("conv_first.weight",),
        "model.0.bias": ("conv_first.bias",),
        "model.1.sub./NB/.weight": ("trunk_conv.weight", "conv_body.weight"),
        "model.1.sub./NB/.bias": ("trunk_conv.bias", "conv_body.bias"),
        r"model.1.sub.\1.RDB\2.conv\3.0.\4": (
            r"RRDB_trunk\.(\d+)\.RDB(\d)\.conv(\d+)\.(weight|bias)",
            r"body\.(\d+)\.rdb(\d)\.conv(\d+)\.(weight|bias)",
        ),
    }

    if "params_ema" in state:
        state = state["params_ema"]

    num_blocks = _get_num_blocks(state, state_map)
    return _new_to_old_arch(state, state_map, num_blocks)


class ESRGANArch(Architecture[ESRGAN]):
    def __init__(self) -> None:
        super().__init__(
            id="ESRGAN",
            detect=KeyCondition.has_any(
                KeyCondition.has_all(
                    "model.0.weight",
                    "model.1.sub.0.RDB1.conv1.0.weight",
                ),
                KeyCondition.has_all(
                    "conv_first.weight",
                    "body.0.rdb1.conv1.weight",
                    "conv_body.weight",
                    "conv_last.weight",
                ),
                KeyCondition.has_all(
                    # BSRGAN/RealSR
                    "conv_first.weight",
                    "RRDB_trunk.0.RDB1.conv1.weight",
                    "trunk_conv.weight",
                    "conv_last.weight",
                ),
                KeyCondition.has_all(
                    # ESRGAN+
                    "model.0.weight",
                    "model.1.sub.0.RDB1.conv1x1.weight",
                ),
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[ESRGAN]:
        # default values
        in_nc: int = 3
        out_nc: int = 3
        num_filters: int = 64
        num_blocks: int = 23
        scale: int = 4
        plus: bool = False
        shuffle_factor: int | None = None

        state_dict = _to_old_arch(state_dict)

        model_seq_len = get_seq_len(state_dict, "model")

        in_nc = state_dict["model.0.weight"].shape[1]
        out_nc = state_dict[f"model.{model_seq_len-1}.weight"].shape[0]

        scale = _get_scale(state_dict)
        num_blocks = get_seq_len(state_dict, "model.1.sub") - 1
        num_filters = state_dict["model.0.weight"].shape[0]

        if any(".conv1x1." in k for k in state_dict.keys()):
            plus = True

        # Detect if pixelunshuffle was used (Real-ESRGAN)
        if in_nc in (out_nc * 4, out_nc * 16) and out_nc in (
            in_nc / 4,
            in_nc / 16,
        ):
            shuffle_factor = int(math.sqrt(in_nc / out_nc))
        else:
            shuffle_factor = None

        model = ESRGAN(
            in_nc=in_nc,
            out_nc=out_nc,
            num_filters=num_filters,
            num_blocks=num_blocks,
            scale=scale,
            plus=plus,
            shuffle_factor=shuffle_factor,
        )
        tags = [
            f"{num_filters}nf",
            f"{num_blocks}nb",
        ]
        if plus:
            tags.insert(0, "ESRGAN+")

        # Adjust these properties for calculations outside of the model
        if shuffle_factor:
            in_nc //= shuffle_factor**2
            scale //= shuffle_factor

        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="Restoration" if scale == 1 else "SR",
            tags=tags,
            supports_half=True,
            supports_bfloat16=True,
            scale=scale,
            input_channels=in_nc,
            output_channels=out_nc,
            size_requirements=SizeRequirements(
                minimum=2,
                multiple_of=4 if shuffle_factor else 1,
            ),
        )


__all__ = ["ESRGANArch", "ESRGAN"]
