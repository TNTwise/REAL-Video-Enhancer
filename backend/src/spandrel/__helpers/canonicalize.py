from __future__ import annotations

from .model_descriptor import StateDict


def remove_common_prefix(state_dict: StateDict, prefixes: list[str]) -> StateDict:
    if len(state_dict) > 0:
        for prefix in prefixes:
            if all(i.startswith(prefix) for i in state_dict.keys()):
                state_dict = {k[len(prefix) :]: v for k, v in state_dict.items()}
    return state_dict


def canonicalize_state_dict(state_dict: StateDict) -> StateDict:
    """
    Canonicalize a state dict.

    This function is used to canonicalize a state dict, so that it can be
    used for architecture detection and loading.

    This function is not intended to be used in production code.
    """

    # the real state dict might be inside a dict with a known key
    unwrap_keys = [
        "model_state_dict",
        "state_dict",
        "params_ema",
        "params-ema",
        "params",
        "model",
        "net",
    ]
    for unwrap_key in unwrap_keys:
        if unwrap_key in state_dict and isinstance(state_dict[unwrap_key], dict):
            state_dict = state_dict[unwrap_key]
            break

    # unwrap single key
    if len(state_dict) == 1:
        single = next(iter(state_dict.values()))
        if isinstance(single, dict):
            state_dict = single

    # remove known common prefixes
    state_dict = remove_common_prefix(state_dict, ["module.", "netG."])

    return state_dict
