"""Layer/Module Helpers
Hacked together by / Copyright 2020 Ross Wightman
"""

from __future__ import annotations

import collections.abc
from itertools import repeat
from typing import Iterable, TypeVar

T = TypeVar("T")


def _to_n(x: T | Iterable[T], n: int):
    if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
        return x  # type: ignore
    return tuple(repeat(x, n))  # type: ignore


def to_1tuple(x: T | Iterable[T]) -> tuple[T]:
    return _to_n(x, 1)  # type: ignore


def to_2tuple(x: T | Iterable[T]) -> tuple[T, T]:
    return _to_n(x, 2)  # type: ignore


def to_3tuple(x: T | Iterable[T]) -> tuple[T, T, T]:
    return _to_n(x, 3)  # type: ignore


def to_4tuple(x: T | Iterable[T]) -> tuple[T, T, T, T]:
    return _to_n(x, 4)  # type: ignore
