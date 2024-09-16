from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Mapping, Sequence

import torch

from .canonicalize import canonicalize_state_dict
from .model_descriptor import ArchId, Architecture, ModelDescriptor, StateDict


class UnsupportedModelError(Exception):
    """
    An error that will be thrown by `ArchRegistry` and `ModelLoader` if a model architecture is not supported.
    """


class DuplicateArchitectureError(ValueError):
    """
    An error that will be thrown by `ArchRegistry` if the same architecture is added twice.
    """


@dataclass(frozen=True)
class ArchSupport:
    """
    An entry in an `ArchRegistry` that describes how to detect and load a model architecture.
    """

    architecture: Architecture[torch.nn.Module]
    """
    The architecture.
    """
    detect: Callable[[StateDict], bool]
    """
    Inspects the given state dict and returns True if this architecture is detected.

    For most architectures, this will be the architecture's `detect` method.
    """
    before: tuple[ArchId, ...] = ()
    """
    This architecture is detected before the architectures with the given IDs.

    See the documentation of `ArchRegistry` for more information on ordering.
    """

    @staticmethod
    def from_architecture(
        arch: Architecture[torch.nn.Module], before: tuple[ArchId, ...] = ()
    ) -> ArchSupport:
        """
        Creates an `ArchSupport` from an `Architecture` by using the architecture's ``detect`` method.
        """
        return ArchSupport(arch, arch.detect, before)


class ArchRegistry:
    """
    A registry of architectures.

    Architectures are detected/loaded in insertion order unless `before` is specified.
    """

    def __init__(self):
        # the registry is copy on write internally
        self._architectures: Sequence[ArchSupport] = []
        self._ordered: Sequence[ArchSupport] = []
        self._by_id: Mapping[ArchId, ArchSupport] = {}

    def copy(self) -> ArchRegistry:
        """
        Returns a copy of the registry.
        """
        new = ArchRegistry()
        new._architectures = self._architectures
        new._ordered = self._ordered
        new._by_id = self._by_id
        return new

    def __contains__(self, id: ArchId | str) -> bool:
        return id in self._by_id

    def __getitem__(self, id: str | ArchId) -> ArchSupport:
        return self._by_id[ArchId(id)]

    def __iter__(self):
        """
        Returns an iterator over all architectures in insertion order.
        """
        return iter(self.architectures("insertion"))

    def __len__(self) -> int:
        return len(self._architectures)

    def get(self, id: str | ArchId) -> ArchSupport | None:
        return self._by_id.get(ArchId(id), None)

    def architectures(
        self,
        order: Literal["insertion", "detection"] = "insertion",
    ) -> list[ArchSupport]:
        """
        Returns a new list with all architectures in the registry.

        The order of architectures in the list is either insertion order or the order in which architectures are detected.
        """
        if order == "insertion":
            return list(self._architectures)
        elif order == "detection":
            return list(self._ordered)
        else:
            raise ValueError(f"Invalid order: {order}")

    def add(
        self,
        *architectures: ArchSupport,
        ignore_duplicates: bool = False,
    ) -> list[ArchSupport]:
        """
        Adds the given architectures to the registry.

        Throws an error if an architecture with the same ID already exists,
        unless `ignore_duplicates` is True, in which case the old architecture is retained.

        Throws an error if a circular dependency of `before` references is detected.

        If an error is thrown, the registry is left unchanged.

        Returns a list of architectures that were added.
        """

        new_architectures = list(self._architectures)
        new_by_id = dict(self._by_id)
        added = []
        for arch in architectures:
            if arch.architecture.id in new_by_id:
                if ignore_duplicates:
                    continue
                raise DuplicateArchitectureError(
                    f"Duplicate architecture: {arch.architecture.id}"
                )

            new_architectures.append(arch)
            new_by_id[arch.architecture.id] = arch
            added.append(arch)

        new_ordered = ArchRegistry._get_ordered(new_architectures)

        self._architectures = new_architectures
        self._ordered = new_ordered
        self._by_id = new_by_id
        return added

    @staticmethod
    def _get_ordered(architectures: list[ArchSupport]) -> list[ArchSupport]:
        inv_before: dict[ArchId, list[ArchId]] = {}
        by_id: dict[ArchId, ArchSupport] = {}
        for arch in architectures:
            by_id[arch.architecture.id] = arch
            for before in arch.before:
                if before not in inv_before:
                    inv_before[before] = []
                inv_before[before].append(arch.architecture.id)

        ordered: list[ArchSupport] = []
        seen: set[ArchSupport] = set()
        stack: list[ArchId] = []

        def visit(arch: ArchSupport):
            if arch.architecture.id in stack:
                raise ValueError(
                    f"Circular dependency in architecture detection: {' -> '.join([*stack, arch.architecture.id])}"
                )
            if arch in seen:
                return
            seen.add(arch)
            stack.append(arch.architecture.id)

            for before in inv_before.get(arch.architecture.id, []):
                visit(by_id[before])

            ordered.append(arch)
            stack.pop()

        for arch in architectures:
            visit(arch)

        return ordered

    def load(self, state_dict: StateDict) -> ModelDescriptor:
        """
        Detects the architecture of the given state dict and loads it.

        This will canonicalize the state dict if it isn't already.

        Throws an `UnsupportedModelError` if the model architecture is not supported.
        """

        state_dict = canonicalize_state_dict(state_dict)

        for arch in self._ordered:
            if arch.detect(state_dict):
                return arch.architecture.load(state_dict)

        raise UnsupportedModelError
