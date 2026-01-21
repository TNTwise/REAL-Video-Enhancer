"""
Spandrel is a library for loading and running pre-trained PyTorch models. It automatically detects the model architecture and hyper parameters from model files, and provides a unified interface for running models.
"""

__version__ = '0.4.0'

from .__helpers.canonicalize import canonicalize_state_dict
from .__helpers.loader import ModelLoader
from .__helpers.main_registry import MAIN_REGISTRY
from .__helpers.model_descriptor import (
    ArchId,
    Architecture,
    ImageModelDescriptor,
    MaskedImageModelDescriptor,
    ModelBase,
    ModelDescriptor,
    ModelTiling,
    Purpose,
    SizeRequirements,
    StateDict,
    UnsupportedDtypeError,
)
from .__helpers.registry import (
    ArchRegistry,
    ArchSupport,
    DuplicateArchitectureError,
    UnsupportedModelError,
)

__all__ = [
    'MAIN_REGISTRY',
    'ArchId',
    'ArchRegistry',
    'ArchSupport',
    'Architecture',
    'DuplicateArchitectureError',
    'ImageModelDescriptor',
    'MaskedImageModelDescriptor',
    'ModelBase',
    'ModelDescriptor',
    'ModelLoader',
    'ModelTiling',
    'Purpose',
    'SizeRequirements',
    'StateDict',
    'UnsupportedDtypeError',
    'UnsupportedModelError',
    'canonicalize_state_dict',
]
