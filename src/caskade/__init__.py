"""
caskade - Build scientific simulators as directed acyclic graphs.

Caskade provides a framework for constructing modular scientific simulators
by composing computational steps into a directed acyclic graph (DAG). It
handles parameter management, caching, and context-dependent evaluation.

Main Public API
---------------
Node : Base class for building computational graph nodes.
Module : High-level container for assembling simulator components.
Param : Declare and manage parameters within nodes.
forward : Decorator to define the forward computation of a node.
active_cache : Decorator for caching intermediate results.
NodeCollection, NodeList, NodeTuple : Collections of nodes.
ActiveContext, ValidContext, OverrideParam : Context managers for evaluation.
backend : Array backend abstraction (NumPy, PyTorch, etc.).
utils : Utility functions.
"""
from ._version import version as VERSION  # noqa

from .base import Node, Memo
from .backend import backend, ArrayLike
from .context import ActiveContext, ValidContext, OverrideParam
from .decorators import forward, active_cache
from .module import Module
from .param import Param
from .collection import NodeCollection, NodeList, NodeTuple
from .tests import test
from .errors import (
    CaskadeException,
    GraphError,
    BackendError,
    LinkToAttributeError,
    NodeConfigurationError,
    ParamConfigurationError,
    ParamTypeError,
    ActiveStateError,
    FillParamsError,
    FillParamsArrayError,
    FillParamsSequenceError,
    FillParamsMappingError,
)
from .warnings import CaskadeWarning, InvalidValueWarning, SaveStateWarning
from . import utils


__version__ = VERSION
__author__ = "Connor Stone and Alexandre Adam"

__all__ = (
    "Node",
    "Memo",
    "backend",
    "ArrayLike",
    "Module",
    "Param",
    "NodeCollection",
    "NodeList",
    "NodeTuple",
    "ActiveContext",
    "ValidContext",
    "OverrideParam",
    "forward",
    "active_cache",
    "test",
    "CaskadeException",
    "GraphError",
    "BackendError",
    "LinkToAttributeError",
    "NodeConfigurationError",
    "ParamConfigurationError",
    "ParamTypeError",
    "ActiveStateError",
    "FillParamsError",
    "FillParamsArrayError",
    "FillParamsSequenceError",
    "FillParamsMappingError",
    "CaskadeWarning",
    "InvalidValueWarning",
    "SaveStateWarning",
    "utils",
)
