from math import prod
from textwrap import dedent

from .backend import backend


class CaskadeException(Exception):
    """Base class for all exceptions in ``caskade``."""


class GraphError(CaskadeException):
    """Class for graph exceptions in ``caskade``."""


class BackendError(CaskadeException):
    """Class for exceptions related to the backend in ``caskade``."""


class LinkToAttributeError(GraphError):
    """Class for exceptions related to linking to an attribute in ``caskade``."""


class NodeConfigurationError(CaskadeException):
    """Class for node configuration exceptions in ``caskade``."""


class ParamConfigurationError(NodeConfigurationError):
    """Class for parameter configuration exceptions in ``caskade``."""


class ParamTypeError(CaskadeException):
    """Class for exceptions related to the type of a parameter in ``caskade``."""


class ActiveStateError(CaskadeException):
    """Class for exceptions related to the active state of a node in ``caskade``."""


class FillParamsError(CaskadeException):
    """Class for exceptions related to filling parameters in ``caskade``"""


class FillParamsArrayError(FillParamsError):
    """Class for exceptions related to filling parameters with ArrayLike objects in ``caskade``."""

    def __init__(self, name, input_params, params):
        fullnumel = sum(max(1, prod(p.shape)) for p in params)
        message = dedent(
            f"""
            For flattened {backend.array_type.__name__} input, the (last) dim of
            the {backend.array_type.__name__} should equal the sum of all
            flattened params ({fullnumel}). Input params shape
            {input_params.shape} does not match params shape of: {name}. 
            
            Registered params (name: shape): 
            {', '.join(f"{repr(p)}: {str(p.shape)}" for p in params)}"""
        )
        super().__init__(message)


class FillParamsSequenceError(FillParamsError):
    """Class for exceptions related to filling parameters with a sequence (list, tuple, etc.) in ``caskade``."""

    def __init__(self, name, input_params, dynamic_params):
        message = dedent(
            f"""
            Input params length ({len(input_params)}) does not match
            params length ({len(dynamic_params)}) of: {name}.
            
            Registered dynamic params:
            {', '.join(repr(p) for p in dynamic_params)}"""
        )
        super().__init__(message)


class FillParamsMappingError(FillParamsError):
    """Class for exceptions related to filling parameters with a mapping (dict) in ``caskade``."""

    def __init__(self, name, children, missing_key=None):
        message = dedent(
            f"""
            Input params key "{missing_key}" not found in children of: {name}. 
            
            Registered children:
            {', '.join(repr(c) for c in children.values())}"""
        )
        super().__init__(message)
