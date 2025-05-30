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


class FillDynamicParamsError(CaskadeException):
    """Class for exceptions related to filling dynamic parameters in ``caskade``."""


class FillDynamicParamsArrayError(FillDynamicParamsError):
    """Class for exceptions related to filling dynamic parameters with ArrayLike objects in ``caskade``."""

    def __init__(self, name, input_params, dynamic_params):
        fullnumel = sum(max(1, prod(p.shape)) for p in dynamic_params)
        message = dedent(
            f"""
            For flattened {backend.array_type.__name__} input, the (last) dim of the {backend.array_type.__name__} should
            equal the sum of all flattened dynamic params ({fullnumel}).
            Input params shape {input_params.shape} does not match dynamic
            params shape of: {name}. 
            
            Registered dynamic params (name: shape):
            {', '.join(f"{repr(p)}: {str(p.shape)}" for p in dynamic_params)}"""
        )
        super().__init__(message)


class FillDynamicParamsSequenceError(FillDynamicParamsError):
    """Class for exceptions related to filling dynamic parameters with a sequence (list, tuple, etc.) in ``caskade``."""

    def __init__(self, name, input_params, dynamic_params, dynamic_modules):
        message = dedent(
            f"""
            Input params length ({len(input_params)}) does not match dynamic
            params length ({len(dynamic_params)}) or number of dynamic
            modules ({len(dynamic_modules)}) of: {name}.
            
            Registered dynamic modules: 
            {', '.join(repr(m) for m in dynamic_modules)}

            Registered dynamic params:
            {', '.join(repr(p) for p in dynamic_params)}"""
        )
        super().__init__(message)


class FillDynamicParamsMappingError(FillDynamicParamsError):
    """Class for exceptions related to filling dynamic parameters with a mapping (dict) in ``caskade``."""

    def __init__(self, name, children, dynamic_modules, missing_key=None, missing_param=None):
        message = dedent(
            f"""
            Input params key "{missing_key}" not found in children of: {name}. 
            
            All registered dynamic modules: 
            {', '.join(repr(m) for m in dynamic_modules)}

            Registered children:
            {', '.join(repr(c) for c in children.values())}"""
        )
        super().__init__(message)
