from math import prod
from textwrap import dedent


class CaskadeException(Exception):
    """Base class for all exceptions in ``caskade``."""


class GraphError(CaskadeException):
    """Class for graph exceptions in ``caskade``."""


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


class FillDynamicParamsTensorError(FillDynamicParamsError):
    """Class for exceptions related to filling dynamic parameters with a tensor in ``caskade``."""

    def __init__(self, name, input_params, dynamic_params):
        fullnumel = sum(max(1, prod(p.shape)) for p in dynamic_params)
        message = dedent(
            f"""
            For flattened Tensor input, the (last) dim of the Tensor should
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
        if missing_key is not None:
            message = dedent(
                f"""
                Input params key "{missing_key}" not found in dynamic modules or children of: {name}. 
                
                Registered dynamic modules: 
                {', '.join(repr(m) for m in dynamic_modules)}

                Registered dynamic children:
                {', '.join(repr(c) for c in children.values() if c.dynamic)}"""
            )
        else:
            message = dedent(
                f"""
                Dynamic param "{missing_param.name}" not filled with given input params dict passed to {name}.

                Dynamic param parent(s):
                {', '.join(repr(p) for p in missing_param.parents)}
                
                Registered dynamic modules: 
                {', '.join(repr(m) for m in dynamic_modules)}

                Registered dynamic children:
                {', '.join(repr(c) for c in children.values() if c.dynamic)}"""
            )
        super().__init__(message)
