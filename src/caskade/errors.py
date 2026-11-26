from math import prod
from textwrap import dedent

from .backend import backend


class CaskadeException(Exception):
    """
    Base class for all exceptions in ``caskade``.

    All custom exceptions raised by ``caskade`` inherit from this class,
    allowing users to catch any ``caskade``-specific error with a single
    except clause.
    """


class GraphError(CaskadeException):
    """
    Exception for graph-related errors in ``caskade``.

    Raised when an operation on the computational graph is invalid, such as
    creating cycles or referencing nonexistent nodes.
    """


class BackendError(CaskadeException):
    """
    Exception for backend-related errors in ``caskade``.

    Raised when the selected numerical backend encounters an unsupported
    operation or configuration issue.
    """


class LinkToAttributeError(GraphError):
    """
    Exception raised when linking to an attribute fails.

    Raised when a child would be linked under the name of a property of the
    parent, such as ``name`` or ``children``. A property decides what its name
    gives, so a child cannot take its place. Any other attribute is replaced by
    the child, with a ``LinkToAttributeWarning``.
    """


class NodeConfigurationError(CaskadeException):
    """
    Exception for node configuration errors in ``caskade``.

    Raised when a node is configured with invalid or incompatible settings.
    """


class ParamConfigurationError(NodeConfigurationError):
    """
    Exception for parameter configuration errors in ``caskade``.

    Raised when a parameter is defined with an invalid shape, type, or
    constraint.
    """


class ParamTypeError(CaskadeException):
    """
    Exception for parameter type errors in ``caskade``.

    Raised when a value assigned to a parameter does not match its
    expected type.
    """


class ActiveStateError(CaskadeException):
    """
    Exception for active-state errors in ``caskade``.

    Raised when an operation requires a node to be in a particular active
    state (enabled or disabled) and that condition is not met.
    """


class FillParamsError(CaskadeException):
    """
    Base exception for errors when filling parameters in ``caskade``.

    Raised when the input data provided to fill node parameters is
    invalid. Subclasses handle specific input types (array, sequence,
    mapping).
    """


class FillParamsArrayError(FillParamsError):
    """
    Exception raised when filling parameters with an array fails.

    Raised when the shape of the input array does not match the total
    number of flattened parameters registered on a node.

    Parameters
    ----------
    name : str
        Name of the node whose parameters are being filled.
    input_params : ArrayLike
        The input array that was provided.
    params : tuple of Param
        Registered parameters whose shapes are compared against the
        input.
    shapes : sequence of tuple of int, optional
        The shape each param takes in the array, where this is not the shape
        of the param. Params below a hierarchical link also have a dimension
        for each link.
    """

    def __init__(self, name, input_params, params, shapes=None):
        if shapes is None:
            shapes = tuple(p.shape for p in params)
        fullnumel = sum(max(1, prod(shape)) for shape in shapes)
        message = dedent(
            f"""
            For flattened {backend.array_type.__name__} input, the (last) dim of
            the {backend.array_type.__name__} should equal the sum of all
            flattened params ({fullnumel}). Input params shape
            {input_params.shape} does not match params shape of: {name}. 
            
            Registered params (name: shape): 
            {', '.join(f"{repr(p)}: {str(shape)}" for p, shape in zip(params, shapes))}"""
        )
        super().__init__(message)


class FillParamsSequenceError(FillParamsError):
    """
    Exception raised when filling parameters with a sequence fails.

    Raised when the length of the input sequence does not match the
    number of dynamic parameters registered on a node.

    Parameters
    ----------
    name : str
        Name of the node whose parameters are being filled.
    input_params : sequence
        The input sequence (list, tuple, etc.) that was provided.
    dynamic_params : tuple of Param
        Registered dynamic parameters expected by the node.
    """

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
    """
    Exception raised when filling parameters with a mapping fails.

    Raised when a key in the input dictionary does not correspond to any
    registered child node.

    Parameters
    ----------
    name : str
        Name of the node whose parameters are being filled.
    children : dict
        Dictionary of registered child nodes.
    missing_key : str, optional
        The key from the input mapping that was not found among the
        node's children.
    """

    def __init__(self, name, children, missing_key=None):
        message = dedent(
            f"""
            Input params key "{missing_key}" not found in children of: {name}. 
            
            Registered children:
            {', '.join(repr(c) for c in children.values())}"""
        )
        super().__init__(message)
