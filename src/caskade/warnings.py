from textwrap import dedent

from .backend import backend


class CaskadeWarning(Warning):
    """
    Base warning class for ``caskade``.

    All custom warnings issued by ``caskade`` inherit from this class,
    allowing users to filter or catch any ``caskade``-specific warning.
    """


class InvalidValueWarning(CaskadeWarning):
    """
    Warning issued when a parameter value is outside its valid range.

    Indicates that the assigned value may cause errors or unexpected
    behavior during computation.

    Parameters
    ----------
    name : str
        Name of the parameter with the out-of-range value.
    value : ArrayLike
        The value that was assigned.
    valid : tuple
        A ``(lower, upper)`` tuple defining the valid range, where
        ``None`` represents negative or positive infinity.
    """

    def __init__(self, name, value, valid):
        message = dedent(
            f"""        
            Value {backend.tolist(value)} for parameter "{name}" is outside the valid range ({backend.tolist(valid[0]) if valid[0] is not None else "-inf"}, {backend.tolist(valid[1]) if valid[1] is not None else "inf"}).
            Likely to cause errors or unexpected behavior!"""
        )
        super().__init__(message)


class SaveStateWarning(CaskadeWarning):
    """
    Warning issued when saving state encounters a non-fatal problem.

    Issued when the state serialization completes but with potential data
    loss or format issues that the user should be aware of.
    """


class LinkToAttributeWarning(CaskadeWarning):
    """
    Warning issued when a child node takes the place of another attribute.

    Linking a node under a name that the parent already uses for something
    that is not a child (a plain value, a placeholder such as ``None``, a
    method) replaces that attribute, as any assignment would. This may be what
    was meant, or it may be a child that was given the name of something the
    parent needs. A name that held a child before does not give this warning,
    so ``node.child = None`` followed by ``node.child = other`` is silent.
    """


class StaticCacheWarning(CaskadeWarning):
    """
    Warning issued when a ``static_cache`` result may not be the right one.

    Issued when the stored result was made from a dynamic param, whose values
    in a simulation it will not follow, and when a stored result is used after
    a param that it was made from has been changed.
    """
