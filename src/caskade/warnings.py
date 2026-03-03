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

    Raised when the state serialization completes but with potential data
    loss or format issues that the user should be aware of.
    """
