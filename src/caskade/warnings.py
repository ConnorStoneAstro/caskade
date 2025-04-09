from textwrap import dedent

from .backend import backend


class CaskadeWarning(Warning):
    """Base warning for ``caskade``."""


class InvalidValueWarning(CaskadeWarning):
    """Warning for values which fall outside the valid range."""

    def __init__(self, name, value, valid):
        message = dedent(
            f"""        
            Value {backend.tolist(value)} for parameter "{name}" is outside the valid range ({backend.tolist(valid[0]) if valid[0] is not None else "-inf"}, {backend.tolist(valid[1]) if valid[1] is not None else "inf"}).
            Likely to cause errors or unexpected behavior!"""
        )
        super().__init__(message)


class SaveStateWarning(CaskadeWarning):
    """Warning for when an issue occurs when a state is saved."""
