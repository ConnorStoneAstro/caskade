from textwrap import dedent


class CaskadeWarning(Warning):
    """Base warning for ``caskade``."""


class InvalidValueWarning(CaskadeWarning):
    """Warning for values which fall outside the valid range."""

    def __init__(self, name, value, valid):
        message = dedent(
            f"""        
            Value {value.detach().cpu().tolist()} for parameter "{name}" is outside the valid range ({valid[0].detach().cpu().tolist() if valid[0] is not None else "-inf"}, {valid[1].detach().cpu().tolist() if valid[1] is not None else "inf"}).
            Likely to cause errors or unexpected behavior!"""
        )
        super().__init__(message)
