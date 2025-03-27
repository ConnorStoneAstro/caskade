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


class AttributeCollisionWarning(CaskadeWarning):
    """Warning for attribute collisions."""

    def __init__(self, name, key, newkey):
        message = dedent(
            f"""        
            Attribute "{key}" already exists in {name}. Overwriting with new name {newkey} to avoid overwrite. This may cause problems on the user side, please choose names that dont collide with Param attributes"""
        )
        super().__init__(message)
