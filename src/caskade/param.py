from typing import Optional, Union, Callable
from warnings import warn

import torch
from torch import Tensor
from torch import pi

from .base import Node
from .errors import ParamConfigurationError, ParamTypeError, ActiveStateError
from .warnings import InvalidValueWarning


class Param(Node):
    """
    Node to represent a parameter in the graph.

    The ``Param`` object is used to represent a parameter in the graph. During
    runtime this will represent a tensor value which can be used in various
    calculations. The ``Param`` object can be set to a constant value (``static``);
    ``None`` meaning the value is to be provided at runtime (``dynamic``); another
    ``Param`` object meaning it will take on that  value at runtime (``pointer``);
    or a function of other ``Param`` objects to be computed at runtime (also
    ``pointer``, see user guides). These options allow users to flexibly set the
    behavior of the simulator.

    Examples
    --------
    Example making some ``Param`` objects::

        p1 = Param("test", (1.0, 2.0)) # constant value, length 2 vector
        p2 = Param("p2", None, (2,2)) # dynamic 2x2 matrix value
        p3 = Param("p3", p1) # pointer to another parameter
        p4 = Param("p4", lambda p: p.children["other"].value * 2) # arbitrary function of another parameter
        p5 = Param("p5", valid=(0.0,2*pi), units="radians", cyclic=True) # parameter with metadata

    Parameters
    ----------
    name: (str)
        The name of the parameter.
    value: (Optional[Union[Tensor, float, int]], optional)
        The value of the parameter. Defaults to None meaning dynamic.
    shape: (Optional[tuple[int, ...]], optional)
        The shape of the parameter. Defaults to () meaning scalar.
    cyclic: (bool, optional)
        Whether the parameter is cyclic, such as a rotation from 0 to 2pi. Defaults to False.
    valid: (Optional[tuple[Union[Tensor, float, int, None]]], optional)
        The valid range of the parameter. Defaults to None meaning all of -inf to inf is valid.
    units: (Optional[str], optional)
        The units of the parameter. Defaults to None.
    """

    graphviz_types = {
        "static": {"style": "filled", "color": "lightgrey", "shape": "box"},
        "dynamic": {"style": "solid", "color": "black", "shape": "box"},
        "pointer": {"style": "filled", "color": "lightgrey", "shape": "cds"},
    }

    def __init__(
        self,
        name: str,
        value: Optional[Union[Tensor, float, int]] = None,
        shape: Optional[tuple[int, ...]] = (),
        cyclic: bool = False,
        valid: Optional[tuple[Union[Tensor, float, int, None]]] = None,
        units: Optional[str] = None,
    ):
        super().__init__(name=name)
        if value is None:
            if shape is None:
                raise ParamConfigurationError("Either value or shape must be provided")
            if not isinstance(shape, tuple):
                raise ParamConfigurationError("Shape must be a tuple")
            self.shape = shape
        elif not isinstance(value, (Param, Callable)):
            value = torch.as_tensor(value)
            if not (shape == () or shape is None or shape == value.shape):
                raise ParamConfigurationError(
                    f"Shape {shape} does not match value shape {value.shape}"
                )
        self.value = value
        self.cyclic = cyclic
        self.valid = valid
        self.units = units

    @property
    def dynamic(self) -> bool:
        return self._type == "dynamic"

    @property
    def pointer(self) -> bool:
        return self._type == "pointer"

    @property
    def static(self) -> bool:
        return self._type == "static"

    @property
    def shape(self) -> tuple:
        return self._shape

    @shape.setter
    def shape(self, shape):
        if self.pointer:
            raise ParamTypeError("Cannot set shape of parameter with type 'pointer'")
        self._shape = shape

    @property
    def value(self) -> Union[Tensor, None]:
        if self.pointer and self._value is None:
            if self.active:
                self._value = self._pointer_func(self)
            else:
                return self._pointer_func(self)
        return self._value

    @value.setter
    def value(self, value):
        # While active no value can be set
        if self.active:
            raise ActiveStateError(f"Cannot set value of parameter {self.name} while active")

        # unlink if pointer to avoid floating references
        if self.pointer:
            for child in tuple(self.children.values()):
                self.unlink(child)

        if value is None:
            self._type = "dynamic"
            self._pointer_func = None
            self._value = None
        elif isinstance(value, Param):
            self._type = "pointer"
            self.link(str(id(value)), value)
            self._pointer_func = lambda p: p[str(id(value))].value
            self._shape = None
            self._value = None
        elif callable(value):
            self._type = "pointer"
            self._shape = None
            self._pointer_func = value
            self._value = None
        else:
            self._type = "static"
            value = torch.as_tensor(value)
            self.shape = value.shape
            self._value = value
            try:
                self.valid = self._valid  # re-check valid range
            except AttributeError:
                pass

        self.update_graph()

    def to(self, device=None, dtype=None):
        """
        Moves and/or casts the values of the parameter.

        Parameters
        ----------
        device: (Optional[torch.device], optional)
            The device to move the values to. Defaults to None.
        dtype: (Optional[torch.dtype], optional)
            The desired data type. Defaults to None.
        """
        super().to(device=device, dtype=dtype)
        if self.static:
            self._value = self._value.to(device=device, dtype=dtype)
        if self.valid[0] is not None:
            self.valid = (self.valid[0].to(device=device, dtype=dtype), self.valid[1])
        if self.valid[1] is not None:
            self.valid = (self.valid[0], self.valid[1].to(device=device, dtype=dtype))

        return self

    @property
    def cyclic(self):
        return self._cyclic

    @cyclic.setter
    def cyclic(self, cyclic: bool):
        self._cyclic = cyclic
        try:
            self.valid = self._valid
        except AttributeError:
            pass

    @property
    def valid(self):
        return self._valid

    @valid.setter
    def valid(self, valid: tuple[Union[Tensor, float, int, None]]):
        if valid is None:
            valid = (None, None)

        if not isinstance(valid, tuple):
            raise ParamConfigurationError("Valid must be a tuple")
        if len(valid) != 2:
            raise ParamConfigurationError("Valid must be a tuple of length 2")

        if valid == (None, None):
            if self.cyclic:
                raise ParamConfigurationError("Cannot set valid to None for cyclic parameter")
            self.to_valid = self._to_valid_base
            self.from_valid = self._from_valid_base
        elif valid[0] is None:
            if self.cyclic:
                raise ParamConfigurationError("Cannot set left valid to None for cyclic parameter")
            self.to_valid = self._to_valid_rightvalid
            self.from_valid = self._from_valid_rightvalid
            valid = (None, torch.as_tensor(valid[1]))
            if self.static and torch.any(self.value > valid[1]):
                warn(InvalidValueWarning(self.name, self.value, valid))
        elif valid[1] is None:
            if self.cyclic:
                raise ParamConfigurationError("Cannot set right valid to None for cyclic parameter")
            self.to_valid = self._to_valid_leftvalid
            self.from_valid = self._from_valid_leftvalid
            valid = (torch.as_tensor(valid[0]), None)
            if self.static and torch.any(self.value < valid[0]):
                warn(InvalidValueWarning(self.name, self.value, valid))
        else:
            if self.cyclic:
                self.to_valid = self._to_valid_cyclic
                self.from_valid = self._from_valid_cyclic
            else:
                self.to_valid = self._to_valid_fullvalid
                self.from_valid = self._from_valid_fullvalid
            valid = (torch.as_tensor(valid[0]), torch.as_tensor(valid[1]))
            if torch.any(valid[0] >= valid[1]):
                raise ParamConfigurationError("Valid range (valid[1] - valid[0]) must be positive")
            if (
                self.static
                and not self.cyclic
                and (torch.any(self.value < valid[0]) or torch.any(self.value > valid[1]))
            ):
                warn(InvalidValueWarning(self.name, self.value, valid))

        self._valid = valid

    def _to_valid_base(self, value):
        if self.pointer:
            raise ParamTypeError("Cannot apply valid transformation to pointer parameter")
        return value

    def _to_valid_fullvalid(self, value):
        value = self._to_valid_base(value)
        return torch.tan((value - self.valid[0]) * pi / (self.valid[1] - self.valid[0]) - pi / 2)

    def _to_valid_cyclic(self, value):
        value = self._to_valid_base(value)
        return (value - self.valid[0]) % (self.valid[1] - self.valid[0]) + self.valid[0]

    def _to_valid_leftvalid(self, value):
        value = self._to_valid_base(value)
        return value - 1.0 / (value - self.valid[0])

    def _to_valid_rightvalid(self, value):
        value = self._to_valid_base(value)
        return value + 1.0 / (self.valid[1] - value)

    def _from_valid_base(self, value):
        if self.pointer:
            raise ParamTypeError("Cannot apply valid transformation to pointer parameter")
        return value

    def _from_valid_fullvalid(self, value):
        value = self._from_valid_base(value)
        value = (torch.atan(value) + pi / 2) * (self.valid[1] - self.valid[0]) / pi + self.valid[0]
        return value

    def _from_valid_cyclic(self, value):
        value = self._from_valid_base(value)
        value = (value - self.valid[0]) % (self.valid[1] - self.valid[0]) + self.valid[0]
        return value

    def _from_valid_leftvalid(self, value):
        value = self._from_valid_base(value)
        value = (value + self.valid[0] + ((value - self.valid[0]) ** 2 + 4).sqrt()) / 2
        return value

    def _from_valid_rightvalid(self, value):
        value = self._from_valid_base(value)
        value = (value + self.valid[1] - ((value - self.valid[1]) ** 2 + 4).sqrt()) / 2
        return value

    def __repr__(self):
        return self.name
