from typing import Optional, Union, Callable

import torch
from torch import Tensor

from .base import Node

__all__ = ("Param", "LiveParam")


class LiveParamBase:
    """Placeholder to identify a parameter as live updating. Like `None` there
    exists only one instance of this class."""

    pass


LiveParam = LiveParamBase()


class Param(Node):
    """
    Node to represent a parameter in the graph.

    The `Param` object is used to represent a parameter in the graph. During
    runtime this will represent a tensor value which can be used in various
    calculations. The `Param` object can be set to a constant value (`static`);
    `None` meaning the value is to be provided at runtime (`dynamic`);
    `LiveParam` meaning the value will be computed internally in the simulator
    during runtime (`live`); another `Param` object meaning it will take on that
    value at runtime (`pointer`); or a function of other `Param` objects to be
    computed at runtime (`function`). These options allow users to flexibly set
    the behavior of the simulator.

    Examples
    --------
    ``` python
    p1 = Param("test", (1.0, 2.0)) # constant value, length 2 vector
    p2 = Param("p2", None, (2,2)) # dynamic 2x2 matrix value
    p3 = Param("fun name", LiveParam) # live updating value
    p4 = Param("p4", p1) # pointer to another parameter
    p5 = Param("p5", lambda p: p.children["other"].value * 2) # function of another parameter
    p5.link("other", p2) # link the other parameter needed for the function
    ```

    Parameters
    ----------
    name: (str)
        The name of the parameter.
    value: (Optional[Union[Tensor, float, int]], optional)
        The value of the parameter. Defaults to None meaning dynamic.
    shape: (Optional[tuple[int, ...]], optional)
        The shape of the parameter. Defaults to () meaning scalar.
    """

    def __init__(
        self,
        name: str,
        value: Optional[Union[Tensor, float, int]] = None,
        shape: Optional[tuple[int, ...]] = (),
    ):
        super().__init__(name=name)
        if value is None:
            if shape is None:
                raise ValueError("Either value or shape must be provided")
            if not isinstance(shape, tuple):
                raise ValueError("Shape must be a tuple")
            self.shape = shape
        elif not isinstance(value, (Param, Callable, LiveParamBase)):
            value = torch.as_tensor(value)
            assert (
                shape == () or shape == value.shape
            ), f"Shape {shape} does not match value shape {value.shape}"
        self.value = value

    @property
    def dynamic(self) -> bool:
        return self._type == "dynamic"

    @property
    def live(self) -> bool:
        return self._type == "live"

    @property
    def shape(self) -> tuple:
        return self._shape

    @shape.setter
    def shape(self, shape):
        if self._type in ["pointer", "function"]:
            raise RuntimeError("Cannot set shape of parameter with type 'pointer' or 'function'")
        self._shape = shape

    @property
    def value(self) -> Union[Tensor, None]:
        if self._type == "pointer":
            return self._value.value
        if self._type == "function":
            return self._value(self)
        return self._value

    @value.setter
    def value(self, value):
        # While active, update silently
        if self.active:
            if self.dynamic or self.live:
                self._value = value
                return
            raise RuntimeError(f"Cannot set value of non-live parameter {self.name} while active")

        # unlink if pointer to avoid floating references
        if self._type == "pointer":
            self.unlink(self._value)
        if self._type == "function":
            for child in tuple(self.children.values()):
                self.unlink(child)

        if value is None:
            self._type = "dynamic"
        elif isinstance(value, LiveParamBase):
            self._type = "live"
        elif isinstance(value, Param):
            self._type = "pointer"
            self.link(value.name, value)
            self._shape = None
        elif callable(value):
            self._type = "function"
            self._shape = None
        else:
            self._type = "static"
            value = torch.as_tensor(value)
            self.shape = value.shape

        self._value = value
        self.update_dynamic_params()

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
        if self._type == "static":
            self._value = self._value.to(device=device, dtype=dtype)
