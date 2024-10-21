from typing import Optional, Union, Callable

import torch
from torch import Tensor

from .base import Node


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
    ```{python}
    p1 = Param("test", (1.0, 2.0)) # constant value, length 2 vector
    p2 = Param("p2", None, (2,2)) # dynamic 2x2 matrix value
    p3 = Param("p3", p1) # pointer to another parameter
    p4 = Param("p4", lambda p: p.children["other"].value * 2) # arbitrary function of another parameter
    p4.link("other", p2) # link the other parameter needed for the function
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
    ):
        super().__init__(name=name)
        if value is None:
            if shape is None:
                raise ValueError("Either value or shape must be provided")
            if not isinstance(shape, tuple):
                raise ValueError("Shape must be a tuple")
            self.shape = shape
        elif not isinstance(value, (Param, Callable)):
            value = torch.as_tensor(value)
            assert (
                shape == () or shape is None or shape == value.shape
            ), f"Shape {shape} does not match value shape {value.shape}"
        self.value = value

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
            raise RuntimeError("Cannot set shape of parameter with type 'pointer'")
        self._shape = shape

    @property
    def value(self) -> Union[Tensor, None]:
        if self.pointer and self._value is None and self.active:
            self._value = self._pointer_func(self)
        return self._value

    @value.setter
    def value(self, value):
        # While active no value can be set
        if self.active:
            raise RuntimeError(
                f"Cannot set value of parameter {self.name}|{self._type} while active"
            )

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

        return self
