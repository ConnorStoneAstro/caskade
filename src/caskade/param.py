from typing import Optional, Union, Callable

import torch
from torch import Tensor

from .base import Node


class LiveParam:
    """Placeholder to identify a parameter as live updating. Like `None` there
    exists only one instance of this class."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LiveParam, cls).__new__(cls)
        return cls._instance


class Param(Node):

    def __init__(
        self,
        name,
        value: Optional[Union[Tensor, float, int]] = None,
        shape: Optional[tuple[int, ...]] = (),
    ):
        super().__init__(name=name)
        if value is None or isinstance(value, LiveParam):
            if shape is None:
                raise ValueError("Either value or shape must be provided")
            if not isinstance(shape, tuple):
                raise ValueError("Shape must be a tuple")
            self.shape = shape
        elif not isinstance(value, (Param, Callable)):
            value = torch.as_tensor(value)
            self.shape = value.shape
            assert shape == () or shape == self.shape, "Shape does not match value shape"
        self.value = value

    @property
    def dynamic(self):
        return self._type == "dynamic"

    @property
    def live(self):
        return self._type == "live"

    @property
    def shape(self):
        if self._type in ["pointer", "function"]:
            return None
        return self._shape

    @shape.setter
    def shape(self, shape):
        self._shape = shape

    @property
    def value(self):
        if self._type == "pointer":
            return self._value.value
        if self._type == "function":
            return self._value(self)
        return self._value

    @value.setter
    def value(self, value):
        # While active, update silently
        if self.active and (self.dynamic or self.live):
            self._value = value
            return

        # unlink if pointer to avoid floating references
        if self._type == "pointer":
            self.unlink(self._value)

        if value is None:
            self._type = "dynamic"
            assert self.shape is not None, "Shape must be provided for dynamic parameters"
        elif isinstance(value, LiveParam):
            self._type = "live"
            assert self.shape is not None, "Shape must be provided for live parameters"
        elif isinstance(value, Param):
            self._type = "pointer"
            self.link(value.name, value)
            self.shape = None
        elif callable(value):
            self._type = "function"
            self.shape = None
        else:
            self._type = "value"
            value = torch.as_tensor(value)
            if value.shape != self.shape:
                raise ValueError(
                    f"Input shape {value.shape} does not match {self.name} shape {self.shape}"
                )

        self._value = value
        self.update_dynamic_params()

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
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
        if self._type == "value":
            self._value = self._value.to(device=device, dtype=dtype)
