from typing import Optional

import torch

from .base import Node


class Param(Node):

    def __init__(self, name, value=None, shape=()):
        super().__init__(name=name)
        if value is None:
            if shape is None:
                raise ValueError("Either value or shape must be provided")
            if not isinstance(shape, tuple):
                raise ValueError("Shape must be a tuple")
            self.shape = shape
        else:
            value = torch.as_tensor(value)
            self.shape = value.shape
        self.value = value

    @property
    def dynamic(self):
        return self._type == "dynamic"

    @property
    def value(self):
        if self._type == "pointer":
            return self._value.value
        if self._type == "function":
            return self._value(self)
        return self._value

    @value.setter
    def value(self, value):
        if value is None:
            self._type = "dynamic"
        elif isinstance(value, Param):
            self._type = "pointer"
        elif callable(value):
            self._type = "function"
        else:
            self._type = "value"
            value = torch.as_tensor(value)
            if value.shape != self.shape:
                raise ValueError(
                    f"Input shape {value.shape} does not match {self.name} shape {self.shape}"
                )
        self._value = value
        self.update_dynamic_params()

    def silent_update_value(self, value):
        self._value = value

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

        if self._type == "pointer":
            self._value.to(device=device, dtype=dtype)
        elif self._type == "function":
            for child in self.children.values():
                child.to(device=device, dtype=dtype)
        else:
            self._value = self._value.to(device=device, dtype=dtype)
