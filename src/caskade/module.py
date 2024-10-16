from typing import Boolean
from collections.abs import Sequence, Mapping

from torch import Tensor

from .base import Node
from .param import Param, LiveParam


class Module(Node):

    def __init__(self, name):
        super().__init__(name=name)
        self.dynamic_params = ()
        self.live_params = ()
        self._type = "module"
        self._batch = False

    @property
    def batch(self):
        return self._batch

    @batch.setter
    def batch(self, value):
        assert isinstance(value, Boolean)
        self._batch = value

    def update_dynamic_params(self):
        super().update_dynamic_params()
        self.dynamic_params = tuple(self.topological_ordering("dynamic"))
        self.live_params = tuple(self.topological_ordering("live"))

    def fill_params(self, params):
        assert self.active, "Module must be active to fill params"

        if isinstance(params, Tensor):
            if self.batch:
                B = params.shape[0]
            pos = 0
            for param in self.dynamic_params:
                size = param.shape.numel()
                if self.batch:
                    param.value = params[:, pos : pos + size].view((B,) + param.shape)
                    pos += size * B
                else:
                    param.value = params[pos : pos + size].view(param.shape)
                    pos += size
        elif isinstance(params, Sequence):
            if len(params) == len(self.dynamic_params):
                for param, value in zip(self.dynamic_params, params):
                    param.value = value
            else:
                raise ValueError(
                    f"Input params length ({len(params)}) does not match dynamic params length ({len(self.dynamic_params)})"
                )
        elif isinstance(params, Mapping):
            for key in params:
                if key in self.children:
                    if isinstance(self.children[key], Param):
                        self.children[key].value = params[key]
                    elif isinstance(self.children[key], Module):
                        self.children[key].fill_params(params[key])
                    else:
                        raise ValueError(f"Key {key} type {type(self.children[key])} not supported")
                else:
                    raise ValueError(f"Key {key} not found in {self.name} children")
        else:
            raise ValueError(
                f"Input params type {type(params)} not supported. Should be Tensor, Sequence or Mapping."
            )

    def clear_params(self):
        assert self.active, "Module must be active to clear params"

        for param in self.dynamic_params:
            param.value = None

        for param in self.live_params:
            param.value = LiveParam()

    def fill_kwargs(self, keys):
        return {key: getattr(self, key).value for key in keys}

    def __setattr__(self, key, value):
        if key in self.children and isinstance(self.children[key], Param):
            self.children[key].value = value
            return
        if isinstance(value, Node):
            self.link(key, value)
            self.update_dynamic_params()

        super().__setattr__(key, value)
