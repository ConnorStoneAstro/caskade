from collections.abs import Sequence, Mapping

from torch import Tensor

from .base import Node
from .param import Param


class Module(Node):

    def __init__(self, name):
        super().__init__(name=name)
        self.dynamic_params = ()
        self._type = "module"

    def update_dynamic_params(self):
        super().update_dynamic_params()
        dynamic = []
        for node in self.topological_ordering:
            if isinstance(node, Param) and node.dynamic:
                dynamic += [node]
        self.dynamic_params = tuple(dynamic)

    def fill_params(self, params):
        if isinstance(params, Tensor):
            pos = 0
            for param in self.dynamic_params:
                size = param.shape.numel()
                param.silent_update_value(params[pos : pos + size].view(param.shape))
                pos += size
        elif isinstance(params, Sequence):
            if len(params) == len(self.dynamic_params):
                for param, value in zip(self.dynamic_params, params):
                    param.silent_update_value(value)
            else:
                raise ValueError(
                    f"Input params length ({len(params)}) does not match dynamic params length ({len(self.dynamic_params)})"
                )
        elif isinstance(params, Mapping):
            for key in params:
                if key in self.children:
                    if isinstance(self.children[key], Param):
                        self.children[key].silent_update_value(
                            params[key].view(self.children[key].shape)
                        )
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
        for param in self.dynamic_params:
            param.value = None

    def fill_kwargs(self, keys):
        return {key: getattr(self, key).value for key in keys}
