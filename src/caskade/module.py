from typing import Sequence, Mapping, Optional
from math import prod

from torch import Tensor

from .base import Node
from .param import Param, LiveParam


class Module(Node):
    """
    Node to represent a simulation module in the graph.

    The `Module` object is used to represent a simulation module in the graph.
    These are python objects that contain the calculations for a simulation,
    they also hold the `Param` objects that are used in the calculations. The
    `Module` object has additional functionality to manage the `Param` objects
    below it in the graph, it keeps track of all `dynamic` `Param` objects so
    that at runtime their values may be filled. The `Module` object has a
    `batch` attribute that can be set to `True` to indicate that the module
    should be run in batch mode. This means that the module will be run with a
    batch of inputs and the `Param` objects will be filled with the
    corresponding batched values. The `Module` object manages its links to other
    nodes through attributes of the class.

    Examples
    --------
    ``` python
    class MySim(Module):
        def __init__(self, a, b=None):
            super().__init__()
            self.a = a
            self.b = Param("b", b)

        @forward
        def myfunc(self, x, b=None):
            return x * self.a.otherfun(x) + b

    class OtherSim(Module):
        def __init__(self, c=None):
            super().__init__()
            self.c = Param("c", c)

        @forward
        def otherfun(self, x, c = None):
            return x + c

    othersim = OtherSim()
    mysim = MySim(a=othersim)
    #                       b                         c
    params = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
    result = mysim.myfunc(3.0, params=params)
    # result is tensor([19.0, 23.0])
    ```
    """

    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name)
        self.dynamic_params = ()
        self.live_params = ()
        self._type = "module"
        self.batch = False

    @property
    def batch(self) -> bool:
        return self._batch

    @batch.setter
    def batch(self, value):
        assert isinstance(value, bool)
        self._batch = value

    def update_dynamic_params(self):
        super().update_dynamic_params()
        self.dynamic_params = tuple(self.topological_ordering("dynamic"))
        self.live_params = tuple(self.topological_ordering("live"))

    def fill_params(self, params):
        assert self.active, "Module must be active to fill params"

        if isinstance(params, Tensor):
            if self.batch:
                *B, _ = params.shape
            pos = 0
            for param in self.dynamic_params:
                if not isinstance(param.shape, tuple):
                    raise ValueError(
                        f"Param {param.name} has no shape. dynamic parameters must have a shape to use Tensor input."
                    )
                size = max(1, prod(param.shape))
                if self.batch:
                    param.value = params[..., pos : pos + size].view(
                        tuple(B) + ((1,) if param.shape == () else param.shape)
                    )
                    pos += size
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
                    else:  # assumed Module
                        self.children[key].fill_params(params[key])
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
            param.value = LiveParam

    def fill_kwargs(self, keys) -> dict[str, Tensor]:
        return {key: getattr(self, key).value for key in keys}

    def __setattr__(self, key, value):
        try:
            if key in self.children and isinstance(self.children[key], Param):
                self.children[key].value = value
                return
            if isinstance(value, Node):
                self.link(key, value)
                self.update_dynamic_params()

            super().__setattr__(key, value)
        except AttributeError:
            super().__setattr__(key, value)
