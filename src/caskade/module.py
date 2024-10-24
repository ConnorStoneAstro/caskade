from typing import Sequence, Mapping, Optional, Union, Any
from math import prod

from torch import Tensor
import torch

from .base import Node
from .param import Param


class Module(Node):
    """
    Node to represent a simulation module in the graph.

    The `Module` object is used to represent a simulation module in the graph.
    These are python objects that contain the calculations for a simulation,
    they also hold the `Param` objects that are used in the calculations. The
    `Module` object has additional functionality to manage the `Param` objects
    below it in the graph, it keeps track of all `dynamic` `Param` objects so
    that at runtime their values may be filled. The `Module` object manages its
    links to other nodes through attributes of the class.

    Examples
    --------
    ```{python}
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

    _module_names = set()
    graphviz_types = {"module": {"style": "solid", "color": "black", "shape": "ellipse"}}

    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name)
        self.dynamic_params = ()
        self.pointer_params = ()
        self._type = "module"
        self.valid_context = False

    def update_graph(self):
        """Maintain a tuple of dynamic and live parameters at all points lower
        in the DAG."""
        self.dynamic_params = tuple(self.topological_ordering("dynamic"))
        self.pointer_params = tuple(self.topological_ordering("pointer"))
        self.local_dynamic_params = tuple(
            p for p in self.children.values() if isinstance(p, Param) and p.dynamic
        )
        self.dynamic_modules = dict(
            (m.name, m) for m in self.topological_ordering("module") if m.dynamic
        )
        super().update_graph()

    @property
    def dynamic(self):
        """Return True if the module has dynamic parameters"""
        return self.local_dynamic_params != ()

    def fill_params(self, params: Union[Tensor, Sequence, Mapping], local=False):
        """
        Fill the dynamic parameters of the module with the input values from
        params.

        Parameters
        ----------
        params: (Union[Tensor, Sequence, Mapping])
            The input values to fill the dynamic parameters with. The input can
            be a Tensor, a Sequence, or a Mapping. If the input is a Tensor, the
            values are filled in order of the dynamic parameters. `params`
            should be a flattened tensor with all parameters concatenated in the
            order of the dynamic parameters. If `len(params.shape)>1` then all
            dimensions but the last one are considered batch dimensions. If the
            input is a Sequence, the values are filled in order of the dynamic
            parameters. If the input is a Mapping, the values are filled by
            matching the keys of the Mapping to the names of the dynamic
            parameters. Note that the system does not check for missing keys in
            the dictionary, but you will get an error eventually if a value is
            missing.
        """
        assert self.active, "Module must be active to fill params"

        if self.valid_context and not local:
            params = self.from_valid(params)

        dynamic_params = self.local_dynamic_params if local else self.dynamic_params

        if isinstance(params, Tensor):
            # check for batch dimension
            batch = len(params.shape) > 1
            B = tuple(params.shape[:-1]) if batch else ()
            pos = 0
            for param in dynamic_params:
                if not isinstance(param.shape, tuple):
                    raise ValueError(
                        f"Param {param.name} has no shape. dynamic parameters must have a shape to use Tensor input."
                    )
                # Handle scalar parameters
                size = max(1, prod(param.shape))
                try:
                    param._value = params[..., pos : pos + size].view(B + param.shape)
                except (RuntimeError, IndexError):
                    fullnumel = sum(max(1, prod(p.shape)) for p in dynamic_params)
                    raise AssertionError(
                        f"Input params shape {params.shape} does not match dynamic params shape of {self.name}. Make sure the last dimension has size equal to the sum of all dynamic params sizes ({fullnumel})."
                    )
                pos += size
            if pos != params.shape[-1]:
                fullnumel = sum(max(1, prod(p.shape)) for p in dynamic_params)
                raise AssertionError(
                    f"Input params length {params.shape} does not match dynamic params length ({fullnumel}) of {self.name}. Not all dynamic params were filled."
                )
        elif isinstance(params, Sequence):
            if len(params) == len(dynamic_params):
                for param, value in zip(dynamic_params, params):
                    param._value = value
            elif len(params) == len(self.dynamic_modules):
                for module, value in zip(self.dynamic_modules.values(), params):
                    module.fill_params(value, local=True)
            else:
                raise AssertionError(
                    f"Input params length ({len(params)}) does not match dynamic params length ({len(dynamic_params)}) or number of dynamic modules ({len(self.dynamic_modules)}) for {self.name}"
                )
        elif isinstance(params, Mapping):
            for key in params:
                if key in self.dynamic_modules:
                    self.dynamic_modules[key].fill_params(params[key], local=True)
                elif key in self.children and self[key].dynamic:
                    self[key]._value = params[key]
                else:
                    raise ValueError(
                        f"Key {key} not found in dynamic modules or {self.name} children"
                    )
        else:
            raise ValueError(
                f"Input params type {type(params)} not supported. Should be Tensor, Sequence or Mapping."
            )

    def clear_params(self):
        """Set all dynamic parameters to None and live parameters to LiveParam.
        This is to be used on exiting an `ActiveContext` and so should not be
        used by a user."""
        assert self.active, "Module must be active to clear params"

        for param in self.dynamic_params + self.pointer_params:
            param._value = None

    def fill_kwargs(self, keys: tuple[str]) -> dict[str, Tensor]:
        """
        Fill the kwargs for an `@forward` method with the values of the dynamic
        parameters. The requested keys are matched to names of `Param` objects
        owned by the `Module`.
        """
        kwargs = {}
        for key in keys:
            if key in self.children and isinstance(self[key], Param):
                kwargs[key] = self[key].value
        return kwargs

    def to_valid(self, params: Union[Tensor, Sequence, Mapping], local=False):
        """Convert input params to valid params."""
        dynamic_params = self.local_dynamic_params if local else self.dynamic_params
        if isinstance(params, Tensor):
            valid_params = torch.zeros_like(params)
            batch = len(params.shape) > 1
            B = tuple(params.shape[:-1]) if batch else ()
            pos = 0
            for param in dynamic_params:
                size = max(1, prod(param.shape))  # Handle scalar parameters
                return_shape = params[..., pos : pos + size].shape
                valid_params[..., pos : pos + size] = param.to_valid(
                    params[..., pos : pos + size].view(B + param.shape)
                ).view(return_shape)
                pos += size
        elif isinstance(params, Sequence):
            valid_params = []
            if len(params) == len(dynamic_params):
                for param, value in zip(dynamic_params, params):
                    valid_params.append(param.to_valid(value))
            elif len(params) == len(self.dynamic_modules):
                for module, value in zip(self.dynamic_modules.values(), params):
                    valid_params.append(module.to_valid(value, local=True))
            else:
                raise AssertionError(
                    f"Input params length ({len(valid_params)}) does not match dynamic params length ({len(dynamic_params)}) or number of dynamic children ({len(self.children)})"
                )
        elif isinstance(params, Mapping):
            valid_params = {}
            for key in params:
                if key in self.dynamic_modules:
                    valid_params[key] = self.dynamic_modules[key].to_valid(params[key], local=True)
                elif key in self.children and self[key].dynamic:
                    valid_params[key] = self[key].to_valid(params[key])
                else:
                    raise ValueError(
                        f"Key {key} not found in dynamic modules or {self.name} children"
                    )
        else:
            raise ValueError(
                f"Input params type {type(params)} not supported. Should be Tensor, Sequence, or Mapping."
            )
        return valid_params

    def from_valid(self, valid_params: Union[Tensor, Sequence, Mapping], local=False):
        """Convert valid params to input params."""

        dynamic_params = self.local_dynamic_params if local else self.dynamic_params

        if isinstance(valid_params, Tensor):
            params = torch.zeros_like(valid_params)
            batch = len(valid_params.shape) > 1
            B = tuple(params.shape[:-1]) if batch else ()
            pos = 0
            for param in dynamic_params:
                size = max(1, prod(param.shape))
                return_shape = valid_params[..., pos : pos + size].shape
                params[..., pos : pos + size] = param.from_valid(
                    valid_params[..., pos : pos + size].view(B + param.shape)
                ).view(return_shape)
                pos += size
        elif isinstance(valid_params, Sequence):
            params = []
            if len(valid_params) == len(dynamic_params):
                for param, value in zip(dynamic_params, valid_params):
                    params.append(param.from_valid(value))
            elif len(valid_params) == len(self.dynamic_modules):
                for module, value in zip(self.dynamic_modules.values(), valid_params):
                    params.append(module.from_valid(value, local=True))
            else:
                raise AssertionError(
                    f"Input params length ({len(params)}) does not match dynamic params length ({len(dynamic_params)}) or number of dynamic children ({len(self.children)})"
                )
        elif isinstance(valid_params, Mapping):
            params = {}
            for key in valid_params:
                if key in self.dynamic_modules:
                    params[key] = self.dynamic_modules[key].from_valid(
                        valid_params[key], local=True
                    )
                elif key in self.children and self[key].dynamic:
                    params[key] = self[key].from_valid(valid_params[key])
                else:
                    raise ValueError(
                        f"Key {key} not found in dynamic modules or {self.name} children"
                    )
        else:
            raise ValueError(
                f"Input params type {type(valid_params)} not supported. Should be Tensor, Sequence or Mapping."
            )
        return params

    @property
    def _name(self) -> str:
        return self.__name

    @_name.setter
    def _name(self, name: str):
        i = 0
        newname = name
        while newname in Module._module_names:
            newname = f"{name}_{i}"
            i += 1
        self._module_names.add(newname)
        self.__name = newname

    def __del__(self):
        """Remove the name from the set of module names when the object is deleted."""
        try:
            self._module_names.remove(self._name)
        except:
            pass

    def __setattr__(self, key: str, value: Any):
        """Intercept attribute setting to update parameters and graph links."""
        try:
            if key in self.children and isinstance(self[key], Param):
                self[key].value = value
                return
            if isinstance(value, Node):
                self.link(key, value)
        except AttributeError:
            pass
        super().__setattr__(key, value)
