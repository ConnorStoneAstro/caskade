from typing import Sequence, Mapping, Optional, Union, Any
from math import prod

from torch import Tensor
import torch

from .base import Node
from .param import Param
from .collection import NodeTuple, NodeList
from .errors import (
    ActiveStateError,
    ParamConfigurationError,
    FillDynamicParamsTensorError,
    FillDynamicParamsSequenceError,
    FillDynamicParamsMappingError,
)


class Module(Node):
    """
    Node to represent a simulation module in the graph.

    The ``Module`` object is used to represent a simulation module in the graph.
    These are python objects that contain the calculations for a simulation,
    they also hold the ``Param`` objects that are used in the calculations. The
    ``Module`` object has additional functionality to manage the ``Param`` objects
    below it in the graph, it keeps track of all ``dynamic`` ``Param`` objects so
    that at runtime their values may be filled. The ``Module`` object manages its
    links to other nodes through attributes of the class.

    Examples
    --------

    Example of a nested pair of ``Module`` objects and how their ``@forward`` methods are called::

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
    """

    _module_names = set()
    _special_tuples = (
        "dynamic_params",
        "pointer_params",
        "local_dynamic_params",
    )  # These tuples will not be converted to NodeTuple objects
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
        self.all_dynamic_value = all("value" in p._type for p in self.dynamic_params)
        self.pointer_params = tuple(self.topological_ordering("pointer"))
        self.local_dynamic_params = tuple(
            p for p in self.children.values() if isinstance(p, Param) and p.dynamic
        )
        self.dynamic_modules = dict(
            (m.name, m) for m in self.topological_ordering(with_isinstance=Module) if m.dynamic
        )
        super().update_graph()

    @property
    def dynamic(self):
        """Return True if the module has dynamic parameters"""
        return self.local_dynamic_params != ()

    def to_dynamic(self, local_only=True, **kwargs):
        if local_only:
            for c in self.children.values():
                if isinstance(c, Param):
                    c.to_dynamic()
        else:
            for n in self.topological_ordering(with_isinstance=Param):
                n.to_dynamic()

    def to_static(self, local_only=True, **kwargs):
        if local_only:
            for c in self.children.values():
                if isinstance(c, Param):
                    c.to_static()
        else:
            for n in self.topological_ordering(with_isinstance=Param):
                n.to_static()

    def _fill_values(
        self, params: Union[Tensor, Sequence, Mapping], local=False, dynamic_values=False
    ):
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

        dynamic_params = self.local_dynamic_params if local else self.dynamic_params
        if len(dynamic_params) == 0 and not dynamic_values:
            return

        if self.valid_context and not local:
            params = self.from_valid(params)

        if isinstance(params, Tensor):
            # check for batch dimension
            batch = len(params.shape) > 1
            B = tuple(params.shape[:-1]) if batch else ()
            pos = 0
            for param in dynamic_params:
                if not isinstance(param.shape, tuple):
                    raise ParamConfigurationError(
                        f"Param {param.name} has no shape. dynamic parameters must have a shape to use Tensor input."
                    )
                # Handle scalar parameters
                size = max(1, prod(param.shape))
                try:
                    val = params[..., pos : pos + size].view(B + param.shape)
                    if dynamic_values:
                        param.dynamic_value = val
                    else:
                        param._value = val
                except (RuntimeError, IndexError):
                    raise FillDynamicParamsTensorError(self.name, params, dynamic_params)

                pos += size
            if pos != params.shape[-1]:
                raise FillDynamicParamsTensorError(self.name, params, dynamic_params)
        elif isinstance(params, Sequence):
            if len(params) == len(dynamic_params):
                for param, value in zip(dynamic_params, params):
                    if dynamic_values:
                        param.dynamic_value = value
                    else:
                        param._value = value
            elif len(params) == len(self.dynamic_modules):
                for module, value in zip(self.dynamic_modules.values(), params):
                    module._fill_values(value, local=True, dynamic_values=dynamic_values)
            else:
                raise FillDynamicParamsSequenceError(
                    self.name, params, dynamic_params, self.dynamic_modules
                )
        elif isinstance(params, Mapping):
            for key in params:
                if key in self.dynamic_modules:
                    self.dynamic_modules[key]._fill_values(
                        params[key], local=True, dynamic_values=dynamic_values
                    )
                elif key in self.children and self[key].dynamic:
                    if dynamic_values:
                        self[key].dynamic_value = params[key]
                    else:
                        self[key]._value = params[key]
                else:
                    raise FillDynamicParamsMappingError(
                        self.name, self.children, self.dynamic_modules, missing_key=key
                    )
            if not local:
                for param in dynamic_params:
                    if param.value is None:
                        raise FillDynamicParamsMappingError(
                            self.name, self.children, self.dynamic_modules, missing_param=param
                        )
        else:
            raise TypeError(
                f"Input params type {type(params)} not supported. Should be Tensor, Sequence, or Mapping."
            )

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
        if not self.active:
            raise ActiveStateError("Module must be active to fill params")

        self._fill_values(params, local=local)

    def clear_params(self):
        """Set all dynamic parameters to None and live parameters to LiveParam.
        This is to be used on exiting an ``ActiveContext`` and so should not be
        used by a user."""
        if not self.active:
            raise ActiveStateError("Module must be active to clear params")

        for param in self.dynamic_params + self.pointer_params:
            param._value = None

    def fill_kwargs(self, keys: tuple[str]) -> dict[str, Tensor]:
        """
        Fill the kwargs for an ``@forward`` method with the values of the dynamic
        parameters. The requested keys are matched to names of ``Param`` objects
        owned by the ``Module``.
        """
        kwargs = {}
        for key in keys:
            if key in self.children and isinstance(self[key], Param):
                kwargs[key] = self[key].value
        return kwargs

    def fill_dynamic_values(self, params: Union[Tensor, Sequence, Mapping], local=False):
        """Fill the dynamic values of the module with the input values from params."""
        if self.active:
            raise ActiveStateError("Cannot fill dynamic values when Module is active")

        self._fill_values(params, local=local, dynamic_values=True)

    def _check_dynamic_values(self, params_type: str = "Tensor"):
        """Check if all dynamic values are set."""
        if not self.all_dynamic_value:
            bad_params = []
            for param in self.dynamic_params:
                if "value" not in param._type:
                    bad_params.append(param.name)
            raise ParamConfigurationError(
                f"Param(s) {bad_params} have no dynamic value, so the params {params_type} cannot be built. Set the `dynamic_value` attribute to use this feature."
            )

    def build_params_tensor(self) -> Tensor:
        """Return an input Tensor for this module's @forward methods by filling with dynamic values."""

        self._check_dynamic_values("Tensor")
        x = []
        for param in self.dynamic_params:
            x.append(param.value.detach().flatten())
        if len(x) == 0:
            return torch.tensor([])
        return torch.cat(x)

    def build_params_list(self) -> list[Tensor]:
        """Return an input list for this module's @forward methods by filling with dynamic values."""

        self._check_dynamic_values("List")
        x = []
        for param in self.dynamic_params:
            x.append(param.value.detach())
        return x

    def build_params_dict(self) -> dict[str, Tensor]:
        """Return an input dict for this module's @forward methods by filling with dynamic values."""

        self._check_dynamic_values("Dict")
        unique_params = set()
        x = {}
        for mod in self.dynamic_modules.values():
            x[mod.name] = {}
            for param in mod.local_dynamic_params:
                if param not in unique_params:
                    unique_params.add(param)
                    x[mod.name][param.name] = param.value.detach()

        return x

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
                raise FillDynamicParamsSequenceError(
                    self.name, params, dynamic_params, self.dynamic_modules
                )
        elif isinstance(params, Mapping):
            valid_params = {}
            for key in params:
                if key in self.dynamic_modules:
                    valid_params[key] = self.dynamic_modules[key].to_valid(params[key], local=True)
                elif key in self.children and self[key].dynamic:
                    valid_params[key] = self[key].to_valid(params[key])
                else:
                    raise FillDynamicParamsMappingError(
                        self.name, self.children, self.dynamic_modules, missing_key=key
                    )
        else:
            raise TypeError(
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
                raise FillDynamicParamsSequenceError(
                    self.name, valid_params, dynamic_params, self.dynamic_modules
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
                    raise FillDynamicParamsMappingError(
                        self.name, self.children, self.dynamic_modules, missing_key=key
                    )
        else:
            raise TypeError(
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
            elif isinstance(value, list):
                if all(isinstance(v, Node) for v in value):
                    self.link(key, NodeList(value))
            elif isinstance(value, tuple) and key not in self._special_tuples:
                if all(isinstance(v, Node) for v in value):
                    self.link(key, NodeTuple(value))
        except AttributeError:
            pass
        super().__setattr__(key, value)

    def __delattr__(self, key: str):
        """Intercept attribute deletion to remove links."""
        if key in self.children:
            self.unlink(key)
        super().__delattr__(key)
