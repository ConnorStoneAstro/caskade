from typing import Sequence, Mapping, Optional, Union, Any
from math import prod

from .backend import backend, ArrayLike
from .base import Node
from .param import Param
from .collection import NodeTuple, NodeList
from .errors import (
    ActiveStateError,
    ParamConfigurationError,
    FillDynamicParamsError,
    FillDynamicParamsArrayError,
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

    _special_tuples = (
        "dynamic_params",
        "pointer_params",
        "static_params",
        "dynamic_modules",
    )  # These tuples will not be converted to NodeTuple objects
    graphviz_types = {"module": {"style": "solid", "color": "black", "shape": "ellipse"}}

    def __init__(self, name: Optional[str] = None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.dynamic_params = ()
        self.pointer_params = ()
        self.child_dynamic_params = {}
        self.node_type = "module"
        self.valid_context = False
        self.clear_state_hooks = set()

    @property
    def all_params(self):
        return self.static_params + self.dynamic_params + self.pointer_params

    def update_graph(self):
        """Maintain a tuple of dynamic and live parameters at all points lower
        in the DAG."""
        T = self.topological_ordering()
        self.dynamic_params = tuple(filter(lambda n: isinstance(n, Param) and n.dynamic, T))
        self.pointer_params = tuple(filter(lambda n: isinstance(n, Param) and n.pointer, T))
        self.static_params = tuple(filter(lambda n: isinstance(n, Param) and n.static, T))
        self.child_dynamic_params = dict(
            (k, p) for k, p in self.children.items() if isinstance(p, Param) and p.dynamic
        )
        self.dynamic_modules = tuple(
            m for m in filter(lambda n: isinstance(n, Module), T) if m.dynamic
        )
        super().update_graph()

    def param_order(self):
        return ", ".join(
            tuple(f"{next(iter(p.parents)).name}: {p.name}" for p in self.dynamic_params)
        )

    @property
    def dynamic(self) -> bool:
        """Return True if the module has dynamic parameters as direct children."""
        return len(self.child_dynamic_params) > 0

    @property
    def static(self) -> bool:
        return not self.dynamic

    def to_dynamic(self, children_only=True, **kwargs):
        """Change all parameters to dynamic parameters. If the parameter has a
        value, this will become a dynamic value parameter.

        Parameters
        ----------
        children_only: (bool, optional)
            If True, only convert the children of this module to dynamic. If False,
            convert all parameters in the graph below this module. Defaults to True.
        """
        if children_only:
            for c in self.children.values():
                if isinstance(c, Param) and not c.pointer:
                    c.to_dynamic()
        else:
            for node in self.topological_ordering():
                if isinstance(node, Param) and not node.pointer:
                    node.to_dynamic()

    def to_static(self, children_only=True, **kwargs):
        """Change all parameters to static parameters. This only works if the
        parameter has a ``dynamic value`` set to become the static value.

        Parameters
        ----------
        children_only: (bool, optional)
            If True, only convert children of this module. If False, convert
            all parameters in the graph below this module. Defaults to True.
        """
        if children_only:
            for c in self.children.values():
                if isinstance(c, Param) and not c.pointer:
                    c.to_static()
        else:
            for node in self.topological_ordering():
                if isinstance(node, Param) and not node.pointer:
                    node.to_static()

    @property
    def valid_context(self) -> bool:
        """Return True if the module is in a valid context."""
        return self._valid_context

    @valid_context.setter
    def valid_context(self, value: bool):
        """Set the valid context of the module."""
        self._valid_context = value
        for node in self.topological_ordering():
            if isinstance(node, Module):
                node._valid_context = value

    def _fill_dict(self, node, params, dynamic_values=False):

        for key in params:
            if key in node.children and isinstance(node[key], Param) and node[key].dynamic:
                if dynamic_values:
                    node[key].dynamic_value(params[key])
                else:
                    node[key]._value = params[key]
            elif (
                key in node.children
                and isinstance(node[key], Module)
                and node[key].dynamic
                and not isinstance(params[key], dict)
            ):
                node[key]._fill_values(
                    params[key], children_only=True, dynamic_values=dynamic_values
                )
            elif key in node.children and isinstance(node[key], Node) and node[key].dynamic:
                self._fill_dict(node[key], params[key], dynamic_values=dynamic_values)
            else:
                raise FillDynamicParamsMappingError(
                    self.name, self.children, self.dynamic_modules, missing_key=key
                )

    def _fill_values(
        self, params: Union[ArrayLike, Sequence, Mapping], children_only=False, dynamic_values=False
    ):
        """
        Fill the dynamic parameters of the module with the input values from
        params.

        Parameters
        ----------
        params: (Union[ArrayLike, Sequence, Mapping])
            The input values to fill the dynamic parameters with. The input can
            be an ArrayLike, a Sequence, or a Mapping. If the input is
            array-like, the values are filled in order of the dynamic
            parameters. `params` should be a flattened array-like object with
            all parameters concatenated in the order of the dynamic parameters.
            If `len(params.shape)>1` then all dimensions but the last one are
            considered batch dimensions. If the input is a Sequence, the values
            are filled in order of the dynamic parameters. If the input is a
            Mapping, the values are filled by matching the keys of the Mapping
            to the names of the dynamic parameters. Note that the system does
            not check for missing keys in the dictionary, but you will get an
            error eventually if a value is missing.
        """

        dynamic_params = (
            self.child_dynamic_params.values() if children_only else self.dynamic_params
        )

        if isinstance(params, backend.array_type):
            if params.shape[-1] == 0:
                return  # No parameters to fill
            # check for batch dimension
            batch = len(params.shape) > 1
            B = tuple(params.shape[:-1]) if batch else ()
            pos = 0
            for param in dynamic_params:
                if not isinstance(param.shape, tuple):
                    raise ParamConfigurationError(
                        f"Param {param.name} has no shape. dynamic parameters must have a shape to use {backend.array_type.__name__} input."
                    )
                # Handle scalar parameters
                size = max(1, prod(param.shape))
                try:
                    val = backend.view(params[..., pos : pos + size], B + param.shape)
                    if dynamic_values:
                        param.dynamic_value(val)
                    else:
                        param._value = val
                except (RuntimeError, IndexError, ValueError, TypeError):
                    raise FillDynamicParamsArrayError(self.name, params, dynamic_params)

                pos += size
            if pos != params.shape[-1]:
                raise FillDynamicParamsArrayError(self.name, params, dynamic_params)
        elif isinstance(params, Sequence):
            if len(params) == 0:
                return
            elif len(params) == len(dynamic_params):
                for param, value in zip(dynamic_params, params):
                    if dynamic_values:
                        param.dynamic_value(value)
                    else:
                        param._value = value
            elif len(params) == len(self.dynamic_modules):
                for module, value in zip(self.dynamic_modules, params):
                    module._fill_values(value, children_only=True, dynamic_values=dynamic_values)
            else:
                raise FillDynamicParamsSequenceError(
                    self.name, params, dynamic_params, self.dynamic_modules
                )
        elif isinstance(params, Mapping):
            self._fill_dict(self, params, dynamic_values=dynamic_values)
        else:
            raise TypeError(
                f"Input params type {type(params)} not supported. Should be {backend.array_type.__name__}, Sequence, or Mapping."
            )

    def fill_params(self, params: Union[ArrayLike, Sequence, Mapping]):
        """
        Fill the dynamic parameters of the module with the input values from
        params.

        Parameters
        ----------
        params: (Union[ArrayLike, Sequence, Mapping])
            The input values to fill the dynamic parameters with. The input can
            be an ArrayLike, a Sequence, or a Mapping. If the input is
            array-like, the values are filled in order of the dynamic
            parameters. `params` should be a flattened array-like object with
            all parameters concatenated in the order of the dynamic parameters.
            If `len(params.shape)>1` then all dimensions but the last one are
            considered batch dimensions. If the input is a Sequence, the values
            are filled in order of the dynamic parameters. If the input is a
            Mapping, the values are filled by matching the keys of the Mapping
            to the names of the dynamic parameters. Note that the system does
            not check for missing keys in the dictionary, but you will get an
            error eventually if a value is missing.
        """
        if not self.active:
            raise ActiveStateError(f"Module {self.name} must be active to fill params")
        if self.valid_context:
            params = self.from_valid(params)
        self._fill_values(params)

    def clear_state(self):
        """Set all dynamic parameters to None and live parameters to LiveParam.
        This is to be used on exiting an ``ActiveContext`` and so should not be
        used by a user."""
        if not self.active:
            raise ActiveStateError(f"Module {self.name} must be active to clear state")

        for param in self.all_params:
            param._value = None

        for hook in list(self.clear_state_hooks):
            hook(self)

    def fill_kwargs(self, keys: tuple[str]) -> dict[str, ArrayLike]:
        """
        Fill the kwargs for an ``@forward`` method with the values of the dynamic
        parameters. The requested keys are matched to names of ``Param`` objects
        owned by the ``Module``.
        """
        kwargs = {}
        for key in keys:
            if key in self.children and isinstance(self[key], Param):
                val = self.children[key].value
                if val is None:
                    raise FillDynamicParamsError(
                        f"Param {key} in Module {self.name} has no value. "
                        "Ensure that the parameter is set before calling the forward method or provided with the params."
                    )
                kwargs[key] = val
        return kwargs

    def fill_dynamic_values(self, params: Union[ArrayLike, Sequence, Mapping]):
        """Fill the dynamic values of the module with the input values from params."""
        if self.active:
            raise ActiveStateError(f"Cannot fill dynamic values when Module {self.name} is active")
        if self.valid_context:
            params = self.from_valid(params)

        self._fill_values(params, dynamic_values=True)

    def _check_dynamic_values(self, params_type: str = "ArrayLike"):
        """Check if all dynamic values are set."""
        bad_params = []
        for param in self.dynamic_params:
            if "value" not in param.node_type:
                bad_params.append(param.name)
        if len(bad_params) > 0:
            raise ParamConfigurationError(
                f"{self.name} Param(s) {bad_params} have no dynamic value, so the params {params_type} cannot be built. Set the `dynamic_value` attribute to use this feature."
            )

    def build_params_array(self) -> ArrayLike:
        """Return an input array-like object for this module's @forward methods by filling with dynamic values."""
        self._check_dynamic_values("ArrayLike")
        x = []
        batch_shape = None
        for param in self.dynamic_params:
            if param.batched:
                B = param.batch_shape
                if batch_shape is None:
                    batch_shape = B
                elif batch_shape != B:
                    raise ParamConfigurationError(
                        f"Batch dimensions must be the same for all params. Got {B} for {param.name} when previous batch shape was {batch_shape}"
                    )

                x.append(backend.copy(param.value).reshape(B + (-1,)))
            else:
                x.append(backend.copy(param.value).flatten())

        if len(x) == 0:
            return backend.make_array([])
        x = backend.concatenate(x, axis=-1)
        if self.valid_context:
            x = self.to_valid(x)
        return x

    def build_params_list(self) -> list[ArrayLike]:
        """Return an input list for this module's @forward methods by filling with dynamic values."""

        self._check_dynamic_values("List")
        x = []
        for param in self.dynamic_params:
            x.append(backend.copy(param.value))
        if self.valid_context:
            x = self.to_valid(x)
        return x

    def _recursive_build_params_dict(self, unique_params: set):
        params = {}
        for link, child in self.children.items():
            if isinstance(child, Param) and child.dynamic and child not in unique_params:
                unique_params.add(child)
                params[link] = backend.copy(child.value)
        for link, child in self.children.items():
            if isinstance(child, Module) and len(child.dynamic_params) > 0:
                params[link] = child._recursive_build_params_dict(unique_params=unique_params)
        return params

    def build_params_dict(self) -> dict[str, Union[dict, ArrayLike]]:
        """Return an input dict for this module's @forward methods by filling with dynamic values."""

        self._check_dynamic_values("Dict")
        unique_params = set()
        x = self._recursive_build_params_dict(unique_params=unique_params)
        if self.valid_context:
            x = self.to_valid(x)
        return x

    def to_valid(
        self, params: Union[ArrayLike, Sequence, Mapping], children_only=False
    ) -> Union[ArrayLike, Sequence, Mapping]:
        """Convert input params to valid params."""
        dynamic_params = (
            self.child_dynamic_params.values() if children_only else self.dynamic_params
        )
        if isinstance(params, backend.array_type):
            valid_params = []  # backend.zeros_like(params)
            batch = len(params.shape) > 1
            B = tuple(params.shape[:-1]) if batch else ()
            pos = 0
            for param in dynamic_params:
                size = max(1, prod(param.shape))  # Handle scalar parameters
                return_shape = (*B, size)
                valid_params.append(
                    backend.view(
                        param.to_valid(
                            backend.view(params[..., pos : pos + size], B + param.shape)
                        ),
                        return_shape,
                    )
                )
                pos += size
            valid_params = backend.concatenate(valid_params, axis=-1)
        elif isinstance(params, Sequence):
            valid_params = []
            if len(params) == len(dynamic_params):
                for param, value in zip(dynamic_params, params):
                    valid_params.append(param.to_valid(value))
            elif len(params) == len(self.dynamic_modules):
                for module, value in zip(self.dynamic_modules, params):
                    valid_params.append(module.to_valid(value, children_only=True))
            else:
                raise FillDynamicParamsSequenceError(
                    self.name, params, dynamic_params, self.dynamic_modules
                )
        elif isinstance(params, Mapping):
            valid_params = {}
            for key in params:
                if key in self.children and isinstance(self[key], Module) and self[key].dynamic:
                    valid_params[key] = self[key].to_valid(params[key], children_only=True)
                elif key in self.children and isinstance(self[key], Param) and self[key].dynamic:
                    valid_params[key] = self[key].to_valid(params[key])
                else:
                    raise FillDynamicParamsMappingError(
                        self.name, self.children, self.dynamic_modules, missing_key=key
                    )
        else:
            raise TypeError(
                f"Input params type {type(params)} not supported. Should be {backend.array_type.__name__}, Sequence, or Mapping."
            )
        return valid_params

    def from_valid(
        self, valid_params: Union[ArrayLike, Sequence, Mapping], children_only=False
    ) -> Union[ArrayLike, Sequence, Mapping]:
        """Convert valid params to input params."""

        dynamic_params = (
            self.child_dynamic_params.values() if children_only else self.dynamic_params
        )

        if isinstance(valid_params, backend.array_type):
            params = []  # backend.zeros_like(valid_params)
            batch = len(valid_params.shape) > 1
            B = tuple(valid_params.shape[:-1]) if batch else ()
            pos = 0
            for param in dynamic_params:
                size = max(1, prod(param.shape))
                return_shape = (*B, size)
                params.append(
                    backend.view(
                        param.from_valid(
                            backend.view(valid_params[..., pos : pos + size], B + param.shape)
                        ),
                        return_shape,
                    )
                )
                pos += size
            params = backend.concatenate(params, axis=-1)
        elif isinstance(valid_params, Sequence):
            params = []
            if len(valid_params) == len(dynamic_params):
                for param, value in zip(dynamic_params, valid_params):
                    params.append(param.from_valid(value))
            elif len(valid_params) == len(self.dynamic_modules):
                for module, value in zip(self.dynamic_modules, valid_params):
                    params.append(module.from_valid(value, children_only=True))
            else:
                raise FillDynamicParamsSequenceError(
                    self.name, valid_params, dynamic_params, self.dynamic_modules
                )
        elif isinstance(valid_params, Mapping):
            params = {}
            for key in valid_params:
                if key in self.children and isinstance(self[key], Module) and self[key].dynamic:
                    params[key] = self[key].from_valid(valid_params[key], children_only=True)
                elif key in self.children and isinstance(self[key], Param) and self[key].dynamic:
                    params[key] = self[key].from_valid(valid_params[key])
                else:
                    raise FillDynamicParamsMappingError(
                        self.name, self.children, self.dynamic_modules, missing_key=key
                    )
        else:
            raise TypeError(
                f"Input params type {type(valid_params)} not supported. Should be {backend.array_type.__name__}, Sequence or Mapping."
            )
        return params

    @property
    def node_str(self) -> str:
        """
        Returns a string representation of the node for graph visualization.
        """
        return f"{self.name}|{self.__class__.__name__}"

    def __setattr__(self, key: str, value: Any):
        """Intercept attribute setting to update parameters and graph links."""
        try:
            if key in self.children and isinstance(self[key], Param):
                self[key].value = value
                return

            if isinstance(value, list) and not isinstance(value, NodeList):
                if len(value) > 0 and all(isinstance(v, Node) for v in value):
                    value = NodeList(value, name=key)
            elif (
                isinstance(value, tuple)
                and not isinstance(value, NodeTuple)
                and key not in self._special_tuples
            ):
                if len(value) > 0 and all(isinstance(v, Node) for v in value):
                    value = NodeTuple(value, name=key)
        except AttributeError:
            pass
        super().__setattr__(key, value)
