from collections.abc import Iterable, Iterator, MutableSet
from typing import Callable, Sequence, Mapping, Optional, Union, Any
from warnings import warn

import numpy as np

from .backend import backend, ArrayLike
from .base import Node, NodeType, MemoNote
from .param import Param
from .collection import NodeTuple, NodeList, NodeDict
from .mixins import GetSetValues, _split_batch_shape
from .errors import (
    ActiveStateError,
    FillParamsError,
    NodeConfigurationError,
    ParamConfigurationError,
)

_SUBGRAPH_KWARGS_MESSAGE = (
    "The '{key}' argument of an @forward method is deprecated and will be removed. Ask the "
    "hierarchical child for its values instead, `child.get_values()` (and `child.get_dims()` "
    "for the dimensions to vmap over), and give them back to its method as `params`."
)


class StaticCache(MutableSet[str]):
    """The names of the ``@forward`` methods of a module whose result is kept
    from one simulation to the next, see ``Module.static_cache``.

    This is a set that looks into its module, so a change made to it is a
    change to the module. It only takes the name of an ``@forward`` method of
    the module, and taking a name out discards the result stored for it.
    """

    __slots__ = ("_module",)

    def __init__(self, module: "Module"):
        self._module = module

    def _check(self, name: object):
        """Raise unless ``name`` is the name of an ``@forward`` method of the module."""
        module = self._module
        method = getattr(type(module), name, None) if isinstance(name, str) else None
        if getattr(method, "_caskade_forward", None) != name:
            raise NodeConfigurationError(
                f"{name!r} is not the name of an @forward method of {type(module).__name__} "
                f"('{module.name}'). Only the result of an @forward method can be kept in the "
                "static_cache, by the name the method was defined with."
            )

    def __contains__(self, name: object) -> bool:
        return name in self._module._static_names

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._module._static_names))

    def __len__(self) -> int:
        return len(self._module._static_names)

    def add(self, name: str):
        """Keep the result of the ``@forward`` method ``name`` between simulations."""
        self._check(name)
        self._module._static_names.add(name)

    def discard(self, name: str):
        """Make ``name`` an ordinary method again and discard its stored result."""
        self._module._static_names.discard(name)
        self._module._static_results.pop(name, None)

    def __repr__(self) -> str:
        return f"StaticCache({list(self)})"


class Module(Node, GetSetValues):
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
        "dynamic_param_groups",
        "static_param_groups",
        "_modules",
        "_deep_params_cache",
    )  # These tuples will not be converted to NodeTuple objects

    #: All dynamic params at or below this module in the graph
    dynamic_params: tuple[Param, ...]
    #: All pointer params at or below this module in the graph
    pointer_params: tuple[Param, ...]
    #: All static params at or below this module in the graph
    static_params: tuple[Param, ...]
    #: The sorted groups that the dynamic params belong to
    dynamic_param_groups: tuple[int, ...]
    #: The sorted groups that the static params belong to
    static_param_groups: tuple[int, ...]
    #: Names of the ``@forward`` arguments filled from hierarchical children (deprecated)
    subgraph_kwargs: tuple[str, ...]

    def __init__(self, name: Optional[str] = None, **kwargs):
        """
        Initialize a Module node.

        Parameters
        ----------
        name : str, optional
            The name of this module node. If not provided, a name is
            automatically assigned by the base ``Node`` class.
        **kwargs
            Additional keyword arguments passed to the ``Node`` base class.
        """
        super().__init__(name=name, **kwargs)
        self.node_type = NodeType.MODULE
        self.valid_context = False
        self._active_cache: dict[Callable, Any] = {}
        # ``static_cache``: the names of the methods, the result stored for each
        # (with the params it was made from), and the ones being run to store them
        self._static_names: set[str] = set()
        self._static_results: dict[str, tuple[Any, tuple]] = {}
        self._static_running: set[str] = set()
        self._graph_updated()  # initialise the param tuples

    @property
    def graphviz_style(self):
        return {"style": "solid", "color": "black", "shape": "ellipse"}

    @property
    def all_params(self):
        """
        All parameters below this module in the DAG.

        Returns
        -------
        tuple of Param
            Concatenation of static, dynamic, and pointer parameters.
        """
        return self.static_params + self.dynamic_params + self.pointer_params

    def _graph_updated(self):
        """Maintain a tuple of dynamic, static, and pointer parameters at all points lower
        in the DAG."""
        T = self.topological_ordering()
        self._modules = tuple(n for n in T if isinstance(n, Module))  # this one included
        self._deep_params_cache: Optional[tuple[Param, ...]] = None
        params = tuple(n for n in T if isinstance(n, Param))
        self.dynamic_params = tuple(p for p in params if p.dynamic)
        self.dynamic_param_groups = tuple(sorted(set(p.group for p in self.dynamic_params)))
        self.pointer_params = tuple(p for p in params if p.pointer)
        self.static_params = tuple(p for p in params if p.static)
        self.static_param_groups = tuple(sorted(set(p.group for p in self.static_params)))
        subgraph_kwargs = []
        if self.subgraphs:
            for key, child in self.children.items():
                if child in self.subgraphs:
                    subgraph_kwargs.append(f"{key}_params")
                    subgraph_kwargs.append(f"{key}_dims")
        self.subgraph_kwargs = tuple(subgraph_kwargs)

    def param_order(self):
        """
        Return a human-readable string of dynamic parameter ordering.

        Each line corresponds to a parameter group and lists the parameters
        in the format ``parent_name: param_name``.

        Returns
        -------
        str
            Multi-line string describing the dynamic parameter order.
        """
        res = []
        for g in self.dynamic_param_groups:
            res.append(
                ", ".join(
                    tuple(
                        f"{next(iter(p.parents)).name}: {p.name}"
                        for p in self.dynamic_params
                        if p.group == g
                    )
                )
            )
        return "\n".join(res)

    @property
    def dynamic(self) -> bool:
        """
        Return True if the module has dynamic parameters as direct children.

        Returns
        -------
        bool
            True if any direct children are dynamic parameters.
        """
        return any(isinstance(n, Param) and n.dynamic for n in self.children.values())

    @property
    def static(self) -> bool:
        """
        Return True if the module has no dynamic parameters as direct children.

        Returns
        -------
        bool
            True if none of the direct children are dynamic parameters.
        """
        return not self.dynamic

    def to_dynamic(self, children_only=True):
        """Change all parameters to dynamic parameters.

        Parameters
        ----------
        children_only: (bool, optional)
            If True, only convert the children of this module to dynamic. If
            False, convert all parameters in the graph below this module.
            Defaults to True.
        """
        node_list = self.children.values() if children_only else self.topological_ordering()
        for node in node_list:
            if isinstance(node, Param) and not node.pointer:
                node.to_dynamic()

    def to_static(self, children_only=True):
        """Change all parameters to static parameters.

        Parameters
        ----------
        children_only: (bool, optional)
            If True, only convert children of this module to static. If False,
            convert all parameters in the graph below this module. Defaults to
            True.
        """
        node_list = self.children.values() if children_only else self.topological_ordering()
        for node in node_list:
            if isinstance(node, Param) and not node.pointer:
                node.to_static()

    def fill_params(self, params: Union[ArrayLike, Sequence, Mapping], dynamic=True):
        """
        Fill the dynamic/static parameters of the module with the input values
        from params.

        Parameters
        ----------
        params: (Union[ArrayLike, Sequence, Mapping])
            The input values to fill the parameters with. The input can be an
            ArrayLike, a Sequence, or a Mapping, laid out as
            ``get_values(dynamic=dynamic)`` returns them.
        dynamic: bool
            Operate on dynamic parameters (True, default) or static parameters
            (False).
        """
        if not self.active:
            raise ActiveStateError(f"Module {self.name} must be active to fill params")

        param_list = self.dynamic_params if dynamic else self.static_params
        groups = self.dynamic_param_groups if dynamic else self.static_param_groups
        if len(groups) > 1:
            for params_g, param_list_g in self._split_groups(params, param_list, groups):
                if self.valid_context:
                    params_g = self._transform_params(self, params_g, param_list_g, "from_valid")
                self._set_values(params_g, param_list_g, attribute="_active_value")
            return
        if self.valid_context:
            params = self._transform_params(self, params, param_list, "from_valid")
        self._set_values(params, param_list, attribute="_active_value")

    def clear_state(self):
        """
        Clear the active state `_active_value` for all params below this Module in the
        DAG. This should not be used by a user under normal circumstances."""

        for param in self.all_params:
            param._active_value = None

    def reset_active_cache(self):
        """Discard the results stored by ``@active_cache`` methods of this
        module, forcing them to be recalculated on their next call."""
        self._active_cache.clear()

    @property
    def static_cache(self) -> "StaticCache":
        """The ``@forward`` methods of this module whose result is kept from one
        simulation to the next, as a set of their names.

        A method named in this set is called outside a simulation to run it and
        store its result. Inside a simulation the stored result is returned and
        the method is not run. The result is replaced by calling the method
        outside a simulation again. This is for an expensive calculation that
        only depends on static params, such as a lookup table or a convolution
        kernel::

            sim.static_cache.add("kernel")
            sim.kernel()  # run and stored
            sim(params)  # every call to self.kernel() in here returns that result
            sim.sigma = 2.0
            sim.kernel()  # stored again, with the new value

        The set belongs to this one module and may be changed at any time. A
        module that should start with a method in it adds the name in its
        ``__init__``. Taking a name out makes the method an ordinary
        ``@forward`` method again, which runs wherever it is called, and
        discards its stored result::

            sim.sigma.to_dynamic()  # the kernel now changes with every simulation
            sim.static_cache.discard("kernel")

        A set of names may also be assigned, ``sim.static_cache = {"kernel"}``.
        Only the name of an ``@forward`` method of the module is accepted,
        anything else raises a ``NodeConfigurationError``.

        Warnings
        --------
        The stored result is never updated for you, and whether it is still the
        right one is yours to judge:

        - Arguments are ignored inside a simulation, as with ``active_cache``.
        - It is made from the values that the params hold when the method is
          called outside a simulation. Inside a simulation it stays what it is,
          whichever values that simulation was given (``params``,
          ``static_params``, an override). For a result that follows the values
          of the simulation use ``active_cache``.
        - It does not follow changes to plain attributes or to the graph.

        Two cases are reported with a ``StaticCacheWarning``. The first is when
        the method read the value of a dynamic param, since the stored result
        belongs to that one value. The second is when a stored result is used
        in a simulation after a static or pointer param that it was made from
        has been given a new value (or has been made dynamic). Nothing is said
        while you change the values, only when a simulation uses a result that
        is out of date, so several values may be changed before the method is
        called again.

        Notes
        -----
        Inside a simulation a method with no stored result raises an
        ``ActiveStateError``, call it outside a simulation first. A method that
        uses other methods with a stored result is called after them.

        The call outside a simulation follows the rules of ``forward``, so with
        batched values and no other argument the stored result is one for the
        whole batch. ``reset_static_cache`` discards the stored results and
        leaves the names in place.

        An ``@forward`` method does not have to take a param, so any method may
        be made one. Stack ``@active_cache`` on top of ``@forward`` for a method
        that should run once in each simulation when its name is not in the set.

        Examples
        --------
        ::

            class Blurred(Module):
                def __init__(self, source, sigma):
                    super().__init__()
                    self.source = source
                    self.sigma = Param("sigma", sigma)  # static
                    self.static_cache.add("kernel")

                @forward
                def kernel(self, sigma):
                    print("Building kernel...")
                    x = torch.arange(-10, 11)
                    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
                    return kernel / kernel.sum()

                @forward
                def image(self):
                    return convolve(self.source.image(), self.kernel())  # no print

            sim = Blurred(source, sigma=1.5)
            sim.kernel()  # prints, once
            for params in chain:
                sim.image(params)
        """
        return StaticCache(self)

    @static_cache.setter
    def static_cache(self, names: Union[str, Iterable[str]]):
        names = {names} if isinstance(names, str) else set(names)
        registry = StaticCache(self)
        for name in names:  # nothing changes unless every name is accepted
            registry._check(name)
        for name in self._static_names - names:
            registry.discard(name)
        self._static_names.update(names)

    def reset_static_cache(self):
        """Discard the results stored for the methods in ``static_cache``. Each
        has to be called outside a simulation again before it can be used in
        one. The names stay in the set."""
        self._static_results.clear()

    def _forget_memo(self, note: MemoNote):
        super()._forget_memo(note)
        # Results cached for a simulation are dropped as soon as it has finished
        if self._active_cache and not self.active:
            self.reset_active_cache()

    def _vmap_dim(self, param: Param) -> Optional[int]:
        """The dimension of the value of ``param`` to map over when calling a
        hierarchical child of this module, ``None`` if every call takes the same
        value. This is what fills the ``{key}_dims`` argument of an ``@forward``
        method."""
        batch, links = _split_batch_shape(self, param)  # also checks the batch shape
        return 0 if batch or links else None

    def _check_link_dims(self):
        """Raise ``ParamConfigurationError`` if a param has batch dimensions,
        but fewer than there are hierarchical links between this module and
        the param. Only a param below two or more links can be in that state,
        so for most graphs there is nothing to look at."""
        deep = self._deep_params_cache
        if deep is None:
            depths = self._hierarchy_depths()
            params = self.dynamic_params + self.static_params
            deep = self._deep_params_cache = tuple(p for p in params if depths.get(p, 0) > 1)
        for param in deep:
            _split_batch_shape(self, param)

    def _batched_values(self, dynamic: bool = True) -> Optional[tuple[list[Param], list, int]]:
        """The current values of the params, when they hold a batch of
        simulations of this module.

        Returns ``None`` when there is no batch. Otherwise returns the dynamic
        and static params that have the batch dimensions, their values with
        those dimensions in front (broadcast to the full batch shape), and the
        number of batch dimensions. Every other param has a single value that
        the whole batch shares.

        With ``dynamic=False`` only the static params are looked at, for when
        the dynamic params are known to hold no batch.
        """
        # Most modules hold no batch, find that out with as little work as possible
        found = []
        # The values are only looked at here, so they are taken straight from
        # where they are kept (a dynamic or static param is never a pointer)
        for param in self.dynamic_params if dynamic else ():
            value = param._active_value
            if value is None:
                value = param._stored_value
                if value is None:
                    return None  # the call that follows says which value is missing
            if param._shape is not None and len(value.shape) != len(param._shape):
                found.append((param, value))
        for param in self.static_params:
            if param._shape is not None:
                value = param._active_value
                if value is None:
                    value = param._stored_value
                if value is not None and len(value.shape) != len(param._shape):
                    found.append((param, value))
        if not found:
            return None

        params, values, batches = [], [], []
        for param, value in found:
            extra = len(value.shape) - len(param.shape)
            # dimensions for hierarchical links below this module are not a batch of it
            batch = _split_batch_shape(self, param, tuple(value.shape[:extra]))[0]
            if batch:
                params.append(param)
                values.append(value)
                batches.append(batch)
        if not params:
            return None
        try:
            full = np.broadcast_shapes(*batches)
        except ValueError:
            shapes = ", ".join(f"{p.name}: {b}" for p, b in zip(params, batches))
            raise ParamConfigurationError(
                f"The params of '{self.name}' hold batches that do not go together, so they "
                f"cannot be run as one batch of simulations. Batch shapes ({shapes})."
            ) from None
        for i, batch in enumerate(batches):
            if batch != full:
                value = values[i]
                values[i] = backend.broadcast_to(value, full + tuple(value.shape[len(batch) :]))
        return params, values, len(full)

    def fill_kwargs(self, keys: tuple[str, ...]) -> dict[str, ArrayLike]:
        """
        Fill the kwargs for an ``@forward`` method with the values of the
        dynamic parameters. The requested keys are matched to names of ``Param``
        objects owned by the ``Module``. This should not be used by the user
        under normal circumstances.
        """
        kwargs = {}
        children = self.children
        for key in keys:
            if key in self.subgraph_kwargs:
                warn(_SUBGRAPH_KWARGS_MESSAGE.format(key=key), DeprecationWarning, stacklevel=3)
                subgraph: Any  # a Module or NodeCollection
                if key.endswith("_params"):
                    subgraph = children[key[:-7]]
                    kwargs[key] = subgraph.get_values("list")
                else:
                    subgraph = children[key[:-5]]
                    kwargs[key] = subgraph.get_values(
                        "list", attribute=self._vmap_dim, respect_valid=False
                    )
                continue
            child = children.get(key)
            if isinstance(child, Param):
                val = child.value
                if val is None:
                    raise FillParamsError(
                        f"Param {key} in Module {self.name} has no value. "
                        "Ensure that the parameter is set before calling the forward method or provided with the params."
                    )
                kwargs[key] = val
        return kwargs

    @property
    def node_str(self) -> str:
        """
        Returns a string representation of the node for graph visualization.
        """
        return f"{self.name}|{self.__class__.__name__}"

    def __setattr__(self, key: str, value: Any):
        """Intercept attribute setting to update parameters and graph links."""
        # ``_children`` only exists once ``Node.__init__`` has run
        child = self.__dict__.get("_children", {}).get(key)
        if isinstance(child, Param):
            child.value = value
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
        elif isinstance(value, dict) and not isinstance(value, NodeDict):
            if len(value) > 0 and all(isinstance(v, Node) for v in value.values()):
                value = NodeDict(value, name=key)
        super().__setattr__(key, value)
