from typing import Sequence, Mapping, Optional, Union, Any

from .backend import ArrayLike
from .base import Node
from .param import Param
from .collection import NodeTuple, NodeList
from .mixins import GetSetValues
from .errors import ActiveStateError, FillParamsError


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
    )  # These tuples will not be converted to NodeTuple objects

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
        self.dynamic_params = ()
        self.pointer_params = ()
        self.static_params = ()
        self.dynamic_param_groups = ()
        super().__init__(name=name, **kwargs)
        self.node_type = "module"
        self.valid_context = False

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

    def update_graph(self):
        """Maintain a tuple of dynamic, static, and pointer parameters at all points lower
        in the DAG."""
        T = self.topological_ordering()
        self.dynamic_params = tuple(filter(lambda n: isinstance(n, Param) and n.dynamic, T))
        self.dynamic_param_groups = tuple(sorted(set(p.group for p in self.dynamic_params)))
        self.pointer_params = tuple(filter(lambda n: isinstance(n, Param) and n.pointer, T))
        self.static_params = tuple(filter(lambda n: isinstance(n, Param) and n.static, T))
        self.subgraph_kwargs = []
        if self.subgraphs:
            for key, child in self.children.items():
                if child in self.subgraphs:
                    self.subgraph_kwargs.append(f"{key}_params")
                    self.subgraph_kwargs.append(f"{key}_dims")
        self.subgraph_kwargs = tuple(self.subgraph_kwargs)
        super().update_graph()

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
            The input values to fill the dynamic parameters with. The input can
            be an ArrayLike, a Sequence, or a Mapping.
        dynamic: bool
            Operate on dynamic parameters (True, default) or static parameters
            (False).
        """
        if not self.active:
            raise ActiveStateError(f"Module {self.name} must be active to fill params")

        param_list = self.dynamic_params if dynamic else self.static_params
        if len(self.dynamic_param_groups) == 1:
            params = (params,)

        for group, params_g in zip(self.dynamic_param_groups, params):
            param_list_g = tuple(p for p in param_list if p.group == group)
            if self.valid_context:
                params_g = self.from_valid(params_g, param_list_g)
            self._set_values(params_g, param_list_g, attribute="_value")

    def clear_state(self):
        """
        Clear the active state `_value` for all params below this Module in the
        DAG. This should not be used by a user under normal circumstances."""

        for param in self.all_params:
            param._value = None

    def remove_memo(self, memo):
        if hasattr(self, "reset_active_cache") and memo == "active":
            self.reset_active_cache()
        return super().remove_memo(memo)

    def fill_kwargs(self, keys: tuple[str]) -> dict[str, ArrayLike]:
        """
        Fill the kwargs for an ``@forward`` method with the values of the
        dynamic parameters. The requested keys are matched to names of ``Param``
        objects owned by the ``Module``. This should not be used by the user
        under normal circumstances.
        """
        kwargs = {}
        for key in keys:
            if key in self.subgraph_kwargs:
                if key.endswith("_params"):
                    kwargs[key] = self[key[:-7]].get_values("list")
                else:
                    kwargs[key] = list(
                        0 if p.batched else None for p in self[key[:-5]].dynamic_params
                    )
            elif key in self.children and isinstance(self[key], Param):
                val = self.children[key].value
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
