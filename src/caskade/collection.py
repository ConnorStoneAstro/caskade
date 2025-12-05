from typing import Union, Sequence, Mapping

from math import prod

from .base import Node
from .backend import backend, ArrayLike
from .errors import (
    ParamConfigurationError,
    FillParamsArrayError,
    FillParamsSequenceError,
    FillParamsMappingError,
)


class NodeCollection(Node):
    def to_dynamic(self, **kwargs):
        for node in self:
            if hasattr(node, "to_dynamic"):
                node.to_dynamic(**kwargs)

    def to_static(self, **kwargs):
        for node in self:
            if hasattr(node, "to_static"):
                node.to_static(**kwargs)

    def set_values(
        self, params: Union[ArrayLike, Sequence, Mapping], node_type="all", attribute="value"
    ):
        if node_type == "all":
            node_type = "dynamic/static"
        if isinstance(params, backend.array_type):
            if params.shape[-1] == 0:
                return  # No parameters to fill
            # check for batch dimension
            batch = len(params.shape) > 1
            B = tuple(params.shape[:-1]) if batch else ()
            pos = 0
            for param in self:
                if param.node_type not in node_type:
                    continue
                if not isinstance(param.shape, tuple):
                    raise ParamConfigurationError(
                        f"Param {param.name} has no shape. dynamic parameters must have a shape to use {backend.array_type.__name__} input."
                    )
                # Handle scalar parameters
                size = max(1, prod(param.shape))
                try:
                    val = backend.view(params[..., pos : pos + size], B + param.shape)
                    setattr(param, attribute, val)
                except (RuntimeError, IndexError, ValueError, TypeError):
                    raise FillParamsArrayError(self.name, params, self)

                pos += size
            if pos != params.shape[-1]:
                raise FillParamsArrayError(self.name, params, self)
        elif isinstance(params, Sequence):
            if len(params) == 0:
                return
            elif len(params) == len(self):
                param_list = filter(lambda p: p.node_type in node_type, self)
                for param, value in zip(param_list, params):
                    setattr(param, attribute, value)
            else:
                raise FillParamsSequenceError(self.name, params, self)
        elif isinstance(params, Mapping):
            params_names = set(params.keys())
            for name, param in self.children.items():
                if name in params:
                    params_names.remove(name)
                    setattr(param, attribute, params[name])
            if len(params_names) > 0:
                raise FillParamsMappingError(self.name, self.children, next(iter(params_names)))
        else:
            raise TypeError(
                f"Input params type {type(params)} not supported. Should be {backend.array_type.__name__}, Sequence, or Mapping."
            )

    def copy(self):
        raise NotImplementedError

    def deepcopy(self):
        raise NotImplementedError

    @property
    def dynamic(self):
        return any(node.dynamic for node in self)

    @property
    def static(self):
        return not self.dynamic

    def __mul__(self, other):
        raise NotImplementedError

    def __eq__(self, other):
        return Node.__eq__(self, other)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name})[{len(self)}]"

    def __hash__(self):
        return Node.__hash__(self)


class NodeTuple(NodeCollection, tuple):
    graphviz_types = {"ntuple": {"style": "solid", "color": "black", "shape": "tab"}}

    def __init__(self, iterable=None, name=None):
        tuple.__init__(iterable)
        Node.__init__(self, name=name)
        self.node_type = "ntuple"

        for node in self:
            if not isinstance(node, Node):
                raise TypeError(f"NodeTuple elements must be Node objects, not {type(node)}")
            self.link(node)

    def __getitem__(self, key):
        if isinstance(key, str):
            return Node.__getitem__(self, key)
        return tuple.__getitem__(self, key)

    def __add__(self, other):
        res = super().__add__(other)
        return NodeTuple(res)


class NodeList(NodeCollection, list):
    graphviz_types = {"nlist": {"style": "solid", "color": "black", "shape": "folder"}}

    def __init__(self, iterable=(), name=None):
        list.__init__(self, iterable)
        Node.__init__(self, name)
        self.node_type = "nlist"

        self._link_nodes()

    def _unlink_nodes(self):
        for node in self:
            self.unlink(node)

    def _link_nodes(self):
        for node in self:
            if not isinstance(node, Node):
                raise TypeError(f"NodeList elements must be Node objects, not {type(node)}")
            self.link(node)

    def append(self, node):
        self._unlink_nodes()
        super().append(node)
        self._link_nodes()

    def insert(self, index, node):
        self._unlink_nodes()
        super().insert(index, node)
        self._link_nodes()

    def extend(self, iterable):
        self._unlink_nodes()
        super().extend(iterable)
        self._link_nodes()

    def clear(self):
        self._unlink_nodes()
        super().clear()
        self._link_nodes()

    def pop(self, index=-1):
        self._unlink_nodes()
        node = super().pop(index)
        self._link_nodes()
        return node

    def remove(self, value):
        self._unlink_nodes()
        super().remove(value)
        self._link_nodes()

    def __getitem__(self, key):
        if isinstance(key, str):
            return Node.__getitem__(self, key)
        if isinstance(key, slice):
            return NodeList(list.__getitem__(self, key), name=self.name)
        return list.__getitem__(self, key)

    def __setitem__(self, key, value):
        self._unlink_nodes()
        super().__setitem__(key, value)
        self._link_nodes()

    def __delitem__(self, key):
        self._unlink_nodes()
        super().__delitem__(key)
        self._link_nodes()

    def __add__(self, other):
        res = super().__add__(other)
        return NodeList(res, name=self.name)

    def __iadd__(self, other):
        self._unlink_nodes()
        ret = super().__iadd__(other)
        self._link_nodes()
        return ret

    def __imul__(self, other):
        raise NotImplementedError
