from .base import Node


class NodeCollection(Node):
    def to_dynamic(self, **kwargs):
        for node in self:
            if hasattr(node, "to_dynamic"):
                node.to_dynamic(**kwargs)

    def to_static(self, **kwargs):
        for node in self:
            if hasattr(node, "to_static"):
                node.to_static(**kwargs)

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
        self._type = "ntuple"

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
        self._type = "nlist"

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
