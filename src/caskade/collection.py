from .base import Node


class NodeTuple(tuple, Node):
    _collections = set()
    graphviz_types = {"ntuple": {"style": "solid", "color": "black", "shape": "tab"}}

    def __init__(self, iterable=None, name=None):
        tuple.__init__(iterable)
        Node.__init__(self, self._get_name(name))
        self._type = "ntuple"

        for n in range(len(self)):
            if not isinstance(self[n], Node):
                raise TypeError(f"NodeTuple elements must be Node objects, not {type(self[n])}")
            self.link(f"Node{n}", self[n])

    def to_dynamic(self, **kwargs):
        for n in range(len(self)):
            self[n].to_dynamic(**kwargs)

    def to_static(self, **kwargs):
        for n in range(len(self)):
            self[n].to_static(**kwargs)

    @classmethod
    def _get_name(cls, name):
        c = 0
        if name is None:
            name = "NodeTuple"
        if name not in cls._collections:
            cls._collections.add(name)
            return name
        while f"{name}_{c}" in cls._collections:
            c += 1
        name = f"{name}_{c}"
        cls._collections.add(name)
        return name

    def __getitem__(self, key):
        if isinstance(key, str):
            return Node.__getitem__(self, key)
        return tuple.__getitem__(self, key)

    def copy(self):
        raise NotImplementedError

    def deepcopy(self):
        raise NotImplementedError

    def __add__(self, other):
        res = super().__add__(other)
        return NodeTuple(res)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name})[{len(self)}]"

    def __del__(self):
        try:
            self._collections.remove(self._name)
        except:
            pass


class NodeList(list, Node):
    _collections = set()
    graphviz_types = {"nlist": {"style": "solid", "color": "black", "shape": "folder"}}

    def __init__(self, iterable=(), name=None):
        list.__init__(self, iterable)
        Node.__init__(self, self._get_name(name))
        self._type = "nlist"

        self._link_nodes()

    def to_dynamic(self, **kwargs):
        for n in range(len(self)):
            self[n].to_dynamic(**kwargs)

    def to_static(self, **kwargs):
        for n in range(len(self)):
            self[n].to_static(**kwargs)

    @classmethod
    def _get_name(cls, name):
        c = 0
        if name is None:
            name = "NodeList"
        if name not in cls._collections:
            cls._collections.add(name)
            return name
        while f"{name}_{c}" in cls._collections:
            c += 1
        name = f"{name}_{c}"
        cls._collections.add(name)
        return name

    def _unlink_nodes(self):
        for n in range(len(self)):
            self.unlink(f"Node{n}")

    def _link_nodes(self):
        for n in range(len(self)):
            if not isinstance(self[n], Node):
                raise TypeError(f"NodeList elements must be Node objects, not {type(self[n])}")
            self.link(f"Node{n}", self[n])

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

    def copy(self):
        raise NotImplementedError

    def deepcopy(self):
        raise NotImplementedError

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
            return NodeList(super().__getitem__(key))
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
        return NodeList(res)

    def __iadd__(self, other):
        self._unlink_nodes()
        ret = super().__iadd__(other)
        self._link_nodes()
        return ret

    def __mul__(self, other):
        raise NotImplementedError

    def __imul__(self, other):
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name})[{len(self)}]"

    def __eq__(self, other):
        return Node.__eq__(self, other)

    def __hash__(self):
        return Node.__hash__(self)

    def __del__(self):
        try:
            self._collections.remove(self._name)
        except:
            pass
