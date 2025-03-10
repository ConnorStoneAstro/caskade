from .base import Node


class NodeTuple(tuple, Node):
    _collections = set()

    def __init__(self, iterable=None):
        tuple.__init__(iterable)
        Node.__init__(self, self._get_name())

        assert all(
            isinstance(n, Node) for n in self
        ), "All elements of a NodeTuple must be Node objects"

        for n in range(len(self)):
            self.link(f"Node{n}", self[n])

    @classmethod
    def _get_name(cls):
        c = 0
        while c in cls._collections:
            c += 1
        cls._collections.add(c)
        return f"NodeTuple{c}"

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
            self._collections.remove(int(self._name[9:]))
        except:
            pass


class NodeList(list, Node):
    _collections = set()

    def __init__(self, *args, **kwargs):
        list.__init__(self, *args, **kwargs)
        Node.__init__(self, self._get_name())

        self._link_nodes()

    @classmethod
    def _get_name(cls):
        c = 0
        while c in cls._collections:
            c += 1
        cls._collections.add(c)
        return f"NodeList{c}"

    def _unlink_nodes(self):
        for n in range(len(self)):
            self.unlink(f"Node{n}")

    def _link_nodes(self):
        for n in range(len(self)):
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
            self._collections.remove(int(self._name[8:]))
        except:
            pass
