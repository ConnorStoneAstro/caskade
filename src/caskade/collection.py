from re import fullmatch

from .base import Node


class NodeTuple(tuple, Node):
    _collections = set()

    def __init__(self, iterable=None):
        tuple.__init__(iterable)
        Node.__init__(self, self._get_name())

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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name})[{len(self)}]"


class NodeList(list, Node):
    _collections = set()

    def __init__(self, iterable=None):
        list.__init__(iterable)
        Node.__init__(self, self._get_name())

        self._rename_nodes()

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
        super().append(self, node)
        self._link_nodes()

    def insert(self, index, node):
        self._unlink_nodes()
        super().insert(self, index, node)
        self._link_nodes()

    def extend(self, iterable):
        self._unlink_nodes()
        super().extend(self, iterable)
        self._link_nodes()

    def clear(self):
        self._unlink_nodes()
        super().clear(self)
        self._link_nodes()

    def __copy__(self):
        raise NotImplementedError

    def __deepcopy__(self, memo):
        raise NotImplementedError

    def pop(self, index=-1):
        self._unlink_nodes()
        node = super().pop(self, index)
        self._link_nodes()
        return node

    def remove(self, value):
        self._unlink_nodes()
        super().remove(self, value)
        self._link_nodes()

    def __getitem__(self, key):
        if isinstance(key, str):
            return Node.__getitem__(self, key)
        return tuple.__getitem__(self, key)

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
