from .base import Node


class TupleCollection(tuple, Node):
    _collections = set()

    def __init__(self, iterable=None):
        list.__init__(iterable)
        Node.__init__(self, self._get_name())

        for n in range(len(self)):
            self.link(f"Node{n}", self[n])

    @classmethod
    def _get_name(cls):
        c = 0
        while c in cls._collections:
            c += 1
        cls._collections.add(c)
        return f"Collection{c}"

    def __getitem__(self, key):
        if isinstance(key, str):
            return Node.__getitem__(self, key)
        return tuple.__getitem__(self, key)

    def copy(self):
        raise NotImplementedError


class ListCollection(list, Node):
    _collections = set()

    def __init__(self, iterable=None):
        list.__init__(iterable)
        Node.__init__(self, self._get_name())

        self._nodes = set()
        for n in range(len(self)):
            self._nodes.add(n)
            self.link(f"Node{n}", self[n])

    @classmethod
    def _get_name(cls):
        c = 0
        while c in cls._collections:
            c += 1
        cls._collections.add(c)
        return f"Collection{c}"

    def _get_nodename(self):
        n = 0
        while n in self._nodes:
            n += 1
        self._nodes.add(n)
        return f"Node{n}"

    def __getitem__(self, key):
        if isinstance(key, str):
            return Node.__getitem__(self, key)
        return tuple.__getitem__(self, key)

    def append(self, node):
        n = self._get_nodename()
        self.link(f"Node{n}", node)
        super().append(self, node)

    def insert(self, index, node):
        n = self._get_nodename()
        self.link(f"Node{n}", node)
        super().insert(self, index, node)

    def extend(self, iterable):
        for node in iterable:
            n = self._get_nodename()
            self.link(f"Node{n}", node)
        super().extend(self, iterable)

    def clear(self):
        for node in self.children:
            self._nodes.remove(node)
            self.unlink(self.children[node])
        super().clear(self)

    def copy(self):
        raise NotImplementedError

    def pop(self, index=-1):
        node = super().pop(self, index)
        for n in self.children:
            if self.children[n] is node:
                self._nodes.remove(n)
                self.unlink(n)
                break
        return node

    def remove(self, value):
        for n in self.children:
            if self.children[n] is value:
                self._nodes.remove(n)
                self.unlink(n)
                break
        super().remove(self, value)
