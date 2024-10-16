

class Node:

    def __init__(self, name):
        self._name = name
        self._children = {}

    def link(self, child, key=None):
        key = child.name if key is None else key
        self._children[key] = child
        super().__setattr__(key, child)

    def unlink(self, key):
        del self._children[key]

    def __setattr__(self, key, value):
        if isinstance(value, Node):
            self.link(value, key = key)
            return
        if key in self.children:
            self.children[key].value = value
            return
        super().__setattr__(key, value)
