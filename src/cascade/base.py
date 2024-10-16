class Node:

    def __init__(self, name):
        self._name = name
        self._children = {}
        self._parents = {}
        self._active = False

    def link(self, child, key=None):
        key = child.name if key is None else key
        self._children[key] = child
        super().__setattr__(key, child)
        self.update_dynamic_params()

    def unlink(self, key):
        del self._children[key]

    @property
    def topological_ordering(self):
        ordering = [self]
        for node in self.children.values():
            for subnode in node.topological_ordering:
                if subnode not in ordering:
                    ordering.append(subnode)
        return ordering

    def update_dynamic_params(self):
        for parent in self.parents.values():
            parent.update_dynamic_params()

    @property
    def active(self):
        return self._active

    @active.setter
    def active(self, value):
        # Avoid unnecessary updates
        if self._active == value:
            return

        # Set self active level
        self._active = value

        # Propagate active level to children
        for child in self._children.values():
            child.active = value

    def __setattr__(self, key, value):
        if isinstance(value, Node):
            self.link(value, key=key)
            return
        if key in self.children:
            self.children[key].value = value
            return
        super().__setattr__(key, value)

    def graph_dict(self, with_type=True, with_object=True):
        value = []
        if with_type:
            value.append(self._type)
        if with_object:
            value.append(self)
        graph = {self.name: value + [{child.graph_dict() for child in self.children.values()}]}
        return graph

    def __str__(self):
        return str(((node.name, node._type) for node in self.topological_ordering))

    def __repr__(self):
        return str(self.graph_dict(True, False))
