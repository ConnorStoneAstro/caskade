from typing import Optional

import torch


class Node:

    def __init__(self, name):
        self._name = name
        self._children = {}
        self._parents = set()
        self._active = False
        self._type = "node"

    @property
    def children(self):
        return self._children

    @property
    def parents(self):
        return self._parents

    def link(self, key, child):
        self._children[key] = child
        child._parents.add(self)
        self.update_dynamic_params()

    def unlink(self, key):
        if isinstance(key, Node):
            for node in self.children:
                if self.children[node] == key:
                    key = node
                    break
        self._children[key]._parents.remove(self)
        del self._children[key]
        self.update_dynamic_params()

    def topological_ordering(self, with_type=None):
        ordering = [self]
        for node in self.children.values():
            for subnode in node.topological_ordering():
                if subnode not in ordering:
                    ordering.append(subnode)
        if with_type is None:
            return ordering
        return tuple(filter(lambda n: n._type == with_type, ordering))

    def update_dynamic_params(self):
        for parent in self.parents:
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

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        """
        Moves and/or casts the values of the parameter.

        Parameters
        ----------
        device: (Optional[torch.device], optional)
            The device to move the values to. Defaults to None.
        dtype: (Optional[torch.dtype], optional)
            The desired data type. Defaults to None.
        """

        for child in self.children.values():
            child.to(device=device, dtype=dtype)

    def graph_dict(self, with_type=True, with_object=True):
        value = []
        if with_type:
            value.append(self._type)
        if with_object:
            value.append(self)
        graph = {self.name: value + [{child.graph_dict() for child in self.children.values()}]}
        return graph

    def __str__(self):
        return str(((node.name, node._type) for node in self.topological_ordering()))

    def __repr__(self):
        return str(self.graph_dict(True, False))
