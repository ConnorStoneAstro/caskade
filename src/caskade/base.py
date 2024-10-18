from typing import Optional, Union


class Node(object):
    """
    Base graph node class for caskade objects.

    The `Node` object is the base class for all caskade objects. It is used to
    construct the directed acyclic graph (DAG). The primary function of the
    `Node` object is to manage the parent-child relationships between nodes in
    the graph. There is limited functionality for the `Node` object, though it
    implements the base versions of the `active` state and `to` /
    `update_dynamic_params` methods. The `active` state is used to communicate
    through the graph that the simulator is currently running. The `to` method
    is used to move and/or cast the values of the parameter. The
    `update_dynamic_params` method is used by `Module` objects to keep track of
    all dynamic `Param` objects below them in the graph.

    Examples
    --------
    ``` python
    n1 = Node()
    n2 = Node()
    n1.link("subnode", n2) # link n2 as a child of n1, may use any str as the key
    n1.unlink("subnode") # alternately n1.unlink(n2) to unlink by object
    ```
    """

    def __init__(self, name: Optional[str] = None):
        if name is None:
            name = self.__class__.__name__
        assert isinstance(name, str), f"{self.__class__.__name__} name must be a string"
        assert "|" not in name, f"{self.__class__.__name__} cannot contain '|'"
        self._name = name
        self._children = {}
        self._parents = set()
        self._active = False
        self._type = "node"

    @property
    def name(self) -> str:
        return self._name

    @property
    def children(self) -> dict:
        return self._children

    @property
    def parents(self) -> set:
        return self._parents

    def link(self, key: str, child: "Node"):
        """Link the current `Node` object to another `Node` object as a child."""
        # Avoid double linking to the same object
        if key in self.children:
            raise ValueError(f"Child key {key} already linked to parent {self.name}")
        for ownchild in self.children.values():
            if ownchild == child:
                raise ValueError(f"Child {child.name} already linked to parent {self.name}")

        self._children[key] = child
        child._parents.add(self)
        self.update_dynamic_params()

    def unlink(self, key: Union[str, "Node"]):
        """Unlink the current `Node` object from another `Node` object which is a child."""
        if isinstance(key, Node):
            for node in self.children:
                if self.children[node] == key:
                    key = node
                    break
        self._children[key]._parents.remove(self)
        self._children[key].update_dynamic_params()
        del self._children[key]
        self.update_dynamic_params()

    def topological_ordering(self, with_type: Optional[str] = None) -> tuple["Node"]:
        """Return a topological ordering of the graph below the current node."""
        ordering = [self]
        for node in self.children.values():
            for subnode in node.topological_ordering():
                if subnode not in ordering:
                    ordering.append(subnode)
        if with_type is None:
            return tuple(ordering)
        return tuple(filter(lambda n: n._type == with_type, ordering))

    def update_dynamic_params(self):
        """Update the dynamic parameters of the current node and all children.
        This is intended to be overridden."""
        for parent in self.parents:
            parent.update_dynamic_params()

    @property
    def active(self) -> bool:
        return self._active

    @active.setter
    def active(self, value: bool):
        # Avoid unnecessary updates
        if self._active == value:
            return

        # Set self active level
        self._active = value

        # Propagate active level to children
        for child in self._children.values():
            child.active = value

    def to(self, device=None, dtype=None):
        """
        Moves and/or casts the PyTorch values of the Node.

        Parameters
        ----------
        device: (Optional[torch.device], optional)
            The device to move the values to. Defaults to None.
        dtype: (Optional[torch.dtype], optional)
            The desired data type. Defaults to None.
        """

        for child in self.children.values():
            child.to(device=device, dtype=dtype)

    def graph_dict(self) -> dict[str, dict]:
        """Return a dictionary representation of the graph below the current
        node."""
        graph = {
            f"{self.name}|{self._type}": {},
        }
        for node in self.children.values():
            graph[f"{self.name}|{self._type}"].update(node.graph_dict())
        return graph

    @staticmethod
    def graph_print(dag: dict, depth: int = 0, indent: int = 4, result: str = "") -> str:
        """Print the graph dictionary in a human-readable format."""
        for key in dag:
            result = f"{result}{' ' * indent * depth}{key}\n"
            result = Node.graph_print(dag[key], depth + 1, indent, result) + "\n"
        if result:  # remove trailing newline
            result = result[:-1]
        return result

    def __str__(self) -> str:
        return self.graph_print(self.graph_dict())

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"
