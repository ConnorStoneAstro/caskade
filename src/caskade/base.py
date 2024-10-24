from typing import Optional, Union


class Node(object):
    """
    Base graph node class for caskade objects.

    The `Node` object is the base class for all caskade objects. It is used to
    construct the directed acyclic graph (DAG). The primary function of the
    `Node` object is to manage the parent-child relationships between nodes in
    the graph. There is limited functionality for the `Node` object, though it
    implements the base versions of the `active` state and `to` /
    `update_graph` methods. The `active` state is used to communicate
    through the graph that the simulator is currently running. The `to` method
    is used to move and/or cast the values of the parameter. The `update_graph`
    method is used signal all parents that the graph below them has changed.

    Examples
    --------
    ```{python}
    n1 = Node()
    n2 = Node()
    n1.link("subnode", n2) # link n2 as a child of n1, may use any str as the key
    n1.unlink("subnode") # alternately n1.unlink(n2) to unlink by object
    ```
    """

    graphviz_types = {"node": {"style": "solid", "color": "black", "shape": "circle"}}

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

    def link(self, key: Union[str, "Node"], child: Optional["Node"] = None):
        """Link the current `Node` object to another `Node` object as a child.

        Parameters
        ----------
        key: (Union[str, Node])
            The key to link the child node with.
        child: (Optional[Node], optional)
            The child `Node` object to link to. Defaults to None in which
            case the key is used as the child.

        Examples
        --------
        ```{python}
        n1 = Node()
        n2 = Node()

        n1.link("subnode", n2) # may use any str as the key
        n1.unlink("subnode")

        # Alternately, link by object
        n1.link(n2)
        n1.unlink(n2)
        ```
        """
        if child is None:
            child = key
            key = child.name
        # Avoid double linking to the same object
        if key in self.children:
            raise ValueError(f"Child key {key} already linked to parent {self.name}")
        if child in self.children.values():
            raise ValueError(f"Child {child.name} already linked to parent {self.name}")
        # avoid cycles
        if self in child.topological_ordering():
            raise ValueError(
                f"Linking {child.name} to {self.name} would create a cycle in the graph"
            )

        self._children[key] = child
        child._parents.add(self)
        self.update_graph()

    def unlink(self, key: Union[str, "Node"]):
        """Unlink the current `Node` object from another `Node` object which is a child."""
        if isinstance(key, Node):
            for node in self.children:
                if self.children[node] == key:
                    key = node
                    break
        self._children[key]._parents.remove(self)
        self._children[key].update_graph()
        del self._children[key]
        self.update_graph()

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

    def update_graph(self):
        """Triggers a call to all parents that the graph below them has been
        updated. The base `Node` object does nothing with this information, but
        other node types may use this to update internal state."""
        for parent in self.parents:
            parent.update_graph()

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

        return self

    def graphviz(self, top_down=True) -> "graphviz.Digraph":
        """Return a graphviz object representing the graph below the current
        node in the DAG.

        Parameters
        ----------
        top_down: (bool, optional)
            Whether to draw the graph top-down (current node at top) or
            bottom-up (current node at bottom). Defaults to True.
        """
        import graphviz

        components = set()

        def add_node(node, dot):
            if node in components:
                return
            dot.attr("node", **node.graphviz_types[node._type])
            dot.node(str(id(node)), f"{node.__class__.__name__}('{node.name}')")
            components.add(node)

            for child in node.children.values():
                add_node(child, dot)
                if top_down:
                    dot.edge(str(id(node)), str(id(child)))
                else:
                    dot.edge(str(id(child)), str(id(node)))

        dot = graphviz.Digraph(strict=True)
        add_node(self, dot)
        return dot

    def graph_dict(self) -> dict[str, dict]:
        """Return a dictionary representation of the graph below the current
        node."""
        graph = {
            f"{self.name}|{self._type}": {},
        }
        for node in self.children.values():
            graph[f"{self.name}|{self._type}"].update(node.graph_dict())
        return graph

    def graph_print(self, dag: dict, depth: int = 0, indent: int = 4, result: str = "") -> str:
        """Print the graph dictionary in a human-readable format."""
        for key in dag:
            result = f"{result}{' ' * indent * depth}{key}\n"
            result = self.graph_print(dag[key], depth + 1, indent, result) + "\n"
        if result:  # remove trailing newline
            result = result[:-1]
        return result

    def __str__(self) -> str:
        return self.graph_print(self.graph_dict())

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"

    def __getitem__(self, key: str) -> "Node":
        return self.children[key]
