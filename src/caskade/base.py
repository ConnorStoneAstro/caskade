from typing import Optional, Union, Any

from .errors import GraphError, NodeConfigurationError, LinkToAttributeError


class Node:
    """
    Base graph node class for ``caskade`` objects.

    The ``Node`` object is the base class for all ``caskade`` objects. It is used to
    construct the directed acyclic graph (DAG). The primary function of the
    ``Node`` object is to manage the parent-child relationships between nodes in
    the graph. There is limited functionality for the ``Node`` object, though it
    implements the base versions of the ``active`` state and ``to`` /
    ``update_graph`` methods. The ``active`` state is used to communicate
    through the graph that the simulator is currently running. The ``to`` method
    is used to move and/or cast the values of the parameter. The ``update_graph``
    method is used signal all parents that the graph below them has changed.

    Examples
    --------

    Example making some ``Node`` objects and then linking/unlinking them::

       n1 = Node()
       n2 = Node()
       n1.link("subnode", n2) # link n2 as a child of n1, may use any str as the key
       n1.unlink("subnode") # alternately n1.unlink(n2) to unlink by object
    """

    graphviz_types = {"node": {"style": "solid", "color": "black", "shape": "circle"}}

    def __init__(self, name: Optional[str] = None):
        if name is None:
            name = self.__class__.__name__
        if not isinstance(name, str):
            raise NodeConfigurationError(f"{self.__class__.__name__} name must be a string")
        if "|" in name:
            raise NodeConfigurationError(f"{self.__class__.__name__} cannot contain '|'")
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

    def _link(self, key: str, child: "Node"):
        if self.active:
            raise GraphError("Cannot link/unlink nodes while the graph is active")
        # Avoid double linking to the same object
        if key in self.children:
            if self.children[key] is child:
                return
            raise GraphError(f"Child key '{key}' already linked to parent {self.name}")
        if child in self.children.values():
            raise GraphError(f"Child {child.name} already linked to parent {self.name}")
        if hasattr(self, key):
            raise LinkToAttributeError(
                f"Child key '{key}' already an attribute of parent {self.name}, use a different name"
            )

        # avoid cycles
        if self in child.topological_ordering():
            raise GraphError(
                f"Linking {child.name} to {self.name} would create a cycle in the graph"
            )

        self._children[key] = child
        child._parents.add(self)
        self.update_graph()

    def link(self, key: Union[str, "Node"], child: Optional["Node"] = None):
        """Link the current ``Node`` object to another ``Node`` object as a child.

        Parameters
        ----------
        key: (Union[str, Node])
            The key to link the child node with.
        child: (Optional[Node], optional)
            The child ``Node`` object to link to. Defaults to None in which
            case the key is used as the child.

        Examples
        --------

        Example making some ``Node`` objects and then linking/unlinking them. demonstrating multiple ways to link/unlink::

            n1 = Node()
            n2 = Node()

            n1.link("subnode", n2) # may use any str as the key
            n1.unlink("subnode")

            # Alternately, link by object
            n1.link(n2)
            n1.unlink(n2)
        """
        if child is None:
            child = key
            key = child.name
        self.__setattr__(key, child)

    def _unlink(self, key: str):
        if self.active:
            raise GraphError(f"Cannot link/unlink nodes while the graph is active ({self.name})")
        self._children[key]._parents.remove(self)
        self._children[key].update_graph()
        del self._children[key]
        self.update_graph()

    def unlink(self, key: Union[str, "Node"]):
        """Unlink the current ``Node`` object from another ``Node`` object which is a child."""
        if isinstance(key, Node):
            for node in self.children:
                if self.children[node] is key:
                    key = node
                    break
        self.__delattr__(key)

    def topological_ordering(
        self, with_type: Optional[str] = None, with_isinstance: Optional[object] = None
    ) -> tuple["Node"]:
        """Return a topological ordering of the graph below the current node."""
        ordering = [self]
        for node in self.children.values():
            for subnode in node.topological_ordering():
                if subnode not in ordering:
                    ordering.append(subnode)
        if with_type is not None:
            ordering = filter(lambda n: with_type in n._type, ordering)
        if with_isinstance is not None:
            ordering = filter(lambda n: isinstance(n, with_isinstance), ordering)
        return tuple(ordering)

    def update_graph(self):
        """Triggers a call to all parents that the graph below them has been
        updated. The base ``Node`` object does nothing with this information, but
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
        Moves and/or casts the PyTorch values of the ``Node``.

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

    def _save_state_hdf5(self, h5group, appendable: bool = False):
        """Save the state of the node and its children to HDF5."""
        for key, child in self.children.items():
            if not hasattr(child, "_h5group"):
                child._h5group = h5group.create_group(key)
            elif key not in h5group:
                h5group[key] = child._h5group
            child._save_state_hdf5(h5group[key], appendable=appendable)

    def save_state(self, saveto: str, appendable: bool = False):
        """Save the state of the node and its children."""
        if saveto.endswith(".h5") or saveto.endswith(".hdf5"):
            import h5py  # noqa

            with h5py.File(saveto, "w") as h5file:
                self._h5group = h5file.create_group(self.name)
                self._save_state_hdf5(h5file[self.name], appendable=appendable)

            for node in self.topological_ordering():
                del node._h5group
        else:
            raise NotImplementedError(
                "Only HDF5 files ('.h5') are currently supported for saving state"
            )

    def _check_append_state_hdf5(self, h5group):
        """Check the state and HDF5 file have the same structure."""
        for key, child in self.children.items():
            if key in h5group:
                child._check_append_state_hdf5(h5group[key])
            else:
                raise GraphError(
                    f"Child '{child.name}' (identified by key '{key}') not found in HDF5 group '{h5group.name}'. Structure of graph changed from last save."
                )

    def _append_state_cleanup(self):
        """Cleanup the state of the node and its children after appending to HDF5."""
        pass

    def _append_state_hdf5(self, h5group):
        """Append the state of the node and its children to an existing HDF5 file."""
        for key, child in self.children.items():
            child._append_state_hdf5(h5group[key])

    def append_state(self, saveto: str):
        """Append the state of the node and its children to an existing HDF5 file."""
        if saveto.endswith(".h5") or saveto.endswith(".hdf5"):
            import h5py  # noqa

            with h5py.File(saveto, "a") as h5file:
                self._check_append_state_hdf5(h5file[self.name])
                self._append_state_hdf5(h5file[self.name])

            for node in self.topological_ordering():
                node._append_state_cleanup()
        else:
            raise NotImplementedError(
                "Only HDF5 files ('.h5') are currently supported for saving state"
            )

    def _check_load_state_hdf5(self, h5group):
        """Check the state and HDF5 file have the same structure."""
        for key, child in self.children.items():
            if key in h5group:
                child._check_load_state_hdf5(h5group[key])
            else:
                raise GraphError(
                    f"Child '{child.name}' (identified by key '{key}') not found in HDF5 group '{h5group.name}'. Structure of graph changed from last save."
                )

    def _load_state_hdf5(self, h5group, index: int = -1):
        """Load the state of the node and its children from HDF5."""
        for key, child in self.children.items():
            child._load_state_hdf5(h5group[key], index=index)

    def load_state(self, loadfrom: str, index: int = -1):
        """Load the state of the node and its children."""
        if loadfrom.endswith(".h5") or loadfrom.endswith(".hdf5"):
            import h5py  # noqa

            with h5py.File(loadfrom, "r") as h5file:
                self._check_load_state_hdf5(h5file[self.name])
                self._load_state_hdf5(h5file[self.name], index=index)
        else:
            raise NotImplementedError(
                "Only HDF5 files ('.h5') are currently supported for loading state"
            )

    def graphviz(self, top_down=True) -> "graphviz.Digraph":
        """Return a graphviz object representing the graph below the current
        node in the DAG.

        Parameters
        ----------
        top_down: (bool, optional)
            Whether to draw the graph top-down (current node at top) or
            bottom-up (current node at bottom). Defaults to True.
        """
        import graphviz  # noqa

        components = set()

        def add_node(node, dot):
            if node in components:
                return
            dot.attr("node", **node.graphviz_types[node._type])
            dot.node(str(id(node)), repr(node))
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

    def __eq__(self, other: "Node") -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)

    def __setattr__(self, key: str, value: Any):
        """Intercept attribute setting to update parameters and graph links."""
        if isinstance(value, Node):
            # check for trying setting an attr with its own setter, allow the setter to handle throwing errors (e.g. value, and dynamic_value)
            if not hasattr(getattr(type(self), key, None), "fset"):
                self._link(key, value)

        super().__setattr__(key, value)

    def __delattr__(self, key: str):
        """Intercept attribute deletion to remove links."""
        if key in self.children:
            self._unlink(key)
        super().__delattr__(key)
