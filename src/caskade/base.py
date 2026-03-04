import os
from typing import Optional, Union, Any
from warnings import warn
from operator import attrgetter
import keyword

try:
    import h5py
except ImportError:
    h5py = None

from .errors import GraphError, NodeConfigurationError, LinkToAttributeError
from .warnings import SaveStateWarning

__all__ = ("Node",)


def attrsetter(obj, attr, value):
    """
    Set an attribute on an object, supporting nested dot-separated paths.

    If the value is the string ``"NONE"``, it is converted to ``None``.
    Dot-separated attribute paths (e.g. ``"a.b.c"``) are resolved
    recursively so that the final attribute is set on the correct object.

    Parameters
    ----------
    obj : object
        The target object on which to set the attribute.
    attr : str
        The attribute name or dot-separated path (e.g. ``"sub.attr"``).
    value : Any
        The value to assign. The string ``"NONE"`` is treated as ``None``.
    """
    if isinstance(value, str) and value == "NONE":
        value = None
    if "." in attr:
        parts = attr.split(".", 1)
        attrsetter(getattr(obj, parts[0]), parts[1], value)
    else:
        setattr(obj, attr, value)


def is_valid_name(name):
    """
    Check whether a string is a valid Python identifier and not a keyword.

    Parameters
    ----------
    name : str
        The candidate name to validate.

    Returns
    -------
    bool
        ``True`` if *name* is a valid Python identifier and is not a
        reserved keyword, ``False`` otherwise.
    """
    return name.isidentifier() and not keyword.iskeyword(name)


class meta:
    """
    Container for meta information attached to a ``Node`` object.

    Each ``Node`` instance carries a ``meta`` attribute that is an instance
    of this class.  Arbitrary attributes may be set on it to store
    auxiliary metadata without polluting the node's own namespace.
    """

    pass


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

    def __init__(
        self,
        name: Optional[str] = None,
        link: Optional[Union["Node", tuple["Node"]]] = None,
        description: str = "",
    ):
        """
        Initialise a new ``Node``.

        Parameters
        ----------
        name : str, optional
            Human-readable name for this node.  Must be a valid Python
            identifier and not a reserved keyword.  Defaults to the class
            name.
        link : Node or tuple of Node, optional
            One or more child nodes to link immediately after construction.
            Each child is linked using its ``name`` as the key.
        description : str, optional
            Free-form text describing the purpose of this node.
        """
        if name is None:
            name = self.__class__.__name__
        if not isinstance(name, str):
            raise NodeConfigurationError(f"{self.__class__.__name__} name must be a string")
        if not is_valid_name(name):
            raise NodeConfigurationError(
                f"{self.__class__.__name__} name is invalid: '{name}'. Must be a valid Python identifier and not a reserved keyword."
            )
        self._name = name
        self._children = {}
        self._parents = set()
        self._subgraphs = set()
        self._memos = set()
        self.node_type = "node"
        self.description = description
        self.meta = meta()
        self.saveattrs = set()
        if link is not None:
            self.link(link)

    @property
    def name(self) -> str:
        """str : The name of this node."""
        return self._name

    @property
    def children(self) -> dict[str, "Node"]:
        """dict[str, Node] : Mapping of link keys to child nodes."""
        return self._children

    @property
    def parents(self) -> set["Node"]:
        """set[Node] : Set of parent nodes that link to this node."""
        return self._parents

    @property
    def subgraphs(self) -> set["Node"]:
        """set[Node] : Subset of children linked hierarchically."""
        return self._subgraphs

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

        self.children[key] = child
        child.parents.add(self)
        self.update_graph()

    def link(
        self,
        key: Union[str, tuple, "Node"],
        child: Optional[Union["Node", tuple]] = None,
    ):
        """
        Link the current ``Node`` object to another ``Node`` object as a child.

        Parameters
        ----------
        key: (Union[str, Node])
            The key to link the child node with. This will also become the
            attribute to access the child node. After linking you will have
            `node.key == child`
        child: (Optional[Node], optional)
            The child ``Node`` object to link to. Defaults to None in which case
            the key is used as the child and the child.name is used as the key.

        Examples
        --------

        Example making some ``Node`` objects and then linking/unlinking them,
        demonstrating multiple ways to link/unlink::

            n1 = Node()
            n2 = Node()

            n1.link("subnode", n2)  # may use any str as the key
            n1.unlink("subnode")

            # Alternatively, link by object
            n1.link(n2)
            n1.unlink(n2)
        """
        if (
            isinstance(key, (tuple, list))
            and not isinstance(key, Node)
            and (child is None or not isinstance(child, Node))
        ):
            if child is None:
                for k in key:
                    self.link(k)
            else:
                for k, c in zip(key, child):
                    self.link(k, c)
            return
        if child is None:
            child = key
            key = child.name

        if not is_valid_name(key):
            raise NodeConfigurationError(
                f"key is invalid: '{key}'. Must be a valid Python identifier and not a reserved keyword."
            )
        self.__setattr__(key, child)

    def hierarchical_link(self, key: str, child: "Node"):
        """
        Link the current ``Node`` object to another ``Node`` object as a child
        in a hierarchical manner. See `link` for more detail on linking. A
        hierarchical link will allow batching internally to the simulator.

        Parameters
        ----------
        key : str
            The key to link the child node with.
        child : Node
            The child ``Node`` object to link to.

        Examples
        --------
        ::

            parent = Node(name="parent")
            child = Node(name="child")
            parent.hierarchical_link("child", child)
        """

        self._subgraphs.add(child)
        self.link(key, child)

    def _unlink(self, key: str):
        if self.active:
            raise GraphError(f"Cannot link/unlink nodes while the graph is active ({self.name})")
        self.children[key].parents.remove(self)
        self.children[key].update_graph()
        self._subgraphs.discard(self.children[key])
        del self.children[key]
        self.update_graph()

    def unlink(self, key: Union[str, "Node", list, tuple]):
        """
        Unlink one or more child nodes from this node.

        Parameters
        ----------
        key : str, Node, list, or tuple
            Identifier of the child(ren) to remove.  May be a link key
            string, the child ``Node`` object itself, or a list/tuple of
            keys or nodes to unlink in bulk.

        Raises
        ------
        GraphError
            If the graph is currently active.
        """
        if isinstance(key, Node):
            for node in self.children:
                if self.children[node] is key:
                    key = node
                    break
        elif isinstance(key, (tuple, list)):
            for k in key:
                self.unlink(k)
            return
        self.__delattr__(key)

    def topological_ordering(self) -> tuple["Node"]:
        """
        Return a topological ordering of the graph below the current node.

        Performs a recursive depth-first search with post-order traversal to
        resolve dependencies. The result starts with this node and proceeds
        to its descendants in dependency order.

        Returns
        -------
        tuple[Node]
            All nodes reachable from (and including) this node, ordered so
            that every parent appears before its children.
        """
        visited = set()
        stack = []

        def visit(node: Node):
            if node in visited:
                return
            visited.add(node)

            # Visit all children first
            for child in reversed(node.children.values()):
                visit(child)

            # Add node to stack only after all children are processed
            stack.append(node)

        visit(self)

        # Reverse the stack to get Parent -> Child ordering
        return tuple(reversed(stack))

    def update_graph(self):
        """Triggers a call to all parents that the graph below them has been
        updated. The base ``Node`` object does nothing with this information, but
        other node types may use this to update internal state."""
        for parent in self.parents:
            parent.update_graph()

    @property
    def active(self) -> bool:
        """bool : ``True`` if the node is currently in an active simulation run."""
        return any(memo.startswith("active") for memo in self._memos)

    @property
    def online(self) -> bool:
        """bool : ``True`` if the node is online within a hierarchical sub-graph."""
        return any(memo.endswith("_active") for memo in self._memos)

    @property
    def memos(self) -> set[str]:
        """set[str] : Current set of memo strings held by this node."""
        return self._memos

    def add_memo(self, memo):
        """
        Add a memo string and propagate it to all children.

        Parameters
        ----------
        memo : str
            The memo message to add.  Children in ``subgraphs`` receive
            the memo with the child name appended (``memo|child_name``).
        """
        self._memos.add(memo)

        # Propagate memo to children
        for child in self.children.values():
            child.add_memo(memo + (f"|{child.name}" if child in self.subgraphs else ""))

    def remove_memo(self, memo):
        """
        Remove a memo string and propagate removal to all children.

        Parameters
        ----------
        memo : str
            The memo message to remove.  The same propagation rules as
            ``add_memo`` apply.
        """
        self._memos.discard(memo)

        # Propagate removal to children
        for child in self.children.values():
            child.remove_memo(memo + (f"|{child.name}" if child in self.subgraphs else ""))

    def to(self, device=None, dtype=None):
        """
        Moves and/or casts the values of the ``Node`` to a particular device and/or dtype.

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

    def _save_state_hdf5(self, h5group, appendable: bool = False, _done_save: set = None):
        """Save the state of the node and its children to HDF5."""
        if id(self) not in _done_save:
            for attr in self.saveattrs:
                value = attrgetter(attr)(self)
                if value is None:
                    value = "NONE"
                try:
                    h5group.attrs[attr] = value
                except TypeError:
                    warn(
                        SaveStateWarning(
                            f"attribute '{attr}' of type {type(value)} cannot be saved to HDF5"
                        )
                    )
                except Exception as e:
                    warn(SaveStateWarning(f"Unable to save attribute '{attr}' due to: {e}"))
            _done_save.add(id(self))
        for key, child in self.children.items():
            if not hasattr(child, "_h5group"):
                child._h5group = h5group.create_group(key)
            elif key not in h5group:
                h5group[key] = child._h5group
            child._save_state_hdf5(h5group[key], appendable=appendable, _done_save=_done_save)

    def save_state(self, saveto: Union[str, "File"], appendable: bool = False):
        """
        Save the state of the node and its children, currently only works for
        HDF5 file types (.h5 and .hdf5).

        The "state" of a node is considered to be the value of its params,
        however it is also possible to save other attributes of the node by
        adding them to the `Node.saveattrs` set. Simply call
        `Node.saveattrs.add('attribute')` and then `Node.attribute` will be
        saved if possible. The HDF5 file will be created with the same structure
        as the graph, even if there are multiple paths to the same node. For
        example if N1 has children N2 and N3, and both N2 and N3 have the child
        N4, the HDF5 file will reflect this. It will be possible to find the N4
        params under both 'N1/N2/N4' and 'N1/N3/N4' if inspecting the HDF5 file
        manually. Specifically, if N4 has the param P1 then you could access its
        value like this:

        .. code-block:: python
            with h5py.File("myfile.h5", "r") as h5file:
                value = h5file["N1/N2/N4/P1/value"][()]
                # or
                value = h5file["N1/N3/N4/P1/value"][()]

        If the save had been set as appendable, then the value will have an
        extra dimension for the number of samples, this will always be the first
        dimension. If appendable was false then the value will simply equal the
        param value.

        Note
        ----
        You need the optional `h5py` package installed to use this method.

        Parameters
        ----------
        saveto: (Union[str, File])
            The file to save the state to. If a string, it should be the path to
            an HDF5 file (ending in '.h5' or '.hdf5'). If a File object, it
            should be an open HDF5 file.
        appendable: (bool, optional)
            Whether to save the state in an appendable format. If True, the
            values will have an extra dimension for the number of samples.
            Defaults to False.

        """
        if isinstance(saveto, str):
            if saveto.endswith(".h5") or saveto.endswith(".hdf5"):
                with h5py.File(saveto, "w") as h5file:
                    self._h5group = h5file.create_group(self.name)
                    self._save_state_hdf5(
                        h5file[self.name], appendable=appendable, _done_save=set()
                    )

                for node in self.topological_ordering():
                    del node._h5group
            else:
                raise NotImplementedError(
                    "Only HDF5 files ('.h5') are currently supported for saving state"
                )
        else:  # assume saveto is an HDF5 File object
            self._h5group = saveto.create_group(self.name)
            self._save_state_hdf5(saveto[self.name], appendable=appendable, _done_save=set())

            for node in self.topological_ordering():
                del node._h5group

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

    def append_state(self, saveto: Union[str, "File"]):
        """
        Append the current state to an existing HDF5 file.

        The file must have been previously created by ``save_state`` with
        ``appendable=True``.  The graph structure in the file is verified
        before appending.

        Parameters
        ----------
        saveto : str or File
            Path to an HDF5 file (``'.h5'`` or ``'.hdf5'``) or an open
            HDF5 ``File`` object.

        Raises
        ------
        GraphError
            If the graph structure no longer matches the file.
        NotImplementedError
            If the file path does not end with a supported extension.
        """
        if isinstance(saveto, str):
            if saveto.endswith(".h5") or saveto.endswith(".hdf5"):
                with h5py.File(saveto, "a") as h5file:
                    self._check_append_state_hdf5(h5file[self.name])
                    self._append_state_hdf5(h5file[self.name])

                for node in self.topological_ordering():
                    node._append_state_cleanup()
            else:
                raise NotImplementedError(
                    "Only HDF5 files ('.h5') are currently supported for saving state"
                )
        else:  # assume saveto is an HDF5 File object
            self._check_append_state_hdf5(saveto[self.name])
            self._append_state_hdf5(saveto[self.name])

            for node in self.topological_ordering():
                node._append_state_cleanup()

    def _check_load_state_hdf5(self, h5group):
        """Check the state and HDF5 file have the same structure."""
        for key, child in self.children.items():
            if key in h5group:
                child._check_load_state_hdf5(h5group[key])
            else:
                raise GraphError(
                    f"Child '{child.name}' (identified by key '{key}') not found in HDF5 group '{h5group.name}'. Structure of graph changed from last save."
                )

    def _load_state_hdf5(self, h5group, index: int = -1, _done_load: set = None):
        """Load the state of the node and its children from HDF5."""

        if id(self) not in _done_load:
            for attr in h5group.attrs:
                attrsetter(self, attr, h5group.attrs[attr])
            _done_load.add(id(self))
        for key, child in self.children.items():
            child._load_state_hdf5(h5group[key], index=index, _done_load=_done_load)

    def load_state(self, loadfrom: Union[str, "File"], index: int = -1, **kwargs):
        """
        Load node state (and children) from an HDF5 file.

        Parameters
        ----------
        loadfrom : str or File
            Path to an HDF5 file (``'.h5'`` or ``'.hdf5'``) or an open
            HDF5 ``File`` object.
        index : int, optional
            Sample index to load when the file was saved in appendable
            mode.  Defaults to ``-1`` (last sample).
        **kwargs
            Additional keyword arguments forwarded to ``h5py.File``
            (e.g. ``driver``).

        Raises
        ------
        GraphError
            If the graph structure no longer matches the file.
        NotImplementedError
            If the file path does not end with a supported extension.
        """
        if isinstance(loadfrom, str):
            if loadfrom.endswith(".h5") or loadfrom.endswith(".hdf5"):
                with h5py.File(loadfrom, "r", **{"driver": "core", **kwargs}) as h5file:
                    self._check_load_state_hdf5(h5file[self.name])
                    self._load_state_hdf5(h5file[self.name], index=index, _done_load=set())
            else:
                raise NotImplementedError(
                    "Only HDF5 files ('.h5') are currently supported for loading state"
                )
        else:  # assume loadfrom is an HDF5 File object
            self._check_load_state_hdf5(loadfrom[self.name])
            self._load_state_hdf5(loadfrom[self.name], index=index, _done_load=set())

    @property
    def graphviz_style(self):
        return {"style": "solid", "color": "black", "shape": "circle"}

    def graphviz(self, saveto: Optional[str] = None) -> "graphviz.Digraph":
        """
        Return a graphviz ``Digraph`` representing the DAG below this node.

        Parameters
        ----------
        saveto : str, optional
            If provided, save the rendered graph to this file path.  The
            file extension determines the output format (e.g. ``'.pdf'``,
            ``'.png'``).  Defaults to ``None``.

        Returns
        -------
        graphviz.Digraph
            The constructed directed-graph object.
        """
        import graphviz  # noqa

        components = set()

        def add_node(node: Node, dot):
            if node in components or node.online:
                return
            dot.attr("node", **node.graphviz_style)
            dot.node(str(id(node)), repr(node))
            components.add(node)

            for child in node.children.values():
                if child in node.subgraphs:
                    with Memo(self, "semi_active"):
                        with dot.subgraph(name=f"cluster_{id(child)}") as subdot:
                            add_node(child, subdot)
                else:
                    add_node(child, dot)

        def add_edges(node: Node, dot):
            for child in node.children.values():
                dot.edge(str(id(node)), str(id(child)))
                add_edges(child, dot)

        dot = graphviz.Digraph(strict=True)
        add_node(self, dot)
        add_edges(self, dot)
        if saveto is not None:
            filename, ext = os.path.splitext(saveto)
            dot.render(graphviz.escape(filename), format=ext.lstrip("."), cleanup=True)
        return dot

    @property
    def node_str(self):
        return f"{self.name}|{self.node_type}"

    def graph_dict(self) -> dict[str, dict]:
        """
        Return a nested dictionary representation of the graph.

        Each key is a string of the form ``"name|node_type"`` and the value
        is a dict containing the same structure for that node's children.

        Returns
        -------
        dict[str, dict]
            Nested dictionary mirroring the DAG hierarchy.
        """
        rep = self.node_str
        graph = {
            rep: {},
        }
        for node in self.children.values():
            graph[rep].update(node.graph_dict())
        return graph

    def graph_print(self, dag: dict, depth: int = 0, indent: int = 4, result: str = "") -> str:
        """
        Recursively render a graph dictionary as an indented string.

        Parameters
        ----------
        dag : dict[str, dict]
            A nested dictionary as returned by ``graph_dict``.
        depth : int, optional
            Current indentation depth (used during recursion).  Defaults
            to ``0``.
        indent : int, optional
            Number of spaces per indentation level.  Defaults to ``4``.
        result : str, optional
            Accumulator string (used during recursion).  Defaults to
            ``""``.

        Returns
        -------
        str
            A human-readable, indented representation of the graph.
        """
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
            # check for trying setting an attr with its own setter, allow the setter to handle throwing errors (e.g. value)
            if not hasattr(getattr(type(self), key, None), "fset"):
                self._link(key, value)

        super().__setattr__(key, value)

    def __delattr__(self, key: str):
        """Intercept attribute deletion to remove links."""
        if key in self.children:
            self._unlink(key)
        super().__delattr__(key)


class Memo:
    """
    Sends a "memo" (a small message) to all nodes below the current one in the
    graph. This can be used to communicate state changes in the graph with all
    lower nodes. By default, the message will skip any subgraphs (hierarchical
    graphs) but this can be changed to ensure all nodes hear the message.

    Note that memos are stored as a python set, so duplicates will be merged.
    Depending on your use case, it may be wise to ensure that your memo is
    unique.

    Parameters
    ----------
    module: Module
        The caskade Module object that will propogate the memo
    memo: str
        The message to send down the graph
    skip_subgraphs: bool
        If True (default) any subgraphs, otherwise known as hierarchical graphs,
        will not get the memo.
    """

    def __init__(self, module: Node, memo: str):
        self.module = module
        self.memo = memo

    def __enter__(self):
        self.module.add_memo(self.memo)

    def __exit__(self, exc_type, exc_value, traceback):
        self.module.remove_memo(self.memo)
