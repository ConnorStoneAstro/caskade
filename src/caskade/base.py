import os
from typing import Optional, Union, Any
from warnings import warn
from operator import attrgetter

try:
    import h5py
except ImportError:
    h5py = None

from .backend import backend
from .errors import GraphError, NodeConfigurationError, LinkToAttributeError, BackendError
from .warnings import SaveStateWarning


def attrsetter(obj, attr, value):
    """Set an attribute on an object."""
    if "." in attr:
        parts = attr.split(".", 1)
        attrsetter(getattr(obj, parts[0]), parts[1], value)
    else:
        setattr(obj, attr, value)


class meta:
    """Meta information for a ``Node`` object."""

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

    graphviz_types = {"node": {"style": "solid", "color": "black", "shape": "circle"}}

    def __init__(
        self,
        name: Optional[str] = None,
        link: Optional[Union["Node", tuple["Node"]]] = None,
    ):
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
        self.meta = meta()
        self.saveattrs = set()
        if link is not None:
            self.link(link)

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

    def link(self, key: Union[str, tuple, "Node"], child: Optional[Union["Node", tuple]] = None):
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
        self.__setattr__(key, child)

    def _unlink(self, key: str):
        if self.active:
            raise GraphError(f"Cannot link/unlink nodes while the graph is active ({self.name})")
        self._children[key]._parents.remove(self)
        self._children[key].update_graph()
        del self._children[key]
        self.update_graph()

    def unlink(self, key: Union[str, "Node", list, tuple]):
        """Unlink the current ``Node`` object from another ``Node`` object which is a child."""
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
        if appendable and backend.backend == "object":
            raise BackendError("Cannot make appendable HDF5 files with the 'object' backend")

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
        """Append the state of the node and its children to an existing HDF5 file."""
        if backend.backend == "object":
            raise BackendError("Cannot append to HDF5 files with the 'object' backend")

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
        """Load the state of the node and its children."""
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

    def graphviz(self, top_down: bool = True, saveto: Optional[str] = None) -> "graphviz.Digraph":
        """Return a graphviz object representing the graph below the current
        node in the DAG.

        Parameters
        ----------
        top_down: (bool, optional)
            Whether to draw the graph top-down (current node at top) or
            bottom-up (current node at bottom). Defaults to True.
        saveto: (Optional[str], optional)
            If provided, save the graph to this file. The file extension
            determines the format (e.g. '.pdf', '.png'). Defaults to None.
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
        if saveto is not None:
            filename, ext = os.path.splitext(saveto)
            dot.render(graphviz.escape(filename), format=ext.lstrip("."), cleanup=True)
        return dot

    @property
    def node_str(self):
        return f"{self.name}|{self._type}"

    def graph_dict(self) -> dict[str, dict]:
        """Return a dictionary representation of the graph below the current
        node."""
        rep = self.node_str
        graph = {
            rep: {},
        }
        for node in self.children.values():
            graph[rep].update(node.graph_dict())
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
