import os
import sys
import enum
import json
from contextlib import contextmanager
from math import prod
from typing import TYPE_CHECKING, NamedTuple, Optional, Union, Any
from warnings import warn
from operator import attrgetter
from threading import get_ident
from types import FrameType
import keyword
import uuid

try:
    import h5py
except ImportError:  # h5py is an optional dependency
    h5py = None

if TYPE_CHECKING:
    import graphviz

import numpy as np

from .backend import backend
from .errors import GraphError, NodeConfigurationError, LinkToAttributeError
from .warnings import LinkToAttributeWarning, SaveStateWarning

__all__ = ("Node", "NodeType", "Memo", "MemoKind", "MemoNote")


#: The nodes that a call to ``Node.to`` in progress has reached already, so that
#: a node which many paths lead to is moved once. Held for each thread that is
#: making such a call (the way ``reprlib.recursive_repr`` keeps track).
_MOVED: dict[int, set["Node"]] = {}


_MISSING = object()


def _user_stacklevel() -> int:
    """The ``stacklevel`` for ``warn`` that points a warning at the line outside
    ``caskade`` which led to it, however many calls inside ``caskade`` lie
    between that line and the function that warns."""
    level = 2  # the caller of the function that warns
    frame: Optional[FrameType] = sys._getframe(2)
    while frame is not None and frame.f_globals.get("__name__", "").split(".")[0] == "caskade":
        frame = frame.f_back
        level += 1
    return level


def _require_h5py():
    """Raise a helpful error if the optional ``h5py`` dependency is missing."""
    if h5py is None:
        raise ImportError(
            "Saving and loading the state of a caskade graph requires the optional "
            "'h5py' package. Install it with `pip install h5py`."
        )


# Names that ``caskade`` keeps for itself among the attributes of the group of
# the node that was saved
_H5_FORMAT = "caskade_format"  # 2 since the count of samples is kept, files without it are 1
_H5_LAYOUT = "caskade_layout"  # "graph" (a group for every node) or "fast" (two arrays)
_H5_APPENDABLE = "caskade_appendable"
_H5_SAMPLES = "caskade_samples"  # number of states an appendable file holds
_H5_RESERVED = (_H5_FORMAT, _H5_LAYOUT, _H5_APPENDABLE, _H5_SAMPLES)
# A file up to this size is read into memory in one go when it is loaded, a
# larger one (a long chain) is read piece by piece
_H5_CORE_DRIVER_LIMIT = 2**28


@contextmanager
def _open_hdf5(target, mode: str, action: str, **kwargs):
    """The HDF5 file (or group) to work in. A path is opened here and closed
    again afterwards, anything else is taken to be an open file or group and is
    left open for its owner."""
    if not isinstance(target, (str, os.PathLike)):
        yield target
        return
    path = os.fspath(target)
    if not path.endswith((".h5", ".hdf5")):
        raise NotImplementedError(
            f"Only HDF5 files ('.h5') are currently supported for {action} state"
        )
    _require_h5py()
    with h5py.File(path, mode, **kwargs) as h5file:
        yield h5file


def _create_rows(h5group, name: str, value, samples: int = 1):
    """Create the dataset for a value that more states will be appended to.
    The first dimension counts the states, ``value`` becomes the last of
    ``samples`` rows and any rows before it are left blank (NaN for floats)."""
    shape = tuple(value.shape)
    dataset = h5group.create_dataset(
        name,
        shape=(samples,) + shape,
        dtype=value.dtype,
        maxshape=(None,) + shape,
        chunks=True,
        compression="gzip",
        fillvalue=float("nan") if value.dtype.kind in "fc" else None,
    )
    dataset[samples - 1] = value
    return dataset


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


class NodeType(enum.StrEnum):
    """The kind of a node, as shown in printouts of the graph and saved to file.

    The members are strings, so comparing with or formatting as text works
    as it would for the plain string: ``node.node_type == "module"`` is
    ``True`` for a ``Module`` and ``f"{node.node_type}"`` gives ``"module"``.
    """

    NODE = "node"
    MODULE = "module"
    STATIC = "static"  #: a ``Param`` with a fixed value
    DYNAMIC = "dynamic"  #: a ``Param`` whose value is given when the simulation is called
    POINTER = "pointer"  #: a ``Param`` whose value is worked out from other nodes
    NTUPLE = "ntuple"
    NLIST = "nlist"
    NDICT = "ndict"


class MemoKind(enum.IntEnum):
    """What a memo tells the nodes that hold it."""

    RUN = 1  #: a simulation is running from the owner (an ``ActiveContext`` is open)
    USER = 2  #: a message from user code, ``caskade`` never acts on these


class MemoNote(NamedTuple):
    """A memo as it is held by one of the nodes that received it."""

    kind: MemoKind
    owner: "Node"  #: the node the memo was sent from
    depth: int = 0  #: fewest hierarchical links on any path from the owner to the holder
    label: str = ""  #: the text of a ``USER`` memo


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
        link: Optional[Union["Node", tuple["Node", ...]]] = None,
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
        self._unique_id = f"{name}_{uuid.uuid4().hex}"
        self._children: dict[str, Node] = {}
        self._parents: set[Node] = set()
        self._subgraphs: set[Node] = set()
        # Names that held a child until a plain value was assigned over it
        self._freed: set[str] = set()
        self._memos: dict[MemoNote, int] = {}  # each note held, and how many times
        self._hierarchy_depths_cache: Optional[dict[Node, int]] = None
        self._run_notes_cache: Optional[list[tuple[Node, MemoNote]]] = None
        self.node_type: str = NodeType.NODE
        self.description = description
        self.meta = meta()
        self.saveattrs: set[str] = set()
        if link is not None:
            self.link(link)

    @property
    def name(self) -> str:
        """str : The name of this node."""
        return self._name

    @property
    def unique_id(self) -> str:
        """str : An id for this node that no other node has, its name followed
        by a random string (``uuid4``).

        The id is made when the node is created and cannot be changed. It goes
        with the node when it is pickled or copied, so a copy of a node has the
        id of the node it was copied from. ``caskade`` makes no use of it.

        A graph that is built again in code has new ids, and ``load_state``
        leaves them alone. To have the id written to an HDF5 file and taken
        back on by ``load_state``, add it to the saved attributes of the node,
        ``node.saveattrs.add("unique_id")`` (the fast layout stores values
        only).
        """
        return self._unique_id

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

    def _link(self, key: str, child: "Node", replace: bool = False):
        """Register ``child`` in the graph under ``key``.

        Nothing is changed unless every check passes. With ``replace=True`` a
        different node already linked under ``key`` is unlinked first, otherwise
        that is an error. A name that is in use for something that is not a
        child is taken over with a ``LinkToAttributeWarning``, except for the
        name of a property, which is an error. A name that held a child before
        (and was given a plain value since) is taken back without a word.
        """
        if self.active:
            raise GraphError("Cannot link/unlink nodes while the graph is active")
        existing = self.children.get(key)
        # Avoid double linking to the same object
        if existing is child:
            return
        if existing is not None and not replace:
            raise GraphError(
                f"Child key '{key}' already linked to parent {self.name}, but with different node {existing.name}"
            )
        if child in self.children.values():
            raise GraphError(
                f"Child {child.name} already linked to parent {self.name}, but not with key '{key}'"
            )
        # The name may be in use for something that is not a child
        replaced = _MISSING
        if existing is None and key not in self._freed:
            on_class = getattr(type(self), key, _MISSING)
            if hasattr(on_class, "__set__"):
                # A property decides what its name gives, a child cannot take its place
                raise LinkToAttributeError(
                    f"Child key '{key}' is a property of {type(self).__name__} "
                    f"('{self.name}'), use a different name for the child"
                )
            replaced = self.__dict__.get(key, on_class)

        # avoid cycles
        if self in child.topological_ordering():
            raise GraphError(
                f"Linking {child.name} to {self.name} would create a cycle in the graph!"
            )

        if replaced is not _MISSING:
            # Allowed, as any assignment to an attribute is, but it may be a mistake
            was = "None" if replaced is None else f"a {type(replaced).__name__}"
            warn(
                LinkToAttributeWarning(
                    f"Linking node '{child.name}' as '{key}' replaces the attribute '{key}' "
                    f"of '{self.name}' (it was {was}). If that is not what you meant, use a "
                    "different name for the child."
                ),
                stacklevel=_user_stacklevel(),
            )
        if existing is not None:
            self._unlink(key)
        self._freed.discard(key)
        self.children[key] = child
        child.parents.add(self)
        self.update_graph()

    def link(
        self,
        key: Union[str, tuple, list, "Node"],
        child: Optional[Union["Node", tuple, list]] = None,
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
            if not isinstance(key, Node):
                raise TypeError(f"child must be a Node object, not {type(key)}")
            child = key
            key = child.name

        if not isinstance(key, str) or not is_valid_name(key):
            raise NodeConfigurationError(
                f"key is invalid: '{key}'. Must be a valid Python identifier and not a reserved keyword."
            )
        if not isinstance(child, Node):
            raise TypeError(f"child must be a Node object, not {type(child)}")
        # An explicit link never replaces an existing child, only assignment does
        self._link(key, child)
        super().__setattr__(key, child)

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

        Notes
        -----
        A hierarchical link says that this node calls the child many times in
        one simulation (usually with ``vmap``), so params below the link carry
        a batch dimension for those calls. For a param whose shallowest path
        from a module ``M`` crosses ``d`` hierarchical links, ``M`` reads the
        param's ``batch_shape`` as::

            (batch of M ..., link 1, ..., link d)

        with the link nearest to ``M`` first. The last ``d`` dimensions are part
        of one simulation of ``M``, so they are flattened into the array that
        ``M.get_values()`` returns and that its ``@forward`` methods take. Any
        dimensions before them are a batch of whole simulations of ``M``. The
        split is always made from the module the array belongs to, so the same
        param is read with one fewer link dimension by the module on the other
        side of each link. An array inside a dictionary of params is a piece of
        the array of the module the dictionary was given to.

        A param below the link with no batch dimensions has one value that is
        shared by every call. A param that does have batch dimensions needs at
        least ``d`` of them, with fewer the modules along the path would not
        agree on which link they are for and a ``ParamConfigurationError`` is
        raised.

        Examples
        --------
        ::

            parent = Node(name="parent")
            child = Node(name="child")
            parent.hierarchical_link("child", child)
        """

        self.link(key, child)
        # Only once the link has succeeded is the child recorded as a subgraph
        self._subgraphs.add(child)
        self.update_graph()

    def _unlink(self, key: str):
        if self.active:
            raise GraphError(f"Cannot link/unlink nodes while the graph is active ({self.name})")
        self.children[key].parents.remove(self)
        self.children[key].update_graph()
        self._subgraphs.discard(self.children[key])
        del self.children[key]
        self.update_graph()

    def unlink(self, key: Union[str, "Node", list, tuple, None] = None):
        """Unlink one or more ``Node`` objects from this ``Node``.

        Parameters
        ----------
        key: (str, Node, list, tuple, or None, optional)
            The key, ``Node`` object, or collection of keys/nodes to unlink.
            If a string, the child with that key is unlinked. If a ``Node``
            object, the matching child is located and unlinked. If a list or
            tuple, each element is unlinked in turn. If ``None`` (the
            default), all children are unlinked.

        Raises
        ------
        GraphError
            If the graph is currently active.
        """
        if key is None:
            self.unlink(list(self.children))
            return
        if isinstance(key, Node):
            for node in self.children:
                if self.children[node] is key:
                    key = node
                    break
            else:
                raise KeyError(f"Node {key.name} not found in parent {self.name}")
        elif isinstance(key, (tuple, list)):
            for k in key:
                self.unlink(k)
            return
        if key not in self.children:
            raise KeyError(f"Child key '{key}' not found in parent {self.name}")
        self.__delattr__(key)

    def topological_ordering(self) -> tuple["Node", ...]:
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
        """Signal this node and every node above it that the graph below them
        has been updated. Each node is notified exactly once, however many paths
        lead to it. Node types that keep state derived from the graph (such as
        ``Module``) refresh it by overriding ``_graph_updated``."""
        seen = {id(self)}
        upstream = [self]
        for node in upstream:  # ``upstream`` grows while it is being iterated
            for parent in node._parents:
                if id(parent) not in seen:
                    seen.add(id(parent))
                    upstream.append(parent)
        for node in upstream:
            node._hierarchy_depths_cache = None
            node._run_notes_cache = None
            node._graph_updated()

    def _graph_updated(self):
        """Called once on a node whenever the graph at or below it has changed.
        The base ``Node`` does nothing with this information."""

    @property
    def active(self) -> bool:
        """bool : ``True`` while a simulation is running from this node or from
        any node above it."""
        return any(note.kind is MemoKind.RUN for note in self._memos)

    @property
    def online(self) -> bool:
        """bool : ``True`` while a simulation is running from this node, or from
        a node above it that reaches this one without crossing a hierarchical
        link. The values held below an online node are those of the simulation
        that is running, so its ``@forward`` methods may be called directly."""
        return any(note.kind is MemoKind.RUN and note.depth == 0 for note in self._memos)

    @property
    def memos(self) -> set[MemoNote]:
        """set[MemoNote] : The memos currently held by this node."""
        return set(self._memos)

    def _hierarchy_depths(self) -> dict["Node", int]:
        """Every node at or below this one, with its hierarchical depth: the
        fewest hierarchical links crossed on any path from this node to it.

        A node used at several levels of a hierarchy belongs to the shallowest
        of them, since that is the outermost level at which its values are
        needed. This is the one place where hierarchical structure is worked
        out. Both the memos sent from this node and the handling of batch
        dimensions (see ``hierarchical_link``) read it from here.

        The depths are worked out the first time they are needed and then kept
        until the graph below this node changes, so sending a memo while a
        simulation runs is a single pass over them.
        """
        depths = self._hierarchy_depths_cache
        if depths is None:
            depths = {self: 0}
            stack = [self]
            while stack:
                node = stack.pop()
                depth = depths[node]
                subgraphs = node._subgraphs
                for child in node._children.values():
                    child_depth = depth + 1 if child in subgraphs else depth
                    # a node is looked at again only when a shallower path to it is found
                    if child_depth < depths.get(child, child_depth + 1):
                        depths[child] = child_depth
                        stack.append(child)
            self._hierarchy_depths_cache = depths
        return depths

    def _notes(self, memo: Union[MemoKind, str]) -> list[tuple["Node", MemoNote]]:
        """Every node that receives ``memo`` when it is sent from this node,
        paired with the note that node holds for it."""
        if memo is MemoKind.RUN:
            # sent on every call of a simulation, so made once per change to the graph
            notes = self._run_notes_cache
            if notes is None:
                notes = self._run_notes_cache = [
                    (node, MemoNote(MemoKind.RUN, self, depth))
                    for node, depth in self._hierarchy_depths().items()
                ]
            return notes
        if not isinstance(memo, str):
            raise TypeError(f"A memo must be a string or MemoKind.RUN, not {type(memo)}")
        return [
            (node, MemoNote(MemoKind.USER, self, depth, memo))
            for node, depth in self._hierarchy_depths().items()
        ]

    def add_memo(self, memo: Union[MemoKind, str]):
        """
        Send a memo to this node and to every node below it.

        Parameters
        ----------
        memo : str or MemoKind
            A string is a message for user code, it is held by each node as a
            ``MemoNote`` of kind ``MemoKind.USER`` with the string as its label
            and ``caskade`` never acts on it. ``MemoKind.RUN`` marks the nodes
            as taking part in a simulation run from this node, it is sent by
            ``ActiveContext``.
        """
        for node, note in self._notes(memo):
            memos = node._memos
            memos[note] = memos.get(note, 0) + 1

    def remove_memo(self, memo: Union[MemoKind, str]):
        """
        Take back one memo that was sent with ``add_memo``.

        Parameters
        ----------
        memo : str or MemoKind
            The same value that was given to ``add_memo``. A memo sent several
            times is held until it has been removed as many times.
        """
        for node, note in self._notes(memo):
            node._forget_memo(note)

    def _forget_memo(self, note: MemoNote):
        """Drop a single memo from this node only. Node types that hold state
        tied to a memo (such as ``Module``) clean it up by overriding this."""
        held = self._memos.get(note, 0)
        if held > 1:
            self._memos[note] = held - 1
        elif held:
            del self._memos[note]

    def to(self, device=None, dtype=None):
        """
        Moves and/or casts the values of the ``Node`` to a particular device and/or dtype.

        Parameters
        ----------
        device: (optional)
            The device to move the values to. Defaults to None.
        dtype: (optional)
            The desired data type. Defaults to None.

        Notes
        -----
        Every node below this one is moved once, however many paths lead to
        it. A subclass that holds arrays of its own overrides this method,
        calls ``super().to(device, dtype)`` and then moves its arrays. Such a
        method is called once as well, with the arguments that the node above
        it passed down.
        """
        thread = get_ident()
        moved = _MOVED.get(thread)
        first = moved is None  # this is the call that starts the move
        if moved is None:
            moved = _MOVED[thread] = {self}
        try:
            for child in self.children.values():
                if child not in moved:
                    moved.add(child)
                    child.to(device=device, dtype=dtype)
        finally:
            if first:
                del _MOVED[thread]
        return self

    # Saving and loading
    #################################################################
    #: Names of what a node of this type writes into its own group of an HDF5
    #: file, next to the groups of its children
    _hdf5_own: tuple[str, ...] = ()

    def _hdf5_top(self, container):
        """The group in ``container`` that this node was saved to."""
        try:
            return container[self.name]
        except KeyError:
            raise GraphError(
                f"No saved state for a node named '{self.name}' was found, the file holds "
                f"{sorted(container.keys())}. The state is stored under the name of the node "
                "that was saved, so it must be loaded from a node with the same name."
            ) from None

    def _hdf5_groups(self, top, create: bool = False) -> list[tuple["Node", Any]]:
        """Pair every node at or below this one with its group in an HDF5 file.

        The file has the structure of the graph, a group for each node under
        the key that its parent holds it by. A node that is reached along
        several paths has a single group, which the file shows at each of
        those paths. Every node is visited once here, however many paths lead
        to it.

        Parameters
        ----------
        top
            The group of this node.
        create : bool
            If ``True`` the groups are made. Otherwise they are looked up, and
            the file is checked against the graph.

        Raises
        ------
        GraphError
            If ``create`` is ``False`` and the file has no group for a child.
        """
        groups = {self: top}
        pending = [self]
        extra: list[str] = []
        for node in pending:  # ``pending`` grows while it is being iterated
            group = groups[node]
            children = node.children
            for key, child in children.items():
                known = groups.get(child)
                if create:
                    if known is None:
                        groups[child] = group.create_group(key)
                        pending.append(child)
                    else:
                        group[key] = known  # the same group, shown at this path as well
                    continue
                if known is None:
                    found = group.get(key)
                    if found is not None:
                        groups[child] = found
                        pending.append(child)
                        continue
                elif key in group:
                    continue
                raise GraphError(
                    f"Child '{child.name}' (identified by key '{key}') not found in HDF5 "
                    f"group '{group.name}'. Structure of graph changed from last save."
                )
            if not create and len(group) != len(children) + len(node._hdf5_own):
                names = set(group.keys()).difference(children, node._hdf5_own)
                extra.extend(f"{group.name}/{name}" for name in sorted(names))
        if extra:
            warn(
                SaveStateWarning(
                    f"The file holds {len(extra)} entries that the graph of '{self.name}' "
                    f"does not have, they are ignored: {', '.join(extra[:5])}"
                    + (", ..." if len(extra) > 5 else "")
                )
            )
        return list(groups.items())

    def _save_hdf5(self, h5group, appendable: bool):
        """Write this node (not its children) into its group."""
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

    def _append_hdf5(self, h5group):
        """Get ready to append the current state of this node (not its children)
        to its group. Everything that could fail is done here, before any node
        has written. Returns the function that does the writing, it is given the
        number of states the file will then hold (``None`` for a file that does
        not keep count). Returns ``None`` if there is nothing to write."""
        return None

    def _read_hdf5(self, h5group, index: int):
        """Read the state of this node (not its children) from its group and
        check that it fits. Returns the function that takes the state on, or
        ``None``. Nothing about the node is changed before that is called, so a
        file that does not fit the graph leaves the whole graph as it was."""
        if not self.saveattrs:
            return None
        stored = h5group.attrs
        values = {attr: stored[attr] for attr in self.saveattrs if attr in stored}

        def take_on():
            for attr, value in values.items():
                if attr == "unique_id":  # read only, apart from here
                    attr = "_unique_id"
                attrsetter(self, attr, value)

        return take_on

    def _fast_params(self) -> dict[str, Any]:
        """Every dynamic and static param at or below this node by its path in
        the graph, the first path if there are several. (They are ``Param``
        objects, a class that is defined on top of this one.)"""
        paths = {self: self.name}
        params: dict[str, Any] = {}
        for node in self.topological_ordering():  # parents come before their children
            path = paths[node]
            if node.node_type is NodeType.DYNAMIC or node.node_type is NodeType.STATIC:
                params[path] = node
            for key, child in node.children.items():
                paths.setdefault(child, f"{path}/{key}")
        return params

    def _fast_state(self) -> tuple[dict[str, list], dict[str, np.ndarray]]:
        """The state for the fast layout. For each of ``"dynamic"`` and
        ``"static"`` a 1D array with the values of those params one after the
        other, and an index with ``[path, shape, dtype]`` for each of the
        params so that the array can be taken apart again. A param with no
        value has ``None`` for its shape and takes no room in the array."""
        index: dict[str, list] = {"dynamic": [], "static": []}
        parts: dict[str, list] = {"dynamic": [], "static": []}
        for path, param in self._fast_params().items():
            kind = param.node_type.value
            value = param.value
            if value is None:
                index[kind].append([path, None, None])
                continue
            value = backend.to_numpy(value)
            index[kind].append([path, list(value.shape), value.dtype.str])
            parts[kind].append(value.reshape(-1))
        arrays = {kind: np.concatenate(p) if p else np.zeros(0) for kind, p in parts.items()}
        return index, arrays

    def _save_fast(self, top, appendable: bool):
        index, arrays = self._fast_state()
        top.create_dataset("index", data=json.dumps(index))
        for kind, array in arrays.items():
            if array.size == 0:
                continue
            if appendable:
                _create_rows(top, kind, array)
            else:
                top.create_dataset(kind, data=array)

    def _append_fast(self, top):
        if not top.attrs[_H5_APPENDABLE]:
            raise IOError(
                f"{self.name} is not appendable. Need to save the HDF5 file with `appendable=True`."
            )
        index, arrays = self._fast_state()
        if index != json.loads(top["index"][()]):
            raise GraphError(
                f"The params of '{self.name}' are not the ones the file was saved with (a "
                "param was added, removed, changed type or shape, or gained or lost its "
                "value). A file with the fast layout only takes states of the same layout."
            )
        samples = int(top.attrs[_H5_SAMPLES]) + 1
        for kind, array in arrays.items():
            if array.size:
                top[kind].resize(samples, axis=0)
                top[kind][samples - 1] = array
        top.attrs[_H5_SAMPLES] = samples

    def _read_fast(self, top, index: int) -> list:
        stored = json.loads(top["index"][()])
        appendable = bool(top.attrs[_H5_APPENDABLE])
        params = self._fast_params()
        missing = set(params)
        plan = []
        for kind in ("dynamic", "static"):
            array = None
            pos = 0
            for path, shape, dtype in stored[kind]:
                param = params.get(path)
                if param is None:
                    pointer = self._fast_pointer(path)
                    if pointer:  # now worked out from other params, as in the graph layout
                        if shape is not None:
                            pos += prod(shape)
                        continue
                    raise GraphError(
                        f"The file holds a value for '{path}', which is not a param of "
                        f"'{self.name}'. Structure of graph changed from last save."
                    )
                missing.discard(path)
                value = None
                if shape is not None:
                    if array is None:
                        array = top[kind][index] if appendable else top[kind][()]
                    size = prod(shape)
                    value = array[pos : pos + size].reshape(shape).astype(dtype, copy=False)
                    pos += size
                plan.append(param._plan_value(kind, value, path))
        if missing:
            raise GraphError(
                f"The file holds no value for {sorted(missing)[:5]}"
                + (f" and {len(missing) - 5} more params" if len(missing) > 5 else "")
                + f" of '{self.name}'. Structure of graph changed from last save."
            )
        return plan

    def _fast_pointer(self, path: str) -> bool:
        """Whether ``path`` leads to a pointer param in the graph of this node."""
        node: Optional[Node] = self
        for key in path.split("/")[1:]:  # the first is the name of this node
            if node is None:
                return False
            node = node.children.get(key)
        return node is not None and node.node_type is NodeType.POINTER

    def save_state(
        self, saveto: Union[str, "h5py.File"], appendable: bool = False, fast: bool = False
    ):
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

        **The fast layout.** A file with the structure of the graph has a group
        and a dataset for every param, and HDF5 takes about a millisecond and
        a few kilobytes for each of those however small the values are. For a
        graph with many params, or a chain with many samples, ``fast=True``
        writes the values of all dynamic params into one array and those of
        all static params into another, each param flattened and placed one
        after the other in the order of the graph:

        .. code-block:: python
            with h5py.File("myfile.h5", "r") as h5file:
                chain = h5file["N1/dynamic"][()]  # (samples, number of values) if appendable
                index = json.loads(h5file["N1/index"][()])  # path, shape and dtype of each param

        For a graph with no batch dimensions a row of ``dynamic`` is what
        ``get_values()`` returns. This layout holds the values and whether each
        param is dynamic or static, and nothing else: pointer params are worked
        out from the others so they are left out, and ``saveattrs``, ``units``
        and ``valid`` ranges are not stored. ``load_state`` and
        ``append_state`` recognise the layout by themselves.

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
        fast: (bool, optional)
            Whether to use the fast layout described above in place of the
            structure of the graph. Defaults to False.

        """
        with _open_hdf5(saveto, "w", "saving") as h5file:
            top = h5file.create_group(self.name)
            top.attrs[_H5_FORMAT] = 2
            top.attrs[_H5_LAYOUT] = "fast" if fast else "graph"
            top.attrs[_H5_APPENDABLE] = appendable
            top.attrs[_H5_SAMPLES] = 1
            if fast:
                self._save_fast(top, appendable)
                return
            for node, group in self._hdf5_groups(top, create=True):
                node._save_hdf5(group, appendable)

    def append_state(self, saveto: Union[str, "h5py.File"]):
        """
        Append the current state to an existing HDF5 file.

        The file must have been previously created by ``save_state`` with
        ``appendable=True``.  The graph structure in the file is verified, and
        every value is collected, before anything is written.

        A param with no value is given a blank row (NaN), so that the same
        index is the same state for every param. A param that had no value
        when the file was saved has blank rows for the states before it got
        one.

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
        with _open_hdf5(saveto, "a", "saving") as h5file:
            top = self._hdf5_top(h5file)
            if top.attrs.get(_H5_LAYOUT) == "fast":
                self._append_fast(top)
                return
            writers = []
            for node, group in self._hdf5_groups(top):
                write = node._append_hdf5(group)
                if write is not None:
                    writers.append(write)
            samples = top.attrs.get(_H5_SAMPLES)  # not kept by older versions of caskade
            if samples is not None:
                samples = int(samples) + 1
            for write in writers:
                write(samples)
            if samples is not None:
                # Written last. If the appending was cut short, the next one
                # brings every param back to the same number of states.
                top.attrs[_H5_SAMPLES] = samples

    def load_state(self, loadfrom: Union[str, "h5py.File"], index: int = -1, **kwargs):
        """
        Load node state (and children) from an HDF5 file.

        The whole state is read and checked against the graph first. Only if
        all of it fits is any of it taken on, so a file that does not match
        leaves the graph as it was.

        An attribute stored with a node is only loaded if the node lists it in
        its ``saveattrs``, the file does not get to choose which attributes
        are set.

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
            (e.g. ``driver``). By default a file of up to 256 MB is read into
            memory in one go (``driver="core"``), which is the quickest on a
            network file system, and a larger file is read piece by piece.

        Raises
        ------
        GraphError
            If the graph structure no longer matches the file.
        NotImplementedError
            If the file path does not end with a supported extension.
        """
        if (
            "driver" not in kwargs
            and isinstance(loadfrom, (str, os.PathLike))
            and os.path.isfile(loadfrom)
            and os.path.getsize(loadfrom) <= _H5_CORE_DRIVER_LIMIT
        ):
            kwargs["driver"] = "core"
        with _open_hdf5(loadfrom, "r", "loading", **kwargs) as h5file:
            top = self._hdf5_top(h5file)
            if top.attrs.get(_H5_LAYOUT) == "fast":
                plan = self._read_fast(top, index)
            else:
                plan = [node._read_hdf5(group, index) for node, group in self._hdf5_groups(top)]
        for take_on in plan:
            if take_on is not None:
                take_on()

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
        depths = self._hierarchy_depths()

        def add_node(node: Node, dot, level: int):
            # ``level`` counts the boxes being drawn inside of. A node used at
            # several levels is drawn at the shallowest, so it is skipped here
            # when this is not that level
            if node in components or depths[node] < level:
                return
            dot.attr("node", **node.graphviz_style)
            dot.node(str(id(node)), repr(node))
            components.add(node)

            for child in node.children.values():
                if child in node.subgraphs:
                    with dot.subgraph(name=f"cluster_{id(child)}") as subdot:
                        add_node(child, subdot, level + 1)
                else:
                    add_node(child, dot, level)

        dot = graphviz.Digraph(strict=True)
        add_node(self, dot, 0)
        for node in self.topological_ordering():  # every node once, whatever the number of paths
            for child in node.children.values():
                dot.edge(str(id(node)), str(id(child)))
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
        graph: dict[str, dict] = {
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

    def __setitem__(self, key: str, value: "Node"):
        self.link(key, value)

    def __setattr__(self, key: str, value: Any):
        """Intercept attribute setting so the graph always matches the attributes.

        Assigning a ``Node`` links it as a child (replacing any child already
        held under that name), and assigning anything else over a child unlinks
        that child. The name stays that of a child, so a node may be assigned
        to it again later (``node.child = None`` and then ``node.child = other``).
        """
        # Values are written to nodes while a simulation runs, so the common
        # case of a value that is not a node is kept as short as possible
        if isinstance(value, Node):
            if "_children" not in self.__dict__:  # ``Node.__init__`` has not run yet
                raise NodeConfigurationError(
                    f"Cannot assign node '{value.name}' to attribute '{key}' before "
                    f"`super().__init__()` has been called in `{type(self).__name__}.__init__`"
                )
            # check for trying setting an attr with its own setter, allow the setter to handle throwing errors (e.g. value)
            if not hasattr(getattr(type(self), key, None), "fset"):
                self._link(key, value, replace=True)
        else:
            try:
                is_child = key in self._children
            except AttributeError:  # ``Node.__init__`` has not run yet
                is_child = False
            if is_child:
                self._unlink(key)
                self._freed.add(key)  # a node may be assigned to this name again

        super().__setattr__(key, value)

    def __delattr__(self, key: str):
        """Intercept attribute deletion to remove links."""
        if key in self.children:
            self._unlink(key)
        self._freed.discard(key)
        super().__delattr__(key)


class Memo:
    """
    Context manager that sends a "memo" (a small message) to a node and to
    every node below it in the graph, and takes it back on exit.

    Each node holds the memo as a ``MemoNote`` in ``node.memos``, which records
    the text (``label``), the node it was sent from (``owner``) and how many
    hierarchical links lie between the two (``depth``). ``caskade`` never acts
    on a memo sent this way, whatever its text, so any string is safe to use.
    The same memo may be sent again while it is already held, each node keeps
    it until every one of them has been taken back.

    Parameters
    ----------
    module: Node
        The node that sends the memo
    memo: str
        The message to send down the graph

    Examples
    --------
    ::

        with Memo(sim, "plotting"):
            assert any(note.label == "plotting" for note in sim.x.memos)
    """

    def __init__(self, module: Node, memo: Union[MemoKind, str]):
        self.module = module
        self.memo = memo

    def __enter__(self):
        self.module.add_memo(self.memo)

    def __exit__(self, exc_type, exc_value, traceback):
        self.module.remove_memo(self.memo)
