from collections.abc import Iterable, Iterator, Mapping, MutableMapping, MutableSequence, Sequence
from typing import Any, Optional, Self, Union, overload

from .base import Node, NodeType
from .param import Param
from .mixins import GetSetValues
from .errors import GraphError


class NodeCollection(Node, GetSetValues):
    """Base class for collections of nodes that track parameters.

    Provides shared functionality for traversing, querying, and converting
    parameters within a graph of nodes. The subclasses ``NodeTuple``,
    ``NodeList`` and ``NodeDict`` add the interface of a Python sequence or
    mapping to it.

    The children of a collection are always exactly its contents, in the same
    order. For that reason the contents may only be changed through the
    sequence / mapping methods of the collection (``append``, ``del``,
    ``update`` and so on). Linking or unlinking a node directly, or assigning a
    node as an attribute, raises a ``TypeError``.
    """

    def _set_children(self, keyed: dict[str, Node]):
        """Make the children of this collection exactly ``keyed``, in that
        order. If this is not possible an error is raised and nothing changes.

        Every change to the contents of a collection goes through this method.
        """
        if self.active:
            raise GraphError(
                f"Cannot change the contents of {type(self).__name__} '{self.name}' "
                "while the graph is active"
            )
        self._check_nodes(keyed.values())
        previous = dict(self._children)
        try:
            self._sync_children(keyed)
        except Exception:
            self._sync_children(previous)  # these were valid before, so this cannot fail
            raise

    def _check_nodes(self, nodes: Iterable):
        for node in nodes:
            if not isinstance(node, Node):
                raise TypeError(
                    f"{type(self).__name__} elements must be Node objects, not {type(node)}"
                )

    def _sync_children(self, keyed: dict[str, Node]):
        for key, child in list(self._children.items()):
            if keyed.get(key) is not child:
                Node.__delattr__(self, key)  # unlinks the child
        for key, child in keyed.items():
            Node.link(self, key, child)  # does nothing if already linked
        # The order of the children sets the order of the params
        if list(self._children) != list(keyed):
            self._children.clear()
            self._children.update(keyed)
            self.update_graph()

    def _closed(self, action: str):
        if isinstance(self, MutableMapping):
            how = "change them with its mapping methods"
        elif isinstance(self, MutableSequence):
            how = "change them with its sequence methods"
        else:
            how = "and they are fixed at construction"
        return TypeError(
            f"Cannot {action} on {type(self).__name__} '{self.name}' directly. The children "
            f"of a collection are always exactly its contents, {how}."
        )

    def link(self, key, child=None):
        """Not supported, change the contents of the collection instead."""
        raise self._closed("link a node")

    def unlink(self, key=None):
        """Not supported, change the contents of the collection instead."""
        raise self._closed("unlink a node")

    def __setattr__(self, key: str, value: Any):
        if isinstance(value, Node):
            raise self._closed(f"assign a node to the attribute '{key}'")
        if key in self.__dict__.get("_children", ()):
            raise self._closed(f"overwrite the node '{key}'")
        super().__setattr__(key, value)

    def __delattr__(self, key: str):
        if key in self._children:
            raise self._closed(f"delete the node '{key}'")
        super().__delattr__(key)

    def to_dynamic(self, children_only=True):
        """Change all parameters to dynamic parameters.

        Parameters
        ----------
        children_only: (bool, optional)
            If True, only convert the children of this module to dynamic. If False,
            convert all parameters in the graph below this module. Defaults to True.
        """
        node_list = self.children.values() if children_only else self.topological_ordering()
        for node in node_list:
            if isinstance(node, Param) and not node.pointer:
                node.to_dynamic()

    def to_static(self, children_only=True):
        """Change all parameters to static parameters.

        Parameters
        ----------
        children_only: (bool, optional)
            If True, only convert children of this module. If False, convert
            all parameters in the graph below this module. Defaults to True.
        """
        node_list = self.children.values() if children_only else self.topological_ordering()
        for node in node_list:
            if isinstance(node, Param) and not node.pointer:
                node.to_static()

    @property
    def dynamic_params(self) -> tuple[Param, ...]:
        """All dynamic parameters in the graph below this node.

        Returns
        -------
        tuple of Param
            Dynamic (non-static, non-pointer) parameters found via
            topological ordering.
        """
        T = self.topological_ordering()
        return tuple(n for n in T if isinstance(n, Param) and n.dynamic)

    @property
    def dynamic_param_groups(self) -> tuple[int, ...]:
        """Sorted unique group identifiers of all dynamic parameters.

        Returns
        -------
        tuple of int
            Sorted group indices present among the dynamic parameters.
        """
        return tuple(sorted(set(p.group for p in self.dynamic_params)))

    @property
    def static_params(self) -> tuple[Param, ...]:
        """All static parameters in the graph below this node.

        Returns
        -------
        tuple of Param
            Static (non-dynamic, non-pointer) parameters found via
            topological ordering.
        """
        T = self.topological_ordering()
        return tuple(n for n in T if isinstance(n, Param) and n.static)

    @property
    def pointer_params(self) -> tuple[Param, ...]:
        """All pointer parameters in the graph below this node.

        Returns
        -------
        tuple of Param
            Parameters that act as pointers to other parameters, found via
            topological ordering.
        """
        T = self.topological_ordering()
        return tuple(n for n in T if isinstance(n, Param) and n.pointer)

    def copy(self):
        raise NotImplementedError

    def deepcopy(self):
        raise NotImplementedError

    @property
    def dynamic(self) -> bool:
        """Whether any node in this collection has dynamic parameters.

        Returns
        -------
        bool
            ``True`` if at least one contained node is dynamic.
        """
        # Every node type that can be held (Param, Module, collection) has ``dynamic``
        return any(node.dynamic for node in self._children.values())  # type: ignore[attr-defined]

    @property
    def static(self) -> bool:
        """Whether all nodes in this collection are static.

        Returns
        -------
        bool
            ``True`` if no contained node is dynamic.
        """
        return not self.dynamic

    def __len__(self) -> int:
        raise NotImplementedError

    def __mul__(self, other):
        raise NotImplementedError

    def __eq__(self, other: object) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name})[{len(self)}]"


class _NodeSequence(NodeCollection, Sequence[Node]):
    """What ``NodeTuple`` and ``NodeList`` share: the nodes are held in a
    private list and linked as children under their own names."""

    def __init__(self, iterable: Optional[Iterable[Node]] = None, name: Optional[str] = None):
        Node.__init__(self, name=name)
        self._nodes: list[Node] = []
        self._set_nodes(() if iterable is None else iterable)

    def _set_nodes(self, nodes: Iterable[Node]):
        """Make ``nodes`` the contents of the sequence, or raise an error and
        change nothing."""
        nodes = list(nodes)
        self._check_nodes(nodes)
        keyed: dict[str, Node] = {}
        for node in nodes:
            if keyed.setdefault(node.name, node) is not node:
                raise GraphError(
                    f"Two different nodes named '{node.name}' cannot be held in the same "
                    f"{type(self).__name__} ({self.name}), as its nodes are linked by name"
                )
        self._set_children(keyed)
        self._nodes = nodes

    def _new(self, nodes: Iterable[Node]) -> Self:
        """A new sequence of the same type and name holding ``nodes``."""
        return type(self)(nodes, name=self.name)

    @overload
    def __getitem__(self, key: Union[int, str]) -> Node: ...
    @overload
    def __getitem__(self, key: slice) -> Self: ...
    def __getitem__(self, key: Union[int, slice, str]) -> Union[Node, Self]:
        if isinstance(key, str):
            return self._children[key]
        if isinstance(key, slice):
            return self._new(self._nodes[key])
        return self._nodes[key]

    def __len__(self) -> int:
        return len(self._nodes)

    def __iter__(self) -> Iterator[Node]:
        return iter(self._nodes)

    def __add__(self, other) -> Self:
        if not isinstance(other, Sequence):
            return NotImplemented
        return self._new(self._nodes + list(other))

    def __radd__(self, other) -> Self:
        if not isinstance(other, Sequence):
            return NotImplemented
        return self._new(list(other) + self._nodes)


class NodeTuple(_NodeSequence):
    """Immutable, ordered collection of nodes.

    Has the interface of a ``tuple`` (it is a ``collections.abc.Sequence``,
    though not a subclass of ``tuple``) and also participates in the caskade
    node graph. All elements must be ``Node`` instances and are linked as
    children, under their names, upon construction.

    Parameters
    ----------
    iterable : iterable of Node, optional
        Nodes to include in the tuple.
    name : str, optional
        Human-readable name for this collection node.
    """

    def __init__(self, iterable: Optional[Iterable[Node]] = None, name: Optional[str] = None):
        super().__init__(iterable, name=name)
        self.node_type = NodeType.NTUPLE

    @property
    def graphviz_style(self):
        return {"style": "solid", "color": "black", "shape": "tab"}

    def __setitem__(self, key, value):
        raise TypeError("'NodeTuple' object does not support item assignment")

    def __delitem__(self, key):
        raise TypeError("'NodeTuple' object does not support item deletion")


class NodeList(_NodeSequence, MutableSequence[Node]):
    """Mutable, ordered collection of nodes.

    Has the interface of a ``list`` (it is a
    ``collections.abc.MutableSequence``, though not a subclass of ``list``) and
    also participates in the caskade node graph. All elements must be ``Node``
    instances, linked as children under their names. The graph is kept in step
    whenever the list is modified, by any of its methods.

    Parameters
    ----------
    iterable : iterable of Node, optional
        Nodes to include in the list.  Defaults to an empty iterable.
    name : str, optional
        Human-readable name for this collection node.
    """

    def __init__(self, iterable: Optional[Iterable[Node]] = None, name: Optional[str] = None):
        super().__init__(iterable, name=name)
        self.node_type = NodeType.NLIST

    @property
    def graphviz_style(self):
        return {"style": "solid", "color": "black", "shape": "folder"}

    # The three methods below, with ``__getitem__`` and ``__len__``, are all that
    # ``MutableSequence`` needs to provide append, pop, remove, +=, and the rest.
    def __setitem__(self, key: Any, value: Any):  # an index with a node, or a slice with nodes
        nodes = list(self._nodes)
        nodes[key] = value
        self._set_nodes(nodes)

    def __delitem__(self, key: Union[int, slice]):
        nodes = list(self._nodes)
        del nodes[key]
        self._set_nodes(nodes)

    def insert(self, index: int, node: Node):
        """Insert a node before the given index and update graph links."""
        nodes = list(self._nodes)
        nodes.insert(index, node)
        self._set_nodes(nodes)

    # These would work without being written here, one node at a time. Doing
    # them in one step means they either fully happen or change nothing.
    def extend(self, iterable: Iterable[Node]):
        """Extend the list with nodes from an iterable and update graph links."""
        self._set_nodes(self._nodes + list(iterable))

    def clear(self):
        """Remove all nodes from the list and update graph links."""
        self._set_nodes(())

    def reverse(self):
        """Reverse the order of the nodes in place."""
        self._set_nodes(self._nodes[::-1])


class NodeDict(NodeCollection, MutableMapping[str, Node]):
    """Mutable, keyed collection of nodes.

    Has the interface of a ``dict`` (it is a
    ``collections.abc.MutableMapping``, though not a subclass of ``dict``) and
    also participates in the caskade node graph. All values must be ``Node``
    instances, linked as children under their keys. The graph is kept in step
    whenever the dict is modified, by any of its methods.

    Parameters
    ----------
    mapping : mapping of str to Node, optional
        Nodes to include in the dict.  Defaults to an empty dict.
    name : str, optional
        Human-readable name for this collection of nodes.
    """

    def __init__(self, mapping: Optional[Mapping[str, Node]] = None, name: Optional[str] = None):
        Node.__init__(self, name=name)
        self.node_type = NodeType.NDICT
        self._set_children(dict(() if mapping is None else mapping))

    @property
    def graphviz_style(self):
        return {"style": "solid", "color": "black", "shape": "component"}

    # The five methods below are all that ``MutableMapping`` needs to provide
    # keys, values, items, get, pop, setdefault, ``in``, and the rest. The nodes
    # are simply the children of this node.
    def __getitem__(self, key: str) -> Node:
        return self._children[key]

    def __setitem__(self, key: str, node: Node):
        self._set_children({**self._children, key: node})

    def __delitem__(self, key: str):
        keyed = dict(self._children)
        del keyed[key]
        self._set_children(keyed)

    def __iter__(self) -> Iterator[str]:
        return iter(self._children)

    def __len__(self) -> int:
        return len(self._children)

    # These would work without being written here, one node at a time. Doing
    # them in one step means they either fully happen or change nothing.
    def update(self, *args, **kwargs):
        """Update the dict with another mapping (i.e. dict) and update graph links."""
        keyed = dict(self._children)
        keyed.update(*args, **kwargs)
        self._set_children(keyed)

    def clear(self):
        """Remove all nodes from the dict and update graph links."""
        self._set_children({})

    def popitem(self) -> tuple[str, Node]:
        """Remove and return the last (key, node) pair that was inserted, as for
        a ``dict``, and update graph links."""
        if not self._children:
            raise KeyError(f"popitem(): {type(self).__name__} is empty")
        key = next(reversed(self._children))
        return key, self.pop(key)
