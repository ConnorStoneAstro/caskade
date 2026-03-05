from .base import Node
from .param import Param
from .mixins import GetSetValues


class NodeCollection(Node, GetSetValues):
    """Base mixin for collections of nodes that track parameters.

    Provides shared functionality for traversing, querying, and converting
    parameters within a graph of nodes. Subclasses such as ``NodeTuple`` and
    ``NodeList`` combine this mixin with a standard Python sequence type.
    """

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
    def dynamic_params(self) -> tuple[Param]:
        """All dynamic parameters in the graph below this node.

        Returns
        -------
        tuple of Param
            Dynamic (non-static, non-pointer) parameters found via
            topological ordering.
        """
        T = self.topological_ordering()
        return tuple(filter(lambda n: isinstance(n, Param) and n.dynamic, T))

    @property
    def dynamic_param_groups(self) -> tuple[int]:
        """Sorted unique group identifiers of all dynamic parameters.

        Returns
        -------
        tuple of int
            Sorted group indices present among the dynamic parameters.
        """
        return tuple(sorted(set(p.group for p in self.dynamic_params)))

    @property
    def static_params(self) -> tuple[Param]:
        """All static parameters in the graph below this node.

        Returns
        -------
        tuple of Param
            Static (non-dynamic, non-pointer) parameters found via
            topological ordering.
        """
        T = self.topological_ordering()
        return tuple(filter(lambda n: isinstance(n, Param) and n.static, T))

    @property
    def pointer_params(self) -> tuple[Param]:
        """All pointer parameters in the graph below this node.

        Returns
        -------
        tuple of Param
            Parameters that act as pointers to other parameters, found via
            topological ordering.
        """
        T = self.topological_ordering()
        return tuple(filter(lambda n: isinstance(n, Param) and n.pointer, T))

    def copy(self):
        raise NotImplementedError

    def deepcopy(self):
        raise NotImplementedError

    @property
    def dynamic(self):
        """Whether any node in this collection has dynamic parameters.

        Returns
        -------
        bool
            ``True`` if at least one contained node is dynamic.
        """
        return any(node.dynamic for node in self)

    @property
    def static(self):
        """Whether all nodes in this collection are static.

        Returns
        -------
        bool
            ``True`` if no contained node is dynamic.
        """
        return not self.dynamic

    def __mul__(self, other):
        raise NotImplementedError

    def __eq__(self, other):
        return Node.__eq__(self, other)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name})[{len(self)}]"

    def __hash__(self):
        return Node.__hash__(self)


class NodeTuple(NodeCollection, tuple):
    """Immutable, ordered collection of nodes.

    Behaves like a standard ``tuple`` but also participates in the caskade
    node graph.  All elements must be ``Node`` instances and are automatically
    linked as children upon construction.

    Parameters
    ----------
    iterable : iterable of Node, optional
        Nodes to include in the tuple.
    name : str, optional
        Human-readable name for this collection node.
    """

    def __init__(self, iterable=None, name=None):
        tuple.__init__(iterable)
        Node.__init__(self, name=name)
        self.node_type = "ntuple"

        for node in self:
            if not isinstance(node, Node):
                raise TypeError(f"NodeTuple elements must be Node objects, not {type(node)}")
            self.link(node)

    @property
    def graphviz_style(self):
        return {"style": "solid", "color": "black", "shape": "tab"}

    def __getitem__(self, key):
        if isinstance(key, str):
            return Node.__getitem__(self, key)
        return tuple.__getitem__(self, key)

    def __add__(self, other):
        res = super().__add__(other)
        return NodeTuple(res)


class NodeList(NodeCollection, list):
    """Mutable, ordered collection of nodes.

    Behaves like a standard ``list`` but also participates in the caskade
    node graph.  All elements must be ``Node`` instances.  Graph links are
    automatically updated whenever the list is modified.

    Parameters
    ----------
    iterable : iterable of Node, optional
        Nodes to include in the list.  Defaults to an empty iterable.
    name : str, optional
        Human-readable name for this collection node.
    """

    def __init__(self, iterable=(), name=None):
        list.__init__(self, iterable)
        Node.__init__(self, name)
        self.node_type = "nlist"

        self._link_nodes()

    @property
    def graphviz_style(self):
        return {"style": "solid", "color": "black", "shape": "folder"}

    def _unlink_nodes(self):
        for node in self:
            self.unlink(node)

    def _link_nodes(self):
        for node in self:
            if not isinstance(node, Node):
                raise TypeError(f"NodeList elements must be Node objects, not {type(node)}")
            self.link(node)

    def append(self, node):
        """Append a node to the list and update graph links."""
        self._unlink_nodes()
        try:
            super().append(node)
        finally:
            self._link_nodes()

    def insert(self, index, node):
        """Insert a node at the given index and update graph links."""
        self._unlink_nodes()
        try:
            super().insert(index, node)
        finally:
            self._link_nodes()

    def extend(self, iterable):
        """Extend the list with nodes from an iterable and update graph links."""
        self._unlink_nodes()
        try:
            super().extend(iterable)
        finally:
            self._link_nodes()

    def clear(self):
        """Remove all nodes from the list and update graph links."""
        self._unlink_nodes()
        try:
            super().clear()
        finally:
            self._link_nodes()

    def pop(self, index=-1):
        """Remove and return a node at the given index, updating graph links."""
        self._unlink_nodes()
        try:
            node = super().pop(index)
        finally:
            self._link_nodes()
        return node

    def remove(self, value):
        """Remove the first occurrence of a node and update graph links."""
        self._unlink_nodes()
        try:
            super().remove(value)
        finally:
            self._link_nodes()

    def __getitem__(self, key):
        if isinstance(key, str):
            return Node.__getitem__(self, key)
        if isinstance(key, slice):
            return NodeList(list.__getitem__(self, key), name=self.name)
        return list.__getitem__(self, key)

    def __setitem__(self, key, value):
        self._unlink_nodes()
        try:
            super().__setitem__(key, value)
        finally:
            self._link_nodes()

    def __delitem__(self, key):
        self._unlink_nodes()
        try:
            super().__delitem__(key)
        finally:
            self._link_nodes()

    def __add__(self, other):
        res = super().__add__(other)
        return NodeList(res, name=self.name)

    def __iadd__(self, other):
        self._unlink_nodes()
        try:
            ret = super().__iadd__(other)
        finally:
            self._link_nodes()
        return ret

    def __imul__(self, other):
        raise NotImplementedError


class NodeDict(NodeCollection, dict):
    """Mutable, keyed collection of nodes.

    Behaves like a standard ``dict`` but also participates in the caskade
    node graph.  All elements must be ``Node`` instances.  Graph links are
    automatically updated whenever the dict is modified.

    Parameters
    ----------
    mapping : mapping of str to Node, optional
        Nodes to include in the dict.  Defaults to an empty dict.
    name : str, optional
        Human-readable name for this collection of nodes.
    """

    def __init__(self, mapping=None, name=None):
        if mapping is None:
            mapping = {}
        dict.__init__(self, mapping)
        Node.__init__(self, name=name)
        self.node_type = "ndict"
        self._link_nodes()

    @property
    def graphviz_style(self):
        return {"style": "solid", "color": "black", "shape": "component"}

    @property
    def dynamic(self):
        return any(node.dynamic for node in dict.values(self))

    def _unlink_nodes(self):
        for node in dict.values(self):
            self.unlink(node)

    def _link_nodes(self):
        for key, node in dict.items(self):
            if not isinstance(node, Node):
                raise TypeError(f"NodeDict values must be Node objects, not {type(node)}")
            self.link(key, node)

    def __getitem__(self, key):
        return dict.__getitem__(self, key)

    def __setitem__(self, key, node):
        self._unlink_nodes()
        try:
            dict.__setitem__(self, key, node)
        finally:
            self._link_nodes()

    def __delitem__(self, key):
        self._unlink_nodes()
        try:
            dict.__delitem__(self, key)
        finally:
            self._link_nodes()

    def update(self, mapping=None, **kwargs):
        """Update the dict with another mapping (i.e. dict) and update graph links."""
        self._unlink_nodes()
        try:
            if mapping is not None:
                dict.update(self, mapping)
            if kwargs:
                dict.update(self, kwargs)
        finally:
            self._link_nodes()

    def pop(self, key, *args):
        """Remove and return a node from the dict and update graph links."""
        self._unlink_nodes()
        try:
            node = dict.pop(self, key, *args)
        finally:
            self._link_nodes()
        return node

    def popitem(self):
        """Remove and return an arbitrary (key, node) pair from the dict (the last one inserted) and update graph links."""
        self._unlink_nodes()
        try:
            key, node = dict.popitem(self)
        finally:
            self._link_nodes()
        return key, node

    def clear(self):
        """Remove all nodes from the dict and update graph links."""
        self._unlink_nodes()
        dict.clear(self)

    def setdefault(self, key, default):
        """If key is in the dictionary, return its value. If not, insert key with a value of default and return default. Update graph links."""
        # Preserve dict.setdefault API shape but enforce NodeDict invariants
        if key in self:
            return self[key]
        if not isinstance(default, Node):
            raise TypeError(f"NodeDict values must be Node objects, not {type(default)}")
        self[key] = default
        return default
