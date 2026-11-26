import copy
import os
import pickle
import warnings

# ``test`` is collected by pytest, which runs the integration check of the package
from caskade import Node, test, GraphError, NodeConfigurationError  # noqa: F401
from caskade import LinkToAttributeError, LinkToAttributeWarning
from caskade import Memo, MemoKind, MemoNote

import pytest


def test_creation():
    node = Node("test")
    assert node._name == "test"
    assert node._children == {}
    assert node._parents == set()
    assert not node.active
    assert node.node_type == "node"


def test_node_type():
    from caskade import Module, NodeDict, NodeList, NodeTuple, NodeType, Param

    param = Param("p", 1.0)
    nodes = {
        NodeType.NODE: Node("n"),
        NodeType.MODULE: Module("m"),
        NodeType.STATIC: param,
        NodeType.NTUPLE: NodeTuple(),
        NodeType.NLIST: NodeList(),
        NodeType.NDICT: NodeDict(),
    }
    for node_type, node in nodes.items():
        assert node.node_type is node_type
        # The members are strings, and print as their text
        assert node.node_type == node_type.value
        assert f"{node.node_type}" == node_type.value

    param.to_dynamic()
    assert param.node_type is NodeType.DYNAMIC and param.dynamic and not param.static
    param.value = Param("q", 2.0)
    assert param.node_type is NodeType.POINTER and param.pointer and not param.dynamic

    # A param is always one of the known types
    with pytest.raises(ValueError):
        param.node_type = "pointr"
    assert param.node_type is NodeType.POINTER

    with pytest.raises(AttributeError):
        node.name = "newname"

    with pytest.raises(NodeConfigurationError):
        Node(1)

    with pytest.raises(NodeConfigurationError):
        Node("test|test")

    with pytest.raises(NodeConfigurationError):
        Node("node 1")

    # Link on creation
    Node("linker", link=node)


def test_meta_link():
    a = Node()
    b = Node()
    a.meta.b = b
    assert len(a.children) == 0
    assert len(b.parents) == 0


def test_linking_with_setitem(node_graph):
    a, b, c, d, e, f, g = node_graph

    # Link using __setitem__
    a["new_child"] = d
    assert "new_child" in a.children
    assert a.children["new_child"] is d
    assert a in d.parents


def test_linking(node_graph):
    a, b, c, d, e, f, g = node_graph

    # Already linked
    nchild = len(a.children)
    a.link(b)
    assert len(a.children) == nchild

    # Bad links
    with pytest.raises(GraphError):
        a.link("c", b)  # key already used
    with pytest.raises(GraphError):
        a.link("children", g)  # key is a property of the node
    with pytest.raises(NodeConfigurationError):
        a.link("bad name", g)  # Name not python identifier
    with pytest.raises(TypeError):
        a.link("acceptable_name", 123)  # value is not a node

    # Double link
    with pytest.raises(GraphError):
        a.link("b2", b)

    # (Un)Link fails while active
    with Memo(a, MemoKind.RUN):
        with pytest.raises(GraphError):
            a.link(g)
        assert "g" not in a.children
        with pytest.raises(GraphError):
            a.unlink(b)
        assert "b" in a.children

    # Make a cycle
    with pytest.raises(GraphError):
        g.link("sneaky_a", a)

    # Test string representations
    a_print = str(a)
    assert all(char in a_print for char in "abcdefg")
    assert repr(a) == "Node(a)"

    # Check identity
    assert a.e == e

    # Check unlink
    a.unlink(e)
    assert "e" not in a.children
    assert a not in e.parents
    assert e in a.topological_ordering()
    with pytest.raises(AttributeError):
        a.e
    with pytest.raises(KeyError):
        a.unlink(e)
    with pytest.raises(KeyError):
        a.unlink("e")
    a.unlink((b, c))

    # Check unlink with no arguments clears all children
    a.link(e)
    assert len(a.children) > 0
    a.unlink()
    assert len(a.children) == 0


def test_graphviz(node_graph, needs_dot):
    a, *_ = node_graph

    graph = a.graphviz(saveto="test_graph.pdf")
    assert graph is not None, "should return a graphviz object"
    assert os.path.exists("test_graph.pdf")
    os.remove("test_graph.pdf")


def test_topological_ordering(node_graph):
    a, b, c, d, e, f, g = node_graph

    # Check topological order
    ordering = a.topological_ordering()
    assert ordering == (a, b, c, d, f, g, e)

    # Unlink changes topological order
    c.unlink(f)
    ordering = a.topological_ordering()
    assert ordering == (a, b, c, d, e)


def test_active(node_graph):
    a, b, c, d, e, f, g = node_graph

    # Default false active and online
    assert all(node.active is False for node in node_graph)
    assert all(node.online is False for node in node_graph)

    # A run reaches every node below the one it starts from
    with Memo(a, MemoKind.RUN):
        assert all(node.active is True for node in node_graph)
        assert all(node.online is True for node in node_graph)

    with Memo(c, MemoKind.RUN):
        for node in (a, b):
            assert node.active is False
            assert node.online is False
        for node in (c, d, e, f, g):
            assert node.active is True
            assert node.online is True

    # Return to deactivated state
    assert all(node.active is False for node in node_graph)
    assert all(node.online is False for node in node_graph)


@pytest.mark.parametrize(
    "text", ["active", "a_active", "semi_active", "active learning", "x|y", "a:semi_set_active"]
)
def test_user_memo_never_changes_state(node_graph, text):
    a, b, c, d, e, f, g = node_graph
    a.unlink(c)
    a.hierarchical_link("c", c)

    with Memo(a, text):
        assert all(node.active is False for node in node_graph)
        assert all(node.online is False for node in node_graph)
        assert b.memos == {MemoNote(MemoKind.USER, a, 0, text)}
        assert f.memos == {MemoNote(MemoKind.USER, a, 1, text)}
        a.link("new", Node("new"))  # the graph is free to change
    assert all(node.memos == set() for node in node_graph)

    with pytest.raises(TypeError):
        a.add_memo(3)
    assert all(node.memos == set() for node in node_graph)


def test_memo_sent_twice():
    a, b = Node("a"), Node("b")
    a.link(b)
    with Memo(a, "note"):
        with Memo(a, "note"):
            assert len(b.memos) == 1
        # still held, it has only been taken back once
        assert b.memos == {MemoNote(MemoKind.USER, a, 0, "note")}
    assert b.memos == set()
    a.remove_memo("note")  # taking back a memo that is not held does nothing
    assert b.memos == set()

    # the same text from two senders is two memos
    top = Node("top")
    top.link(a)
    with Memo(top, "note"), Memo(a, "note"):
        assert b.memos == {
            MemoNote(MemoKind.USER, top, 0, "note"),
            MemoNote(MemoKind.USER, a, 0, "note"),
        }
        assert a.memos == b.memos
        assert top.memos == {MemoNote(MemoKind.USER, top, 0, "note")}


def test_subgraph(node_graph):
    a, b, c, d, e, f, g = node_graph
    a.unlink(c)
    a.hierarchical_link("c", c)

    # Nodes that are only reached through the hierarchical link are part of the
    # run, but are not online
    with Memo(a, MemoKind.RUN):
        assert all(node.active is True for node in node_graph)
        assert all(node.online is True for node in (a, b, d, e))
        assert all(node.online is False for node in (c, f, g))

    # A run from below the link is online for its own nodes
    with Memo(a, MemoKind.RUN), Memo(c, MemoKind.RUN):
        assert all(node.online is True for node in node_graph)
    assert all(node.active is False for node in node_graph)


def test_hierarchy_depths():
    top, mid, low, shared = Node("top"), Node("mid"), Node("low"), Node("shared")
    top.hierarchical_link("mid", mid)
    mid.hierarchical_link("low", low)
    low.link(shared)
    assert top._hierarchy_depths() == {top: 0, mid: 1, low: 2, shared: 2}
    assert mid._hierarchy_depths() == {mid: 0, low: 1, shared: 1}
    assert low._hierarchy_depths() == {low: 0, shared: 0}

    # A node used at several levels belongs to the shallowest of them
    mid.link(shared)
    assert top._hierarchy_depths() == {top: 0, mid: 1, low: 2, shared: 1}
    assert mid._hierarchy_depths()[shared] == 0
    top.link(shared)
    assert top._hierarchy_depths() == {top: 0, mid: 1, low: 2, shared: 0}

    # Whichever order the paths are found in
    top.unlink(shared)
    mid.unlink(shared)
    assert top._hierarchy_depths()[shared] == 2
    side = Node("side")
    top.link(side)
    side.link(low)  # ``low`` is now also reached with no hierarchical link
    assert top._hierarchy_depths() == {top: 0, mid: 1, low: 0, shared: 0, side: 0}

    # Each node holds one note for a memo, at its depth
    top.add_memo("note")
    assert mid.memos == {MemoNote(MemoKind.USER, top, 1, "note")}
    assert shared.memos == {MemoNote(MemoKind.USER, top, 0, "note")}
    top.remove_memo("note")


def test_subgraph_graphviz(node_graph, needs_dot):
    a, _, c, *_ = node_graph
    a.unlink(c)
    a.hierarchical_link("c", c)

    graph = a.graphviz(saveto="test_graph.pdf")
    assert graph is not None, "should return a graphviz object"
    assert os.path.exists("test_graph.pdf")
    os.remove("test_graph.pdf")


def test_assignment_keeps_graph_in_sync():
    a, b, c = Node("a"), Node("b"), Node("c")
    a.child = b

    # Overwriting a child with a plain value unlinks it
    a.child = 5
    assert a.child == 5
    assert a.children == {}
    assert a not in b.parents
    del a.child

    # Assigning another node over a child replaces it
    a.child = b
    a.child = c
    assert a.child is c
    assert a.children == {"child": c}
    assert a not in b.parents
    assert a in c.parents

    # Assigning the same node again changes nothing
    a.child = c
    assert a.children == {"child": c}
    assert c.parents == {a}


def test_name_of_a_child_can_take_a_node_again():
    a, b, c = Node("a"), Node("b"), Node("c")
    a.child = b
    a.child = None  # unlinks, the attribute holds None
    assert a.child is None and a.children == {} and a not in b.parents
    a.child = c
    assert a.child is c and a.children == {"child": c} and a in c.parents

    # Whatever the plain value was, and with an explicit link as well
    a.child = "nothing here"
    a.child = 5
    a.link("child", b)
    assert a.child is b and a.children == {"child": b}

    # Nothing was said so far, the name was that of a child all along
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        a.child = None
        a.child = c
    assert a.children == {"child": c}


def test_node_takes_the_place_of_another_attribute():
    a, b, c = Node("a"), Node("b"), Node("c")

    # A placeholder for a node that comes later
    a.later = None
    with pytest.warns(LinkToAttributeWarning, match=r"'b' as 'later'.*it was None") as caught:
        a.later = b
    assert caught[0].filename == __file__  # the line that did it, not one inside caskade
    assert a.later is b and a.children == {"later": b} and a in b.parents

    # Any other value, by assignment or with an explicit link
    a.number = 5.0
    with pytest.warns(LinkToAttributeWarning, match="'number'.*it was a float") as caught:
        a.link("number", c)
    assert caught[0].filename == __file__
    assert a.number is c and a.children == {"later": b, "number": c}

    # A name that was a child, but was deleted since, is a name like any other
    a.later = None
    del a.later
    a.later = "text"
    with pytest.warns(LinkToAttributeWarning, match="'later'.*it was a str"):
        a.later = b

    # This includes what caskade keeps on the node and the methods of the node,
    # that is what the warning is for
    d, e = Node("d"), Node("e")
    with pytest.warns(LinkToAttributeWarning, match="'description'.*it was a str"):
        d.description = e
    with pytest.warns(LinkToAttributeWarning, match="'unlink'.*it was a function"):
        d.unlink = Node("f")
    assert set(d.children) == {"description", "unlink"}

    # With warnings turned into errors nothing is changed
    g = Node("g")
    g.spot = None
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(LinkToAttributeWarning):
            g.spot = e
    assert g.spot is None and g.children == {} and g not in e.parents


def test_property_name_cannot_be_a_child():
    a, b = Node("a"), Node("b")
    for name in ("children", "name", "active"):
        with pytest.raises(LinkToAttributeError, match=f"'{name}' is a property of Node"):
            a.link(name, b)
    assert a.children == {} and a not in b.parents and a.name == "a"
    with pytest.raises(AttributeError):  # plain assignment is answered by the property
        a.children = b
    assert a.children == {}


def test_failed_replacement_changes_nothing():
    a, b, c = Node("a"), Node("b"), Node("c")
    a.first = b
    a.second = c

    def unchanged():
        return a.first is b and a.children == {"first": b, "second": c} and a in b.parents

    # New child is already linked under a different key
    with pytest.raises(GraphError):
        a.first = c
    assert unchanged()

    # New child would create a cycle
    top = Node("top")
    top.a = a
    with pytest.raises(GraphError):
        a.first = top
    assert unchanged()

    # An explicit link never replaces a child, only assignment does
    with pytest.raises(GraphError):
        a.link("first", Node("d"))
    with pytest.raises(GraphError):
        a["first"] = Node("d")
    assert unchanged()

    # Nothing may be replaced or removed while active
    with Memo(a, MemoKind.RUN):
        with pytest.raises(GraphError):
            a.first = Node("d")
        with pytest.raises(GraphError):
            a.first = 5
    assert unchanged()


def test_assign_node_before_init():
    class Early(Node):
        def __init__(self):
            self.plain = 1  # ordinary attributes are fine
            self.child = Node("child")
            super().__init__("early")

    with pytest.raises(NodeConfigurationError, match=r"super\(\).__init__\(\)"):
        Early()

    class Fine(Node):
        def __init__(self):
            self.plain = 1
            super().__init__("fine")
            self.child = Node("child")

    fine = Fine()
    assert fine.plain == 1
    assert fine.children == {"child": fine.child}


class Counter(Node):
    def __init__(self, name):
        super().__init__(name)
        self.updates = 0
        self.forgotten = 0

    def _graph_updated(self):
        self.updates += 1

    def _forget_memo(self, memo):
        super()._forget_memo(memo)
        self.forgotten += 1


def test_shared_nodes_handled_once():
    # Two nodes per layer, each linked to both nodes of the next layer. There
    # are 2**40 paths from the top to the bottom, so this only finishes if every
    # node is handled once rather than once per path.
    top = Counter("top")
    nodes = [top]
    layer = [top]
    for depth in range(40):
        next_layer = [Counter(f"a{depth}"), Counter(f"b{depth}")]
        for parent in layer:
            parent.link(next_layer)
        nodes += next_layer
        layer = next_layer

    top.add_memo(MemoKind.RUN)
    assert all(node.active for node in nodes)
    top.remove_memo(MemoKind.RUN)
    assert not any(node.active for node in nodes)
    assert all(node.forgotten == 1 for node in nodes)

    before = [node.updates for node in nodes]
    layer[0].update_graph()
    after = [node.updates for node in nodes]
    # every node sits above the bottom one, except the other bottom node
    assert all(a == b + 1 for a, b, node in zip(after, before, nodes) if node is not layer[1])
    assert layer[1].updates == before[-1]


def test_memo_follows_each_kind_of_path():
    top, a, b, shared = Node("top"), Node("a"), Node("b"), Node("shared")
    top.link((a, b))
    a.link(shared)
    b.hierarchical_link("shared", shared)

    # reached both with and without a hierarchical link, one note at the least depth
    top.add_memo("note")
    assert shared.memos == {MemoNote(MemoKind.USER, top, 0, "note")}
    top.remove_memo("note")
    assert shared.memos == set()


def test_memo_follows_graph_changes():
    top, a, b = Node("top"), Node("a"), Node("b")
    top.link(a)
    top.add_memo("note")
    top.remove_memo("note")

    # The graph changes below ``top`` after it has already sent memos
    a.link(b)
    top.add_memo("note")
    assert b.memos == {MemoNote(MemoKind.USER, top, 0, "note")}
    top.remove_memo("note")

    a.unlink(b)
    top.add_memo("note")
    assert b.memos == set()
    top.remove_memo("note")

    # An existing link becomes hierarchical
    top.hierarchical_link("a", a)
    top.add_memo("note")
    assert a.memos == {MemoNote(MemoKind.USER, top, 1, "note")}
    top.remove_memo("note")
    assert a.memos == set()


def test_link_needs_a_node():
    node = Node("node")
    with pytest.raises(TypeError, match="child must be a Node object, not <class 'str'>"):
        node.link("not_a_node")
    assert node.children == {}


def test_self_test_reports_a_wrong_result(monkeypatch):
    # ``caskade.test()`` is what users run to check an installation, so a
    # wrong result has to be an error
    from caskade import backend

    real = backend.make_array
    monkeypatch.setattr(backend, "make_array", lambda value: real(value + 1.0))
    with pytest.raises(RuntimeError, match="integration test failed: expected 13.0, got 16.0"):
        test()


def test_unique_id():
    a, b = Node("same"), Node("same")
    assert a.unique_id != b.unique_id  # the name is not what makes it unique
    name, _, tail = a.unique_id.rpartition("_")
    assert name == "same" and len(tail) == 32 and int(tail, 16) >= 0
    assert a.unique_id == a.unique_id

    with pytest.raises(AttributeError):
        a.unique_id = "mine"
    assert a.unique_id.startswith("same_")

    # Nodes made one after another, each gone before the next
    ids = set()
    for _ in range(200):
        ids.add(Node("again").unique_id)
    assert len(ids) == 200


@pytest.mark.parametrize("how", ["pickle", "deepcopy"])
def test_unique_id_goes_with_a_copy(how):
    top, mid, low = Node("top"), Node("mid"), Node("low")
    top.link(mid)
    mid.link(low)
    other = pickle.loads(pickle.dumps(top)) if how == "pickle" else copy.deepcopy(top)
    assert other is not top
    for node, copied in zip(top.topological_ordering(), other.topological_ordering()):
        assert copied.unique_id == node.unique_id
    assert len({node.unique_id for node in other.topological_ordering()}) == 3
