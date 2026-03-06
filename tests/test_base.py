import os

from caskade import Node, test, GraphError, NodeConfigurationError, Memo

import pytest


def test_creation():
    node = Node("test")
    assert node._name == "test"
    assert node._children == {}
    assert node._parents == set()
    assert node.active == False
    assert node.node_type == "node"

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
        a.link("link", g)  # key is attribute
    with pytest.raises(NodeConfigurationError):
        a.link("bad name", g)  # Name not python identifier

    # Double link
    with pytest.raises(GraphError):
        a.link("b2", b)

    # (Un)Link fails while active
    with Memo(a, "active"):
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


def test_graphviz(node_graph):
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

    # Propagate memo state
    with Memo(a, "active"):
        assert all(node.active is True for node in node_graph)
        assert all(node.online is False for node in node_graph)

    with Memo(a, "semi_active"):
        assert all(node.active is False for node in node_graph)
        assert all(node.online is True for node in node_graph)

    with Memo(c, "active"):
        assert a.active is False
        assert b.active is False
        assert c.active is True
        assert d.active is True
        assert e.active is True
        assert f.active is True
        assert g.active is True
        assert all(node.online is False for node in node_graph)

    with Memo(c, "semi_active"):
        assert all(node.active is False for node in node_graph)
        assert a.online is False
        assert b.online is False
        assert c.online is True
        assert d.online is True
        assert e.online is True
        assert f.online is True
        assert g.online is True

    # Return to deactivated state
    assert all(node.active is False for node in node_graph)
    assert all(node.online is False for node in node_graph)


def test_subgraph(node_graph):
    a, b, c, d, e, f, g = node_graph
    a.unlink(c)
    a.hierarchical_link("c", c)

    # Propagate memo state
    with Memo(a, "active"):
        assert all(node.active is True for node in node_graph)
        assert all(node.online is False for node in node_graph)

    with Memo(a, "semi_active"):
        assert all(node.online is True for node in (a, b, d, e))
        assert all(node.online is False for node in (c, f, g))


def test_subgraph_graphviz(node_graph):
    a, _, c, *_ = node_graph
    a.unlink(c)
    a.hierarchical_link("c", c)

    graph = a.graphviz(saveto="test_graph.pdf")
    assert graph is not None, "should return a graphviz object"
    assert os.path.exists("test_graph.pdf")
    os.remove("test_graph.pdf")
