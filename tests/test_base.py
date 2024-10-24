from caskade import Node, test

import pytest


def test_creation():
    node = Node("test")
    assert node._name == "test"
    assert node._children == {}
    assert node._parents == set()
    assert node._active == False
    assert node._type == "node"

    with pytest.raises(AttributeError):
        node.name = "newname"


def test_link():
    node1 = Node("node1")
    node2 = Node("node2")
    node1.link("subnode", node2)

    # Already linked
    with pytest.raises(ValueError):
        node1.link("subnode", node2)

    # Double link
    with pytest.raises(ValueError):
        node1.link("subnode2", node2)

    # Make a cycle
    node3 = Node("node3")
    node2.link("subnode", node3)
    with pytest.raises(ValueError):
        node3.link("subnode", node1)

    assert "subnode" in node1._children
    assert node1._children["subnode"] == node2
    assert node1._parents == set()
    assert node2._parents == set([node1])

    str(node1)
    repr(node1)

    node1.unlink(node2)
    assert "subnode" not in node1._children
    assert node2._parents == set()
    assert node1._parents == set()


def test_topological_ordering():
    node1 = Node("node1")
    node2 = Node("node2")
    node3 = Node("node3")
    node4 = Node("node4")
    node5 = Node("node5")
    node6 = Node("node6")

    node1.link("subnode1", node2)
    node1.link("subnode2", node3)
    node2.link("subnode3", node4)
    node2.link("subnode4", node5)
    node3.link("subnode5", node6)

    ordering = node1.topological_ordering()
    assert ordering == (node1, node2, node4, node5, node3, node6)

    ordering = node1.topological_ordering(with_type="node")
    assert ordering == (node1, node2, node4, node5, node3, node6)

    ordering = node1.topological_ordering(with_type="dynamic")
    assert ordering == ()

    graph = node1.graphviz()
    assert graph is not None, "should return a graphviz object"

    node1.unlink("subnode1")
    ordering = node1.topological_ordering()
    assert ordering == (node1, node3, node6)

    node1.unlink(node3)
    ordering = node1.topological_ordering()
    assert ordering == (node1,)


@pytest.mark.parametrize("linkbyname", [True, False])
@pytest.mark.parametrize("graphviz_order", [True, False])
def test_active(linkbyname, graphviz_order):
    node1 = Node("node1")
    node2 = Node("node2")
    node3 = Node("node3")
    node4 = Node("node4")
    node5 = Node("node5")
    node6 = Node("node6")

    if linkbyname:
        node1.link("subnode1", node2)
        node1.link("subnode2", node3)
        node2.link("subnode3", node4)
        node2.link("subnode4", node5)
        node3.link("subnode5", node6)
    else:
        node1.link(node2)
        node1.link(node3)
        node2.link(node4)
        node2.link(node5)
        node3.link(node6)

    node1.active = True
    assert node1.active == True
    assert node2.active == True
    assert node3.active == True
    assert node4.active == True
    assert node5.active == True
    assert node6.active == True

    node2.active = False
    assert node1.active == True
    assert node2.active == False
    assert node3.active == True
    assert node4.active == False
    assert node5.active == False
    assert node6.active == True

    node1.active = False
    assert node1.active == False
    assert node2.active == False
    assert node3.active == False
    assert node4.active == False
    assert node5.active == False
    assert node6.active == False

    graph = node1.graphviz(graphviz_order)
    assert graph is not None, "should return a graphviz object"


def test_test():
    test()
