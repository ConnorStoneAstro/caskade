import pytest

from caskade import NodeList, NodeTuple, Param, Module


def test_node_tuple_creation():

    # Minimal creation
    n1 = NodeTuple()
    assert n1.name == "NodeTuple"
    assert len(n1) == 0

    # Creation with list of param nodes
    params = [Param("ptest1", 1), Param("ptest2", 2)]
    n2 = NodeTuple(params)
    assert len(n2) == 2
    assert n2[0] is params[0]
    assert n2.ptest1 is params[0]
    assert n2[1] is params[1]
    assert n2.ptest2 is params[1]

    # Creation with list of module nodes
    modules = [Module("mtest1"), Module("mtest2"), Module("mtest3")]
    n3 = NodeTuple(modules)
    assert len(n3) == 3
    assert n3[0] is modules[0]
    assert n3.mtest1 is modules[0]
    assert n3[1] is modules[1]
    assert n3["mtest2"] is modules[1]
    assert n3[2] is modules[2]
    assert n3["mtest3"] is modules[2]

    # Adding node tuples
    n4 = n1 + n2 + n3
    assert len(n4) == 5
    assert n4[0] is params[0]
    assert n4[1] is params[1]
    assert n4[2] is modules[0]
    assert n4[3] is modules[1]
    assert n4[4] is modules[2]

    # Check repr
    assert isinstance(repr(n4), str)
    assert "[5]" in repr(n4)

    # Check copy
    with pytest.raises(NotImplementedError):
        n4.copy()
    with pytest.raises(NotImplementedError):
        n4.deepcopy()

    # Check bad init
    with pytest.raises(TypeError):
        NodeTuple(modules + [1])

    # Check to static/dynamic
    n4.to_dynamic()
    n4.to_static()


def test_node_list_creation():

    # Minimal creation
    n1 = NodeList()
    assert n1.name.startswith("NodeList")
    assert len(n1) == 0

    # Creation with list of param nodes
    params = [Param("ptest1"), Param("ptest2")]
    n2 = NodeList(params)
    assert len(n2) == 2
    assert n2[0] is params[0]
    assert n2.ptest1 is params[0]
    assert n2[1] is params[1]
    assert n2["ptest2"] is params[1]

    # Creation with list of module nodes
    modules = [Module("mtest1"), Module("mtest2"), Module("mtest3")]
    n3 = NodeList(modules)
    assert len(n3) == 3
    assert n3[0] is modules[0]
    assert n3.mtest1 is modules[0]
    assert n3[1] is modules[1]
    assert n3["mtest2"] is modules[1]
    assert n3[2] is modules[2]
    assert n3["mtest3"] is modules[2]

    # Adding node Lists
    n4 = n1 + n2 + n3
    assert len(n4) == 5
    assert n4[0] is params[0]
    assert n4[1] is params[1]
    assert n4[2] is modules[0]
    assert n4[3] is modules[1]
    assert n4[4] is modules[2]

    # Check repr
    assert isinstance(repr(n4), str)
    assert "[5]" in repr(n4)

    # Check copy
    with pytest.raises(NotImplementedError):
        n4.copy()
    with pytest.raises(NotImplementedError):
        n4.deepcopy()

    # Check bad init
    with pytest.raises(TypeError):
        NodeList(modules + [1])
    with pytest.raises(TypeError):
        n4.append(1)


def test_node_list_manipulation():

    params = [Param("ptest1", 1), Param("ptest2", 2)]
    modules = [Module("mtest1"), Module("mtest2"), Module("mtest3")]
    n1 = NodeList(params)

    # Append
    n1.append(Param("ptest3", 3))
    assert len(n1) == 3
    assert n1[2].name == "ptest3"
    assert n1.ptest3.name == "ptest3"

    # Insert
    n1.insert(1, Module("mtest4"))
    assert len(n1) == 4
    assert n1[1].name == "mtest4"

    # Extend
    n1.extend(modules)
    assert len(n1) == 7
    assert n1[4].name == "mtest1"
    assert n1.mtest1.name == "mtest1"

    # Check to static/dynamic
    n1.to_dynamic()
    n1.to_static()

    # Clear
    n1.clear()
    assert len(n1) == 0

    # iadd
    n1 += params
    assert len(n1) == 2
    assert n1[0].name == "ptest1"

    # Pop
    assert n1.pop().name == "ptest2"
    assert len(n1) == 1

    # Remove
    n1 += modules
    n1.remove(modules[1])
    assert len(n1) == 3
    assert n1[1].name == "mtest1"

    # Set item
    n1[1] = modules[1]
    assert n1[1].name == "mtest2"
    n1[1:3] = [Module("mtest5"), Module("mtest6")]
    assert len(n1) == 3
    assert n1[1].name == "mtest5"
    assert n1[2].name == "mtest6"

    # Get item
    n2 = NodeList(modules)
    assert n2[1] is modules[1]
    assert all(n == m for n, m in zip(n2[:2], modules[:2]))
    assert all(n == m for n, m in zip(n2[1:], modules[1:]))

    # del item
    del n2[1]
    assert len(n2) == 2
    assert n2[1].name == "mtest3"

    # mul
    with pytest.raises(NotImplementedError):
        n2 * 2
    with pytest.raises(NotImplementedError):
        n2 *= 2


def test_collection_in_module():

    l1 = [Param("ptest1"), Param("ptest2"), Module("mtest1"), Module("mtest2")]
    t1 = (Param("ptest3"), Param("ptest4"), Module("mtest3"), Module("mtest4"))

    m1 = Module("test")
    m1.l = l1
    m1.t = t1

    assert m1["l"][2] == l1[2]
    assert m1["t"][2] == t1[2]
    assert m1.l[3] == l1[3]
    assert m1.t[3] == t1[3]
