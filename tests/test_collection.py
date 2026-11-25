import os
import pickle
from collections.abc import MutableMapping, MutableSequence, Sequence

import pytest

from caskade import (
    ActiveContext,
    GraphError,
    NodeList,
    NodeTuple,
    NodeDict,
    Param,
    Module,
    backend,
    forward,
    ValidContext,
    FillParamsArrayError,
    FillParamsSequenceError,
    FillParamsMappingError,
)


@pytest.mark.parametrize("node_type", [NodeTuple, NodeList])
def test_node_collection_creation(node_type):

    # Minimal creation
    n1 = node_type()
    assert n1.name.startswith(node_type.__name__)
    assert len(n1) == 0

    # Creation with list of param nodes
    params = [Param("ptest1"), Param("ptest2")]
    n2 = node_type(params)
    assert len(n2) == 2
    assert n2[0] is params[0]
    assert n2.ptest1 is params[0]
    assert n2[1] is params[1]
    assert n2["ptest2"] is params[1]

    # Creation with list of module nodes
    modules = [Module("mtest1"), Module("mtest2"), Module("mtest3")]
    n3 = node_type(modules)
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

    # Make a slice
    assert isinstance(n4[1:4], node_type)

    # Check repr
    assert isinstance(repr(n4), str)
    assert "[5]" in repr(n4)

    # Check to static/dynamic
    n4.to_dynamic(False)
    assert len(n4.static_params) == 0
    n4.to_static(False)
    assert len(n4.static_params) == 2
    assert len(n4.pointer_params) == 0

    # Graphviz
    graph = n4.graphviz(saveto="test_graph.pdf")
    assert graph is not None, "should return a graphviz object"
    assert os.path.exists("test_graph.pdf")
    os.remove("test_graph.pdf")

    # Check copy
    with pytest.raises(NotImplementedError):
        n4.copy()
    with pytest.raises(NotImplementedError):
        n4.deepcopy()

    # Check bad init
    with pytest.raises(TypeError):
        node_type(modules + [1])
    if "List" in node_type.__name__:
        with pytest.raises(TypeError):
            n4.append(1)
    else:
        with pytest.raises(AttributeError):
            n4.append(1)


@pytest.mark.parametrize("node_type", [NodeTuple, NodeList])
def test_node_collection_param_values(node_type):
    NL = node_type([Param("p1"), Param("p2"), Param("p3")])

    NL.set_values([1, 2, 3])

    assert NL[0].value.item() == 1.0
    assert NL[1].value.item() == 2.0
    assert NL[2].value.item() == 3.0


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
    d1 = {"ptest5": Param("ptest5"), "ptest6": Param("ptest6"), "mtest5": Module("mtest5")}

    m1 = Module("test")
    m1.l = l1
    m1.t = t1
    m1.d = d1

    assert m1["l"][2] == l1[2]
    assert m1["t"][2] == t1[2]
    assert m1.l[3] == l1[3]
    assert m1.t[3] == t1[3]
    assert m1["d"]["ptest5"] == d1["ptest5"]
    assert m1.d["mtest5"] == d1["mtest5"]


@pytest.mark.parametrize("node_type", [NodeTuple, NodeList])
def test_collection_fill(node_type):
    NL = node_type([Param("p1"), Param("p2"), Param("p3")])

    # Bad params
    with pytest.raises(TypeError):
        NL.set_values(lambda p: "bad idea")

    # List params
    NL.set_values([1, 2, 3])
    NL.set_values([])
    with pytest.raises(FillParamsSequenceError):
        NL.set_values([1, 2])

    # Dict params
    NL.set_values({"p1": 4, "p2": 5, "p3": 6})
    with pytest.raises(FillParamsMappingError):
        NL.set_values({"p1": 4, "p2": 5, "p3": 6, "p4": 7})

    # Array params
    NL.set_values(backend.as_array([7, 8, 9]))
    NL.set_values(backend.as_array([]))

    with pytest.raises(FillParamsArrayError):
        NL.set_values(backend.as_array([7, 8]))

    with pytest.raises(FillParamsArrayError):
        NL.set_values(backend.as_array([7, 8, 9, 10]))

    NL[1].to_static()
    NL.set_values(backend.as_array([7, 8]), "dynamic")
    NL[1].to_dynamic()

    NL[1].value = None
    NL[1].shape = None
    NL.set_values(backend.as_array([7, 8, 9]))
    with pytest.raises(FillParamsArrayError):
        NL.set_values(backend.as_array([7, 8]))


@pytest.mark.parametrize("group", [0, 1])
@pytest.mark.parametrize("params_type", ["array", "list", "dict"])
def test_valid_list(node_list, params_type, group):
    node_list.to_dynamic(False)

    node_list[2].helper.h1.group = group
    for i in range(5):
        node_list[1].workers[i].w1.group = i * group

    init_params = node_list.get_values()

    round_trip_params = node_list.from_valid(node_list.to_valid(init_params))

    with ValidContext(node_list):
        params = node_list.get_values(params_type)
        node_list.set_values(params)

    if group == 0:
        assert backend.module.allclose(init_params, round_trip_params)
        assert backend.module.allclose(init_params, node_list.get_values())
    else:
        assert len(node_list.dynamic_param_groups) > 1
        final_params = node_list.get_values()
        for i in range(len(node_list.dynamic_param_groups)):
            assert backend.module.allclose(init_params[i], round_trip_params[i])
            assert backend.module.allclose(init_params[i], final_params[i])


@pytest.mark.parametrize("group", [0, 1])
@pytest.mark.parametrize("params_type", ["array", "list", "dict"])
def test_valid_tuple(node_tuple, params_type, group):
    node_tuple.to_dynamic(False)

    node_tuple[2].helper.h1.group = group
    for i in range(5):
        node_tuple[1].workers[i].w1.group = i * group

    init_params = node_tuple.get_values()

    round_trip_params = node_tuple.from_valid(node_tuple.to_valid(init_params))

    with ValidContext(node_tuple):
        params = node_tuple.get_values(params_type)
        node_tuple.set_values(params)

    if group == 0:
        assert backend.module.allclose(init_params, round_trip_params)
        assert backend.module.allclose(init_params, node_tuple.get_values())
    else:
        assert len(node_tuple.dynamic_param_groups) > 1
        final_params = node_tuple.get_values()
        for i in range(len(node_tuple.dynamic_param_groups)):
            assert backend.module.allclose(init_params[i], round_trip_params[i])
            assert backend.module.allclose(init_params[i], final_params[i])


def test_node_tuple_immutable():
    params = [Param("p1"), Param("p2"), Param("p3")]
    modules = [Module("m1"), Module("m2"), Module("m3")]
    nt = NodeTuple(params + modules)

    # Attempt to modify the NodeTuple
    with pytest.raises(TypeError):
        nt[0] = Param("new_param")

    with pytest.raises(AttributeError):
        nt.append(Param("new_param"))

    with pytest.raises(AttributeError):
        nt.extend([Module("new_module")])

    with pytest.raises(AttributeError):
        nt.insert(1, Param("new_param"))

    with pytest.raises(TypeError):
        del nt[0]

    with pytest.raises(AttributeError):
        nt.pop()

    with pytest.raises(AttributeError):
        nt.remove(modules[0])

    with pytest.raises(TypeError):
        nt.link(Module("new_module"))


def test_node_dict_creation():

    # Minimal creation
    n1 = NodeDict()
    assert n1.name.startswith("NodeDict")
    assert len(n1) == 0

    # Creation with dict of param nodes
    params = {"p1": Param("p1"), "p2": Param("p2")}
    n2 = NodeDict(params)
    assert len(n2) == 2
    assert n2["p1"] is params["p1"]
    assert n2.p1 is params["p1"]
    assert n2["p2"] is params["p2"]

    # Creation with dict of module nodes
    modules = {"m1": Module("m1"), "m2": Module("m2"), "m3": Module("m3")}
    n3 = NodeDict(modules)
    assert len(n3) == 3
    assert n3["m1"] is modules["m1"]
    assert n3.m1 is modules["m1"]
    assert n3["m2"] is modules["m2"]

    # Check repr
    assert isinstance(repr(n3), str)
    assert "[3]" in repr(n3)

    # Check to static/dynamic
    n2.to_dynamic(False)
    assert len(n2.static_params) == 0
    n2.to_static(False)
    assert len(n2.static_params) == 2
    assert len(n2.pointer_params) == 0

    # Graphviz
    graph = n3.graphviz(saveto="test_graph_dict.pdf")
    assert graph is not None, "should return a graphviz object"
    assert os.path.exists("test_graph_dict.pdf")
    os.remove("test_graph_dict.pdf")

    # Check copy
    with pytest.raises(NotImplementedError):
        n3.copy()
    with pytest.raises(NotImplementedError):
        n3.deepcopy()

    # Check bad init
    with pytest.raises(TypeError):
        NodeDict({"bad": 1})


def test_node_dict_manipulation():

    params = {"p1": Param("p1", 1), "p2": Param("p2", 2)}
    modules = {"m1": Module("m1"), "m2": Module("m2"), "m3": Module("m3")}
    nd = NodeDict(params)

    # Set item
    p3 = Param("p3", 3)
    nd["p3"] = p3
    assert len(nd) == 3
    assert nd["p3"] is p3
    assert nd.p3 is p3

    # Update
    nd.update(modules)
    assert len(nd) == 6
    assert nd["m1"] is modules["m1"]
    assert nd.m1 is modules["m1"]

    # Pop
    popped = nd.pop("m3")
    assert popped is modules["m3"]
    assert len(nd) == 5
    assert "m3" not in nd

    # Del item
    del nd["m2"]
    assert len(nd) == 4
    assert "m2" not in nd

    # Popitem
    key, _ = nd.popitem()
    assert key not in nd

    # Clear
    nd.clear()
    assert len(nd) == 0

    # Setdefault
    nd2 = NodeDict({"p1": Param("p1")})
    p_new = Param("p_new")
    nd2.setdefault("new_key", p_new)
    assert nd2["new_key"] is p_new
    assert len(nd2) == 2
    # setdefault should not overwrite existing
    existing = nd2["p1"]
    nd2.setdefault("p1", Param("p1_other"))
    assert nd2["p1"] is existing
    with pytest.raises(TypeError):
        nd2.setdefault("bad_key", "not a node")

    # Check to static/dynamic
    nd3 = NodeDict({"p1": Param("p1", 1), "p2": Param("p2", 2)})
    nd3.to_dynamic()
    nd3.to_static()

    # dynamic property
    assert nd3.static
    assert not nd3.dynamic
    nd3.to_dynamic()
    assert nd3.dynamic
    assert not nd3.static

    # Update with kwargs
    nd4 = NodeDict({"p1": Param("p1")})
    m_kw = Module("mkw")
    nd4.update(mkw=m_kw)
    assert len(nd4) == 2
    assert nd4["mkw"] is m_kw
    assert nd4.mkw is m_kw

    # mul raises NotImplementedError
    with pytest.raises(NotImplementedError):
        nd3 * 2


def test_node_dict_param_values():
    nd = NodeDict({"p1": Param("p1"), "p2": Param("p2"), "p3": Param("p3")})

    nd.set_values([1, 2, 3])

    assert nd["p1"].value.item() == 1.0
    assert nd["p2"].value.item() == 2.0
    assert nd["p3"].value.item() == 3.0


def _in_step(collection, top=None):
    """The graph matches the contents of the collection exactly."""
    nodes = list(collection.values() if isinstance(collection, NodeDict) else collection)
    assert list(collection.children.values()) == list(dict.fromkeys(nodes))
    for node in nodes:
        assert collection in node.parents
    if isinstance(collection, NodeDict):
        assert all(getattr(collection, key) is node for key, node in collection.items())
    else:
        assert all(getattr(collection, node.name) is node for node in nodes)
    if top is not None:  # the order of the params follows the order of the collection
        assert list(top.static_params) == list(dict.fromkeys(nodes))


def test_collections_are_abstract_containers():
    assert isinstance(NodeTuple(), Sequence) and not isinstance(NodeTuple(), MutableSequence)
    assert isinstance(NodeList(), MutableSequence)
    assert isinstance(NodeDict(), MutableMapping)
    assert not isinstance(NodeTuple(), tuple)
    assert not isinstance(NodeList(), list)
    assert not isinstance(NodeDict(), dict)


def test_node_list_every_method_updates_graph():
    a, b, c, d, e = (Param(name, i) for i, name in enumerate("abcde"))
    nl = NodeList([a, b, c], name="nl")
    top = Module("top")
    top.items = nl
    _in_step(nl, top)

    nl.reverse()  # none of these were kept in step when NodeList was a ``list``
    assert list(nl) == [c, b, a]
    _in_step(nl, top)
    nl += [d]
    nl.insert(1, nl.pop())
    assert list(nl) == [c, d, b, a]
    _in_step(nl, top)
    nl[0:2] = nl[0:2][::-1]
    assert list(nl) == [d, c, b, a]
    _in_step(nl, top)
    del nl[::2]
    assert list(nl) == [c, a]
    _in_step(nl, top)
    assert nl not in b.parents and nl not in d.parents and not hasattr(nl, "b")
    nl[0] = e
    assert list(nl) == [e, a] and nl not in c.parents
    _in_step(nl, top)
    nl.extend((b, c))
    nl.remove(a)
    assert list(nl) == [e, b, c] and nl.index(b) == 1 and nl.count(c) == 1 and a not in nl
    _in_step(nl, top)
    assert [n.name for n in reversed(nl)] == ["c", "b", "e"]
    nl.clear()
    assert len(nl) == 0 and not nl.children and not top.static_params

    # adding makes a new list, from either side
    nl.extend((a, b))
    assert list(nl + [c]) == [a, b, c] and isinstance(nl + (c,), NodeList)
    assert list([c] + nl) == [c, a, b] and isinstance([c] + nl, NodeList)
    with pytest.raises(TypeError):
        nl + 1


def test_node_list_failed_change_changes_nothing():
    a, b = Param("a", 1), Param("b", 2)
    nl = NodeList([a, b], name="nl")
    top = Module("top")
    top.items = nl

    changes = [
        (TypeError, lambda: nl.append(3)),
        (TypeError, lambda: nl.extend([Param("z"), 3])),  # nothing is added, not even z
        (GraphError, lambda: nl.append(Param("a"))),  # another node with a name in use
        (GraphError, lambda: nl.extend([Param("z"), top])),  # would make a cycle
        (TypeError, lambda: nl.link(Param("q"))),
        (TypeError, lambda: nl.unlink(a)),
        (TypeError, lambda: setattr(nl, "q", Param("q"))),
        (TypeError, lambda: setattr(nl, "a", 3)),
        (TypeError, lambda: delattr(nl, "a")),
        (TypeError, lambda: nl.__setitem__("a", Param("q"))),
    ]
    for error, change in changes:
        with pytest.raises(error):
            change()
        assert list(nl) == [a, b]
        _in_step(nl, top)

    with ActiveContext(top):
        with pytest.raises(GraphError):
            nl.reverse()
    assert list(nl) == [a, b]
    _in_step(nl, top)


def test_node_sequence_repeated_node():
    a, b = Param("a", 1), Param("b", 2)
    for node_type in (NodeTuple, NodeList):
        ns = node_type([a, b, a])
        assert len(ns) == 3 and ns.count(a) == 2
        assert list(ns.children.values()) == [a, b]  # a single child
    nl = NodeList([a, b, a])
    nl.pop()
    assert list(nl) == [a, b] and nl in a.parents
    nl.pop(0)
    assert list(nl) == [b] and nl not in a.parents


def test_node_tuple_fixed():
    a, b = Param("a", 1), Param("b", 2)
    nt = NodeTuple([a, b], name="nt")
    for change in (
        lambda: nt.unlink(a),
        lambda: setattr(nt, "c", Param("c")),
        lambda: setattr(nt, "a", 3),
        lambda: delattr(nt, "a"),
    ):
        with pytest.raises(TypeError):
            change()
        assert list(nt) == [a, b]
        _in_step(nt)
    assert not hasattr(nt, "reverse") and not hasattr(nt, "extend")
    assert (nt + [a])[2] is a and isinstance(nt + nt, NodeTuple)


def test_node_dict_every_method_updates_graph():
    a, b, c, d = (Param(name, i) for i, name in enumerate("abcd"))
    nd = NodeDict({"x": a, "y": b, "z": c}, name="nd")
    top = Module("top")
    top.items = nd
    _in_step(nd, top)

    nd["y"] = d  # replacing a node keeps its place, as for a dict
    assert list(nd.items()) == [("x", a), ("y", d), ("z", c)]
    assert nd not in b.parents
    _in_step(nd, top)
    assert nd.popitem() == ("z", c)  # the last one, as for a dict
    assert nd.get("nope") is None and "x" in nd and "z" not in nd
    assert list(nd.keys()) == ["x", "y"] and list(nd.values()) == [a, d]
    _in_step(nd, top)

    with pytest.raises(TypeError):
        nd.update({"u": b, "v": 3})  # nothing is added, not even u
    with pytest.raises(GraphError):
        nd["w"] = d  # one node under two keys
    for change in (
        lambda: nd.link("q", Param("q")),
        lambda: nd.unlink("x"),
        lambda: setattr(nd, "q", Param("q")),
        lambda: setattr(nd, "x", 3),
        lambda: delattr(nd, "x"),
    ):
        with pytest.raises(TypeError):
            change()
    assert list(nd.items()) == [("x", a), ("y", d)]
    _in_step(nd, top)

    nd.update([("u", b)], v=c)
    assert list(nd) == ["x", "y", "u", "v"]
    _in_step(nd, top)
    with pytest.raises(KeyError):
        NodeDict().popitem()


class PickleSim(Module):  # pickle needs the class to be defined at the top of a file
    def __init__(self):
        super().__init__("sim")
        self.x = Param("x", 2.0)
        self.y = Param("y", self.x)  # pointer to another param
        self.l = [Param("l1", 1.0), Param("l2", 2.0)]
        self.t = (Param("t1", 3.0),)
        self.d = {"k": Param("k", 5.0)}

    @forward
    def total(self, x, y):
        return x + y + sum(p.value for p in self.l) + self.t[0].value + self.d["k"].value


def test_pickle_collections_and_pointers():
    sim = PickleSim()
    copy = pickle.loads(pickle.dumps(sim))
    assert sim.total().item() == copy.total().item() == 15.0
    assert isinstance(copy.l, NodeList) and isinstance(copy.t, NodeTuple)
    assert isinstance(copy.d, NodeDict)
    assert copy.y.pointer and copy.x is not sim.x
    for collection in (copy.l, copy.t, copy.d):
        _in_step(collection)

    # the copy is a working graph of its own
    copy.x = 10.0
    copy.l.append(Param("l3", 100.0))
    assert copy.total().item() == 131.0 and sim.total().item() == 15.0
    assert [p.name for p in copy.static_params] == ["x", "l1", "l2", "l3", "t1", "k"]
