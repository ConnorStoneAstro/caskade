import itertools
import numpy as np

from caskade import Node, Param, Module, forward, backend
import pytest


@pytest.fixture
def node_graph():

    a = Node("a", description="node a")
    b = Node("b", description="node b")
    c = Node("c", description="node c")
    d = Node("d", description="node d")
    e = Node("e", description="node e")
    f = Node("f", description="node f")
    g = Node("g", description="node g")

    a.link((b, c))
    a.link(e)

    b.link("d1", d)
    b.link(e)

    c.link(("d", "eff"), (d, f))

    f.link(g)

    g.link(e)

    return a, b, c, d, e, f, g


#####################################################################

# 1. Define your parameter domains
names = [None, "test_param"]
values = [None, 1, [1.0, 2.0]]
shapes = [None, (), (2,)]
cyclics = [True, False]
valids = [None, (0, None), (None, 3), (0, 3)]
units_list = [None, "arcsec"]
dynamics = [None, True, False]
groups = [0, 1]

# 2. Generate the Cartesian product (all possible combinations)
# This creates a list of tuples: [(None, None, None, True...), (None, None, None, False...), ...]
param_combinations = list(
    itertools.product(names, values, shapes, cyclics, valids, units_list, dynamics, groups)
)

# 3. Remove invalid combinations
i = 0
while i < len(param_combinations):
    if param_combinations[i][3] and param_combinations[i][4] != [0, 3]:
        param_combinations.pop(i)
        continue
    if param_combinations[i][1] == 1 and param_combinations[i][2] == (2,):
        param_combinations.pop(i)
        continue
    i += 1


# 4. Pass the combinations to the fixture
@pytest.fixture(params=param_combinations)
def many_param(request):
    # 4. Unpack the tuple from request.param
    name, value, shape, cyclic, valid, units, dynamic, group = request.param

    # Return your initialized object
    return (
        Param(name, value, shape, cyclic, valid, units, dynamic, group),
        name,
        value,
        shape,
        cyclic,
        valid,
        units,
        dynamic,
        group,
    )


#####################################################################


@pytest.fixture
def sim():
    class Helper(Module):
        def __init__(self, h1=1, h2=(2, 3), name=None):
            super().__init__(name)
            self.h1 = Param("h1", h1, shape=())
            self.h2 = Param("h2", h2, shape=(2,))

        @forward
        def get_help(self, x, h1, h2):
            return backend.sum(x + h1 + h2)

    class Worker(Module):
        def __init__(self, helper: Helper, w1=4, w2=[[5, 6], [7, 8]], name=None):
            super().__init__(name)
            self.helper = helper
            self.w1 = Param("w1", w1, shape=())
            self.w2 = Param("w2", w2, shape=(2, 2))

        @forward
        def sub_work(self, w2):
            return backend.sum(w2)

        @forward
        def do_work(self, a, w1, w2):
            return backend.sum(a + w1 + w2) + self.sub_work() + self.helper.get_help(a + w1)

    class Simulator(Module):
        def __init__(self, helper: Helper, workers: list[Worker], s1=9, name=None):
            super().__init__(name)
            self.helper = helper
            self.workers = workers
            self.s1 = Param("s1", s1, shape=())

        @forward
        def sub_sim(self, s1):
            return s1

        @forward
        def run_sim(self, helper, a):
            return (
                self.sub_sim()
                + self.helper.get_help(helper)
                + sum(worker.do_work(a) for worker in self.workers)
            )

    H = Helper()
    sim = Simulator(H, list(Worker(H, name=f"worker_{i}") for i in range(5)), name="sim")
    sim.workers[3].w1 = sim.workers[0].w1
    return sim


#####################################################################


@pytest.fixture
def hierarchical_sim():
    if backend.backend != "torch":
        pytest.skip()

    class Helper(Module):
        def __init__(self, h1=1, h2=(2, 3), name=None):
            super().__init__(name)
            self.h1 = Param("h1", h1, shape=())
            self.h2 = Param("h2", h2, shape=(2,))

        @forward
        def get_help(self, x, h1, h2):
            return backend.sum(x + h1 + h2)

    class Worker(Module):
        def __init__(self, helper: Helper, w1=4, w2=[[5, 6], [7, 8]], name=None):
            super().__init__(name)
            self.helper = helper
            self.w1 = Param("w1", w1, shape=())
            self.w2 = Param("w2", w2, shape=(2, 2))

        @forward
        def sub_work(self, w2):
            return backend.sum(w2)

        @forward
        def do_work(self, a, w1, w2):
            return backend.sum(a + w1 + w2) + self.sub_work() + self.helper.get_help(a + w1)

    class Simulator(Module):
        def __init__(self, helper: Helper, worker: Worker, s1=9, name=None):
            super().__init__(name)
            self.helper = helper
            self.hierarchical_link("worker", worker)
            self.s1 = Param("s1", s1, shape=())

        @forward
        def sub_sim(self, a, s1, worker_params, worker_dims):
            batched_worker = backend.module.vmap(self.worker.do_work, in_dims=(None, worker_dims))
            return s1 + batched_worker(a, worker_params).sum()

        @forward
        def run_sim(self, helper, a):
            return self.sub_sim(a) + self.helper.get_help(helper)

    H = Helper()
    sim = Simulator(H, Worker(H, name="worker"), name="sim")
    sim.worker.w1 = 4 * np.ones(5)
    assert sim.worker.w1.batched
    sim.worker.w2 = np.array([[5, 6], [7, 8]]) * np.ones((5, 2, 2))
    assert sim.worker.w2.batched
    return sim
