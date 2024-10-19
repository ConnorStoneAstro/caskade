import torch

from caskade import Module, Param, forward


def test_full_integration():

    class TestSim(Module):
        def __init__(self, a, b, c, c_shape, m1):
            super().__init__("test_sim")
            self.a = a
            self.b = Param("b", b)
            self.c = Param("c", c, c_shape)
            self.m1 = m1

        @forward
        def testfun(self, x, b=None, c=None):
            c.value = b + x
            y = self.m1()
            return x + self.a + b + y

    class TestSubSim(Module):
        def __init__(self, d, e, f):
            super().__init__("test_sub_sim")
            self.d = Param("d", d)
            self.e = Param("e", e)
            self.f = Param("f", f)

        @forward
        def __call__(self, d=None, e=None, f=None):
            return d + e + f

    sub1 = TestSubSim(d=1.0, e=lambda s: s.children["flink"].value, f=None)
    sub1.e.link("flink", sub1.f)
    main1 = TestSim(a=2.0, b=None, c=None, c_shape=(), m1=sub1)
    main1.c = main1.b
    sub1.f = main1.c

    main1.to(dtype=torch.float32)

    b_value = torch.tensor(3.0)
    res = main1.testfun(1.0, params=[b_value])
    assert res.item() == 13.0


def test_full_integration_v2():
    class MyMainSim(Module):
        def __init__(self, a_utility, b_action, c_param=None):
            super().__init__()
            self.a_utility = a_utility  # This will hold a module
            self.b_action = b_action  # this is a list of modules, so we have to link them manually
            for sim in self.b_action:
                self.link(sim.name, sim)
            self.c_param = Param("c", c_param)  # regular parameter
            self.d_param = Param("d", None)  # live parameter

        @forward
        def mymainfunction(
            self, x, c_param=None, d_param=None
        ):  # note we use the attribute name, not the parameter name
            u = self.a_utility.myutilityfunction(x + 2)
            s = self.mysecondfunction(u + d_param)
            for sim in self.b_action:
                s = s + sim.myactionfunction(s)
            return s * c_param

        @forward
        def mysecondfunction(self, y, d_param=None):
            u = self.a_utility.myutilityfunction(y + 2)
            return u + y + d_param

    class MyActionSim(Module):
        def __init__(self, a_utility, a=None, b=None):
            super().__init__()
            self.a_utility = a_utility  # same module as in MyMainSim
            self.a = Param("a", a)
            self.b = Param("b", b)

        @forward
        def myactionfunction(self, w, a=None, b=None):
            u = self.a_utility.myutilityfunction(w + 3)
            return u * a + b

    class MyUtilitySim(Module):
        def __init__(self, u=None):
            super().__init__()
            self.u = Param("u", u)

        @forward
        def myutilityfunction(self, z, u=None):
            return u * z

    util = MyUtilitySim()
    #                      u for MyUtilitySim
    params = [torch.tensor(1.0)]
    actions = []
    for i in range(3):
        actions.append(MyActionSim(util))
        #                     a for MyActionSim, b for MyActionSim
        params = params + [torch.tensor(i), torch.tensor(i + 1)]

    main = MyMainSim(util, actions)

    main.d_param = lambda p: p["utility u"].myutilityfunction(p["c"].value) * 2
    main.d_param.link("utility u", util)
    main.d_param.link("c", main.c_param)

    #                      c for MyMainSim
    params = params + [torch.tensor(3.0)]

    assert main.mymainfunction(1.0, params).item() == 558.0

    graph = main.graphviz()
    assert graph is not None, "should return a graphviz object"
