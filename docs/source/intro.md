# caskade

[![CI](https://github.com/ConnorStoneAstro/caskade/actions/workflows/ci.yml/badge.svg)](https://github.com/ConnorStoneAstro/caskade/actions/workflows/ci.yml)
[![CD](https://github.com/ConnorStoneAstro/caskade/actions/workflows/cd.yml/badge.svg)](https://github.com/ConnorStoneAstro/caskade/actions/workflows/cd.yml)
[![codecov](https://codecov.io/gh/ConnorStoneAstro/caskade/graph/badge.svg?token=7YAEJJZ4GM)](https://codecov.io/gh/ConnorStoneAstro/caskade)
[![PyPI - Version](https://img.shields.io/pypi/v/caskade)](https://pypi.org/project/caskade/)
[![Documentation Status](https://readthedocs.org/projects/caskade/badge/?version=latest)](https://caskade.readthedocs.io/en/latest/?badge=latest)

Build scientific simulators, treating them as a directed acyclic graph. Handles
argument passing for complex nested simulators.

## Install

``` bash
pip install caskade
```

## Usage

Make a `Module` object which may have some `Param`s. Define a `forward` method
using the decorator.

``` python
from caskade import Module, Param, forward

class MySim(Module):
    def __init__(self, a, b=None):
        super().__init__()
        self.a = a
        self.b = Param("b", b)

    @forward
    def myfun(self, x, b=None):
        return x + self.a + b
```

We may now create instances of the simulator and pass the dynamic parameters.

``` python
import torch

sim = MySim(1.0)

params = [torch.tensor(2.0)]

print(sim.myfun(3.0, params=params))
```

Which will print `6` by automatically filling `b` with the value from `params`.

### Why do this?

The above example is not very impressive, the real power comes from the fact
that `Module` objects can be nested arbitrarily making a much more complicated
analysis graph. Further, the `Param` objects can be linked or have other complex
relationships. All of the complexity of the nested structure and argument
passing is abstracted away so that at the top one need only pass a list of
tensors for each parameter, a single large 1d tensor, or a dictionary with the
same structure as the graph.

## Documentation

For a quick start, jump right to the [Jupyter notebook
tutorial](notebooks/BeginnersGuide.ipynb)!