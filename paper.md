---
title: 'caskade: building Pythonic scientific models'
tags:
  - Python
  - astronomy
  - inference
  - simulation
authors:
  - name: Connor Stone
    orcid: 0000-0002-9086-6398
    corresponding: true
    equal-contrib: true
    affiliation: "1, 2, 3"
  - name: Alexandre Adam
    orcid: 0000-0001-8806-7936
    affiliation: "1, 2, 3"
  - name: Adam Coogan
    orcid: 0000-0002-0055-1780
    affiliation: "1, 2, 3, a"
  - name: Laurence Perreault-Levasseur
    orcid: 0000-0003-3544-3939
    affiliation: "1, 2, 3, 4, 5, 6"
  - name: Yashar Hezaveh
    orcid: 0000-0002-8669-5733
    affiliation: "1, 2, 3, 4, 5, 6"
affiliations:
  - name:
      Ciela Institute - Montréal Institute for Astrophysical Data Analysis and
      Machine Learning, Montréal, Québec, Canada
    index: 1
  - name:
      Department of Physics, Université de Montréal, Montréal, Québec, Canada
    index: 2
  - name:
      Mila - Québec Artificial Intelligence Institute, Montréal, Québec, Canada
    index: 3
  - name:
      Center for Computational Astrophysics, Flatiron Institute, 162 5th Avenue,
      10010, New York, NY, USA
    index: 4
  - name: Perimeter Institute for Theoretical Physics, Waterloo, Canada
    index: 5
  - name: Trottier Space Institute, McGill University, Montréal, Canada
    index: 6
  - name: Work done while at UdeM, Ciela, and Mila
    index: a
date: 17 Januray 2025
bibliography: paper.bib

---

# Summary

Scientific code is now often written in Python due to its readability,
flexibility, and large package inventory. Many scientists are self taught in
programming and produce code that is difficult to read, scale, and maintain
causing a large amount of technical debt in the community. Here we present a
formalism for developing scientific simulators, and a package `caskade` which
utilizes this formalism to support highly flexible parameter representation.

# Statement of need

Python is widely used in the scientific community to develop analysis code that
can perform inference, handle complex calculations, and build flexible
simulators. Previously no unified framework existed for developing such codes and
a number of common problems needed to be solved many times in different teams.
Most simulators used in complex calculations and inference have some number of
parameters which must be managed throughout the simulator. Common problems
encountered when developing scientific simulators include passing parameters
through the simulation, selectively fixing/freeing inference parameters,
post-hoc reparametrizations, and linking parameters. Similarly, once a simulator
has been constructed, one often needs to write tedious wrappers and
modifications to interface with other scientific analysis codes. Any simulator
may be understood as a directed acyclic graph (DAG) of individual calculations,
though no framework previously existed for formalizing this conception.
`caskade` presents a highly generalized format for constructing scientific
simulators which promotes good coding practices and abstracts the passing of
parameters through the DAG model.

Originally developed for `caustics` [Stone2024], the `caskade` system is now
used in multiple scientific analysis codes.

# `caskade` Formalism

`caskade` promotes an object oriented approach to simulator construction.
Parameters that may be involved in inference are represented as `Param` objects
which are held by `Module`s. Analysis code is modularized and organized in
`Module` objects, which are Python classes that inherit from the
`caskade.Module` class. Methods of a `Module` that use `Param` values are
decorated with `@forward` to automatically collect and distribute the values.

To see this in action, let us consider the case of a star image model to
represent a star in an astronomical image.

```python
class AstroObject(Module):
    def __init__(self):
        self.x0 = Param("x0")
        self.y0 = Param("y0")

    @forward
    def center_coords(self, x, y, x0, y0):
        return x - x0, y - y0

class Star(AstroObject):
    def __init__(self):
        self.sigma = Param("sigma")
        self.brightness = Param("brightness")

    @forward
    def image(self, x, y, sigma, brightness):
        x, y = self.center_coords(x, y)
        return brightness * torch.exp(-0.5 * (x**2 + y**2) / sigma**2)
```

See how the `Module` inheritance is used to give the simulators the `caskade`
capabilities. Notice that the `Star` class can use the `center_coords` method
without needing to pass any `Param`s like the `x0` and `y0` that are needed for
the method. One can now create a `Star` instance and call the `image` method to
sample an image of a Gaussian.

```python
coords = torch.linspace(-1,1,100)
X, Y = torch.meshgrid(coords, coords, "xy")
mystar = Star()
#                      x0   y0    brightness
params = torch.tensor([0.1, -0.2, 1.0])
mystar.sigma = 0.3
image = mystar.image(X, Y, params)
```

Notice that we pass all the registered `Param` values when calling the
`Star.image` method; the `@forward` decorator collects and organizes them such
that the values go to the correct places. The `sigma` parameter is fixed to
`0.3` and so does not need to be passed at the call to the `image` method. One
may now make simulators for other astronomical objects like galaxies and by
conforming to the above coding pattern the analysis will remain scalable and
flexible. If one now wishes to make `mystar` move across the sky as a function
of time, one would traditionally need to rewrite some elements of the code,
however with `caskade` this may be accomplished straightforwardly.

```python
t = Param("t")
mystar.x0 = lambda p: -2 * p["t"].value
mystar.x0.link(t)
mystar.y0 = lambda p: 0.5 * p["t"].value
mystar.y0.link(t)
#                      t    brightness
params = torch.tensor([1.5, 1.0])
image = mystar.image(X, Y, params)
```

Now instead of providing `x0` and `y0` in the `params` tensor, one provides the
time `t` and the position will be automatically computed using the provided
functions. Many more capabilities of `caskade` make tuning, scaling, and
maintaining scientific code far easier. As a further benefit, all `caskade`
simulators can interface with each other as larger simulators with all
parameters passing as expected. Thus one may join an "ecosystem" of powerful
Pythonic scientific code.