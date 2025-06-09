---
title: 'caskade: building Pythonic scientific simulators'
tags:
  - Python
  - astronomy
  - inference
  - simulation
authors:
  - name: Connor Stone
    orcid: 0000-0002-9086-6398
    corresponding: true
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
date: 2 June 2025
bibliography: paper.bib

---

# Summary

Scientific simulators and pipelines form the core of many research projects.
Writing high quality, modular code allows for efficiently scaling a project, but
this can be challenging in a research context. Research project goals and
solutions to those goals are constantly in flux, often requiring many
refactoring rounds to meet these changes. The result can be a progressively more
unwieldy interconnected code; we dub this, "the args problem" in scientific
simulator construction. Here we present a system, `caskade`, which allows users
to focus on modular components of a simulator, these are small and testable to
ensure robustness. With `caskade` one can turn these modular components into
abstracted blocks that stack to form a powerful simulator. This is inspired by
the `PyTorch` framework `nn.Module` which allows for near effortless
construction of machine learning models. We generalize the object oriented
framework to apply to almost any scientific forward model, simulator, analysis
pipeline, and so on; `caskade` manages the flow of parameters through these
models.

# Features

The core features of `caskade` are the `Module` base class, `Param` parameter,
and `forward` decorator. To construct a `caskade` simulator, one subclasses
`Module` then adds some number of `Param` objects as attributes of the class,
finally any number of class methods may be decorated with `@forward`. As modules
are combined into a larger simulator, `caskade` builds a directed acyclic graph
(DAG) representation. This allows it to automatically manage the flow (cascade)
of parameters through the simulator and encode arbitrary relationships between
them. In some ways `caskade` is reminiscent of `Hydra` [@hydra], however `Hydra`
efficiently builds configuration files for arbitrary applications while
`caskade` focuses on numerical parameters and scientific inference. The two may
even be used in tandem.

For the `Param` values managed by `caskade`, a number of features are available
which are designed to be useful in inference and analysis contexts. Any
parameter may be transformed between "static" and "dynamic" where static has a
fixed value and dynamic is provided at call time. The dynamic parameters are
those that would be sampled or optimized using external packages like emcee
[@emcee], scipy.optimize [@scipy], Pyro [@pyro], dynesty [@dynesty], torch.Optim
[@pytorch], etc. `caksade` can automatically unravel and concatenate all
parameters into a single 1D vector for easy interfacing with these external
packages. Individual parameters, whole modules, or whole simulators may be
switched between static and dynamic. Parameters may be synced with arbitrary
functional relationships between them. New parameters may be added dynamically
to allow for sophisticated transformations. For example, an entire simulator may
be turned into a function of time without modifying the underlying simulator by
adding a time parameter and linking appropriately. 

![Example `caksade` DAG representation of a gravitational lensing simulator. Ovals represent Modules, boxes represent parameters, arrow boxes represent parameters which are functionally dependent on another parameter, and thin arrows show the direction of the graph flow for parameters passed at the top level.\label{fig:graph}](media/model_graph.png)

Many more creative uses of the dynamic parameter management system have been
tested, with positive results. Note that `caksade` simply manages how parameters
enter class methods and so the user has complete freedom to design the internals
of the simulator. In fact one could write thin wrapper classes over an existing
code base to quickly allow access to `caskade` parameter management. Our
suggested design flow is to build out a functional programming base for the
package, then use `Module`s as wrappers for the functional base to design a
convenient user interface. This design encourages modular development and is
supportive of users who wish to expand functionality at different levels (core
functionality or interface level). The `caustics` package [@Stone2024]
implements this code design to great effect. \autoref{fig:graph} shows an
example `caskade` graph[^1] from `caustics`. In this graph the redshift
parameters of each lens are linked to ensure consistent evaluation despite the
functional backed having no explicit enforcement of this.

[^1]: visual generated by `graphviz` [@graphviz]

`caskade` has grown beyond its initial development goal of supporting `caustics`
and now includes many parameter related convenience features. There is a utility
to save sampling chains for a simulator into HDF5 format and it is possible to
load the state of the simulator back to a given point. It is now possible to use
`caskade` with `NumPy` [@numpy], `JAX` [@jax], or `PyTorch` [@pytorch] numerical
backends. Parameters may be extracted/provided as a 1D array, a list, or a
dictionary. One may store metadata alongside parameters. Many more convenience
features make for a seamless experience. One critical aspect of `caskade` is
that it is rigorously tested to ensure reliability. We maintain 100% unit
testing coverage over the entire code base, and provide highly informative error
messages to users. These features give users confidence to push the limits of
their simulators and iterate quickly while doing so.

# Acknowledgements

This research was enabled by a generous donation by Eric and Wendy Schmidt with
the recommendation of the Schmidt Futures Foundation. CS acknowledges the
support of a NSERC Postdoctoral Fellowship and a CITA National Fellowship. This
research was enabled in part by support provided by Calcul Québec and the
Digital Research Alliance of Canada. The work of A.A. was partially funded by
NSERC CGS D scholarships. Y.H. and L.P. acknowledge support from the National
Sciences and Engineering Council of Canada grants RGPIN-2020-05073 and 05102,
the Fonds de recherche du Québec grants 2022-NC-301305 and 300397, and the
Canada Research Chairs Program. 