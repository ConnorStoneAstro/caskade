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
Most scientists writing such code are primarily self-taught and tend to produce
software that scales poorly and is difficult to follow, colloquially known as
"spaghetti code". A significant reason for these development challenges is the
ever evolving nature of a research project as goals change and progressively
more realism is added to a simulator. Chief among these is the need to manage
parameters for the model and ensuring they pass correctly from input to the
downstream functions that use them. Over time some parameters may need to be
fixed, share values, share complex relations, take multiple values, change
unit/coordinate systems, and more. Refactoring code to manually enforce these
dynamic relationships is time consuming and error prone. Further, one must also
often write wrappers to interface with other codes that expect alternate
parameter formats. We dub this the "args problem" as managing arguments to a
simulator often requires considerable refactoring for what should be small
changes. Here we present a fully featured solution to the args problem:
`caskade`, which represents any scientific simulator as a directed acyclic graph
(DAG).

# Features

The core features of `caskade` are the `Module` base class, `Param`
registration, and `forward` decorator. To construct a `caskade` simulator, one
subclasses `Module` then add some number of `Param` objects as attributes of the
class, finally any number of functions may be decorated with `forward` and any
parameters that have been registered with `Param` objects will be managed
automatically. 

When any class method is decorated by `@forward`, one need only provide the
non-`Param` arguments (if any), `caskade` will manage the `Param` values. Any
parameter may be transformed between "static" and "dynamic" where static has a
fixed value and dynamic must be provided at call time. The dynamic parameters
are those that would be sampled or optimized using external packages like emcee
[@emcee], scipy.optimize [@scipy], Pyro [@pyro], dynesty [@dynesty], torch.Optim [@pytorch], etc.
`caksade` will automatically unravel all parameters into a single 1D vector for
easy interfacing with these external packages. Individual parameters, whole
modules, or whole simulators may be switched between static and dynamic.
Parameters from multiple models may be synced. New parameters may be added
dynamically to allow for coordinate/unit transformations. An entire simulator
may even be turned from a function of many parameters into a function of time
without modifying the underlying simulator by adding a time parameter. 

![Example `caksade` DAG representation of a gravitational lensing simulator. Ovals represent Modules, boxes represent parameters, arrow boxes represent parameters which are functionally dependent on another parameter, and arrows show the direction of the graph flow for parameters passed at the top level.\label{fig:graph}](media/model_graph.png)

Many more creative uses of the dynamic parameter management system have been
tested, with positive results. Note that `caksade` simply manages how parameters
enter class methods and so the user has complete freedom to design the internals
of the simulator. In fact one could write thin wrapper classes over existing
code bases to quickly allow access to `caskade` parameter management. Our
suggested design flow is to build out a functional programming base for the
package, then use `caksade` classes as wrappers for the functional base to
design a convenient user interface. This design encourages modular development
and is supportive of users who wish to expand functionality at different levels
(core functionality, or interface level). The `caustics` package [@Stone2024]
implements this code design to great effect, \autoref{fig:graph} shows an
example `caskade` graph[^1]. In this graph the redshift parameters of each lens
are linked to ensure consistent evaluation despite the functional backed having
no explicit enforcement of this.

[^1]: visual generated by `graphviz` [@graphviz]

`caskade` has grown beyond its initial development goal of supporting `caustics`
and now includes many parameter related convenience features. There is a utility
to save sampling chains for a simulator into HDF5 format and it is possible to
load the state of the simulator back to a given point. It is now possible to use
`caskade` with `numpy` [@numpy], `jax` [@jax], or `pytorch` [@pytorch] numerical
backends. Parameters may be extracted, or provided as a 1D array, a list, or a
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