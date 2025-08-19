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
solutions to those goals are constantly in flux, requiring many refactoring
rounds to meet these changes. The result can be a progressively more unwieldy
interconnected code. Here we present a system, `caskade`, which allows users to
focus on modular components of a simulator, these are small and testable to
ensure robustness. With `caskade` one can turn these modular components into
abstracted blocks that connect to form a powerful simulator. `caskade` manages
the flow of parameter values through such a simulator. 

# Statement of Need

Science is an intrinsically iterative process, and so is the development of
scientific code. Well written code is flexible and scalable while being
performant, this is difficult to achieve in a scientific context where goals
often evolve rapidly, requiring code refactoring. A major aspect of this is the
parameters of a scientific model, the values that will ultimately be sampled
and/or optimized to represent some data. A value may need to alternately be
fixed, then allowed to vary (e.g. in Gibbs sampling). Some parameters that were
initially separate may need to share a value or some functional relationship. In
the extreme, a whole simulator may become a function of a single variable, such
as time. Meta-data such as the uncertainty or valid range of a parameter may
need to be stored. One may need to represent all parameters as a single 1D
vector to interface with external tools, such as emcee [@emcee], scipy.optimize
[@scipy], Pyro [@pyro], dynesty [@dynesty], and torch.Optim [@pytorch]. Large
projects and correspondingly large teams require the ability to break projects
into manageable subtasks which can later be naturally combined into a complete
analysis suite. Most importantly, as all of the above needs change, it is
critical to meaningfully re-use older code without "code debt" or "software
entropy" growing unsustainably.

# Features

The core features of `caskade` are the `Module` base class, `Param` object, and
`forward` decorator. To construct a `caskade` simulator, one subclasses `Module`
then adds some number of `Param` objects as attributes of the class. Any number
of class methods may be decorated with `@forward`, meaning `caskade` will manage
the `Param` arguments of that function. As modules are combined into a larger
simulator, `caskade` builds a directed acyclic graph (DAG) representation. This
allows it to automatically manage the flow (cascade) of parameters through the
simulator and encode arbitrary relationships between them. This is inspired by
the `PyTorch` framework `nn.Module` which allows for near-effortless
construction of machine learning models. We generalize the object oriented
framework to apply to almost any scientific forward model, simulator, analysis
pipeline, and so on; `caskade` manages the flow of parameters through these
models.

Thus the primary capability of `caskade` is the management of `Param` values as
they enter `@forward` methods of `Modules`. Any parameter may be transformed
between "static" and "dynamic" where static has a fixed value and dynamic is
provided at call time. Individual parameters or whole branches of the DAG may be
switched between static and dynamic. Parameters may be synced with arbitrary
functional relationships between them. New parameters may be added dynamically
to allow for sophisticated transformations. For example, an entire simulator may
be turned into a function of time without modifying the underlying simulator by
adding a time parameter and linking appropriately. It is possible to use
`caskade` with `NumPy` [@numpy], `JAX` [@jax], or `PyTorch` [@pytorch] numerical
backends.

![Example `caskade` DAG representation of a gravitational lensing simulator. Ovals represent Modules, boxes represent dynamic parameters, shaded boxes represent fixed parameters, arrow boxes represent parameters which are functionally dependent on another parameter, and thin arrows show the direction of the graph flow for parameters passed at the top level.\label{fig:graph}](media/model_graph.pdf)

Our suggested design flow is to build out a functional programming base for the
package, then use `Module`s as wrappers for the functional base to design a
convenient user interface. This design encourages modular development and is
supportive of users who wish to expand functionality at different levels. The
`caustics` package [@Stone2024] implements this code design to great effect.
\autoref{fig:graph} shows an example `caskade` graph[^1] from `caustics`. In
this graph the redshift parameters of each lens are linked to ensure consistent
evaluation despite the functional backed having no explicit enforcement of this.
See also that all of the lens objects (`ExternalShear`, `SIE`, and
`SinglePlane`) point to a single cosmology `Module` and so share the same
cosmological parameters automatically.

[^1]: visual generated by `graphviz` [@graphviz]

# State of the Field

In some ways `caskade` is reminiscent of `Hydra` [@hydra], however `caskade`
focuses on numerical parameters and scientific inference, while `Hydra` focuses
on configuration management and large scale process organization. The two may
even be used in tandem. Another package, `tesseract-core` [@Hafner2025] focuses
more on containerization and distribution of simulations to interface different
ecosystems (`PyTorch` and `JAX` as well as `Python` and `C++`) and on different
compute engines (HPC clusters or in cloud). The `SimFrame` [@Stammler2022]
package shares `caskade`'s modular and extensible core design, though is focused
exclusively on solving differential equations. Encoding the Functional Mockup
Interface standard [@Blochwitz2012] is the `Ecos` package [@Hatledal2025] which
is also designed for building modular simulators though in the more strict FMI
standard which requires auxiliary `.xml` specification files, `caskade` focuses
on lean and active research development which thrives on minimal overhead.
Finally, `PathSim` also shares the `caskade` modular simulator building
framework, though it focuses exclusively on time-domain dynamical systems.
Clearly, many fields of research and development desire such modular
simulation-building frameworks; `caskade` fulfills the role very generally,
though not so abstractly as to require overhead schema or meta-data files. It is
lean and extensible so as to grow with a project.


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