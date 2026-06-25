# Install

In a recent update we made it so that a basic ``caskade`` install just defaults
to the numpy version. This is part of our quest to make ``caskade`` as easy to
use and nimble as possible. ``torch`` is actually a pretty heavy package to
install (taking a lot of space/time to install and several seconds just to
import), so by fully eliminating it as a requirement, anyone with the numpy or
jax versions don't need to wait for an import they wont use.

## Basic install

To get the ``numpy`` version, just directly pip install:

``` bash
pip install caskade
```


## Install with PyTorch backend

```bash
pip install caskade[torch]
```

This will simply install ``torch`` along with ``numpy``. You can also always just
pip install ``torch`` yourself after a basic ``caskade`` install.

> **Note:** PyTorch is not compatible with Python 3.12 on all systems, you may need 3.9 - 3.11

## Install with JAX backend

```bash
pip install caskade[jax]
```

This will simply install ``jax`` along with ``numpy``. You can also always just
pip install ``jax`` yourself after a basic ``caskade`` install.

> **Note:** For M1 Mac users there can be compatibility issues with jax/jaxlib. See [discussion here](https://stackoverflow.com/questions/68327863/importing-jax-fails-on-mac-with-m1-chip) and consider installing `jaxlib==0.4.35`.

## Install with PyTorch and JAX

```bash
pip install caskade[torch,jax]
```

All the extra ``[torch,jax]`` bit does is add those to the list of dependencies, so you can always just install them yourself with pip.

## Install from source

1. Fork the repo on [GitHub](https://github.com/ConnorStoneAstro/caskade)
1. Clone your version of the repo to your machine: `git clone git@github.com:YourGitHubName/caskade.git`
1. Move into the `caksade` git repo: `cd caksade`
1. Install the editable version: `pip install -e .[dev]`
1. Make a new branch: `git checkout -b mynewbranch`