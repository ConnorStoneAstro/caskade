# Install

## Basic install

``` bash
pip install caskade
```

> **Note:** PyTorch is not compatible with Python 3.12 on all systems, you may need 3.9 - 3.11


## Install with jax backend

```bash
pip install caskade[jax]
```

This will simply install `jax`/`jaxlib` along with the other dependencies. It is
always possible to use the `torch` and `numpy` backends since they are core
requirements.

> **Note:** For M1 Mac users there can be compatibility issues with jax/jaxlib. See [discussion here](https://stackoverflow.com/questions/68327863/importing-jax-fails-on-mac-with-m1-chip) and consider installing `jaxlib==0.4.35`.

## Install from source

1. Fork the repo on [GitHub](https://github.com/ConnorStoneAstro/caskade)
1. Clone your version of the repo to your machine: `git clone git@github.com:YourGitHubName/caskade.git`
1. Move into the `caksade` git repo: `cd caksade`
1. Install the editable version: `pip install -e .[dev]`
1. Make a new branch: `git checkout -b mynewbranch`