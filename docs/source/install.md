# Install

## Basic install

``` bash
pip install caskade
```

## Install with jax backend

```bash
pip install caskade[jax]
```

This will simply install `jax`/`jaxlib` along with the other dependencies. It is
always possible to use the `torch` and `numpy` backends since they are core
requirements.

## Install from source

1. Fork the repo on [GitHub](https://github.com/ConnorStoneAstro/caskade)
1. Clone your version of the repo to your machine: `git clone git@github.com:YourGitHubName/caskade.git`
1. Move into the `caksade` git repo: `cd caksade`
1. Install the editable version: `pip install -e .[dev]`
1. Make a new branch: `git checkout -b mynewbranch`