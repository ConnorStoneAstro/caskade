from typing import Union, Mapping, Sequence

from torch import Tensor

from .module import Module


class ActiveContext:
    def __init__(self, module: Module):
        self.module = module

    def __enter__(self):
        self.module.active = True

    def __exit__(self, exc_type, exc_value, traceback):
        self.module.clear_params()
        self.module.active = False
