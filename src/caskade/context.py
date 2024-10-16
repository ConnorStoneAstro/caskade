from typing import Union, Mapping, Sequence

from torch import Tensor

from .module import Module


class ActiveContext:
    def __init__(
        self, module: Module, params: Union[Sequence[Tensor], Mapping[str, Tensor], Tensor]
    ):
        self.module = module
        self.params = params

    def __enter__(self):
        self.module.active = True
        self.module.fill_params(self.params)

    def __exit__(self, exc_type, exc_value, traceback):
        self.module.clear_params()
        self.module.active = False
