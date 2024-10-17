from .module import Module


class ActiveContext:
    """
    Context manager to activate a module for a simulation. Only inside an
    ActiveContext is it possible to fill/clear the dynamic and live parameters.
    """

    def __init__(self, module: Module):
        self.module = module

    def __enter__(self):
        self.module.active = True

    def __exit__(self, exc_type, exc_value, traceback):
        self.module.clear_params()
        self.module.active = False
