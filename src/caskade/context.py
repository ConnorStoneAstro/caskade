from .module import Module
from .param import Param


class ActiveContext:
    """
    Context manager to activate a module for a simulation. Only inside an
    ActiveContext is it possible to fill/clear the dynamic and live parameters.
    """

    def __init__(self, module: Module, active: bool = True):
        self.module = module
        self.active = active

    def __enter__(self):
        self.outer_active = self.module.active
        if self.outer_active and not self.active:
            self.outer_params = list(p.value for p in self.module.dynamic_params)
            self.module.clear_params()
        self.module.active = self.active

    def __exit__(self, exc_type, exc_value, traceback):
        if not self.outer_active and self.active:
            self.module.clear_params()
        self.module.active = self.outer_active
        if self.outer_active and not self.active:
            self.module.fill_params(self.outer_params)


class ValidContext:
    """
    Context manager to set valid values for parameters. Only inside a
    ValidContext will parameters automatically be assumed valid.
    """

    def __init__(self, module: Module):
        self.module = module

    def __enter__(self):
        self.module.valid_context = True

    def __exit__(self, exc_type, exc_value, traceback):
        self.module.valid_context = False


class OverrideParam:
    """
    Context manager to override a parameter value. Only inside an
    OverrideParam will the parameter be set to the new value.
    """

    def __init__(self, param, value):
        self.param = param
        self.value = value

    def __enter__(self):
        # Store the old value
        self.old_values = {str(id(self.param)): self.param._value}
        # Set the new value
        self.param._value = self.value
        # Clear the pointer values as they may have updated
        for node in self.param.parents:
            if isinstance(node, Param) and node.pointer:
                self.old_values[str(id(node))] = node._value
                node._value = None

    def __exit__(self, exc_type, exc_value, traceback):
        # Reset the old value
        self.param._value = self.old_values[str(id(self.param))]
        # Clear the pointer values as they may have updated
        for node in self.param.parents:
            if isinstance(node, Param) and node.pointer:
                node._value = self.old_values[str(id(node))]
