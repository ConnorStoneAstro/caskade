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
        self.init_valid = self.module.valid_context
        self.module.valid_context = True

    def __exit__(self, exc_type, exc_value, traceback):
        self.module.valid_context = self.init_valid


class OverrideParam:
    """
    Context manager to override a parameter value. Only inside an
    OverrideParam will the parameter be set to the new value.
    """

    def __init__(self, param, value):
        self.param = param
        self.value = value

    @staticmethod
    def _collect_old_values(param):
        # Recursively collect the old values for any pointer affected by the override
        old_values = [(param, param._value)]
        for node in param.parents:
            if isinstance(node, Param) and node.pointer:
                old_values += OverrideParam._collect_old_values(node)
                node._value = None
        return old_values

    def __enter__(self):
        # Store the old value(s) of the parameter and any pointers that may need updating
        self.old_values = OverrideParam._collect_old_values(self.param)
        # Set the new value
        self.param._value = self.value

    def __exit__(self, exc_type, exc_value, traceback):
        # Reset the param and pointer values as they were before the override
        for node, value in self.old_values:
            node._value = value
