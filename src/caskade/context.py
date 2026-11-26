from .base import MemoKind
from .module import Module
from .param import Param


class ActiveContext:
    """
    Context manager to run a simulation from a module.

    Only inside an ``ActiveContext`` is it possible to fill the params with the
    values of a simulation. On exit the values are cleared again.

    A simulation may be started while another one is running, from the same
    module or from one above it. The values, the pointer values and the
    ``@active_cache`` results of the outer simulation are then set aside for
    as long as the inner one runs, and are put back when it has finished. Any
    value that the inner simulation is not given is the one the outer
    simulation holds.

    Parameters
    ----------
    module : Module
        The module to run the simulation from.

    Examples
    --------
    Activate a module, fill parameters, and run a forward pass::

        with ActiveContext(my_module):
            my_module.fill_params(params)
            result = my_module.my_forward(x)
    """

    def __init__(self, module: Module):
        self.module = module

    def __enter__(self):
        module = self.module
        # Below an online module every node is marked for the run already
        self.send = not module.online
        self.nested = module.active
        if self.nested:
            self.state = [p._active_value for p in module.all_params]
            for p in module.pointer_params:
                p._active_value = None  # worked out from the values of the outer simulation
            self.caches = [(m, m._active_cache) for m in module._modules if m._active_cache]
            for m, _ in self.caches:
                m._active_cache = {}
        if self.send:
            module.add_memo(MemoKind.RUN)

    def __exit__(self, exc_type, exc_value, traceback):
        module = self.module
        if not self.nested:
            module.clear_state()
            module.remove_memo(MemoKind.RUN)  # this also drops the cached results
            return
        for p, s in zip(module.all_params, self.state):
            p._active_value = s
        if self.send:
            module.remove_memo(MemoKind.RUN)
        for m in module._modules:
            if m._active_cache:
                m._active_cache = {}
        for m, cache in self.caches:
            m._active_cache = cache


class ValidContext:
    """
    Context manager that transforms parameter values to an unconstrained space.

    Inside a ``ValidContext``, all parameter values are automatically
    mapped into the range ``(-inf, inf)`` via each parameter's
    ``to_valid`` / ``from_valid`` transformations. This is useful when
    interfacing with samplers or optimizers that expect unconstrained
    parameters—any value they propose will be mapped back into the
    parameter's original valid range on exit.

    Parameters
    ----------
    module : Module
        The module whose parameters should be transformed.

    Examples
    --------
    Get unconstrained parameter values for use with an optimizer::

        with ValidContext(my_module):
            unconstrained_params = my_module.get_values()
            # unconstrained_params live in (-inf, inf)
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
    Context manager to override a parameter value.

    Only inside an ``OverrideParam`` will the parameter be set to the new
    value. The original value (and the values of any parent pointer
    parameters) are saved on entry and restored on exit.

    Parameters
    ----------
    param : Param
        The parameter whose value should be temporarily overridden.
    value : object
        The temporary value to assign to *param*.

    Examples
    --------
    Override a parameter inside a ``@forward`` method so that it uses
    ``new_value`` regardless of what was passed via ``params``::

        class MySim(Module):
            def __init__(self):
                super().__init__()
                self.a = Param("a", None)
                self.b = Param("b", None)

            @forward
            def __call__(self, x, a=None, b=None):
                with OverrideParam(self.b, 5.0):
                    # b will always be 5.0 here, ignoring params
                    return x + a + self.b.value
    """

    def __init__(self, param: Param, value):
        self.param = param
        self.value = value

    @staticmethod
    def _collect_old_values(param):
        # Recursively collect the old values for any pointer affected by the override
        old_values = [(param, param._active_value)]
        for node in param.parents:
            if isinstance(node, Param) and node.pointer:
                old_values += OverrideParam._collect_old_values(node)
                node._active_value = None
        return old_values

    def __enter__(self):
        # Store the old value(s) of the parameter and any pointers that may need updating
        self.old_values = OverrideParam._collect_old_values(self.param)
        # Set the new value
        self.param._active_value = self.value

    def __exit__(self, exc_type, exc_value, traceback):
        # Reset the param and pointer values as they were before the override
        for node, value in self.old_values:
            node._active_value = value
