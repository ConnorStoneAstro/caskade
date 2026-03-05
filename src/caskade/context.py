from .module import Module
from .param import Param
from .errors import ActiveStateError


class ActiveContext:
    """
    Context manager to activate a module for a simulation.

    Only inside an ``ActiveContext`` is it possible to fill or clear the
    dynamic and live parameters. On entry, the module is marked as active
    (or its current parameter state is saved if already active). On exit,
    the state is restored.

    Parameters
    ----------
    module : Module
        The module to activate for the duration of the context.

    Raises
    ------
    ActiveStateError
        If the module is already running a simulation (``module.online``
        is ``True``).

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
        if self.module.online:
            raise ActiveStateError(f"Module '{self.module.name}' is already running a simulation")
        if self.module.active:
            self.state = list(p._value for p in self.module.all_params)
        else:
            self.state = None
            self.module.add_memo("active")
        self.module.add_memo(f"{self.module.name}_active")

    def __exit__(self, exc_type, exc_value, traceback):
        self.module.clear_state()
        self.module.remove_memo(f"{self.module.name}_active")
        if self.state is not None:
            for p, s in zip(self.module.all_params, self.state):
                p._value = s
        else:
            self.module.remove_memo("active")


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
