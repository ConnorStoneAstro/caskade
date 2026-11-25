import inspect
import functools
from contextlib import ExitStack

from .backend import backend
from .context import ActiveContext, OverrideParam
from .param import Param
from .errors import FillParamsError

__all__ = ("forward", "active_cache")

_NOT_GIVEN = object()  # ``None`` could be a value that the user gives


def forward(method):
    """
    Decorator to define a forward method for a module.

    Manages parameter passing and activation for the decorated method. When
    called, it automatically fills keyword arguments from the module's
    ``Param`` children and handles parameter overrides and active context.

    Parameters
    ----------
    method: (Callable)
        The forward method to be decorated.

    Returns
    -------
    Callable
        The decorated forward method.

    Examples
    --------
    Standard usage of the forward decorator::

        class ExampleSim(Module):
            def __init__(self, a, b, c):
                super().__init__("example_sim")
                self.a = a
                self.b = Param("b", b, shape=())
                self.c = Param("c", c)

            @forward
            def example_func(self, x, b=None):
                return x + self.a + b

        E = ExampleSim(a=1, b=None, c=3)
        print(E.example_func(4, params=[5]))
        # Output: 10

    Notes
    -----
    A call from outside a simulation starts one. Its values are given in one of
    three ways:

    - As the last positional argument, or as the ``params`` keyword. The values
      are handed to the method exactly as they are given.
    - Not at all, with every other argument passed by keyword. The current
      values of the params are used, exactly as they are.
    - Not at all, in a call with no arguments. The current values are used, and
      the method is vectorised when they hold a batch of simulations (below).

    The ``static_params`` keyword gives values for the static params in the
    same way, laid out as ``module.get_values(dynamic=False)`` returns them.
    With both keywords the result depends on nothing but the arguments of the
    call, which is what ``vmap`` or ``jit`` need to know about every value.

    A call made from inside a running simulation is a step of that simulation
    and uses its values. There, a keyword argument with the name of a param
    replaces the value of that param for the duration of the call. In a call
    that starts a simulation this is an error, the value belongs in ``params``.

    **A batch of simulations.** In a call with no arguments the batch
    dimensions are read from the values, as the dimensions in front of the
    declared ``shape`` of each param (a param with no declared shape never has
    any). They are those that ``module.get_values()`` shows in front of its
    last dimension. A dynamic or static param with no batch dimensions is
    shared by the whole batch::

        @forward
        def no_arguments(self, b):  # one more method of ExampleSim
            return 4 + self.a + b

        E.b = [5.0, 6.0, 7.0]  # three simulations
        print(E.no_arguments())
        # Output: [10., 11., 12.], as vmap(E.no_arguments)(E.get_values())

    This uses ``torch.vmap`` or ``jax.vmap``, with ``numpy`` the calls are made
    one at a time. A call with any argument is never vectorised, since there is
    no telling which of the arguments belong to the batch. Go over the values
    yourself in that case::

        vmap(lambda p: E.example_func(4, params=p))(E.get_values())

    **A simulation inside a simulation.** ``params`` and ``static_params`` may
    also be given to a call made from inside a running simulation. The method
    is then run as a simulation of its own with those values, and the outer
    simulation carries on unchanged afterwards. This is how a module is run
    many times with values that the simulation works out as it goes, such as
    one row for each copy of a hierarchical child::

        @forward
        def total(self, x):
            rows = self.child.get_values()  # values of the running simulation
            return vmap(lambda p: self.child.example_func(x, params=p))(rows).sum(0)

    Pass the params through a ``lambda`` as above. ``torch.vmap`` does not go
    over keyword arguments and ``jax.vmap`` goes over all of them. A
    hierarchical child is never a part of the simulation of its parent, so
    every call to it follows the rules for a call from outside: positional
    params, ``params``, or no arguments at all for one result per copy.

    **Compiling with jit.** A compiled function only sees a value change when
    that value is one of its arguments. Every value that the method reads from
    the module is fixed at the time it is compiled, which includes the static
    params, any dynamic param that is not passed in, and plain attributes. To
    compile one method of a simulation, make a function of the values::

        heavy = jax.jit(lambda x, p, s: sim.heavy(x, params=p, static_params=s))
        result = heavy(x, sim.get_values(), sim.get_values(dynamic=False))

    Do not apply ``jit`` to the method in the class body, it would be handed
    the module itself as an argument.
    """

    # Get arguments from function signature
    sig = inspect.signature(method)
    method_params = tuple(sig.parameters.keys())

    @functools.wraps(method)
    def wrapped(self, *args, **kwargs):
        online = self.online
        if online and "params" not in kwargs and "static_params" not in kwargs:
            # The call is a step of the simulation that is running
            with ExitStack() as stack:
                # User override of parameters for single function call
                for kwarg, kval in list(kwargs.items()):
                    if kwarg in self.children and isinstance(self.children[kwarg], Param):
                        stack.enter_context(OverrideParam(self.children[kwarg], kval))
                        del kwargs[kwarg]
                kwargs = {**self.fill_kwargs(method_params), **kwargs}
                return method(self, *args, **kwargs)

        # The call starts a simulation
        params = kwargs.pop("params", _NOT_GIVEN)
        static_params = kwargs.pop("static_params", _NOT_GIVEN)
        if kwargs:
            children = self.children
            for kwarg in kwargs:
                if isinstance(children.get(kwarg), Param):
                    raise FillParamsError(
                        f"Param '{kwarg}' of '{self.name}' was given by keyword in a call to "
                        f"'{method.__name__}' that starts a simulation. The values of a "
                        "simulation are given with its params, a keyword only replaces the "
                        "value of a param in a call made from inside a running simulation."
                    )

        trailing_params = False
        if params is _NOT_GIVEN and not online:
            if args:
                if self.dynamic_params:
                    params = args[-1]
                    args = args[:-1]
                else:  # there may be an empty params as the last argument
                    trailing_params = True
            elif not kwargs and static_params is _NOT_GIVEN:
                # No arguments at all, the current values of the params are
                # used. They may hold a batch of simulations.
                batched = self._batched_values()
                if batched is not None:
                    targets, values, batch_ndim = batched

                    def simulation(values):
                        with ActiveContext(self):
                            for param, value in zip(targets, values):
                                param._active_value = value
                            return method(self, **self.fill_kwargs(method_params))

                    # The method is written for one simulation
                    for _ in range(batch_ndim):
                        simulation = backend.vmap(simulation, [0] * len(values))
                    return simulation(values)

        with ActiveContext(self):
            if params is not _NOT_GIVEN:
                self.fill_params(params)
            if static_params is not _NOT_GIVEN:
                self.fill_params(static_params, dynamic=False)
            self._check_link_dims()
            kwargs = {**self.fill_kwargs(method_params), **kwargs}
            if trailing_params:
                try:
                    sig.bind(self, *args, **kwargs)
                except TypeError:  # user supplied empty params as last arg
                    args = args[:-1]
            return method(self, *args, **kwargs)

    return wrapped


def active_cache(method):
    """
    Caches the first evaluated result of a Module method for the duration of a
    simulation.

    This decorator ensures that an expensive method is executed exactly once per
    active simulation run. Once calculated, subsequent calls to the decorated
    method will return the stored value, ignoring any arguments passed to it.

    Warnings
    --------
    If the method is called multiple times with different arguments in one
    simulation, the cached result will still be returned, which may lead to
    unexpected behavior. Use with caution!

    Notes
    -----
    The results are stored on the ``Module`` instance and are discarded as soon
    as the simulation ends. They may also be discarded by hand with
    ``Module.reset_active_cache()``.

    If you are stacking multiple decorators on a method (such as ``@forward``),
    place ``@active_cache`` outermost (on top), so that a cached result is
    returned without entering the other decorators at all.

    A simulation that is started inside another one (see ``forward``) has a
    cache of its own, it is never handed a result of the outer simulation.

    Examples
    --------
    ::

        class FluxModel(Module):
            def __init__(self, nodes, x, M):
                super().__init__()
                self.nodes = nodes
                self.x = Param("x", x)
                self.M = Param("M", M)

            @active_cache  # Notice active_cache is placed at the top
            @forward
            def compute_intrinsic_sed(self, w, x, M):
                print("Computing SED...")
                return jnp.interp(w, self.nodes, x * M)

            @forward
            def compute_flux(self, wavelengths):
                sed = self.compute_intrinsic_sed(wavelengths)  # Cached after first call
                flux = jnp.sum(sed)
                sed = self.compute_intrinsic_sed(wavelengths)  # Returns cached result, no print
                peak = jnp.max(sed)
                return flux, peak

        model = FluxModel(np.linspace(400, 700, 10), x=1.0, M=np.random.rand(10))

        # Compute flux only calls compute_intrinsic_sed once due to caching
        flux, peak = model.compute_flux(wavelengths)
    """

    @functools.wraps(method)
    def wrapped(self, *args, **kwargs):
        # If not in a simulation, just call the method without caching
        if not self.active:
            return method(self, *args, **kwargs)

        # One result per decorated method, keyed by the decorated method itself
        # so that a method and the parent-class method it overrides never collide
        cache = self._active_cache
        try:
            return cache[wrapped]
        except KeyError:
            pass

        # Run the method, save the output, and return it
        result = cache[wrapped] = method(self, *args, **kwargs)
        return result

    return wrapped
