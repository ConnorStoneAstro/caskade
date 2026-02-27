import inspect
import functools
from contextlib import ExitStack

from .context import ActiveContext, OverrideParam
from .param import Param

__all__ = ("forward", "active_cache")


def forward(method):
    """
    Decorator to define a forward method for a module.

    Parameters
    ----------
    method: (Callable)
        The forward method to be decorated.

    Examples
    --------
    Standard usage of the forward decorator::

        class ExampleSim(Module):
            def __init__(self, a, b, c):
                super().__init__("example_sim")
                self.a = a
                self.b = Param("b", b)
                self.c = Param("c", c)

            @forward
            def example_func(self, x, b=None):
                return x + self.a + b

        E = ExampleSim(a=1, b=None, c=3)
        print(E.example_func(4, params=[5]))
        # Output: 10

    Returns
    -------
    Callable
        The decorated forward method.
    """

    # Get arguments from function signature
    sig = inspect.signature(method)
    method_params = tuple(sig.parameters.keys())

    @functools.wraps(method)
    def wrapped(self, *args, **kwargs):
        if self.online:
            with ExitStack() as stack:
                # User override of parameters for single function call
                for kwarg, kval in list(kwargs.items()):
                    if kwarg in self.children and isinstance(self.children[kwarg], Param):
                        stack.enter_context(OverrideParam(self.children[kwarg], kval))
                        del kwargs[kwarg]
                kwargs = {**self.fill_kwargs(method_params), **kwargs}
                return method(self, *args, **kwargs)

        # Extract params from the arguments
        if "params" in kwargs:
            params = kwargs.pop("params")
            with ActiveContext(self):
                self.fill_params(params)
                kwargs = {**self.fill_kwargs(method_params), **kwargs}
                return method(self, *args, **kwargs)
        elif len(self.dynamic_params) == 0:
            with ActiveContext(self):
                kwargs = {**self.fill_kwargs(method_params), **kwargs}
                try:
                    sig.bind(self, *args, **kwargs)
                    empty_params = False
                except TypeError:  # user supplied empty params as last arg
                    empty_params = True

                if empty_params:
                    return method(self, *args[:-1], **kwargs)
                else:
                    return method(self, *args, **kwargs)
        elif args:
            params = args[-1]
            args = args[:-1]
            with ActiveContext(self):
                self.fill_params(params)
                kwargs = {**self.fill_kwargs(method_params), **kwargs}
                return method(self, *args, **kwargs)
        else:
            with ActiveContext(self):
                kwargs = {**self.fill_kwargs(method_params), **kwargs}
                return method(self, *args, **kwargs)

    return wrapped


class active_cache:
    """
    Caches the first evaluated result of a Module method for the duration of a
    simulation.

    This decorator ensures that an expensive method is executed exactly once per
    active simulation run. Once calculated, subsequent calls to the decorated
    method will return the stored value, ignoring any arguments passed to it.

    **WARNING**:
        If the method is called multiple times with different arguments in one
        simulation, the cached result will still be returned, which may lead to
        unexpected behavior. Use with caution!

    Note:
        If you are stacking multiple decorators on a method (such as `@forward`
        or `@jax.jit`), `@active_cache` MUST be the outermost (top) decorator.

    Example::
        class FluxModel(Module):
            def __init__(self, nodes, x, M):
                super().__init__()
                self.nodes = nodes
                self.x = Param("x", x)
                self.M = Param("M", M)

            @active_cache
            @jax.jit  # Notice active_cache is placed at the top
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

    def __init__(self, func):
        self.func = func
        # Unique attribute name to store the single result
        self.cache_attr = f"_active_cache_{func.__name__}"

        # Update wrapper to preserve function metadata
        functools.update_wrapper(self, func)

    def __set_name__(self, owner, name):
        """Injects the reset function when the class is created."""
        if "_cache_attrs" not in owner.__dict__:
            # Start with a copy of any inherited cache attributes from parent classes
            inherited_attrs = set()
            for base in owner.__bases__:
                if hasattr(base, "_cache_attrs"):
                    inherited_attrs.update(base._cache_attrs)

            # Assign the independent set to this specific subclass
            owner._cache_attrs = inherited_attrs
        owner._cache_attrs.add(self.cache_attr)

        if "reset_active_cache" not in owner.__dict__:

            def reset_active_cache(instance):
                """Deletes the cached attributes to force a recalculation."""
                for attr in instance.__class__._cache_attrs:
                    if hasattr(instance, attr):
                        delattr(instance, attr)

            owner.reset_active_cache = reset_active_cache

    def __get__(self, instance, owner):
        if instance is None:
            return self

        @functools.wraps(self.func)
        def wrapper(*args, **kwargs):
            # If not in simulation, just call the function without caching
            if not instance.active:
                return self.func(instance, *args, **kwargs)

            # If we already have the attribute, return it immediately
            if hasattr(instance, self.cache_attr):
                return getattr(instance, self.cache_attr)

            # Run the function, save the output, and return it
            result = self.func(instance, *args, **kwargs)
            setattr(instance, self.cache_attr, result)
            return result

        return wrapper
