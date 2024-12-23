import inspect
import functools

from .context import ActiveContext

__all__ = ("forward",)


def _get_arguments(method):
    sig = inspect.signature(method)
    return tuple(sig.parameters.keys())


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
    method_params = _get_arguments(method)

    @functools.wraps(method)
    def wrapped(self, *args, **kwargs):
        if self.active:
            kwargs = {**self.fill_kwargs(method_params), **kwargs}
            return method(self, *args, **kwargs)

        # Extract params from the arguments
        if len(self.dynamic_params) == 0:
            params = {}
        elif "params" in kwargs:
            params = kwargs.pop("params")
        elif args:
            params = args[-1]
            args = args[:-1]
        else:
            raise ValueError(
                f"Params must be provided for a top level @forward method. Either by keyword 'method(params=params)' or as the last positional argument 'method(a, b, c, params)'"
            )

        with ActiveContext(self):
            self.fill_params(params)
            kwargs = {**self.fill_kwargs(method_params), **kwargs}
            return method(self, *args, **kwargs)

    return wrapped
