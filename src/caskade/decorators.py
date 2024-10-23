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
                f"Params must be provided for dynamic modules. Expected {len(self.dynamic_params)} params."
            )

        with ActiveContext(self):
            self.fill_params(params)
            kwargs = {**self.fill_kwargs(method_params), **kwargs}
            return method(self, *args, **kwargs)

    return wrapped
