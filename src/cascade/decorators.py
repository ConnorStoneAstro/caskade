import functools

from .context import ActiveContext


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

    @functools.wraps(method)
    def wrapped(self, *args, **kwargs):
        if method.active:
            return method(self, *args, **kwargs)
        with ActiveContext(self):
            return method(self, *args, **kwargs)

    return wrapped
