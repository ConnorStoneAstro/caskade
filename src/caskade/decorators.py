import inspect
import functools
from contextlib import ExitStack

from .backend import backend
from .context import ActiveContext, OverrideParam
from .param import Param

__all__ = ("forward",)


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
        if self.active:
            with ExitStack() as stack:
                # User override of parameters for single function call
                used_kwargs = []
                for cname, cval in self.children.items():
                    if not isinstance(cval, Param):
                        continue
                    for kwarg, kval in kwargs.items():
                        if kwarg == cname:
                            stack.enter_context(OverrideParam(cval, kval))
                            used_kwargs.append(kwarg)
                # Remove used kwargs from kwargs
                for kwarg in used_kwargs:
                    kwargs.pop(kwarg)
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
