API Reference
=============

This page documents the public API of ``caskade``. The library is organized
around a few core concepts: the computational graph (nodes and modules),
parameters, decorators for defining forward methods, context managers for
controlling evaluation, and a backend abstraction layer.

Core Classes
------------

Module
~~~~~~

.. autoclass:: caskade.Module
   :members:
   :show-inheritance:

Param
~~~~~

.. autoclass:: caskade.Param
   :members:
   :show-inheritance:

Node
~~~~

.. autoclass:: caskade.Node
   :members:
   :show-inheritance:

Decorators
----------

.. autofunction:: caskade.forward

.. autoclass:: caskade.active_cache
   :members:
   :show-inheritance:

Context Managers
----------------

.. autoclass:: caskade.ActiveContext
   :members:

.. autoclass:: caskade.ValidContext
   :members:

.. autoclass:: caskade.OverrideParam
   :members:

Collections
-----------

.. autoclass:: caskade.NodeCollection
   :members:
   :show-inheritance:

.. autoclass:: caskade.NodeList
   :members:
   :show-inheritance:

.. autoclass:: caskade.NodeTuple
   :members:
   :show-inheritance:

Graph Communication
-------------------

.. autoclass:: caskade.Memo
   :members:

Backend
-------

.. autoclass:: caskade.backend.Backend
   :members:

.. autodata:: caskade.backend.backend
Exceptions
----------

.. autoclass:: caskade.CaskadeException
   :show-inheritance:

.. autoclass:: caskade.GraphError
   :show-inheritance:

.. autoclass:: caskade.BackendError
   :show-inheritance:

.. autoclass:: caskade.LinkToAttributeError
   :show-inheritance:

.. autoclass:: caskade.NodeConfigurationError
   :show-inheritance:

.. autoclass:: caskade.ParamConfigurationError
   :show-inheritance:

.. autoclass:: caskade.ParamTypeError
   :show-inheritance:

.. autoclass:: caskade.ActiveStateError
   :show-inheritance:

.. autoclass:: caskade.FillParamsError
   :show-inheritance:

.. autoclass:: caskade.FillParamsArrayError
   :show-inheritance:

.. autoclass:: caskade.FillParamsSequenceError
   :show-inheritance:

.. autoclass:: caskade.FillParamsMappingError
   :show-inheritance:

Warnings
--------

.. autoclass:: caskade.CaskadeWarning
   :show-inheritance:

.. autoclass:: caskade.InvalidValueWarning
   :show-inheritance:

.. autoclass:: caskade.SaveStateWarning
   :show-inheritance:

Utilities
---------

.. automodule:: caskade.utils
   :members:

Testing
-------

.. autofunction:: caskade.test