Usage
=====

API
---

Import the ehrapy API as follows:

.. code:: python

   import ehrapy.api as eh

You can then access the respective modules like:

.. code:: python

   eh.pl.cool_fancy_plot()

.. contents::
    :local:
    :backlinks: none

Preprocessing
~~~~~~~~~~~~~

.. automodule:: ehrapy.api.preprocessing
   :members:

Tools
~~~~~

.. automodule:: ehrapy.api.tools
   :members:

Plotting
~~~~~~~~

.. automodule:: ehrapy.api.plot
   :members:

Command-line interface
-----------------------

.. click:: ehrapy.__main__:ehrapy_cli
   :prog: ehrapy
   :nested: full
