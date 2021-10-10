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

Reading and writing
~~~~~~~~~~~~~~~~~~~~

.. module:: ehrapy.api
.. currentmodule:: ehrapy.api

.. autosummary::
    :toctree: io

    io.read

    io.write

Encoding
~~~~~~~~

.. currentmodule:: ehrapy.api

.. autosummary::
    :toctree: encode

    encode.encode

Data
~~~~~

.. currentmodule:: ehrapy.api

.. autosummary::
    :toctree: data

    data.mimic_2

Preprocessing
~~~~~~~~~~~~~

Any transformation of the data matrix that is not a tool.
Other than tools, preprocessing steps usually don’t return an easily interpretable annotation, but perform a basic transformation on the data matrix.

.. currentmodule:: ehrapy.api

.. autosummary::
    :toctree: preprocessing

    preprocessing.replace_explicit
    preprocessing.log1p
    preprocessing.pca

Tools
~~~~~

Any transformation of the data matrix that is not preprocessing.
In contrast to a preprocessing function, a tool usually adds an easily interpretable annotation to the data matrix, which can then be visualized with a corresponding plotting function.

.. currentmodule:: ehrapy.api

.. autosummary::
    :toctree: tools

    tools.DeepL

Plotting
~~~~~~~~

The plotting module scanpy.pl largely parallels the tl.* and a few of the pp.* functions.
For most tools and for some preprocessing functions, you’ll find a plotting function with the same name.

.. automodule:: ehrapy.api.plot
   :members:

Command-line interface
-----------------------

.. click:: ehrapy.__main__:ehrapy_cli
   :prog: ehrapy
   :nested: full
