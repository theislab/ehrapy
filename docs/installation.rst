.. highlight:: shell

============
Installation
============


Stable release
--------------

To install ehrapy, run this command in your terminal:

.. code-block:: console

    $ pip install ehrapy

This is the preferred method to install ehrapy, as it will always install the most recent stable release.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide you through the process.

To install MedCAT/Spacy language models you can run the installation with extra dependency groups like:

.. code-block:: console

    $ pip install ehrapy[en_core_web_md]

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


From sources
------------

The sources for ehrapy can be downloaded from the `Github repo`_.
Please note that you require `poetry`_ to be installed.

You can either clone the public repository:

.. code-block:: console

    $ git clone git://github.com/theislab/ehrapy

Or download the `tarball`_:

.. code-block:: console

    $ curl -OJL https://github.com/theislab/ehrapy/tarball/master

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ make install

To install MedCAT/Spacy language models you can run the installation with extra dependency groups like:

.. code-block:: console

    $ poetry install -E en_core_web_md

MedCAT/Spacy language models
----------------------------

Available language models are

- en_core_web_md
- en-core-sci-sm  (pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.4.0/en_core_sci_sm-0.4.0.tar.gz)
- en-core-sci-md  (pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.4.0/en_core_sci_md-0.4.0.tar.gz)
- en-core-sci-lg  (pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.4.0/en_core_sci_lg-0.4.0.tar.gz)
-


.. _Github repo: https://github.com/theislab/ehrapy
.. _tarball: https://github.com/theislab/ehrapy/tarball/master
.. _poetry: https://python-poetry.org/
