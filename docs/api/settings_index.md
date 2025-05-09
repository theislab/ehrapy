# Settings

```{eval-rst}
.. module:: ehrapy
    :no-index:
```

```{eval-rst}

.. currentmodule:: ehrapy._settings

.. autosummary::
    :toctree: settings
    :nosignatures:

    EhrapyConfig
```

```python
import ehrapy as ep

ep.settings.set_figure_params(dpi=150)
```

## Dependency Versions

ehrapy is complex software with many dependencies. To ensure a consistent runtime environment you should save all
versions that were used for an analysis. This comes in handy when trying to diagnose issues and to reproduce results.

Call the function via:

```python
import ehrapy as ep

ep.print_versions()
```
