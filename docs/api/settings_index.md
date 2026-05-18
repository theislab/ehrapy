# Settings

```{eval-rst}
.. module:: ehrapy
    :no-index:
```

```{eval-rst}
.. autosummary::
   :toctree: generated

   settings
   settings.override
```

## Dependency Versions

ehrapy is complex software with many dependencies. To ensure a consistent runtime environment you should save all
versions that were used for an analysis. This comes in handy when trying to diagnose issues and to reproduce results.

Call the function via:

```python
import ehrapy as ep

ep.print_versions()
```
