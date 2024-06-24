
# Since we might check whether an object is an instance of dask.array.Array
# without requiring dask installed in the environment.
# This would become obsolete should dask become a requirement for ehrapy
try:
    from dask.array import Array as DaskArray
except ImportError:

    class DaskArray:
        pass