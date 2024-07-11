# Since we might check whether an object is an instance of dask.array.Array
# without requiring dask installed in the environment.
# This would become obsolete should dask become a requirement for ehrapy


try:
    import dask.array as da

    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False


def is_dask_array(array):
    if DASK_AVAILABLE:
        return isinstance(array, da.Array)
    else:
        return False
