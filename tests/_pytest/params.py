def asarray(a):
    import numpy as np
    return np.asarray(a)

def as_dense_dask_array(a):
    import dask.array as da
    return da.asarray(a)

ARRAY_TYPES = tuple((asarray, as_dense_dask_array))