import numpy as np


def remove_diagonal(arrays):
    numpy_arrays = np.array(arrays)
    n_of_arrays = numpy_arrays.shape[0]
    strided = np.lib.stride_tricks.as_strided
    stride_0, stride_1 = numpy_arrays.strides
    return strided(numpy_arrays.ravel()[1:],
                   shape=(n_of_arrays-1, n_of_arrays),
                   strides=(stride_0+stride_1, stride_1)).reshape(n_of_arrays, -1)
