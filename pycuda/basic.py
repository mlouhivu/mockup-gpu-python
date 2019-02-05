from pycuda import gpuarray
import numpy

def numpify(x, dtype=None):
    if type(x) == numpy.ndarray:
        return x.astype(dtype)
    elif x is None:
        return numpy.random.random(4).astype(dtype)
    try:
        x = numpy.array(x).astype(dtype)
    except TypeError:
        x = numpy.random.random(4).astype(dtype)
    return x

def sum(a=None, b=None):
    a = numpify(a)
    b = numpify(b)
    c = a + b
    print('Original: c = a + b')
    print('  a={0}'.format(a))
    print('  b={0}'.format(b))
    print('  c={0}'.format(c))
    print('')

    a_gpu = gpuarray.to_gpu(a)
    b_gpu = gpuarray.to_gpu(b)
    c_gpu = a_gpu + b_gpu
    print('GPU: c = a + b')
    print('  a={0}'.format(a_gpu))
    print('  b={0}'.format(b_gpu))
    print('  c={0}'.format(c_gpu))
    print('')

