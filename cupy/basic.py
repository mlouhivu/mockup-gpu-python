import cupy
import numpy

def numpify(x, dtype=None):
    if x is not None:
        try:
            return numpy.array(x).astype(dtype)
        except TypeError:
            pass
    if dtype == complex:
        return numpy.random.random(4) + numpy.random.random(4) * 1j
    return numpy.random.random(4).astype(dtype)

def sum(a=None, b=None):
    a = numpify(a)
    b = numpify(b)
    c = a + b
    print('Original: c = a + b')
    print('  a={0}'.format(a))
    print('  b={0}'.format(b))
    print('  c={0}'.format(c))
    print('')

    a_gpu = cupy.asarray(a)
    b_gpu = cupy.asarray(b)
    c_gpu = a_gpu + b_gpu
    print('GPU: c = a + b')
    print('  a={0}'.format(a_gpu))
    print('  b={0}'.format(b_gpu))
    print('  c={0}'.format(c_gpu))
    print('')

