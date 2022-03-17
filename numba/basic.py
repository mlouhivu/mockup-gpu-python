import numba
import numba.cuda
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

@numba.vectorize(
        ['float32(float32, float32)', 'float64(float64, float64)'],
        target='cuda')
def _sum(a, b):
    return a + b

def sum(a=None, b=None):
    a = numpify(a)
    b = numpify(b)
    c = a + b
    print('Original: c = a + b')
    print('  a={0}'.format(a))
    print('  b={0}'.format(b))
    print('  c={0}'.format(c))
    print('')

    a_gpu = numba.cuda.to_device(a)
    b_gpu = numba.cuda.to_device(b)
    c_gpu = _sum(a_gpu, b_gpu)
    a = a_gpu.copy_to_host()
    b = b_gpu.copy_to_host()
    c = c_gpu.copy_to_host()
    print('GPU: c = a + b')
    print('  a={0}'.format(a))
    print('  b={0}'.format(b))
    print('  c={0}'.format(c))
    print('')

if __name__ == '__main__':
    sum()
